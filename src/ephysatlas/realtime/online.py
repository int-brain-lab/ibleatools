"""The online pipeline: localize a probe while its recording is being acquired.

Every ``cadence_s`` of acquisition the pipeline takes the snippet that has just been acquired
(``snippet_s`` of AP and LF, the release's snippet length) and computes its features in a pool of
worker processes with the release code (:func:`ephysatlas.realtime.features.snippet_features`).
Completed snippets are used in acquisition order: after each one the per-channel median over all
snippets so far is denoised along the probe (:func:`~ephysatlas.realtime.features.aggregate_snippets`)
and the probe is localized again with :class:`~ephysatlas.alignment.EphysOnlyLocalizer`
(``feature_set="lf_ap_no_detection"``), starting from the planned trajectory, or without one
with a whole-brain search (:func:`ephysatlas.realtime.whole_brain.localize_without_plan`).

When the workers fall behind (more than ``max_in_flight`` snippets waiting) new snippets are
dropped and counted, so the pipeline always works on recent data. With a source that has
everything available at once (a replay with ``speed=None``) nothing is dropped: snippets are
submitted as workers free up, and the updates are the same as a real-time run that keeps up.

Every update records when its newest snippet finished being acquired and how long the features
and the localization took after that (``latency_s``), the measure of real-time operation.

Probes other than a standard Neuropixels 1.0 (the source's
:class:`~ephysatlas.realtime.probe.ProbeLayout`) are localized one shank at a time: the
shank's channel features are resampled onto the localizer's rows, a probe without an LF band
gets it from its wideband AP band, and the localizer's probe length is scaled to the shank's
recorded span. The planned positions given to the pipeline are those of the rows
(:meth:`~ephysatlas.realtime.probe.ProbeLayout.plan_rows`).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .features import (
    LF_LATENCY_SAMPLES,
    SNIPPET_S,
    aggregate_snippets,
    feature_matrix,
    snippet_samples,
)

# Peak memory of one worker computing a 5 s Neuropixels 1.0 snippet (measured: 3.6 GB).
WORKER_PEAK_GB = 3.6


def suggest_workers(max_workers: int = 8, reserve_gb: float = 4.0) -> int:
    """Workers whose simultaneous peaks fit in the RAM available now (at least 1)."""
    import psutil

    free_gb = psutil.virtual_memory().available / 1e9
    return int(max(1, min(max_workers, (free_gb - reserve_gb) // WORKER_PEAK_GB)))


@dataclass
class OnlineConfig:
    """Settings of the online pipeline.

    Attributes:
        cadence_s: A snippet starts every ``cadence_s`` of acquisition.
        snippet_s: Snippet length (AP and LF).
        first_snippet_s: Acquisition time of the first snippet's start.
        n_workers: Worker processes computing snippet features (each peaks at
            :data:`WORKER_PEAK_GB`; see :func:`suggest_workers`).
        threads_per_worker: Numerical-library threads per worker.
        max_in_flight: Snippets submitted but not finished beyond which new ones are dropped
            (default ``2 * n_workers``); when the source has everything available, at most
            ``n_workers`` are submitted at a time and none is dropped.
        localize_every: Localize again after this many new snippets.
        poll_s: Polling interval of the main loop.
    """

    cadence_s: float = 10.0
    snippet_s: float = SNIPPET_S
    first_snippet_s: float = 0.0
    n_workers: int = 6
    threads_per_worker: int = 2
    max_in_flight: Optional[int] = None
    localize_every: int = 1
    poll_s: float = 0.1


@dataclass
class Snippet:
    """One snippet's bookkeeping (times in seconds; ``*_wall`` since the pipeline started)."""

    index: int
    t_start: float
    acquired_wall: float = np.nan  # its last sample was acquired
    done_wall: float = np.nan  # its features were ready
    compute_s: float = np.nan  # feature time in the worker
    dropped: bool = False  # not computed: workers behind, or the computation failed
    error: str = ""
    features: Optional[pd.DataFrame] = None


@dataclass
class Update:
    """The localization after the first ``n_snippets`` usable snippets."""

    index: int
    n_snippets: int
    acquired_s: float  # acquisition time at the end of the newest snippet used
    wall_s: float  # when the update was ready
    latency_s: float  # wall_s - when the newest snippet finished being acquired
    aggregate_s: float
    localize_s: float
    features: np.ndarray  # [R, 43] in row order
    result: object  # AlignmentResult (positions of the localizer's rows)
    snippet_starts: list = field(default_factory=list)
    recorded_xyz: Optional[np.ndarray] = None  # [n, 3] the channels used (source.channels order)


def _init_worker(threads: int) -> None:
    import logging
    import warnings

    from threadpoolctl import threadpool_limits

    threadpool_limits(int(threads))
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")


def _compute_snippet(
    raw_ap, raw_lf, s2v_ap, s2v_lf, fs_ap, fs_lf, geometry, neuropixel_version=1
):
    """Worker: int16 snippets -> volts -> snippet features; returns (table, seconds). Without
    ``raw_lf`` (a wideband probe) the LF band is derived from the AP snippet."""
    from .features import snippet_features
    from .probe import lf_from_wideband

    t0 = time.perf_counter()
    ap = (raw_ap.T * s2v_ap[:, None]).astype(np.float32)
    if raw_lf is None:
        lf, fs_lf = lf_from_wideband(ap, fs_ap)
    else:
        lf = (raw_lf.T * s2v_lf[:, None]).astype(np.float32)
    df = snippet_features(
        ap, lf, fs_ap, fs_lf, geometry, neuropixel_version=neuropixel_version
    )
    return df, time.perf_counter() - t0


class OnlinePipeline:
    """Localize a probe from a source as its data arrive.

    Args:
        source: A source (see :mod:`ephysatlas.realtime.sources`).
        localizer: :class:`~ephysatlas.alignment.EphysOnlyLocalizer` (the real-time feature set).
        planned_xyz: ``[R, 3]`` planned positions of the localizer's rows (m, top first: the
            channels of a standard Neuropixels 1.0, else ``source.layout.plan_rows(...)``), or
            None to search the whole brain at every update (slower; left hemisphere).
        config: :class:`OnlineConfig`.
        on_update: Called with each :class:`Update` as soon as it is ready.
        pid: Insertion id passed to the localizer ('' for a new recording).
    """

    def __init__(
        self,
        source,
        localizer,
        planned_xyz: Optional[np.ndarray],
        config: Optional[OnlineConfig] = None,
        on_update: Optional[Callable[[Update], None]] = None,
        pid: str = "",
    ):
        self.source = source
        self.layout = getattr(source, "layout", None)
        # A probe other than a standard 1.0 gets the localizer with its probe length scaled.
        self.localizer = (
            self.layout.localizer(localizer) if self.layout is not None else localizer
        )
        self.planned_xyz = (
            None if planned_xyz is None else np.asarray(planned_xyz, dtype=np.float64)
        )
        self.config = config or OnlineConfig()
        self.on_update = on_update
        self.pid = str(pid)
        self.snippets: list[Snippet] = []
        self.updates: list[Update] = []

    def _localize(self, used: list, t0: float) -> Update:
        cfg = self.config
        t_agg = time.perf_counter()
        table = aggregate_snippets([s.features for s in used], self.source.geometry)
        if self.layout is not None:
            x = self.layout.feature_rows(table)
        else:
            x = feature_matrix(table, n_channels=self.source.n_channels)
        t_loc = time.perf_counter()
        if self.planned_xyz is None:
            from .whole_brain import localize_without_plan

            result = localize_without_plan(self.localizer, x, pid=self.pid)
        else:
            result = self.localizer.localize(x, self.planned_xyz, pid=self.pid)
        now = time.perf_counter() - t0
        newest = max(used, key=lambda s: s.t_start)
        update = Update(
            index=len(self.updates),
            n_snippets=len(used),
            acquired_s=float(newest.t_start + cfg.snippet_s),
            wall_s=now,
            latency_s=now - newest.acquired_wall,
            aggregate_s=t_loc - t_agg,
            localize_s=time.perf_counter() - t_loc,
            features=x,
            result=result,
            snippet_starts=[float(s.t_start) for s in used],
            recorded_xyz=(
                self.layout.channel_xyz(result.channel_xyz)
                if self.layout is not None
                else None
            ),
        )
        self.updates.append(update)
        if self.on_update is not None:
            self.on_update(update)
        return update

    def run(self) -> list:
        """Run until the source's data are used up (a replay) or :meth:`stop` (a live source)."""
        cfg, src = self.config, self.source
        unlimited = getattr(src, "speed", 1.0) is None
        cap = cfg.n_workers if unlimited else (cfg.max_in_flight or 2 * cfg.n_workers)
        has_lf = bool(getattr(src, "has_lf", True))
        version = self.layout.neuropixel_version if self.layout is not None else 1
        ns_ap, ns_lf = (
            snippet_samples(src.fs_ap, cfg.snippet_s),
            snippet_samples(src.fs_lf, cfg.snippet_s),
        )
        # A snippet is complete once its last AP and LF samples are acquired (a derived LF band
        # comes from the AP samples).
        need_s = (
            max(ns_ap / src.fs_ap, (ns_lf + LF_LATENCY_SAMPLES) / src.fs_lf)
            if has_lf
            else ns_ap / src.fs_ap
        )
        duration = getattr(src, "duration_s", np.inf)
        self._stop = False
        pending, used = {}, []
        next_t, next_use, since_localized = float(cfg.first_snippet_s), 0, 0
        with ProcessPoolExecutor(
            max_workers=cfg.n_workers,
            initializer=_init_worker,
            initargs=(cfg.threads_per_worker,),
        ) as pool:
            src.start()
            t0 = time.perf_counter()
            while True:
                acquired = src.acquired_s()
                while (
                    next_t + need_s <= acquired + 1e-9
                    and next_t + need_s <= duration + 1e-9
                ):
                    if len(pending) >= cap and unlimited:
                        break
                    snip = Snippet(index=len(self.snippets), t_start=next_t)
                    snip.acquired_wall = time.perf_counter() - t0
                    self.snippets.append(snip)
                    next_t += cfg.cadence_s
                    if len(pending) >= cap:
                        snip.dropped = True
                        continue
                    n_ap = int(round(snip.t_start * src.fs_ap))
                    n_lf = int(round(snip.t_start * src.fs_lf)) + LF_LATENCY_SAMPLES
                    future = pool.submit(
                        _compute_snippet,
                        src.read("ap", n_ap, ns_ap),
                        src.read("lf", n_lf, ns_lf) if has_lf else None,
                        src.s2v_ap,
                        src.s2v_lf,
                        src.fs_ap,
                        src.fs_lf,
                        src.geometry,
                        version,
                    )
                    pending[future] = snip
                for future in [f for f in pending if f.done()]:
                    snip = pending.pop(future)
                    snip.done_wall = time.perf_counter() - t0
                    try:
                        snip.features, snip.compute_s = future.result()
                    except Exception as exc:  # noqa: BLE001 - one failed snippet must not stop the run
                        snip.dropped, snip.error = True, repr(exc)[:300]
                # Use completed snippets in acquisition order. A live run localizes once on all
                # of them; a run with everything available localizes after each one, so its
                # updates are those of a real-time run that keeps up.
                while next_use < len(self.snippets):
                    snip = self.snippets[next_use]
                    if snip.dropped:
                        next_use += 1
                        continue
                    if snip.features is None:
                        break
                    used.append(snip)
                    next_use += 1
                    since_localized += 1
                    if unlimited and since_localized >= cfg.localize_every:
                        self._localize(list(used), t0)
                        since_localized = 0
                exhausted = next_t + need_s > duration + 1e-9
                finished = (
                    (exhausted or self._stop)
                    and not pending
                    and next_use == len(self.snippets)
                )
                if (
                    used
                    and since_localized
                    and (since_localized >= cfg.localize_every or finished)
                ):
                    self._localize(list(used), t0)
                    since_localized = 0
                if finished:
                    break
                time.sleep(cfg.poll_s)
            src.stop()
        return self.updates

    def stop(self) -> None:
        """Stop taking new snippets (a live source); :meth:`run` returns once pending work ends."""
        self._stop = True

    # -- outputs ------------------------------------------------------------------------------

    def snippet_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(
                    index=s.index,
                    t_start=s.t_start,
                    dropped=s.dropped,
                    acquired_wall=s.acquired_wall,
                    done_wall=s.done_wall,
                    compute_s=s.compute_s,
                    error=s.error,
                )
                for s in self.snippets
            ]
        )

    def update_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(
                    index=u.index,
                    n_snippets=u.n_snippets,
                    acquired_s=u.acquired_s,
                    wall_s=u.wall_s,
                    latency_s=u.latency_s,
                    aggregate_s=u.aggregate_s,
                    localize_s=u.localize_s,
                    score=float(u.result.diagnostics["score"]),
                )
                for u in self.updates
            ]
        )

    def save(self, out_dir) -> None:
        """Write ``snippets.csv``, ``updates.csv``, ``updates.npz`` (row / trace positions and
        features of every update; for a probe other than a standard 1.0 also the positions of the
        channels used, ``recorded_channel_xyz``, with their stream indices and the rows' heights)
        and ``snippet_features.pqt`` (every snippet's raw features)."""
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.snippet_table().to_csv(out / "snippets.csv", index=False)
        self.update_table().to_csv(out / "updates.csv", index=False)
        if self.updates:
            np.savez_compressed(
                out / "updates.npz",
                channel_xyz=np.stack([u.result.channel_xyz for u in self.updates]),
                trace_xyz=np.stack([u.result.trace_xyz for u in self.updates]),
                features=np.stack([u.features for u in self.updates]),
                planned_xyz=(  # NaN without a plan
                    self.planned_xyz
                    if self.planned_xyz is not None
                    else np.full_like(self.updates[0].result.channel_xyz, np.nan)
                ),
                valid=np.stack([u.result.valid for u in self.updates]),
                **(
                    dict(
                        recorded_channel_xyz=np.stack(
                            [u.recorded_xyz for u in self.updates]
                        ),
                        recorded_channels=self.layout.channels,
                        row_height_um=self.layout.row_height_um,
                        shank=self.layout.shank,
                    )
                    if self.layout is not None
                    else {}
                ),
            )
        done = [s for s in self.snippets if s.features is not None]
        if done:
            pd.concat(
                [s.features.assign(snippet=s.index, t_start=s.t_start) for s in done],
                ignore_index=True,
            ).to_parquet(out / "snippet_features.pqt")
