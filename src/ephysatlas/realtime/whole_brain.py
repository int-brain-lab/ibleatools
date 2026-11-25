"""Localization without a planned trajectory: a whole-brain search.

An extension of :class:`ephysatlas.alignment.EphysOnlyLocalizer` for recordings without a plan:

1. **scan** -- every straight trajectory of the nominal probe length centred on every other node
   of the candidate lattice (400 µm apart), at 7 angles from vertical over 2-35° and every 30°
   of azimuth, tip down, scored by its likelihood with the outside-brain penalty and the
   direction prior (~320,000 trajectories);
2. **candidates** -- the best trajectory of each part of the brain: best first, each one kept if
   its mean bin distance to every one kept so far is at least 1 mm, up to 5;
3. **refinement** -- each candidate is annealed as the localizer anneals from a plan (all
   restarts, angles from vertical within the same 2-35°), under the plan prior without its
   centre term and with the direction prior in place of its direction term
   (:class:`NoPlanPrior`), and one search scale ``c_p`` shared by all, so that their scores
   compare; the lowest score wins.

**Direction prior** (:class:`DirectionPrior`). Without a plan nothing says which way the probe
points, and a probe through the right regions at the wrong angle can explain the features almost
as well as the right one. The plan prior's direction term (a bivariate Student-t of the angle to
the planned axis, fitted ``nu`` and ``s``) becomes a mixture over the true axes ``a_i`` of the
model's TRAINING insertions,

    D(a) = -w log( mean_i exp(-rho_t2(angle(a, a_i), nu, s)) )   (>= 0),

added to the score of every trajectory, in the scan and in the refinement. The axes are cached
with the localizer's fit (``fit_dir/train_axes.npz``, :func:`training_axes`). ``w = 1`` and the
2-35° range were chosen on the validation insertions of a model retrained without the 34 test
mice of LFPformer (Dey et al., NeurIPS 2026), mean channel distance 1,190 -> 1,170 µm. On their
119 test probes the mean went from 1,083 to 963 µm, the median angle between the found and the
true axis from 20° to 8°, and the probes more than 30° off from 25 to 2. ``direction_weight=0``
with ``min_angle_deg=10, max_angle_deg=30, n_thetas=5`` is the search without it (as in the Open
Ephys plugin, ``Source/Localization/Realtime/EphysOnlyLocalizer.cpp``, which has no direction
prior yet).

Without a plan the hemisphere cannot be told: positions are given in the left one.
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from ephysatlas.alignment.ephys_only import (
    _CORNERS,
    PlanPrior,
    ProbeScore,
    TraceParams,
    _to_metres,
    _unit_rows,
    anneal,
    channel_positions,
    joint_angles_deg,
    mad,
    outside_penalty,
    plan_axis_um,
    plan_neighbourhood_traces,
    probe_axis_from_angles,
    rho_t2,
    section_bin_indices,
    stable_seed,
    trace_points,
)
from ephysatlas.alignment.progress import ProgressCallback, report
from ephysatlas.alignment.result import AlignmentResult

AXES_FILE = "train_axes.npz"  # in the localizer's fit_dir (see training_axes)


@dataclass
class WholeBrainConfig:
    """Settings of the whole-brain search (see the module doc)."""

    centre_step_um: float = 400.0
    # Angle from vertical of the scanned and the refined trajectories (None: the localizer's
    # annealing range, 10-30°).
    min_angle_deg: Optional[float] = 2.0
    max_angle_deg: Optional[float] = 35.0
    n_thetas: int = 7
    phi_step_deg: float = 30.0
    direction_weight: float = 1.0  # w of the direction prior; 0 = no direction prior
    n_seeds: int = 5
    min_seed_distance_um: float = 1000.0
    points_per_chunk: int = 60_000  # bin points scored at a time by a thread (memory)
    threads: int = 4  # scan threads (the feature workers of the online pipeline keep the rest)


def training_axes(localizer, dataset=None) -> np.ndarray:
    """``[n, 3]`` unit directions (top to tip, left hemisphere) of the model's TRAINING
    insertions, from their histology-aligned channels (:func:`plan_axis_um` of ``human_xyz``).

    Cached in ``fit_dir/train_axes.npz`` for the localizer's model. Computing them reads the
    feature release (``dataset``, a :class:`~ephysatlas.alignment.data.ChannelDataset`, else
    loaded: ONE credentials), as the localizer's own fit does.
    """
    path = Path(localizer.fit_dir) / AXES_FILE
    commit = str(localizer.model_commit)
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if str(f["model_commit"]) == commit:
                return f["axes"].astype(np.float64)
    if dataset is None:
        from ephysatlas.alignment.data import ChannelDataset
        from ephysatlas.alignment.models import VINTAGE

        cm = localizer.channel_model
        vintage = getattr(cm, "data_vintage", None) or getattr(cm, "vintage", None) or VINTAGE
        dataset = ChannelDataset.load(str(vintage), exclude_pids=[])
    have = set(dataset.pids.astype(str))
    pids = [p for p in map(str, localizer.channel_model.split()["train_pids"]) if p in have]
    if not pids:
        raise RuntimeError("none of the model's training insertions is in the dataset")
    axes = np.array([plan_axis_um(dataset.probe(p)["human_xyz"])[0] for p in pids])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, model_commit=np.asarray(commit), pids=np.asarray(pids), axes=axes)
    return axes


class DirectionPrior:
    """``D(a) = -w log mean_i exp(-rho_t2(angle(a, a_i), nu, s))`` (≥ 0) of probe directions
    ``a``: the plan prior's direction term as a mixture over the axes ``a_i`` (module doc).

    Args:
        axes: ``[n, 3]`` directions (top to tip, left hemisphere), e.g. :func:`training_axes`.
        nu, s: The Student-t of the angle in degrees: the plan prior's ``axis`` parameters.
        weight: ``w``; 0 = no penalty.
    """

    def __init__(self, axes, nu: float, s: float, weight: float = 1.0):
        self.axes = _unit_rows(np.asarray(axes, dtype=np.float64).reshape(-1, 3))
        if not len(self.axes):
            raise ValueError("the direction prior needs at least one axis")
        self.nu, self.s, self.weight = float(nu), float(s), float(weight)

    @classmethod
    def for_localizer(cls, localizer, weight: float = 1.0, dataset=None) -> "DirectionPrior":
        """The prior of ``localizer``: its training axes and its plan prior's angle term."""
        nu, s = localizer.params["prior"]["axis"]
        return cls(training_axes(localizer, dataset), nu, s, weight)

    def __call__(self, directions: np.ndarray) -> np.ndarray:
        """``[N]`` penalties of ``[N, 3]`` directions (top to tip, left hemisphere; any length)."""
        d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        if self.weight == 0.0:
            return np.zeros(len(d))
        cos = np.clip(_unit_rows(d) @ self.axes.T, -1.0, 1.0)
        x = -rho_t2(np.degrees(np.arccos(cos)), self.nu, self.s)  # [N, n]
        m = x.max(axis=1, keepdims=True)
        return -self.weight * (m[:, 0] + np.log(np.exp(x - m).mean(axis=1)))


_DIRECTION_CACHE: dict = {}


def _direction_prior(localizer, weight: float) -> Optional[DirectionPrior]:
    """The localizer's :class:`DirectionPrior` (built once per fit), or None with a warning when
    the training axes are neither cached nor computable here."""
    key = (str(localizer.fit_dir), str(localizer.model_commit), float(weight))
    if key not in _DIRECTION_CACHE:
        try:
            _DIRECTION_CACHE[key] = DirectionPrior.for_localizer(localizer, weight)
        except Exception as e:  # no cached axes and no feature release on this machine
            warnings.warn(
                f"whole-brain search without the direction prior ({e}): put the model's "
                f"{AXES_FILE} in {localizer.fit_dir} (whole_brain.training_axes writes it)",
                stacklevel=3,
            )
            _DIRECTION_CACHE[key] = None
    return _DIRECTION_CACHE[key]


def search_anneal_config(localizer, config: Optional[WholeBrainConfig] = None):
    """The localizer's :class:`~ephysatlas.alignment.ephys_only.AnnealConfig` with the search's
    range of angles from vertical."""
    cfg = config or WholeBrainConfig()
    acfg = localizer.anneal_cfg
    return replace(
        acfg,
        min_angle_deg=acfg.min_angle_deg if cfg.min_angle_deg is None else float(cfg.min_angle_deg),
        max_angle_deg=acfg.max_angle_deg if cfg.max_angle_deg is None else float(cfg.max_angle_deg),
    )


def scan_angles(acfg, config: Optional[WholeBrainConfig] = None) -> list:
    """``[(theta, phi), ...]`` (degrees) of the scanned directions: ``n_thetas`` angles from
    vertical over ``acfg``'s range times every ``phi_step_deg`` of azimuth."""
    cfg = config or WholeBrainConfig()
    lo, hi = float(acfg.min_angle_deg), float(acfg.max_angle_deg)
    n_t = max(1, int(cfg.n_thetas))
    thetas = [0.5 * (lo + hi)] if n_t == 1 else np.linspace(lo, hi, n_t)
    phis = np.arange(0.0, 360.0 - 1e-9, max(float(cfg.phi_step_deg), 1.0))
    return [(float(t), float(p)) for t in thetas for p in phis]


class NoPlanPrior(PlanPrior):
    """The plan prior without a plan: the length term (against the span of the depth bins) and
    the bend terms, no centre to stay close to, and for the direction the
    :class:`DirectionPrior` when one is given (else none)."""

    def __init__(self, params: dict, cfg, direction: Optional[DirectionPrior] = None):
        super().__init__(params, cfg)
        self.direction = direction

    def reference(self, planned_xyz_m, bins) -> dict:
        span = float(np.ptp(bins.centers_um)) if len(bins) > 1 else 1.0
        return dict(
            c_plan=np.zeros(3),
            axis_plan=np.array([0.0, 0.0, -1.0]),
            L_plan=max(span, 1.0),
            degenerate=False,
            span_um=0.0,
            widen=1.0,
            sections=section_bin_indices(bins.centers_um, self.cfg.n_sections),
        )

    def evaluate(self, traces_um: np.ndarray, ref: dict) -> np.ndarray:
        tr = np.asarray(traces_um, dtype=np.float64)
        w = float(ref["widen"])
        ratio = np.linalg.norm(tr[:, -1] - tr[:, 0], axis=1) / ref["L_plan"]
        m_l, s_l = self.p["length"]
        nu_b, s_b = self.p["bend"]
        bends = rho_t2(joint_angles_deg(tr, ref["sections"]), nu_b, s_b).sum(axis=1)
        cost = 0.5 * ((ratio - m_l) / (s_l * w)) ** 2 + bends
        if self.direction is not None:
            cost = cost + self.direction(tr[:, -1] - tr[:, 0])
        return cost


def _locate(grid, pts_um: np.ndarray, out_free_um: float):
    """:meth:`DenseGrid.locate` for finite points, querying the nearest voxel only where it can
    matter: a point whose 8 surrounding nodes are all candidate voxels is within half a cell
    diagonal of one, which the outside penalty does not count (d_nn reported as 0)."""
    q = np.asarray(pts_um, dtype=np.float64)
    u = (q - grid.origin) / grid.step
    b = np.floor(u).astype(np.int64)
    f = u - b
    c = b[:, None, :] + _CORNERS[None, :, :]
    inside = np.all((c >= 0) & (c < grid.shape[None, None, :]), axis=2)
    cc = np.clip(c, 0, grid.shape[None, None, :] - 1)
    vid = np.where(inside, grid.index[cc[..., 0], cc[..., 1], cc[..., 2]], -1)
    wf = np.where(_CORNERS[None, :, :] == 1, f[:, None, :], 1.0 - f[:, None, :]).prod(
        axis=2
    )
    w = np.where(vid >= 0, wf, 0.0)
    tot = w.sum(axis=1)
    d_nn = np.zeros(len(q))
    nearest = np.zeros(len(q), dtype=np.int64)
    need = ~(vid >= 0).all(axis=1)
    if 0.5 * np.sqrt(3.0) * grid.step > out_free_um:
        need[:] = True  # a coarse lattice: the shortcut would not be exact
    if need.any():
        d_nn[need], nearest[need] = grid.tree.query(q[need], k=1)
    empty = tot <= 1e-12
    w = np.where(empty[:, None], 0.0, w / np.where(empty, 1.0, tot)[:, None])
    ids = np.where(vid >= 0, vid, 0)
    if empty.any():
        ids[empty, 0] = nearest[empty]
        w[empty, 0] = 1.0
    return ids, w, d_nn


def whole_brain_seeds(
    localizer,
    prep: dict,
    config: Optional[WholeBrainConfig] = None,
    direction: Optional[DirectionPrior] = None,
) -> tuple[list, int]:
    """``(seeds, n_scanned)``: the starting trajectories of the search (:class:`TraceParams`,
    best first) and how many straight trajectories the scan scored; ``direction`` adds the
    direction prior to every trajectory's score."""
    cfg = config or WholeBrainConfig()
    acfg, scfg = search_anneal_config(localizer, cfg), localizer.score_cfg
    grid, bins = localizer.grid, prep["bins"]
    wg, wx = prep["wg"], prep["wx"]
    nu, T, F = float(localizer.params["nu"]), float(localizer.T), wx.shape[1]

    # Bins along a straight trajectory, relative to its centre (trace_points without bends).
    length = float(
        np.clip(acfg.probe_length_um, acfg.min_probe_length_um, acfg.max_probe_length_um)
    )
    d = np.asarray(bins.centers_um, dtype=np.float64)
    offset = (
        np.clip((d - d.min()) / max(d.max() - d.min(), 1e-6), 0.0, 1.0) * length
        - 0.5 * length
    )
    angles = scan_angles(acfg, cfg)
    dirs = np.array(
        [probe_axis_from_angles(t, p, "negative") for t, p in angles], dtype=np.float64
    )
    rel = dirs[:, None, :] * offset[None, :, None]  # [D, B, 3]
    # The direction prior of each direction (the same for every centre), on the scale of the
    # likelihood over T as in the refinement.
    dpen = (
        direction(rel[:, -1] - rel[:, 0]) if direction is not None else np.zeros(len(dirs))
    )

    # Centres: the candidate voxels on every stride-th lattice node.
    stride = max(1, int(round(cfg.centre_step_um / grid.step)))
    ijk = np.rint((grid.xyz_um - grid.origin) / grid.step).astype(np.int64)
    centres = grid.xyz_um[(ijk % stride == 0).all(axis=1)]
    n_c, n_d, n_b = len(centres), len(dirs), len(offset)

    # Likelihood and outside penalty over T of every straight trajectory, plus its direction
    # prior (the prior's length and bend terms are the same for all of them), in chunks of
    # centres scored on a few threads (numpy and the KD-tree release the GIL for most of it).
    scores = np.empty((n_c, n_d))
    per = max(1, int(cfg.points_per_chunk) // (n_d * n_b))
    half = 0.5 * (nu + F)

    def score_chunk(c0: int) -> None:
        pts = centres[c0 : c0 + per, None, None, :] + rel[None]  # [n, D, B, 3]
        pts[..., 0] = -np.abs(pts[..., 0])
        n = pts.shape[0]
        ids, w, d_nn = _locate(grid, pts.reshape(-1, 3), scfg.out_free_um)
        g = np.einsum("nk,nkf->nf", w, wg[ids]).reshape(n, n_d, n_b, F)
        e = wx[None, None] - g
        lik = (half * np.log1p((e * e).sum(axis=3) / (nu - 2.0))).sum(axis=2)
        out = scfg.lambda_out * outside_penalty(d_nn, scfg).reshape(n, n_d, n_b).sum(
            axis=2
        )
        scores[c0 : c0 + n] = (lik + out) / T + dpen[None, :]

    with ThreadPoolExecutor(max_workers=max(1, int(cfg.threads))) as pool:
        list(pool.map(score_chunk, range(0, n_c, per)))

    # Best first; each one taken if it is at least min_seed_distance_um (mean over bins) from
    # every trajectory already taken: the best trajectory of each part of the brain.
    flat = scores.ravel()
    taken, seeds = [], []
    for idx in np.argsort(flat, kind="stable"):
        if len(seeds) >= cfg.n_seeds or not np.isfinite(flat[idx]):
            break
        ci, di = divmod(int(idx), n_d)
        trace = centres[ci] + rel[di]
        trace[:, 0] = -np.abs(trace[:, 0])
        if any(
            np.linalg.norm(trace - t, axis=1).mean() < cfg.min_seed_distance_um
            for t in taken
        ):
            continue
        taken.append(trace)
        seeds.append(
            TraceParams(
                x=-abs(float(centres[ci, 0])),
                y=float(centres[ci, 1]),
                z=float(centres[ci, 2]),
                theta=angles[di][0],
                phi=angles[di][1],
                length=length,
                bends=np.zeros((int(acfg.n_sections) - 1, 2), dtype=np.float32),
            )
        )
    return seeds, n_c * n_d


def localize_without_plan(
    localizer,
    features: np.ndarray,
    *,
    pid: str = "",
    config: Optional[WholeBrainConfig] = None,
    progress: Optional[ProgressCallback] = None,
    direction: Optional[DirectionPrior] = None,
) -> AlignmentResult:
    """Localize one probe without a planned trajectory (see the module doc).

    Args:
        localizer: :class:`~ephysatlas.alignment.EphysOnlyLocalizer` (any feature set).
        features: ``[C, F]`` recorded channel features as for
            :meth:`~ephysatlas.alignment.EphysOnlyLocalizer.localize`.
        pid: Insertion id ('' for a new recording).
        config: :class:`WholeBrainConfig`.
        progress: ``progress(fraction, message)`` callback.
        direction: The :class:`DirectionPrior` to use; by default the localizer's
            (:meth:`DirectionPrior.for_localizer`) with ``config.direction_weight``, none when that
            is 0 (or, with a warning, when the training axes are not available).

    Returns:
        AlignmentResult: ``method="ephys_only_no_plan"``, positions in the left hemisphere.
        ``diagnostics`` as :meth:`~ephysatlas.alignment.EphysOnlyLocalizer.localize` where they
        apply (``score``, ``c_p``, ``mad_S``, ``n_restarts``, ``theta_deg``, ``phi_deg``,
        ``length_um``, ``n_bins``, ...), and for the search ``n_scanned``, ``scan_s``,
        ``seed_initial_score`` / ``seed_score`` ``[S]`` (each candidate's straight-start and
        annealed score), ``seed_xyz`` ``[S, B, 3]`` (annealed candidates, m, left),
        ``selected_seed``, ``direction_weight`` (0 without the direction prior),
        ``direction_prior`` (its value for the result, part of ``prior``) and
        ``angle_range_deg``.
    """
    t0 = time.time()
    cfg = config or WholeBrainConfig()
    scfg, acfg = localizer.score_cfg, search_anneal_config(localizer, cfg)
    if direction is None and cfg.direction_weight > 0:
        direction = _direction_prior(localizer, cfg.direction_weight)
    recorded = np.asarray(features, dtype=np.float64)
    report(progress, 0.0, "Whole-brain search: preparing the probe")
    prep = localizer._prepare(recorded, str(pid))
    valid, bins = prep["valid"], prep["bins"]
    prior = NoPlanPrior(localizer.params["prior"], scfg, direction)
    ref = prior.reference(None, bins)
    score = ProbeScore(
        localizer.grid,
        prep["wg"],
        prep["wx"],
        float(localizer.params["nu"]),
        localizer.T,
        prior,
        ref,
        scfg,
    )

    report(progress, 0.05, "Whole-brain search: scanning straight trajectories")
    ts = time.time()
    seeds, n_scanned = whole_brain_seeds(localizer, prep, cfg, direction)
    t_scan = time.time() - ts
    if not seeds:
        raise RuntimeError("the whole-brain scan found no trajectory inside the brain")
    starts = [trace_points(s, bins.centers_um, "negative", acfg)[0] for s in seeds]

    # One search scale for all candidates (from trajectories around the best one).
    rng = np.random.default_rng(
        stable_seed("sa_scale_pool", scfg.scale_pool_version, str(pid))
    )
    pool = plan_neighbourhood_traces(starts[0], scfg.scale_pool_size, rng)
    mad_s = mad(score.components(pool)["raw"])
    c_p = scfg.scale_K / mad_s if np.isfinite(mad_s) and mad_s > 1e-12 else 1.0
    score.scale = (
        float(np.clip(c_p, 1e-8, 1e8)) if np.isfinite(c_p) and c_p > 0 else 1.0
    )

    initial, finals = [], []
    for s, (seed, start) in enumerate(zip(seeds, starts)):
        report(
            progress,
            0.3 + 0.6 * s / len(seeds),
            f"Whole-brain search: refining candidate {s + 1} of {len(seeds)}",
        )
        start_score = score(start)
        best = None
        for r in range(localizer.n_restarts):
            res = anneal(
                score,
                seed,
                start,
                start_score,
                bins.centers_um,
                "negative",
                acfg,
                rng_seed=localizer.seed + 1000003 * r + 7919 * s,
            )
            if best is None or res.score < best.score:
                best = res
        initial.append(start_score)
        finals.append(best)
    final_scores = np.array([f.score for f in finals])
    if not np.isfinite(final_scores).any():
        raise RuntimeError("every annealing restart ended with a non-finite score")
    k = int(np.argmin(np.where(np.isfinite(final_scores), final_scores, np.inf)))
    final = finals[k]
    t_search = time.time() - t0

    channel_um = channel_positions(
        final.trace_um, bins.centers_um, len(recorded), scfg.channel_spacing_um
    )
    channel_xyz = _to_metres(channel_um, False)
    report(progress, 0.9, "Predicting features at the inferred positions")
    t1 = time.time()
    predicted_std = localizer.channel_model.predict_std(channel_xyz, str(pid))
    p_good = localizer.channel_model.confidence(recorded, channel_xyz, predicted_std)
    p_good[~valid] = np.nan
    comp = {k_: float(v[0]) for k_, v in score.components(final.trace_um[None]).items()}
    bin_of_channel = np.full(len(recorded), -1, dtype=np.int32)
    for b, rows in enumerate(bins.channel_idx):
        bin_of_channel[rows] = b
    tr = final.trace_um
    diagnostics = dict(
        score=float(final.score),
        temperature_T=float(localizer.T),
        c_p=float(score.scale),
        scale_K=float(scfg.scale_K),
        mad_S=float(mad_s),
        n_restarts=int(localizer.n_restarts),
        likelihood_over_T=comp["lik_over_T"],
        likelihood_sum=comp["lik_sum"],
        outside_penalty=comp["outside"],
        prior=comp["prior"],
        planned_score=np.nan,
        n_bins=len(bins),
        nu=float(localizer.params["nu"]),
        plan_hemisphere="unknown (left shown)",
        theta_deg=float(final.params.theta),
        phi_deg=float(final.params.phi),
        length_um=float(final.params.length),
        bin_centre_um=np.asarray(bins.centers_um, dtype=np.float64),
        bin_channel_index=bin_of_channel,
        n_scanned=int(n_scanned),
        scan_s=float(t_scan),
        seed_initial_score=np.asarray(initial, dtype=np.float64),
        seed_score=final_scores,
        seed_xyz=np.stack([f.trace_um for f in finals]).astype(np.float64) * 1e-6,
        selected_seed=k,
        direction_weight=float(direction.weight) if direction is not None else 0.0,
        direction_prior=(
            float(direction((tr[-1] - tr[0])[None])[0]) if direction is not None else 0.0
        ),
        angle_range_deg=(float(acfg.min_angle_deg), float(acfg.max_angle_deg)),
    )
    report(progress, 1.0, "Whole-brain search done")
    return AlignmentResult(
        method="ephys_only_no_plan",
        pid=str(pid),
        channel_xyz=channel_xyz,
        recorded=recorded,
        predicted_std=predicted_std,
        recorded_std=localizer.channel_model.standardize(recorded),
        p_good=p_good,
        valid=valid,
        trace_xyz=_to_metres(final.trace_um, False),
        feature_names=list(localizer.channel_model.features),
        diagnostics=diagnostics,
        timings=dict(
            scan=t_scan,
            anneal=t_search - t_scan,
            predict=time.time() - t1,
            total=time.time() - t0,
        ),
    )
