"""Localization without a planned trajectory: a whole-brain search.

An extension of :class:`ephysatlas.alignment.EphysOnlyLocalizer` for recordings without a plan,
the same as the Open Ephys plugin's (``Source/Localization/Realtime/EphysOnlyLocalizer.cpp``):

1. **scan** -- every straight trajectory of the nominal probe length centred on every other node
   of the candidate lattice (400 µm apart), at 5 angles from vertical over the annealer's range
   (10-30°) and every 30° of azimuth, tip down, scored by its likelihood with the outside-brain
   penalty (~230,000 trajectories);
2. **candidates** -- the best trajectory of each part of the brain: best first, each one kept if
   its mean bin distance to every one kept so far is at least 1 mm, up to 5;
3. **refinement** -- each candidate is annealed as the localizer anneals from a plan (all
   restarts), under the plan prior without its centre and direction terms (:class:`NoPlanPrior`:
   length and bends only) and one search scale ``c_p`` shared by all, so that their scores
   compare; the lowest score wins.

Without a plan the hemisphere cannot be told: positions are given in the left one.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ephysatlas.alignment.ephys_only import (
    _CORNERS,
    PlanPrior,
    ProbeScore,
    TraceParams,
    _to_metres,
    anneal,
    channel_positions,
    joint_angles_deg,
    mad,
    outside_penalty,
    plan_neighbourhood_traces,
    probe_axis_from_angles,
    rho_t2,
    section_bin_indices,
    stable_seed,
    trace_points,
)
from ephysatlas.alignment.progress import ProgressCallback, report
from ephysatlas.alignment.result import AlignmentResult


@dataclass
class WholeBrainConfig:
    """Settings of the whole-brain search (see the module doc)."""

    centre_step_um: float = 400.0
    n_thetas: int = 5
    phi_step_deg: float = 30.0
    n_seeds: int = 5
    min_seed_distance_um: float = 1000.0
    points_per_chunk: int = 60_000  # bin points scored at a time by a thread (memory)
    threads: int = 4  # scan threads (the feature workers of the online pipeline keep the rest)


class NoPlanPrior(PlanPrior):
    """The plan prior without a plan: the length term (against the span of the depth bins) and
    the bend terms; no centre or direction to stay close to."""

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
        return 0.5 * ((ratio - m_l) / (s_l * w)) ** 2 + bends


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
    localizer, prep: dict, config: Optional[WholeBrainConfig] = None
) -> tuple[list, int]:
    """``(seeds, n_scanned)``: the starting trajectories of the search (:class:`TraceParams`,
    best first) and how many straight trajectories the scan scored."""
    cfg = config or WholeBrainConfig()
    acfg, scfg = localizer.anneal_cfg, localizer.score_cfg
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
    lo, hi = float(acfg.min_angle_deg), float(acfg.max_angle_deg)
    n_t = max(1, int(cfg.n_thetas))
    thetas = [0.5 * (lo + hi)] if n_t == 1 else np.linspace(lo, hi, n_t)
    phis = np.arange(0.0, 360.0 - 1e-9, max(float(cfg.phi_step_deg), 1.0))
    angles = [(float(t), float(p)) for t in thetas for p in phis]
    dirs = np.array(
        [probe_axis_from_angles(t, p, "negative") for t, p in angles], dtype=np.float64
    )
    rel = dirs[:, None, :] * offset[None, :, None]  # [D, B, 3]

    # Centres: the candidate voxels on every stride-th lattice node.
    stride = max(1, int(round(cfg.centre_step_um / grid.step)))
    ijk = np.rint((grid.xyz_um - grid.origin) / grid.step).astype(np.int64)
    centres = grid.xyz_um[(ijk % stride == 0).all(axis=1)]
    n_c, n_d, n_b = len(centres), len(dirs), len(offset)

    # Likelihood and outside penalty over T of every straight trajectory (the prior's length and
    # bend terms are the same for all of them), in chunks of centres scored on a few threads
    # (numpy and the KD-tree release the GIL for most of the work).
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
        scores[c0 : c0 + n] = (lik + out) / T

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
) -> AlignmentResult:
    """Localize one probe without a planned trajectory (see the module doc).

    Args:
        localizer: :class:`~ephysatlas.alignment.EphysOnlyLocalizer` (any feature set).
        features: ``[C, F]`` recorded channel features as for
            :meth:`~ephysatlas.alignment.EphysOnlyLocalizer.localize`.
        pid: Insertion id ('' for a new recording).
        config: :class:`WholeBrainConfig`.
        progress: ``progress(fraction, message)`` callback.

    Returns:
        AlignmentResult: ``method="ephys_only_no_plan"``, positions in the left hemisphere.
        ``diagnostics`` as :meth:`~ephysatlas.alignment.EphysOnlyLocalizer.localize` where they
        apply (``score``, ``c_p``, ``mad_S``, ``n_restarts``, ``theta_deg``, ``phi_deg``,
        ``length_um``, ``n_bins``, ...), and for the search ``n_scanned``, ``scan_s``,
        ``seed_initial_score`` / ``seed_score`` ``[S]`` (each candidate's straight-start and
        annealed score), ``seed_xyz`` ``[S, B, 3]`` (annealed candidates, m, left) and
        ``selected_seed``.
    """
    t0 = time.time()
    cfg = config or WholeBrainConfig()
    scfg, acfg = localizer.score_cfg, localizer.anneal_cfg
    recorded = np.asarray(features, dtype=np.float64)
    report(progress, 0.0, "Whole-brain search: preparing the probe")
    prep = localizer._prepare(recorded, str(pid))
    valid, bins = prep["valid"], prep["bins"]
    prior = NoPlanPrior(localizer.params["prior"], scfg)
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
    seeds, n_scanned = whole_brain_seeds(localizer, prep, cfg)
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
