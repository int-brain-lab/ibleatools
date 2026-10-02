"""Ephys-only localization: place a probe from its recording and its planned trajectory alone.

No histology is used. The recorded channel features are averaged in 100 µm depth bins and the
method searches the (left-hemisphere) brain for the trajectory whose model predictions best
explain the bin profiles, under a prior on how far real insertions end up from their plan. It is
method ``student_t_mahalanobis_plan_posterior`` (research section 5.3) of the localization study,
ported from ``ephys-localization-agents`` and refitted on the released channel model:

- **Features** -- the 30 LF + AP features (by name), clipped to the release percentiles and
  standardised; bin means ``x_b`` over the channels with a signal.
- **Predictions** -- the channel model's standardised predictions ``G`` on a 200 µm lattice of the
  left hemisphere (Cosmos id >= 1), interpolated trilinearly at any point (:class:`DenseGrid`).
- **Likelihood** -- whitened multivariate Student-t of each bin's residual,
  ``m_b = |W (x_b - mu_r - mu(p_b))|^2`` and ``t_b = (nu + F)/2 log(1 + m_b/(nu - 2))``, with
  ``W = (kappa^2 Sigma)^-1/2`` (Ledoit-Wolf ``Sigma`` and ``mu_r`` of TRAIN leave-probe-out
  residuals, ``nu`` by maximum likelihood, ``kappa`` the VALIDATION/TRAIN scale ratio), plus
  ``20 ((d_nn - 300 µm)_+ / 500 µm)^2`` for points away from the lattice.
- **Prior** -- :class:`PlanPrior` (fitted on TRAIN plan-vs-truth offsets): Student-t terms on
  the trajectory centre offset (with the plan's systematic bias), the angle to the planned axis,
  the length ratio and the three bends.
- **Score** -- ``S = sum_b (t_b + out_b) / T + Pi``; the temperature ``T`` is tuned on the
  VALIDATION probes. The search minimises ``c_p S`` with ``c_p = K / MAD(S)`` over 64 seeded
  trajectories around the plan: a per-probe scale that never moves the minimum but makes the
  fixed annealing schedule equally greedy for every probe.
- **Search** -- simulated annealing over a 4-section piecewise-linear trajectory (centre,
  inclination in [10, 30] deg, azimuth, length in [3.5, 4.5] mm, three bends <= 10 deg), started
  from the straight planned trajectory, restarted 10 times with different random streams; the
  best-scoring restart wins.

Fitted parameters and the grid predictions are cached per model release in ``fit_dir``
(:mod:`.ephys_only_fit` fits them). All positions are handled in the left hemisphere, like the
model; the result is returned in the hemisphere of the planned trajectory.
"""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np

from .progress import ProgressCallback, report, sub_progress
from .result import AlignmentResult

# The likelihood features: every LF and AP feature, selected by name (the release order varies).
FEATURE_GROUPS_USED = ("lf", "ap")
# Annealing snapshots kept by ``localize(record_history=True)``: every this many iterations.
HISTORY_EVERY = 10
# Files in the fit directory.
GRID_FILE = "grid.npz"
FIT_FILE = "fit.npz"
SUMMARY_FILE = "fit_summary.json"


class PlannedTrajectoryUnavailable(ValueError):
    """The planned trajectory is a placeholder (all channels at one point) or shorter than 100 µm,
    so the search has nowhere to start."""


@dataclass
class ScoreConfig:
    """The trajectory score. Values from ``ephys-localization-agents`` (``scoring_components``
    unless noted), unchanged by the port except ``scale_K``."""

    # Candidate lattice (localization_common.AtlasGridConfig): 200 µm, left hemisphere, Cosmos >= 1.
    grid_um: float = 200.0
    grid_mapping: str = "Cosmos"
    valid_rid_min: int = 1
    # Depth bins (ChannelScoringConfig): channel depth = row index x 10 µm, 100 µm bins from the
    # first channel with a signal, bins with fewer than 2 such channels dropped.
    channel_spacing_um: float = 10.0
    depth_bin_um: float = 100.0
    min_channels_per_bin: int = 2
    # Outside-lattice penalty lambda_out ((d_nn - free)_+ / scale)^2, inside the 1/T.
    lambda_out: float = 20.0
    out_free_um: float = 300.0
    out_scale_um: float = 500.0
    # Plan prior: plans spanning less than this are "degenerate" and get centre / axis / length
    # scales widened by ``degenerate_prior_widen``; the bends use ``n_sections`` equal-arc sections.
    plan_min_span_um: float = 1000.0
    degenerate_prior_widen: float = 5.0
    n_sections: int = 4
    # Planned trajectories spanning less than this end to end are unusable
    # (localization_diagnostics.planned_trace_is_available).
    min_usable_plan_span_um: float = 100.0
    # Search scale c_p = scale_K / MAD(S) over ``scale_pool_size`` plan-neighbourhood trajectories.
    # The source used c_p = MAD_ref / MAD(S), MAD_ref the MAD of its baseline score on the same
    # pool; scale_K = 0.167 is the median MAD_ref over its test probes (the baseline is not ported).
    scale_pool_size: int = 64
    scale_pool_version: int = 1
    scale_K: float = 0.167
    # Likelihood temperature T; None uses the value tuned on the validation probes.
    temperature_T: Optional[float] = None


@dataclass
class AnnealConfig:
    """Simulated annealing (``localization_common.SimulatedAnnealingConfig`` with the overrides of
    the study's ``experiment_setup.build_loc_cfg``). The length and bend penalties of the source
    annealer have weight 0 in that configuration and are left out."""

    n_iterations: int = 400
    initial_temperature: float = 0.02
    final_temperature: float = 1e-4
    temperature_decay_iters: int = 400
    # Proposal widths, decayed exponentially from init to final over the iterations.
    init_xyz_sigma_um: float = 500.0  # RMS of the 3-D translation
    final_xyz_sigma_um: float = 100.0
    init_angle_sigma_deg: float = 5.0  # inclination theta
    final_angle_sigma_deg: float = 1.0
    init_azimuth_sigma_deg: float = 20.0  # azimuth phi
    final_azimuth_sigma_deg: float = 4.0
    init_length_sigma_um: float = 300.0
    final_length_sigma_um: float = 50.0
    init_joint_angle_sigma_deg: float = 1.8
    final_joint_angle_sigma_deg: float = 0.1
    # Move probabilities: translation, orientation (theta or phi), length, one joint.
    move_probabilities: tuple = (0.30, 0.20, 0.10, 0.40)
    # Trajectory: sections of equal arc length, bend norm cap, inclination from vertical, length.
    n_sections: int = 4
    max_joint_angle_deg: float = 10.0
    min_angle_deg: float = 10.0  # localization_common.NLLTraceConfig.min_angle_deg
    max_angle_deg: float = 30.0
    probe_length_um: float = 3840.0
    min_probe_length_um: float = 3500.0
    max_probe_length_um: float = 4500.0
    # Reheating: after ``reheat_patience`` iterations without improvement (at most ``max_reheats``
    # times) the temperature restarts at max(factor x current, minimum) and the proposal widths are
    # multiplied by ``reheat_sigma_factor``, decaying by ``reheat_decay`` per iteration.
    reheat_patience: int = 80
    max_reheats: int = 1
    reheat_temperature_factor: float = 1.5
    reheat_min_temperature: float = 0.01
    reheat_sigma_factor: float = 1.75
    reheat_decay: float = 0.97
    # A restart stops after this many iterations without a new best.
    early_stop_patience: int = 100


def _configs(overrides: dict) -> tuple[ScoreConfig, AnnealConfig]:
    """The two configurations with ``overrides`` (field name -> value) applied."""
    configs = (ScoreConfig(), AnnealConfig())
    for key, value in overrides.items():
        target = next((c for c in configs if key in {f.name for f in fields(c)}), None)
        if target is None:
            raise TypeError(f"unknown ephys-only setting {key!r}")
        setattr(target, key, value)
    return configs


def default_fit_dir(model_commit: str) -> Path:
    """``~/ephys-atlas/results/alignment/ephys_only_fit/<model_commit>``."""
    from ephysatlas.unit_level_encoder.config import DEFAULT_RESULTS_DIR

    return Path(DEFAULT_RESULTS_DIR) / "alignment" / "ephys_only_fit" / str(model_commit)


def used_feature_names() -> list[str]:
    """The 30 likelihood features (LF then AP)."""
    from ephysatlas.spatial_encoder.utils import FEATURE_GROUPS

    return [name for group in FEATURE_GROUPS_USED for name in FEATURE_GROUPS[group]]


def stable_seed(*parts) -> int:
    """A process-independent RNG seed from ``parts`` (CRC32 of their ``|``-joined text)."""
    return int(zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0xFFFFFFFF)


def mad(values: np.ndarray) -> float:
    """Normal-consistent median absolute deviation of the finite values (NaN if none)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


def mirror_left(xyz: np.ndarray) -> np.ndarray:
    """A copy with x -> -|x| (the model's left hemisphere)."""
    out = np.array(xyz, copy=True)
    out[..., 0] = -np.abs(out[..., 0])
    return out


def valid_positions(xyz: np.ndarray) -> np.ndarray:
    """Positions that are finite and not the all-zero placeholder."""
    xyz = np.asarray(xyz)
    return np.isfinite(xyz).all(axis=-1) & ~np.all(np.nan_to_num(xyz) == 0.0, axis=-1)


# ---------------------------------------------------------------------------------------------
# Recording preprocessing
# ---------------------------------------------------------------------------------------------


def signal_channels(recorded_used: np.ndarray) -> np.ndarray:
    """``[C]`` channels with a signal: finite and not all-zero on the used features."""
    x = np.asarray(recorded_used, dtype=np.float64)
    return np.isfinite(x).all(axis=1) & ~np.all(np.nan_to_num(x) == 0.0, axis=1)


@dataclass
class DepthBins:
    """100 µm depth bins of a probe's channels with a signal (row order, top first).

    Attributes:
        channel_idx: Channel rows of each bin.
        centers_um: ``[B]`` bin centres, µm below the top channel (channel depth = row x spacing).
    """

    channel_idx: list
    centers_um: np.ndarray

    @classmethod
    def from_valid(cls, valid: np.ndarray, cfg: ScoreConfig) -> "DepthBins":
        """Bins as ``localization_common.make_channel_depth_bins``."""
        valid = np.asarray(valid, dtype=bool)
        depth = np.arange(len(valid), dtype=np.float32) * float(cfg.channel_spacing_um)
        if not valid.any():
            raise ValueError("no channel carries a signal")
        d0, d1 = float(depth[valid].min()), float(depth[valid].max())
        edges = np.arange(d0, d1 + cfg.depth_bin_um, cfg.depth_bin_um, dtype=np.float32)
        if len(edges) < 2:
            edges = np.array([d0, d0 + cfg.depth_bin_um], dtype=np.float32)
        bin_id = np.clip(np.digitize(depth, edges) - 1, 0, len(edges) - 2)
        idx, centers = [], []
        for b in range(len(edges) - 1):
            rows = np.flatnonzero((bin_id == b) & valid)
            if len(rows) >= cfg.min_channels_per_bin:
                idx.append(rows)
                centers.append(0.5 * (edges[b] + edges[b + 1]))
        if len(idx) < 2:
            raise ValueError(f"only {len(idx)} depth bin(s) with signal; at least 2 are needed")
        return cls(channel_idx=idx, centers_um=np.asarray(centers, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.channel_idx)

    def means(self, values: np.ndarray) -> np.ndarray:
        """``[B, F]`` NaN-mean of per-channel ``values`` in each bin."""
        v = np.asarray(values)
        return np.stack([np.nanmean(v[rows], axis=0) for rows in self.channel_idx]).astype(
            np.float64
        )

    def mean_xyz_um(self, xyz_m: np.ndarray) -> np.ndarray:
        """``[B, 3]`` mean valid channel position of each bin (µm), NaN where none is valid."""
        xyz = np.asarray(xyz_m, dtype=np.float64) * 1e6
        ok = valid_positions(xyz)
        out = np.full((len(self), 3), np.nan)
        for b, rows in enumerate(self.channel_idx):
            rows = rows[ok[rows]]
            if len(rows):
                out[b] = xyz[rows].mean(axis=0)
        return out


def clip_standardize(recorded: np.ndarray, clip_lo, clip_hi, e_mean, e_std) -> np.ndarray:
    """The release preprocessing: clip to the release percentiles, then standardise (float32)."""
    x = np.clip(np.asarray(recorded, dtype=np.float32), clip_lo[None, :], clip_hi[None, :])
    return ((x - e_mean[None, :]) / np.maximum(e_std[None, :], 1e-6)).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Candidate lattice and trilinear interpolation
# ---------------------------------------------------------------------------------------------


def candidate_grid(brain_atlas, cfg: ScoreConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(xyz_m [V, 3], region_ids [V])``: the lattice nodes in the left hemisphere of the brain.

    As ``localization_common.build_left_hemisphere_valid_voxel_grid_xyz_m``: a ``grid_um`` lattice
    over the atlas bounding box, x < 0, region id (``grid_mapping``) >= ``valid_rid_min``.
    """
    from .geometry import region_ids

    bc = brain_atlas.bc
    lims = [np.asarray(lim, dtype=np.float64) * 1e6 for lim in (bc.xlim, bc.ylim, bc.zlim)]
    (x0, x1), (y0, y1), (z0, z1) = [(float(np.min(v)), float(np.max(v))) for v in lims]
    step = float(cfg.grid_um)
    xs = np.arange(x0, min(0.0, x1) + 0.5 * step, step)
    xs = xs[xs < 0.0]
    ys = np.arange(y0, y1 + 0.5 * step, step)
    zs = np.arange(z0, z1 + 0.5 * step, step)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    xyz_m = (np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]) * 1e-6).astype(np.float32)
    rids = region_ids(brain_atlas, xyz_m, cfg.grid_mapping)
    keep = rids >= int(cfg.valid_rid_min)
    return xyz_m[keep], rids[keep].astype(np.int64)


_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)


class DenseGrid:
    """Trilinear interpolation of voxel values on the regular candidate lattice.

    For a point the 8 surrounding lattice nodes are gathered; nodes that are not candidate voxels
    are dropped and the remaining weights renormalised; if none remains the nearest voxel is used.
    ``d_nn`` is the distance to the nearest candidate voxel (for the outside penalty).
    """

    def __init__(self, xyz_m: np.ndarray, step_um: float = 200.0):
        from scipy.spatial import cKDTree

        xyz = np.asarray(xyz_m, dtype=np.float64) * 1e6
        self.step = float(step_um)
        self.origin = xyz.min(axis=0)
        ijk = np.rint((xyz - self.origin) / self.step).astype(np.int64)
        err = np.abs(self.origin + ijk * self.step - xyz).max()
        if err > 1.0:
            raise ValueError(
                f"candidate voxels are not on a {step_um} µm lattice (error {err:.2f} µm)"
            )
        self.shape = ijk.max(axis=0) + 1
        self.index = np.full(tuple(self.shape), -1, dtype=np.int64)
        self.index[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(len(xyz))
        self.xyz_um = xyz
        self.n_voxels = len(xyz)
        self.tree = cKDTree(xyz)

    def locate(self, pts_um: np.ndarray):
        """``(ids [N, 8], weights [N, 8], d_nn [N])`` of points (µm); weights rows sum to 1."""
        p = np.asarray(pts_um, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(p).all(axis=1)
        q = np.where(finite[:, None], p, self.origin[None, :])
        u = (q - self.origin) / self.step
        b = np.floor(u).astype(np.int64)
        f = u - b
        c = b[:, None, :] + _CORNERS[None, :, :]
        inside = np.all((c >= 0) & (c < self.shape[None, None, :]), axis=2)
        cc = np.clip(c, 0, self.shape[None, None, :] - 1)
        vid = np.where(inside, self.index[cc[..., 0], cc[..., 1], cc[..., 2]], -1)
        wf = np.where(_CORNERS[None, :, :] == 1, f[:, None, :], 1.0 - f[:, None, :]).prod(axis=2)
        w = np.where(vid >= 0, wf, 0.0)
        tot = w.sum(axis=1)
        d_nn, nearest = self.tree.query(q, k=1)
        empty = tot <= 1e-12
        w = np.where(empty[:, None], 0.0, w / np.where(empty, 1.0, tot)[:, None])
        ids = np.where(vid >= 0, vid, 0)
        if empty.any():
            ids[empty, 0] = nearest[empty]
            w[empty, 0] = 1.0
        return ids, w, np.where(finite, d_nn, np.inf)

    @staticmethod
    def interp(values: np.ndarray, ids: np.ndarray, w: np.ndarray) -> np.ndarray:
        """``values [V, F]`` at the located points -> ``[N, F]``."""
        return np.einsum("nk,nkf->nf", w, values[ids])


# ---------------------------------------------------------------------------------------------
# Trajectory geometry (localization_sa)
# ---------------------------------------------------------------------------------------------


def _unit_vector(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, -1.0])
    return v / n


def probe_axis_from_angles(
    theta_deg: float, phi_deg: float, z_sign: str = "negative"
) -> np.ndarray:
    """Unit axis at inclination ``theta`` from the vertical and azimuth ``phi`` (float32)."""
    theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    uz = np.cos(theta)
    u = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            -uz if z_sign == "negative" else uz,
        ],
        dtype=np.float32,
    )
    return u / np.linalg.norm(u)


def clip_joint_bend(bend_uv_deg, max_joint_angle_deg: float) -> np.ndarray:
    """A 2-D joint bend with its norm (the bend angle) capped at ``max_joint_angle_deg``."""
    bend = np.asarray(bend_uv_deg, dtype=np.float64).reshape(2)
    max_angle = max(0.0, float(max_joint_angle_deg))
    norm = float(np.linalg.norm(bend))
    if norm > max_angle and norm > 0.0:
        bend *= max_angle / norm
    return bend.astype(np.float32)


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``np.cross`` of two 3-vectors: the same products and differences, without its overhead."""
    return np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    )


def _bend_direction(direction, bend_uv_deg, max_joint_angle_deg: float) -> np.ndarray:
    """``direction`` rotated by the 2-D bend expressed in a local perpendicular basis."""
    d = _unit_vector(direction)
    bend = clip_joint_bend(bend_uv_deg, max_joint_angle_deg).astype(np.float64)
    angle = float(np.linalg.norm(bend))
    if angle <= 1e-12:
        return d.astype(np.float32)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(d, reference))) > 0.90:
        reference = np.array([1.0, 0.0, 0.0])
    u = _unit_vector(_cross3(d, reference))
    v = _unit_vector(_cross3(u, d))
    tangent = _unit_vector(bend[0] * u + bend[1] * v)
    rad = np.deg2rad(angle)
    return _unit_vector(np.cos(rad) * d + np.sin(rad) * tangent).astype(np.float32)


@dataclass
class TraceParams:
    """A piecewise-linear trajectory: arc-length midpoint (µm, left), inclination ``theta`` from
    the vertical, azimuth ``phi`` (deg), length (µm), ``[n_sections - 1, 2]`` joint bends (deg)."""

    x: float
    y: float
    z: float
    theta: float
    phi: float
    length: float
    bends: np.ndarray


def trace_points(
    params: TraceParams, depth_centers_um: np.ndarray, z_sign: str, cfg: AnnealConfig
) -> tuple[np.ndarray, np.ndarray]:
    """``(points [B, 3] µm, clipped bends)``: the bin positions along a trajectory.

    As ``localization_sa.trace_from_probe_params``: ``n_sections`` sections of equal arc length,
    each joint turning the direction by its (capped) 2-D bend; the bin depth centres are mapped
    linearly onto the arc length, top bin at the start; the points are mirrored to the left.
    """
    n = max(1, int(cfg.n_sections))
    bends = np.asarray(params.bends, dtype=np.float32).reshape(n - 1, 2)
    bends = (
        np.stack([clip_joint_bend(b, cfg.max_joint_angle_deg) for b in bends])
        if n > 1
        else np.zeros((0, 2), dtype=np.float32)
    )
    directions = [probe_axis_from_angles(params.theta, params.phi, z_sign)]
    for j in range(n - 1):
        directions.append(_bend_direction(directions[-1], bends[j], cfg.max_joint_angle_deg))
    directions = np.asarray(directions, dtype=np.float32)
    length = max(float(params.length), 1e-6)
    seg = length / float(n)
    vertices = np.empty((n + 1, 3), dtype=np.float32)
    vertices[0] = np.array([-abs(params.x), params.y, params.z], dtype=np.float32)
    remaining, section = 0.5 * length, 0
    while remaining > 1e-6 and section < n:  # walk back half the length from the midpoint
        step = min(seg, remaining)
        vertices[0] -= step * directions[section]
        remaining -= step
        section += 1
    for s in range(n):
        vertices[s + 1] = vertices[s] + seg * directions[s]
    depth = np.asarray(depth_centers_um, dtype=np.float32)
    d0, d1 = float(np.nanmin(depth)), float(np.nanmax(depth))
    arc = np.clip((depth - d0) / max(d1 - d0, 1e-6), 0.0, 1.0) * length
    sid = np.minimum((arc / seg).astype(np.int64), n - 1)
    local = arc - sid.astype(np.float32) * seg
    points = vertices[sid] + local[:, None] * directions[sid]
    points[:, 0] = -np.abs(points[:, 0])
    return points.astype(np.float32), bends


def planned_seed(planned_xyz_m: np.ndarray, cfg: AnnealConfig) -> tuple[TraceParams, str]:
    """``(params, z_sign)`` of the straight trajectory fitted to the planned channels.

    As ``localization_sa._planned_trace_seed``: centre = mean of the planned channels (mirrored
    left), axis = their first principal direction oriented top -> tip, inclination clipped to
    [min_angle_deg, max_angle_deg], nominal length, no bends.
    """
    target = mirror_left(np.asarray(planned_xyz_m, dtype=np.float32))
    pts = target[valid_positions(target)] * 1e6
    center = np.nanmean(pts, axis=0).astype(np.float32)
    axis = np.linalg.svd(pts - center[None, :], full_matrices=False)[2][0].astype(np.float32)
    if float(np.dot(axis, pts[-1] - pts[0])) < 0:
        axis = -axis
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    z_sign = "positive" if axis[2] >= 0 else "negative"
    theta = float(np.rad2deg(np.arccos(np.clip(abs(float(axis[2])), 0.0, 1.0))))
    phi = float(np.rad2deg(np.arctan2(float(axis[1]), float(axis[0]))) % 360.0)
    theta = float(np.clip(theta, cfg.min_angle_deg, cfg.max_angle_deg))
    length = float(np.clip(cfg.probe_length_um, cfg.min_probe_length_um, cfg.max_probe_length_um))
    params = TraceParams(
        x=-abs(float(center[0])),
        y=float(center[1]),
        z=float(center[2]),
        theta=theta,
        phi=phi,
        length=length,
        bends=np.zeros((cfg.n_sections - 1, 2), dtype=np.float32),
    )
    return params, z_sign


def check_planned_trajectory(planned_xyz_m: np.ndarray, min_span_um: float = 100.0) -> None:
    """Raise :class:`PlannedTrajectoryUnavailable` for placeholder or too-short plans."""
    xyz = np.asarray(planned_xyz_m, dtype=np.float64)
    ok = valid_positions(xyz)
    if ok.sum() < 2:
        raise PlannedTrajectoryUnavailable(
            f"planned trajectory unavailable: {int(ok.sum())} valid planned channel position(s)"
        )
    pts = xyz[ok] * 1e6
    span = float(np.linalg.norm(pts[-1] - pts[0]))
    if span < min_span_um:
        raise PlannedTrajectoryUnavailable(
            f"planned trajectory unavailable: the planned channels span {span:.0f} µm end to end "
            f"(first at {np.round(pts[0]).tolist()} µm), a placeholder plan; the ephys-only search "
            "starts from the plan and cannot run without one"
        )


def channel_positions(
    trace_um: np.ndarray, depth_centers_um: np.ndarray, n_channels: int, spacing_um: float
) -> np.ndarray:
    """``[C, 3]`` channel positions (µm) along the bin trace, by channel depth (row x spacing).

    Between bin centres as the source (``interpolate_xyz_by_depth``: per-axis linear in depth).
    Channels above the first / below the last bin centre, which the source leaves NaN, are
    extrapolated linearly along the first / last trace step, so every channel gets a position.
    """
    tr = np.asarray(trace_um, dtype=np.float64)
    d = np.asarray(depth_centers_um, dtype=np.float64)
    q = np.arange(int(n_channels), dtype=np.float64) * float(spacing_um)
    out = np.column_stack([np.interp(q, d, tr[:, k]) for k in range(3)])
    top, bottom = q < d[0], q > d[-1]
    out[top] = tr[0] + (q[top] - d[0])[:, None] * (tr[1] - tr[0]) / (d[1] - d[0])
    out[bottom] = tr[-1] + (q[bottom] - d[-1])[:, None] * (tr[-1] - tr[-2]) / (d[-1] - d[-2])
    return out


# ---------------------------------------------------------------------------------------------
# Plan prior (scoring_components 4.5)
# ---------------------------------------------------------------------------------------------


def rho_t(x, nu, loc, s):
    """Univariate Student-t negative log-density, up to a constant."""
    z = (np.asarray(x, dtype=np.float64) - loc) / s
    return 0.5 * (nu + 1.0) * np.log1p(z * z / nu)


def rho_t2(r, nu, s):
    """Isotropic bivariate Student-t negative log-density of a 2-D vector of norm ``r``."""
    r = np.asarray(r, dtype=np.float64)
    return 0.5 * (nu + 2.0) * np.log1p(r * r / (nu * s * s))


def _unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def section_bin_indices(depth_centers_um: np.ndarray, n_sections: int) -> list:
    """Bins of each equal-arc section by depth fraction (boundary bins belong to both sections)."""
    d = np.asarray(depth_centers_um, dtype=np.float64)
    s = (d - d.min()) / max(float(np.ptp(d)), 1e-6)
    return [
        np.flatnonzero((s >= k / n_sections - 1e-9) & (s <= (k + 1) / n_sections + 1e-9))
        for k in range(n_sections)
    ]


def joint_angles_deg(traces_um: np.ndarray, sections: list) -> np.ndarray:
    """``[N, n_sections - 1]`` angles between the principal directions of consecutive sections."""
    tr = np.asarray(traces_um, dtype=np.float64)
    n = tr.shape[0]
    dirs = np.full((n, len(sections), 3), np.nan)
    ok = [k for k, idx in enumerate(sections) if len(idx) >= 2]
    if ok:
        pts = [tr[:, sections[k]] for k in ok]
        cov = np.stack(
            [
                np.einsum("nbi,nbj->nij", c, c)
                for c in (p - p.mean(axis=1, keepdims=True) for p in pts)
            ],
            axis=1,
        )
        vec = np.linalg.eigh(cov)[1][..., -1]  # [N, len(ok), 3], one batched call
        chord = np.stack([p[:, -1] - p[:, 0] for p in pts], axis=1)
        sign = np.where(np.einsum("nki,nki->nk", vec, chord) < 0, -1.0, 1.0)
        dirs[:, ok] = vec * sign[..., None]
    dots = np.einsum("nki,nki->nk", dirs[:, :-1], dirs[:, 1:])
    ang = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return np.where(np.isfinite(ang), ang, 0.0)


def plan_axis_um(planned_xyz_m: np.ndarray) -> tuple[np.ndarray, float]:
    """Planned axis (first principal direction of the valid planned channels, left, oriented in
    channel order) and the planned channels' end-to-end span (µm)."""
    xyz = np.asarray(planned_xyz_m, dtype=np.float64)
    ok = valid_positions(xyz)
    if ok.sum() < 2:
        return np.array([0.0, 0.0, -1.0]), 0.0
    pts = mirror_left(xyz[ok] * 1e6)
    a = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)[2][0]
    if np.dot(a, pts[-1] - pts[0]) < 0:
        a = -a
    return a / np.linalg.norm(a), float(np.linalg.norm(pts[-1] - pts[0]))


class PlanPrior:
    """Pi(trajectory) of research section 4.5, a function of the bin points only.

    ``Pi = sum_k rho_t(dc_k) + rho_t2(axis angle) + 0.5 ((L / L_plan - m) / s)^2``
    ``+ sum_j rho_t2(bend_j)``

    - ``dc`` = mean of the trajectory's bin points - mean of the binned planned channels (left):
      per-axis Student-t with a location, which corrects the plan's systematic bias;
    - axis angle = angle between the trajectory's end-to-end axis and the planned axis;
    - ``L``, ``L_plan`` = end-to-end spans of the trajectory and of the binned plan;
    - bends = angles between the principal directions of the ``n_sections`` depth sections.

    ``params``: ``centre`` {axis: (nu, loc, s)}, ``axis`` (nu, s), ``length`` (median, sd),
    ``bend`` (nu, s), fitted on TRAIN probes by :func:`.ephys_only_fit.fit_plan_prior`.
    """

    def __init__(self, params: dict, cfg: ScoreConfig):
        self.p = params
        self.cfg = cfg

    def reference(self, planned_xyz_m: np.ndarray, bins: DepthBins) -> dict:
        """The plan quantities the prior compares a trajectory with (plan only, no truth)."""
        axis_plan, span = plan_axis_um(planned_xyz_m)
        P = bins.mean_xyz_um(mirror_left(np.asarray(planned_xyz_m, dtype=np.float64)))
        okp = np.isfinite(P).all(axis=1)
        degenerate = bool(span < self.cfg.plan_min_span_um or okp.sum() < 2)
        if degenerate:  # nominal span of the binned channels
            depth = (
                np.array([np.mean(rows) for rows in bins.channel_idx]) * self.cfg.channel_spacing_um
            )
            L_plan = float(np.ptp(depth)) if depth.size > 1 else 3770.0
            c_plan = np.nanmean(P, axis=0) if okp.any() else np.zeros(3)
        else:
            L_plan = float(np.linalg.norm(P[okp][-1] - P[okp][0]))
            c_plan = P[okp].mean(axis=0)
        return dict(
            c_plan=np.asarray(c_plan, dtype=np.float64),
            axis_plan=axis_plan,
            L_plan=max(L_plan, 1.0),
            degenerate=degenerate,
            span_um=span,
            widen=self.cfg.degenerate_prior_widen if degenerate else 1.0,
            sections=section_bin_indices(bins.centers_um, self.cfg.n_sections),
        )

    def evaluate(self, traces_um: np.ndarray, ref: dict) -> np.ndarray:
        """``[N]`` prior cost of ``[N, B, 3]`` trajectories (µm, left)."""
        tr = np.asarray(traces_um, dtype=np.float64)
        w = float(ref["widen"])
        dc = tr.mean(axis=1) - ref["c_plan"][None, :]
        centre = 0.0
        for k, name in enumerate("xyz"):
            nu, loc, s = self.p["centre"][name]
            centre = centre + rho_t(dc[:, k], nu, loc, s * w)
        at = _unit_rows(tr[:, -1] - tr[:, 0])
        angle = np.degrees(np.arccos(np.clip(at @ ref["axis_plan"], -1.0, 1.0)))
        nu_a, s_a = self.p["axis"]
        ratio = np.linalg.norm(tr[:, -1] - tr[:, 0], axis=1) / ref["L_plan"]
        m_l, s_l = self.p["length"]
        nu_b, s_b = self.p["bend"]
        bends = rho_t2(joint_angles_deg(tr, ref["sections"]), nu_b, s_b).sum(axis=1)
        return (
            centre + rho_t2(angle, nu_a, s_a * w) + 0.5 * ((ratio - m_l) / (s_l * w)) ** 2 + bends
        )


# ---------------------------------------------------------------------------------------------
# The trajectory score of one probe
# ---------------------------------------------------------------------------------------------


def outside_penalty(d_nn_um: np.ndarray, cfg: ScoreConfig) -> np.ndarray:
    """``((d_nn - free)_+ / scale)^2`` (unweighted); 1e6 for non-finite distances."""
    d = np.asarray(d_nn_um, dtype=np.float64)
    return np.where(
        np.isfinite(d), (np.maximum(0.0, d - cfg.out_free_um) / cfg.out_scale_um) ** 2, 1e6
    )


class ProbeScore:
    """``c_p S`` of trajectories of one probe (bin points in µm, left hemisphere).

    Args:
        grid: The lattice.
        wg: ``[V, F]`` whitened lattice predictions ``G W^T`` (leave-probe-out for TRAIN probes).
        wx: ``[B, F]`` whitened bin means ``(x_b - mu_r) W^T``.
        nu: Student-t degrees of freedom.
        T: Likelihood temperature.
        prior: The plan prior and ``prior_ref`` its reference for this probe's plan.
    """

    def __init__(
        self,
        grid: DenseGrid,
        wg: np.ndarray,
        wx: np.ndarray,
        nu: float,
        T: float,
        prior: PlanPrior,
        prior_ref: dict,
        cfg: ScoreConfig,
    ):
        self.grid, self.wg, self.wx = grid, wg, wx
        self.nu, self.T = float(nu), float(T)
        self.prior, self.prior_ref, self.cfg = prior, prior_ref, cfg
        self.scale = 1.0

    def components(self, traces_um: np.ndarray) -> dict:
        """``lik_sum``, ``outside``, ``lik_over_T``, ``prior``, ``raw`` (= S) and ``score``
        (= c_p S) of ``[N, B, 3]`` trajectories; non-finite trajectories score inf."""
        tr = np.asarray(traces_um, dtype=np.float64)
        n, b, _ = tr.shape
        ids, w, d_nn = self.grid.locate(tr.reshape(-1, 3))
        e = self.wx[None] - DenseGrid.interp(self.wg, ids, w).reshape(n, b, -1)
        m = (e * e).sum(axis=2)
        lik = (0.5 * (self.nu + self.wx.shape[1]) * np.log1p(m / (self.nu - 2.0))).sum(axis=1)
        outside = self.cfg.lambda_out * outside_penalty(d_nn, self.cfg).reshape(n, b).sum(axis=1)
        lik_over_T = (lik + outside) / self.T
        prior = self.prior.evaluate(tr, self.prior_ref)
        raw = lik_over_T + prior
        raw = np.where(~np.isfinite(tr).all(axis=(1, 2)) | ~np.isfinite(raw), np.inf, raw)
        return dict(
            lik_sum=lik,
            outside=outside,
            lik_over_T=lik_over_T,
            prior=prior,
            raw=raw,
            score=self.scale * raw,
        )

    def __call__(self, trace_um: np.ndarray) -> float:
        return float(self.components(np.asarray(trace_um, dtype=np.float32)[None])["score"][0])


def _rotation_matrices(omega_deg: np.ndarray) -> np.ndarray:
    th = np.deg2rad(np.linalg.norm(omega_deg, axis=1))
    ax = np.where(
        th[:, None] > 1e-12,
        omega_deg / np.maximum(np.linalg.norm(omega_deg, axis=1, keepdims=True), 1e-12),
        0,
    )
    K = np.zeros((len(th), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -ax[:, 2], ax[:, 1], -ax[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = ax[:, 2], -ax[:, 1], ax[:, 0]
    return (
        np.eye(3)[None] + np.sin(th)[:, None, None] * K + (1 - np.cos(th))[:, None, None] * (K @ K)
    )


def plan_neighbourhood_traces(base_um: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``[n, B, 3]`` rigid + stretch perturbations of the trajectory ``base_um`` (family A).

    Shift ~ t3 x (300, 500, 500) µm per axis (norm capped at 2.5 mm); rotation about the centre
    by |t3| x 5 deg about an axis perpendicular to the trajectory; stretch ~ N(1, 0.05) clipped to
    [0.85, 1.2]; mirrored left.
    """
    base = np.asarray(base_um, dtype=np.float64)
    c = base.mean(axis=0)
    axis = np.linalg.svd(base - c, full_matrices=False)[2][0]
    shift = rng.standard_t(3, size=(n, 3)) * np.array([300.0, 500.0, 500.0])
    nr = np.linalg.norm(shift, axis=1, keepdims=True)
    shift = np.where(nr > 2500.0, shift / np.maximum(nr, 1e-12) * 2500.0, shift)
    om = rng.normal(size=(n, 3))
    om -= (om @ axis)[:, None] * axis[None]
    om /= np.maximum(np.linalg.norm(om, axis=1, keepdims=True), 1e-12)
    om *= (np.abs(rng.standard_t(3, size=n)) * 5.0)[:, None]
    stretch = np.clip(1.0 + rng.normal(0.0, 0.05, n), 0.85, 1.2)
    tr = (
        np.einsum("nij,bj->nbi", _rotation_matrices(om), base - c) * stretch[:, None, None]
        + (c + shift)[:, None, :]
    )
    return mirror_left(tr)


# ---------------------------------------------------------------------------------------------
# Simulated annealing (localization_sa.optimize_probe_trace_simulated_annealing)
# ---------------------------------------------------------------------------------------------


def _exp_schedule(start: float, end: float, alpha: float) -> float:
    start, end = float(start), float(end)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if start <= 0.0 or end <= 0.0:
        return start * (1.0 - alpha) + end * alpha
    return start * (end / start) ** alpha


@dataclass
class RestartResult:
    """The best trajectory visited by one annealing restart, every proposal it scored and, when
    requested, snapshots of its state: ``history`` columns ``iteration``, ``temperature``,
    ``score`` (current), ``best_score`` and ``best_trace_um`` ([H, B, 3])."""

    params: TraceParams
    trace_um: np.ndarray
    score: float
    proposal_scores: np.ndarray
    proposal_traces_um: np.ndarray
    history: Optional[dict] = None


def anneal(
    score: ProbeScore,
    seed: TraceParams,
    seed_trace_um: np.ndarray,
    seed_score: float,
    depth_centers_um: np.ndarray,
    z_sign: str,
    cfg: AnnealConfig,
    rng_seed: int,
    progress: Optional[ProgressCallback] = None,
    label: str = "",
    history_every: int = 0,
) -> RestartResult:
    """One simulated-annealing restart from ``seed`` (whose trace and score are given).

    The source algorithm step for step -- schedule, block proposals, reheating, early stop and
    the order of the random draws -- without its per-iteration history records. With
    ``history_every`` > 0 the state after every ``history_every``-th and the last iteration is
    kept.
    """
    rng = np.random.default_rng(int(rng_seed))
    lo, hi = float(cfg.min_angle_deg), float(cfg.max_angle_deg)
    n_joints = int(cfg.n_sections) - 1
    moves = np.asarray(cfg.move_probabilities, dtype=float)
    n_it = int(cfg.n_iterations)
    decay_iters = max(1, int(cfg.temperature_decay_iters))

    def evaluate(p: TraceParams):
        # Bends are capped here and again in trace_points, as in the source.
        bends = (
            np.stack([clip_joint_bend(b, cfg.max_joint_angle_deg) for b in p.bends])
            if n_joints
            else p.bends
        )
        p = TraceParams(
            x=-abs(p.x),
            y=p.y,
            z=p.z,
            theta=float(np.clip(p.theta, lo, hi)),
            phi=p.phi,
            length=float(p.length),
            bends=bends,
        )
        trace, p.bends = trace_points(p, depth_centers_um, z_sign, cfg)
        return p, trace, score(trace)

    cur = best = seed
    cur_score = best_score = float(seed_score)
    best_trace = seed_trace_um
    prop_scores = np.full(n_it, np.nan)
    prop_traces = np.full((n_it, len(depth_centers_um), 3), np.nan, dtype=np.float32)
    last_improve = last_best_improve = 0
    n_reheats, boost = 0, 1.0
    last_reheat, reheat_start, prev_temp = None, np.nan, None
    n_done = 0
    history = {k: [] for k in ("iteration", "temperature", "score", "best_score", "best_trace_um")}
    for it in range(n_it):
        base_temp = _exp_schedule(
            cfg.initial_temperature, cfg.final_temperature, min(1.0, it / decay_iters)
        )
        a = it / max(1, n_it - 1)
        sig_xyz = _exp_schedule(cfg.init_xyz_sigma_um, cfg.final_xyz_sigma_um, a)
        sig_theta = _exp_schedule(cfg.init_angle_sigma_deg, cfg.final_angle_sigma_deg, a)
        sig_phi = _exp_schedule(cfg.init_azimuth_sigma_deg, cfg.final_azimuth_sigma_deg, a)
        sig_len = _exp_schedule(cfg.init_length_sigma_um, cfg.final_length_sigma_um, a)
        sig_joint = _exp_schedule(
            cfg.init_joint_angle_sigma_deg, cfg.final_joint_angle_sigma_deg, a
        )
        if (it - last_improve) >= cfg.reheat_patience and n_reheats < cfg.max_reheats:
            n_reheats += 1
            last_improve = last_reheat = it
            boost = float(cfg.reheat_sigma_factor)
        else:
            boost = max(1.0, boost * cfg.reheat_decay)
        if last_reheat is not None:
            if it == last_reheat:
                previous = prev_temp if prev_temp is not None else base_temp
                reheat_start = max(
                    previous * cfg.reheat_temperature_factor, cfg.reheat_min_temperature
                )
            temp = _exp_schedule(
                reheat_start, cfg.final_temperature, min(1.0, (it - last_reheat) / decay_iters)
            )
        else:
            temp = float(base_temp)
        if boost > 1.0:
            sig_xyz, sig_theta, sig_phi, sig_len, sig_joint = (
                s * boost for s in (sig_xyz, sig_theta, sig_phi, sig_len, sig_joint)
            )

        move = int(rng.choice(4, p=moves))
        prop = TraceParams(
            x=cur.x,
            y=cur.y,
            z=cur.z,
            theta=cur.theta,
            phi=cur.phi,
            length=cur.length,
            bends=cur.bends.copy(),
        )
        if move == 0:  # translation, isotropic with RMS sig_xyz
            step = rng.normal(0.0, sig_xyz / np.sqrt(3.0), size=3)
            prop.x, prop.y, prop.z = (
                -abs(prop.x + float(step[0])),
                prop.y + float(step[1]),
                prop.z + float(step[2]),
            )
        elif move == 1:  # inclination or azimuth
            if rng.random() < 0.5:
                prop.theta = float(np.clip(cur.theta + rng.normal(0.0, sig_theta), lo, hi))
            else:
                prop.phi = (cur.phi + rng.normal(0.0, sig_phi)) % 360.0
        elif move == 2:  # length
            prop.length = float(
                np.clip(
                    cur.length + rng.normal(0.0, sig_len),
                    cfg.min_probe_length_um,
                    cfg.max_probe_length_um,
                )
            )
        elif n_joints:  # one joint
            j = int(rng.integers(0, n_joints))
            prop.bends[j] += rng.normal(0.0, sig_joint, size=2).astype(np.float32)
            prop.bends[j] = clip_joint_bend(prop.bends[j], cfg.max_joint_angle_deg)
        prop, prop_trace, s = evaluate(prop)
        prop_scores[it], prop_traces[it] = s, prop_trace
        n_done = it + 1

        delta = s - cur_score
        p_accept = 1.0 if delta <= 0.0 else float(np.exp(-delta / max(temp, 1e-12)))
        if delta <= 0.0 or rng.random() < p_accept:
            cur, cur_score = prop, s
            if s < best_score:
                best, best_score, best_trace = prop, s, prop_trace
                last_improve = last_best_improve = it
        prev_temp = temp
        if progress is not None and it % 50 == 0:
            report(progress, it / n_it, f"Simulated annealing: {label}, iteration {it}/{n_it}")
        stop = (it - last_best_improve) >= cfg.early_stop_patience
        if history_every > 0 and (it % history_every == 0 or stop or it == n_it - 1):
            for k, v in zip(history, (it, temp, cur_score, best_score, best_trace)):
                history[k].append(v)
        if stop:
            break
    return RestartResult(
        params=best,
        trace_um=np.asarray(best_trace, dtype=np.float32),
        score=float(best_score),
        proposal_scores=prop_scores[:n_done],
        proposal_traces_um=prop_traces[:n_done],
        history={k: np.asarray(v) for k, v in history.items()} if history_every > 0 else None,
    )


# ---------------------------------------------------------------------------------------------
# The localizer
# ---------------------------------------------------------------------------------------------


@dataclass
class _Search:
    """The outcome of the search for one probe (before the model is queried at the result)."""

    valid: np.ndarray
    trace_um: np.ndarray  # [B, 3] left
    channel_um: np.ndarray  # [C, 3] left
    right_hemisphere: bool
    diagnostics: dict
    timings: dict


class EphysOnlyLocalizer:
    """Localize probes without histology (see the module doc for the method).

    Args:
        channel_model: :class:`~ephysatlas.alignment.models.ChannelModel`.
        brain_atlas: ``iblatlas.atlas.AllenAtlas``.
        fit_dir: Where the fitted parameters and the lattice predictions are cached; default
            ``~/ephys-atlas/results/alignment/ephys_only_fit/<model_commit>``. Fitted with
            :meth:`fit` when absent (needs the feature release, i.e. ONE credentials).
        n_restarts: Annealing restarts from the planned trajectory.
        seed: Base seed of the restarts' random streams (restart r uses ``seed + 1000003 r``).
        progress: ``progress(fraction, message)`` callback for the first use, when the lattice
            predictions are computed and the fit runs.
        **overrides: Any field of :class:`ScoreConfig` or :class:`AnnealConfig`, e.g.
            ``temperature_T=20``.
    """

    def __init__(
        self,
        channel_model,
        brain_atlas,
        *,
        fit_dir=None,
        n_restarts: int = 10,
        seed: int = 0,
        progress: Optional[ProgressCallback] = None,
        **overrides,
    ):
        self.channel_model = channel_model
        self.brain_atlas = brain_atlas
        self.n_restarts = int(n_restarts)
        self.seed = int(seed)
        self.score_cfg, self.anneal_cfg = _configs(overrides)
        self.model_commit = str(channel_model.model_commit)
        self.fit_dir = Path(fit_dir) if fit_dir is not None else default_fit_dir(self.model_commit)
        self.feature_names = used_feature_names()
        missing = [f for f in self.feature_names if f not in channel_model.features]
        if missing:
            raise ValueError(f"the channel model does not predict {missing}")
        self.feature_idx = np.array([channel_model.features.index(f) for f in self.feature_names])
        stats = channel_model.encoder.preprocessing_stats()
        self._stats = {
            k: np.asarray(stats[k], dtype=np.float32)[self.feature_idx]
            for k in ("rec_ephys_low_pctl", "rec_ephys_high_pctl", "e_mean", "e_std")
        }
        self.neighbour_radius_um = float(
            (channel_model.encoder.config.get("neighbourhood") or {}).get("radius_um", 500.0)
        )
        self._bank_um = None
        self._load_or_build_grid(sub_progress(progress, 0.0, 0.1))
        self.params = None
        if not self._load_fit():
            self.fit(progress=sub_progress(progress, 0.1, 1.0))
        report(progress, 1.0, "Ephys-only localizer ready")

    # -- cached lattice predictions and fitted parameters -------------------------------------

    def _load_or_build_grid(self, progress: Optional[ProgressCallback] = None) -> None:
        path = self.fit_dir / GRID_FILE
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                if (
                    str(f["model_commit"]) == self.model_commit
                    and f["feature_names"].astype(str).tolist() == self.feature_names
                    and float(f["grid_um"]) == float(self.score_cfg.grid_um)
                ):
                    self.grid_xyz_m = f["xyz_m"]
                    self.grid_region_ids = f["region_ids"]
                    self.grid_pred = f["pred_std"].astype(np.float64)
                    self.grid = DenseGrid(self.grid_xyz_m, self.score_cfg.grid_um)
                    return
        report(progress, 0.0, "Building the candidate lattice")
        xyz_m, rids = candidate_grid(self.brain_atlas, self.score_cfg)
        report(progress, 0.1, f"Predicting features on {len(xyz_m):,} lattice voxels")
        # A pid in no bank: nothing is excluded from the neighbours of a lattice voxel.
        pred = self.channel_model.predict_std(xyz_m, "__grid__")[:, self.feature_idx].astype(
            np.float32
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            model_commit=np.asarray(self.model_commit),
            grid_um=np.float64(self.score_cfg.grid_um),
            feature_names=np.asarray(self.feature_names),
            xyz_m=xyz_m,
            region_ids=rids,
            pred_std=pred,
        )
        self.grid_xyz_m, self.grid_region_ids, self.grid_pred = xyz_m, rids, pred.astype(np.float64)
        self.grid = DenseGrid(xyz_m, self.score_cfg.grid_um)

    def _load_fit(self) -> bool:
        """Load ``fit.npz``; True when it belongs to this model and a temperature is available."""
        path = self.fit_dir / FIT_FILE
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as f:
            if (
                str(f["model_commit"]) != self.model_commit
                or f["feature_names"].astype(str).tolist() != self.feature_names
            ):
                return False
            params = {k: f[k] for k in ("mu_r", "W", "nu", "kappa", "T")}
            params["prior"] = json.loads(str(f["prior_json"]))
        self.set_params(params)
        return np.isfinite(self.T)

    def set_params(self, params: dict) -> None:
        """Use fitted parameters: ``mu_r``, ``W``, ``nu``, ``kappa``, ``prior`` and ``T``."""
        self.params = dict(params)
        self.prior = PlanPrior(self.params["prior"], self.score_cfg)
        self.wg = self.grid_pred @ np.asarray(self.params["W"], dtype=np.float64).T

    @property
    def T(self) -> float:
        """The likelihood temperature in use (the override, else the tuned value)."""
        if self.score_cfg.temperature_T is not None:
            return float(self.score_cfg.temperature_T)
        return float(self.params["T"]) if self.params is not None else float("nan")

    def fit(self, dataset=None, progress: Optional[ProgressCallback] = None) -> dict:
        """Fit the residual model and plan prior on the TRAIN probes, ``kappa`` and the temperature
        on the VALIDATION probes; write them to ``fit_dir`` (see :mod:`.ephys_only_fit`).

        Args:
            dataset: :class:`~ephysatlas.alignment.data.ChannelDataset` (loaded when None).
            progress: ``progress(fraction, message)`` callback.

        Returns:
            dict: The fit summary (also ``fit_dir/fit_summary.json``).
        """
        from .ephys_only_fit import fit_localizer

        summary = fit_localizer(self, dataset, progress)
        if not self._load_fit():
            raise RuntimeError(f"the fit in {self.fit_dir} did not produce a temperature")
        return summary

    # -- leave-probe-out lattice for training probes -----------------------------------------

    def _bank_channels_um(self, pid: str) -> np.ndarray:
        """``[n, 3]`` positions (µm, left) of ``pid``'s channels in the model's neighbour bank."""
        if self._bank_um is None:
            bank = self.channel_model.encoder._neighbor_bank()
            pids, inverse = np.unique(bank["pid"].astype(str), return_inverse=True)
            order = np.argsort(inverse, kind="stable")
            bounds = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(pids)))]
            xyz = np.asarray(bank["xyz"], dtype=np.float64) * 1e6
            self._bank_um = {p: xyz[order[bounds[i] : bounds[i + 1]]] for i, p in enumerate(pids)}
        return self._bank_um.get(str(pid), np.zeros((0, 3)))

    def loo_voxels(self, pid: str) -> np.ndarray:
        """Lattice voxels whose model neighbourhood can include ``pid``'s bank channels."""
        pts = self._bank_channels_um(pid)
        if not len(pts):
            return np.zeros(0, dtype=np.int64)
        hits = self.grid.tree.query_ball_point(pts, r=self.neighbour_radius_um + 1.0)
        return np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits]))

    def lattice_for(self, pid: str) -> np.ndarray:
        """``[V, F]`` whitened lattice predictions for ``pid``: the cached lattice, with the voxels
        near a training probe's own bank channels re-predicted without that probe."""
        vox = self.loo_voxels(pid)
        if not len(vox):
            return self.wg
        pred = self.channel_model.predict_std(self.grid_xyz_m[vox], str(pid))[:, self.feature_idx]
        wg = self.wg.copy()
        wg[vox] = pred @ np.asarray(self.params["W"], dtype=np.float64).T
        return wg

    # -- search ----------------------------------------------------------------------------

    def _prepare(self, recorded: np.ndarray, pid: str, wg: Optional[np.ndarray] = None) -> dict:
        """Channels with a signal, depth bins, whitened bin means ``wx`` and lattice ``wg``."""
        used = np.asarray(recorded, dtype=np.float64)[:, self.feature_idx]
        valid = signal_channels(used)
        bins = DepthBins.from_valid(valid, self.score_cfg)
        st = self._stats
        x = bins.means(
            clip_standardize(
                used, st["rec_ephys_low_pctl"], st["rec_ephys_high_pctl"], st["e_mean"], st["e_std"]
            )
        )
        W = np.asarray(self.params["W"], dtype=np.float64)
        wx = (x - np.asarray(self.params["mu_r"], dtype=np.float64)[None, :]) @ W.T
        return dict(valid=valid, bins=bins, wx=wx, wg=self.lattice_for(pid) if wg is None else wg)

    def bin_likelihood_maps(
        self, features: np.ndarray, *, pid: str = ""
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each depth bin's Student-t likelihood term over the lattice (no prior, before the 1/T).

        Args:
            features: ``[C, F]`` recorded channel features, as for :meth:`localize`.
            pid: Insertion id (a training probe gets its leave-probe-out lattice).

        Returns:
            tuple: ``(grid_xyz_m [V, 3], nll [B, V])``: the lattice voxels (m, left hemisphere) and
            ``t_b(v) = (nu + F)/2 log(1 + |W (x_b - mu_r) - W G_v|^2 / (nu - 2))`` for every bin
            ``b`` (top first, see ``diagnostics["bin_centre_um"]`` of a result) and voxel ``v``.
        """
        prep = self._prepare(np.asarray(features, dtype=np.float64), str(pid))
        wx, wg = prep["wx"], prep["wg"]
        m = (wx**2).sum(axis=1)[:, None] + (wg**2).sum(axis=1)[None, :] - 2.0 * (wx @ wg.T)
        nu = float(self.params["nu"])
        nll = 0.5 * (nu + wx.shape[1]) * np.log1p(np.maximum(m, 0.0) / (nu - 2.0))
        return self.grid_xyz_m.copy(), nll

    def _search(
        self,
        recorded: np.ndarray,
        planned_xyz: np.ndarray,
        pid: str,
        T: float,
        progress: Optional[ProgressCallback] = None,
        wg: Optional[np.ndarray] = None,
        record_history: bool = False,
    ) -> _Search:
        """Run the search for one probe at temperature ``T`` (no model query at the result)."""
        t0 = time.time()
        cfg, acfg = self.score_cfg, self.anneal_cfg
        recorded = np.asarray(recorded, dtype=np.float64)
        planned_xyz = np.asarray(planned_xyz, dtype=np.float64)
        check_planned_trajectory(planned_xyz, cfg.min_usable_plan_span_um)
        report(progress, 0.0, "Preparing the bins and the lattice predictions")
        prep = self._prepare(recorded, pid, wg)
        valid, bins = prep["valid"], prep["bins"]
        prior_ref = self.prior.reference(planned_xyz, bins)
        score = ProbeScore(
            self.grid,
            prep["wg"],
            prep["wx"],
            float(self.params["nu"]),
            T,
            self.prior,
            prior_ref,
            cfg,
        )

        seed, z_sign = planned_seed(planned_xyz, acfg)
        seed_trace, _ = trace_points(seed, bins.centers_um, z_sign, acfg)
        # Search scale c_p = K / MAD(S) over seeded trajectories around the planned seed.
        rng = np.random.default_rng(stable_seed("sa_scale_pool", cfg.scale_pool_version, pid))
        pool = plan_neighbourhood_traces(seed_trace, cfg.scale_pool_size, rng)
        mad_s = mad(score.components(pool)["raw"])
        c_p = cfg.scale_K / mad_s if np.isfinite(mad_s) and mad_s > 1e-12 else 1.0
        score.scale = float(np.clip(c_p, 1e-8, 1e8)) if np.isfinite(c_p) and c_p > 0 else 1.0
        seed_score = score(seed_trace)
        t_prepare = time.time() - t0

        restarts = []
        for r in range(self.n_restarts):
            label = f"restart {r + 1}/{self.n_restarts}"
            report(progress, 0.05 + 0.95 * r / self.n_restarts, f"Simulated annealing: {label}")
            restarts.append(
                anneal(
                    score,
                    seed,
                    seed_trace,
                    seed_score,
                    bins.centers_um,
                    z_sign,
                    acfg,
                    rng_seed=self.seed + 1000003 * r,
                    label=label,
                    history_every=HISTORY_EVERY if record_history else 0,
                    progress=sub_progress(
                        progress,
                        0.05 + 0.95 * r / self.n_restarts,
                        0.05 + 0.95 * (r + 1) / self.n_restarts,
                    ),
                )
            )
        restart_scores = np.array([res.score for res in restarts])
        if not np.isfinite(restart_scores).any():
            raise RuntimeError("every annealing restart ended with a non-finite score")
        best = int(np.argmin(np.where(np.isfinite(restart_scores), restart_scores, np.inf)))
        final = restarts[best]
        comp = {k: float(v[0]) for k, v in score.components(final.trace_um[None]).items()}

        # Every trajectory the search scored: the planned seed (stage -1), then each restart's
        # proposals (stage = restart index), with its mean pointwise distance to the result.
        traces = np.concatenate([seed_trace[None]] + [res.proposal_traces_um for res in restarts])
        cand_score = np.concatenate([[seed_score]] + [res.proposal_scores for res in restarts])
        stage = np.concatenate(
            [[-1]] + [np.full(len(res.proposal_scores), r) for r, res in enumerate(restarts)]
        )
        dist = np.linalg.norm(traces.astype(np.float64) - final.trace_um[None], axis=2).mean(axis=1)
        keep = _subsample(cand_score, 5000)
        bin_of_channel = np.full(len(recorded), -1, dtype=np.int32)
        for b, rows in enumerate(bins.channel_idx):
            bin_of_channel[rows] = b
        right = bool(np.median(planned_xyz[valid_positions(planned_xyz), 0]) > 0)
        diagnostics = dict(
            candidate_score=cand_score[keep],
            candidate_distance_um=dist[keep],
            candidate_stage=stage[keep].astype(np.int32),
            restart_score=restart_scores,
            restart_iterations=np.array(
                [len(res.proposal_scores) for res in restarts], dtype=np.int32
            ),
            init_xyz=seed_trace.astype(np.float64) * 1e-6,
            bin_centre_um=bins.centers_um.astype(np.float64),
            bin_channel_index=bin_of_channel,
            score=float(final.score),
            temperature_T=float(T),
            c_p=float(score.scale),
            scale_K=float(cfg.scale_K),
            mad_S=float(mad_s),
            n_restarts=int(self.n_restarts),
            best_restart=best,
            likelihood_over_T=comp["lik_over_T"],
            likelihood_sum=comp["lik_sum"],
            outside_penalty=comp["outside"],
            prior=comp["prior"],
            planned_score=float(seed_score),
            n_bins=len(bins),
            nu=float(self.params["nu"]),
            plan_degenerate=bool(prior_ref["degenerate"]),
            plan_span_um=float(prior_ref["span_um"]),
            plan_hemisphere="right" if right else "left",
            theta_deg=float(final.params.theta),
            phi_deg=float(final.params.phi),
            length_um=float(final.params.length),
        )
        if record_history:
            hist = [res.history for res in restarts]
            diagnostics.update(
                history_restart=np.concatenate(
                    [np.full(len(h["iteration"]), r) for r, h in enumerate(hist)]
                ).astype(np.int32),
                history_iteration=np.concatenate([h["iteration"] for h in hist]).astype(np.int32),
                history_temperature=np.concatenate([h["temperature"] for h in hist]).astype(
                    np.float64
                ),
                history_score=np.concatenate([h["score"] for h in hist]).astype(np.float64),
                history_best_score=np.concatenate([h["best_score"] for h in hist]).astype(
                    np.float64
                ),
                history_best_xyz=np.concatenate([h["best_trace_um"] for h in hist]).astype(
                    np.float64
                )
                * 1e-6,
            )
        channel_um = channel_positions(
            final.trace_um, bins.centers_um, len(recorded), cfg.channel_spacing_um
        )
        return _Search(
            valid=valid,
            trace_um=final.trace_um.astype(np.float64),
            channel_um=channel_um,
            right_hemisphere=right,
            diagnostics=diagnostics,
            timings=dict(prepare=t_prepare, anneal=time.time() - t0 - t_prepare),
        )

    def localize(
        self,
        features: np.ndarray,
        planned_xyz: np.ndarray,
        *,
        pid: str = "",
        progress: Optional[ProgressCallback] = None,
        record_history: bool = False,
    ) -> AlignmentResult:
        """Localize one probe from its recording and planned trajectory.

        Args:
            features: ``[C, F]`` recorded channel features, feature units, the channel model's
                feature order, row order (top first).
            planned_xyz: ``[C, 3]`` planned channel positions (m), row order.
            pid: Insertion id. A training probe of the model's neighbour bank is localized on a
                leave-probe-out lattice (voxels near its own bank channels re-predicted without it).
            progress: ``progress(fraction, message)`` callback.
            record_history: Also keep snapshots of every annealing restart (every
                ``HISTORY_EVERY`` iterations and the last one).

        Returns:
            AlignmentResult: ``method="ephys_only"``; ``trace_xyz`` the inferred bin trajectory
            ``[B, 3]`` (39 points for a full probe), ``channel_xyz`` every channel along it (those
            beyond the first / last bin centre extrapolated along the trajectory), both in the
            planned trajectory's hemisphere. ``diagnostics`` arrays:

            - ``candidate_score`` / ``candidate_distance_um`` / ``candidate_stage`` ``[n]``: every
              trajectory the search scored (score ``c_p S``; mean pointwise distance to the result,
              µm; restart index, -1 for the planned seed), at most 5000;
            - ``restart_score`` / ``restart_iterations`` ``[n_restarts]``;
            - ``init_xyz`` ``[B, 3]``: the planned seed trajectory (m, left hemisphere);
            - ``bin_centre_um`` ``[B]`` (µm below the top channel) and ``bin_channel_index`` ``[C]``
              (bin of each channel row, -1 when in none);
            - with ``record_history``: ``history_restart``, ``history_iteration``,
              ``history_temperature``, ``history_score`` (current), ``history_best_score`` ``[H]``
              and ``history_best_xyz`` ``[H, B, 3]`` (best-so-far trajectory, m, left hemisphere);

            and scalars ``score``, ``temperature_T``, ``c_p``, ``scale_K``, ``mad_S``,
            ``n_restarts``, ``best_restart``, ``likelihood_over_T``, ``likelihood_sum``,
            ``outside_penalty``, ``prior``, ``planned_score``, ``n_bins``, ``nu``,
            ``plan_degenerate``, ``plan_span_um``, ``plan_hemisphere``, ``theta_deg``,
            ``phi_deg``, ``length_um``.

        Raises:
            PlannedTrajectoryUnavailable: If the plan is a placeholder or spans < 100 µm.
        """
        t0 = time.time()
        recorded = np.asarray(features, dtype=np.float64)
        report(progress, 0.0, "Ephys-only localization: preparing the probe")
        search = self._search(
            recorded,
            planned_xyz,
            str(pid),
            self.T,
            sub_progress(progress, 0.0, 0.9),
            record_history=record_history,
        )
        channel_xyz = _to_metres(search.channel_um, search.right_hemisphere)
        report(progress, 0.9, "Predicting features at the inferred positions")
        t1 = time.time()
        predicted_std = self.channel_model.predict_std(channel_xyz, str(pid))
        report(progress, 0.95, "Scoring alignment confidence")
        p_good = self.channel_model.confidence(recorded, channel_xyz, predicted_std)
        p_good[~search.valid] = np.nan
        timings = dict(search.timings, predict=time.time() - t1, total=time.time() - t0)
        report(progress, 1.0, "Ephys-only localization done")
        return AlignmentResult(
            method="ephys_only",
            pid=str(pid),
            channel_xyz=channel_xyz,
            recorded=recorded,
            predicted_std=predicted_std,
            recorded_std=self.channel_model.standardize(recorded),
            p_good=p_good,
            valid=search.valid,
            trace_xyz=_to_metres(search.trace_um, search.right_hemisphere),
            feature_names=list(self.channel_model.features),
            diagnostics=search.diagnostics,
            timings=timings,
        )


def _to_metres(xyz_um: np.ndarray, right_hemisphere: bool) -> np.ndarray:
    """µm (left) -> m, mirrored to the right hemisphere when the plan is there."""
    out = np.asarray(xyz_um, dtype=np.float64) * 1e-6
    if right_hemisphere:
        out[..., 0] = np.abs(out[..., 0])
    return out


def _subsample(scores: np.ndarray, n_max: int) -> np.ndarray:
    """Indices of at most ``n_max`` evenly spaced candidates, always keeping the best one."""
    n = len(scores)
    if n <= n_max:
        return np.arange(n)
    keep = np.unique(np.linspace(0, n - 1, n_max - 1).round().astype(int))
    best = int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))
    return np.union1d(keep, [best])


def localize_ephys_only(
    channel_model,
    brain_atlas,
    features: np.ndarray,
    planned_xyz: np.ndarray,
    *,
    pid: str = "",
    progress: Optional[ProgressCallback] = None,
    **kwargs,
) -> AlignmentResult:
    """One-call form of :meth:`EphysOnlyLocalizer.localize` (``kwargs`` go to the localizer).
    Reuse an :class:`EphysOnlyLocalizer` for many probes."""
    localizer = EphysOnlyLocalizer(channel_model, brain_atlas, **kwargs)
    return localizer.localize(features, planned_xyz, pid=pid, progress=progress)
