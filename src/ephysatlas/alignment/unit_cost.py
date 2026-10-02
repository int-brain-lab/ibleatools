"""Channel + unit histology-based alignment: add the spike-sorted units' evidence to the cost.

For every recorded channel row, the units the spike sorter placed within ``half_width_um`` of that
channel's physical axial position are scored at every trace sample under the unit-level model:
the log-likelihood of their 60-d latents under the context-conditioned mixture
``sum_k gamma_k(x) N(z; mu_k, Sigma_k)`` at the sample's molecular context. The row cost is their
mean negative log-likelihood per latent dimension, minus the row's minimum (only the variation
along the trace matters). Rows without units stay empty -- nothing is interpolated.

The channel cost is calibrated to a Gaussian NLL per feature (0.5 x squared standardised error /
n_features, row-relative), the unit cost is matched to its contrast with one factor per probe
(median row 10-90% spread), and the two are summed with weight ``unit_weight`` on the unit rows
that exist. Warping then uses penalties relative to the fused cost's 10-90% spread.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import logsumexp

from .geometry import np1_axial_um_in_row_order
from .histology import align_histology, valid_channels
from .progress import ProgressCallback, report, sub_progress

UNIT_HALF_WIDTH_UM = 100.0
UNIT_WEIGHT = 0.5
SCALE_MATCH_BOUNDS = (0.25, 20.0)
# DTW penalties on the calibrated, row-relative fused cost, as fractions of its 10-90% spread.
FUSED_DTW_PENALTY = 0.5
FUSED_MIN_OVERLAP_FRACTION = 0.75


def row_relative(cost: np.ndarray) -> np.ndarray:
    """Subtract each row's minimum finite value (rows without finite values stay NaN)."""
    x = np.asarray(cost, dtype=np.float64).copy()
    for i in range(x.shape[0]):
        finite = np.isfinite(x[i])
        if finite.any():
            x[i, finite] -= np.min(x[i, finite])
    return x


def calibrated_channel_cost(feature_cost: np.ndarray, n_features: int) -> np.ndarray:
    """Squared standardised error as a row-relative Gaussian NLL per feature."""
    return row_relative(0.5 * np.asarray(feature_cost, dtype=np.float64) / max(int(n_features), 1))


def unit_cost_matrix(
    channel_model,
    unit_model,
    pid: str,
    trace_xyz: np.ndarray,
    valid: np.ndarray,
    *,
    half_width_um: float = UNIT_HALF_WIDTH_UM,
) -> tuple[Optional[np.ndarray], dict]:
    """``[C_valid, L]`` row-relative unit NLL per latent dimension (NaN rows without units)."""
    z, axial = unit_model.units_of(pid)
    info = {"unit_n_units": int(len(z))}
    if len(z) == 0:
        info["unit_reason"] = "no prepared units on this probe"
        return None, info
    from ephysatlas.unit_level_encoder.gmm_models import component_log_prob

    channel_axial = np1_axial_um_in_row_order(len(valid))[np.asarray(valid, dtype=bool)]
    log_w = unit_model.log_mixture_weights(channel_model.context_raw(trace_xyz))  # [L, K]
    component_lp = component_log_prob(unit_model.bundle.gmm, z)  # [n_units, K]
    latent_dim = max(int(z.shape[1]), 1)
    cost = np.full((len(channel_axial), len(trace_xyz)), np.nan)
    counts = []
    for r, centre in enumerate(channel_axial):
        near = np.flatnonzero(np.abs(axial - centre) <= float(half_width_um))
        if len(near) == 0:
            continue
        lp = logsumexp(log_w[:, None, :] + component_lp[near][None, :, :], axis=2)
        row = -np.mean(lp, axis=1) / latent_dim
        cost[r] = row - np.nanmin(row)
        counts.append(len(near))
    rows = np.isfinite(cost).any(axis=1)
    info.update(
        unit_rows=int(rows.sum()),
        unit_row_fraction=float(rows.mean()) if len(rows) else 0.0,
        unit_units_per_row_median=float(np.median(counts)) if counts else np.nan,
    )
    if not rows.any():
        info["unit_reason"] = "no channel row had a unit within reach"
        return None, info
    info["unit_reason"] = ""
    return cost, info


def _median_row_contrast(cost: np.ndarray, rows: np.ndarray) -> float:
    contrasts = []
    for r in np.flatnonzero(rows):
        vals = cost[r, np.isfinite(cost[r])]
        if len(vals) >= 2:
            q10, q90 = np.percentile(vals, [10.0, 90.0])
            contrasts.append(max(q90 - q10, 0.0))
    return float(np.median(contrasts)) if contrasts else np.nan


def fuse_costs(
    channel_cost: np.ndarray,
    unit_cost: np.ndarray,
    *,
    unit_weight: float = UNIT_WEIGHT,
    scale_bounds: tuple = SCALE_MATCH_BOUNDS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Contrast-match the unit cost to the channel cost and fuse them.

    Returns:
        tuple: ``(fused, unit_scaled, info)``; the channel coefficient is ``1 - unit_weight``
        on every row, and missing unit rows add nothing.
    """
    rows = np.isfinite(unit_cost).any(axis=1)
    ch = _median_row_contrast(channel_cost, rows)
    un = _median_row_contrast(unit_cost, rows)
    if np.isfinite(ch) and np.isfinite(un) and un > 1e-6:
        scale = float(np.clip(ch / un, *scale_bounds))
    else:
        scale = 1.0
    unit_scaled = unit_cost * scale
    fused = (1.0 - unit_weight) * channel_cost
    ok = np.isfinite(unit_scaled)
    fused[ok] += unit_weight * unit_scaled[ok]
    return row_relative(fused), unit_scaled, dict(
        unit_scale_factor=scale, unit_weight=float(unit_weight), fused_rows_with_units=int(rows.sum())
    )


def align_histology_with_units(
    channel_model,
    unit_model,
    recorded: np.ndarray,
    trace_xyz: np.ndarray,
    *,
    pid: str,
    channel_result=None,
    unit_weight: float = UNIT_WEIGHT,
    half_width_um: float = UNIT_HALF_WIDTH_UM,
    progress: Optional[ProgressCallback] = None,
):
    """Channel-only and channel + unit alignments of one probe along its histology trace.

    Args:
        channel_result: The channel-only result, if already computed (its trace predictions are
            reused).

    Returns:
        tuple: ``(channel_result, combined_result)``. ``combined_result`` is None when the probe
        has no usable unit rows; otherwise it carries in ``diagnostics`` the calibrated channel,
        unit and fused costs and the fusion summary.
    """
    if channel_result is None:
        channel_result = align_histology(
            channel_model, recorded, trace_xyz, pid=pid, progress=sub_progress(progress, 0.0, 0.6)
        )
    report(progress, 0.6, "Scoring the units under the unit-level model along the trace")
    valid = valid_channels(recorded)
    n_features = len(channel_model.features)
    channel_cost = calibrated_channel_cost(channel_result.diagnostics["feature_cost"], n_features)
    unit_cost, info = unit_cost_matrix(
        channel_model, unit_model, pid, trace_xyz, valid, half_width_um=half_width_um
    )
    if unit_cost is None:
        report(progress, 1.0, "No unit evidence on this probe")
        channel_result.diagnostics.update(info)
        return channel_result, None
    fused, unit_scaled, fusion = fuse_costs(channel_cost, unit_cost, unit_weight=unit_weight)
    info.update(fusion)
    report(progress, 0.8, "Warping the fused channel + unit cost")
    combined = align_histology(
        channel_model,
        recorded,
        trace_xyz,
        pid=pid,
        cost=fused,
        method="histology_unit",
        dtw_penalty_scale="q10_q90",
        dtw_vertical_penalty=FUSED_DTW_PENALTY,
        dtw_horizontal_penalty=FUSED_DTW_PENALTY,
        min_overlap_fraction=FUSED_MIN_OVERLAP_FRACTION,
        predicted_trace_std=channel_result.diagnostics["predicted_trace_std"],
    )
    if combined.diagnostics["rigid_fallback"]:
        # The fused path covered too little of the probe: keep the channel-only alignment rather
        # than a rigid shift of the fused cost.
        combined = align_histology(
            channel_model,
            recorded,
            trace_xyz,
            pid=pid,
            method="histology_unit",
            predicted_trace_std=channel_result.diagnostics["predicted_trace_std"],
        )
        info["fused_path_rejected"] = True
    combined.diagnostics.update(
        info,
        channel_cost=channel_cost.astype(np.float32),
        unit_cost=unit_scaled.astype(np.float32),
        fused_cost=fused.astype(np.float32),
    )
    report(progress, 1.0, "Channel + unit alignment done")
    return channel_result, combined
