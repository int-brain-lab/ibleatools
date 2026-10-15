"""Scores of an alignment against the human (reference) alignment of the same probe."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import region_ids, valid_xyz_mask

CONFIDENCE_THRESHOLD = 0.5


def _subset_metrics(true_xyz, est_xyz, mask, brain_atlas) -> dict:
    out = {"n_channels": int(mask.sum()), "mean_distance_um": np.nan,
           "median_distance_um": np.nan, "cosmos_acc": np.nan, "beryl_acc": np.nan}
    if not mask.any():
        return out
    d = np.linalg.norm(est_xyz[mask] - true_xyz[mask], axis=1) * 1e6
    out["mean_distance_um"] = float(np.mean(d))
    out["median_distance_um"] = float(np.median(d))
    for mapping, key in (("Cosmos", "cosmos_acc"), ("Beryl", "beryl_acc")):
        t = region_ids(brain_atlas, true_xyz[mask], mapping)
        p = region_ids(brain_atlas, est_xyz[mask], mapping)
        inside = t != 0
        out[key] = float(np.mean(t[inside] == p[inside])) if inside.any() else np.nan
    return out


def alignment_metrics(
    true_xyz: np.ndarray,
    est_xyz: np.ndarray,
    brain_atlas,
    *,
    p_good: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """Per-probe scores of estimated channel positions against the reference positions.

    Channels count when both positions are finite and non-zero (and ``valid`` when given). The
    region accuracies are over channels whose reference position is inside the brain; an estimate
    outside the brain counts as wrong. Distances are Euclidean, per channel. With ``p_good`` the
    same scores are also given for the high-confidence (``p_good >= threshold``) and
    low-confidence channels, with the high-confidence fraction.

    Returns:
        dict: ``cosmos_acc``, ``beryl_acc``, ``mean_distance_um``, ``median_distance_um``,
        ``n_channels``, and the ``*_high_conf`` / ``*_low_conf`` variants and
        ``high_conf_fraction`` when ``p_good`` is given.
    """
    true_xyz = np.asarray(true_xyz, dtype=float)
    est_xyz = np.asarray(est_xyz, dtype=float)
    mask = valid_xyz_mask(true_xyz) & valid_xyz_mask(est_xyz)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    out = _subset_metrics(true_xyz, est_xyz, mask, brain_atlas)
    if p_good is not None:
        p_good = np.asarray(p_good, dtype=float)
        scored = mask & np.isfinite(p_good)
        high = scored & (p_good >= threshold)
        low = scored & (p_good < threshold)
        for suffix, subset in (("high_conf", high), ("low_conf", low)):
            for key, value in _subset_metrics(true_xyz, est_xyz, subset, brain_atlas).items():
                out[f"{key}_{suffix}"] = value
        out["high_conf_fraction"] = float(high.sum() / max(scored.sum(), 1))
    return out


def result_metrics(result, true_xyz, brain_atlas, prefix: str = "") -> dict:
    """:func:`alignment_metrics` of an :class:`~.result.AlignmentResult`, keys prefixed."""
    m = alignment_metrics(
        true_xyz, result.channel_xyz, brain_atlas, p_good=result.p_good, valid=result.valid
    )
    return {f"{prefix}{k}": v for k, v in m.items()}
