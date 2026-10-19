"""Histology-based alignment: place a probe's channels along its reconstructed histology trace.

The channel-level model predicts the features expected at every sample of the trace (extended to
the brain boundary). A cost matrix compares each recorded channel with each trace sample (squared
Euclidean distance of the standardised feature vectors), and dynamic time warping finds the
cheapest monotonic channel -> trace correspondence, with open ends, free diagonal steps and
penalised stretching/compression. If the warped path covers too little of the probe, the best
rigid shift is used instead. This is the method of the paper's Figure 5 and of the alignment GUI.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from .progress import ProgressCallback, report
from .result import AlignmentResult

# Dynamic time warping step penalties, as fractions of the median cost (vertical: several channels
# on one trace sample; horizontal: trace samples skipped between consecutive channels).
DTW_VERTICAL_PENALTY = 0.5
DTW_HORIZONTAL_PENALTY = 0.1
# Below this fraction of the recorded channels covered by the warped path, align rigidly.
MIN_OVERLAP_FRACTION = 0.9


def cost_matrix(recorded_std: np.ndarray, predicted_std: np.ndarray) -> np.ndarray:
    """``[C, L]`` squared Euclidean distance between each recorded channel and each trace sample."""
    a = np.asarray(recorded_std, dtype=np.float64)
    b = np.asarray(predicted_std, dtype=np.float64)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return (aa + bb - 2.0 * a @ b.T).clip(min=0.0)


def dynamic_time_warping(
    cost: np.ndarray,
    *,
    lam_d: float = 0.0,
    lam_u: float = 0.1,
    lam_l: float = 0.1,
    open_begin: bool = True,
):
    """Monotonic warping path through ``cost`` (channels x trace samples).

    Moves: diagonal (next channel, next sample; penalty ``lam_d``), vertical (next channel, same
    sample; ``lam_u``), horizontal (same channel, next sample; ``lam_l``). With ``open_begin`` the
    first channel may start anywhere on the trace, and the last channel always ends at its best
    sample (open end).

    Returns:
        tuple: ``(j_start, j_end, path, total_cost)``, ``path`` a list of (channel, sample) pairs.
    """
    c = np.asarray(cost, dtype=np.float64)
    c = np.where(np.isfinite(c), c, np.inf)
    n, m = c.shape
    d = np.full((n, m), np.inf)
    p = np.full((n, m), -1, dtype=np.int8)
    d[0, 0] = c[0, 0]
    for j in range(1, m):
        if open_begin:
            d[0, j] = c[0, j]
        else:
            d[0, j] = c[0, j] + d[0, j - 1] + lam_l
            p[0, j] = 2
    for i in range(1, n):
        d[i, 0] = c[i, 0] + d[i - 1, 0] + lam_u
        p[i, 0] = 1
    for i in range(1, n):
        prev, row, ci = d[i - 1], d[i], c[i]
        diag = prev[:-1] + lam_d
        up = prev[1:] + lam_u
        for j in range(1, m):
            best, k = diag[j - 1], 0
            if up[j - 1] < best:
                best, k = up[j - 1], 1
            left = row[j - 1] + lam_l
            if left < best:
                best, k = left, 2
            row[j] = ci[j] + best
            p[i, j] = k
    j_end = int(np.nanargmin(d[n - 1]))
    total = float(d[n - 1, j_end])
    i, j = n - 1, j_end
    path = [(i, j)]
    while i > 0 or (not open_begin and j > 0):
        k = p[i, j]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        elif k == 2:
            j -= 1
        else:
            break
        path.append((i, j))
    path.reverse()
    return int(path[0][1]), j_end, path, total


def rigid_assignment(recorded_std: np.ndarray, predicted_std: np.ndarray):
    """Best constant shift of the channels along the trace (mean squared error)."""
    a, b = np.asarray(recorded_std), np.asarray(predicted_std)
    n = a.shape[0]
    best_k, best = 0, np.inf
    for k in range(0, b.shape[0] - n + 1):
        mse = ((b[k : k + n] - a) ** 2).mean()
        if mse < best:
            best, best_k = mse, k
    return best_k, best_k + n - 1, [(i, best_k + i) for i in range(n)]


def path_to_channel_map(path, valid: np.ndarray, trace_len: int) -> np.ndarray:
    """``[C]`` trace index of every channel: valid channels from the path, others interpolated."""
    valid = np.asarray(valid, dtype=bool)
    i_seq, j_seq = np.asarray(path, dtype=int).T
    j_for_i = np.full(int(valid.sum()), np.nan)
    j_for_i[i_seq] = j_seq
    idx = np.arange(len(j_for_i))
    ok = np.isfinite(j_for_i)
    j_for_i = np.interp(idx, idx[ok], j_for_i[ok]) if ok.any() else np.zeros(len(idx))
    j_map = np.interp(np.arange(len(valid)), np.flatnonzero(valid), j_for_i)
    return np.clip(np.round(j_map).astype(int), 0, trace_len - 1)


def valid_channels(recorded: np.ndarray) -> np.ndarray:
    """Channels with a signal: finite and not all-zero features."""
    recorded = np.asarray(recorded, dtype=float)
    return np.isfinite(recorded).all(axis=1) & ~np.all(np.nan_to_num(recorded) == 0.0, axis=1)


def align_histology(
    channel_model,
    recorded: np.ndarray,
    trace_xyz: np.ndarray,
    *,
    pid: str = "",
    feature_indices: Optional[Sequence[int]] = None,
    cost: Optional[np.ndarray] = None,
    method: str = "histology",
    dtw_penalty_scale: str = "median",
    dtw_vertical_penalty: float = DTW_VERTICAL_PENALTY,
    dtw_horizontal_penalty: float = DTW_HORIZONTAL_PENALTY,
    min_overlap_fraction: float = MIN_OVERLAP_FRACTION,
    predicted_trace_std: Optional[np.ndarray] = None,
    progress: Optional[ProgressCallback] = None,
) -> AlignmentResult:
    """Align a probe's recorded channels to a histology trace.

    Args:
        channel_model: :class:`~ephysatlas.alignment.models.ChannelModel`.
        recorded: ``[C, F]`` recorded channel features in feature units, row order (top first).
        trace_xyz: ``[L, 3]`` trace samples (m), top to bottom, usually already extended to the
            brain boundary with :func:`~ephysatlas.alignment.geometry.extend_trace_to_brain`.
        pid: Insertion id; its own channels are excluded from the model's neighbours.
        feature_indices: Features the cost uses (all by default).
        cost: A precomputed ``[C_valid, L]`` cost to warp instead of the feature cost (the
            channel + unit variant passes its fused cost here).
        method: Label stored in the result.
        dtw_penalty_scale: ``"median"`` (penalties are fractions of the median cost, the published
            method) or ``"q10_q90"`` (of the 10-90% cost spread, for calibrated, row-relative costs).
        dtw_vertical_penalty, dtw_horizontal_penalty: Penalty fractions.
        min_overlap_fraction: Minimum covered fraction of the valid channels before falling back
            to a rigid shift.
        predicted_trace_std: ``[L, F]`` standardised predictions along the trace, if already
            computed (reused by the channel + unit variant).
        progress: ``progress(fraction, message)`` callback.

    Returns:
        AlignmentResult: with ``cost_matrix`` / ``path``, and in ``diagnostics`` the trace
        predictions (``predicted_trace_std``), the channel -> trace map (``channel_to_trace``),
        ``j_start`` / ``j_end``, the warping total cost and whether the rigid fallback was used.
    """
    t0 = time.time()
    recorded = np.asarray(recorded, dtype=np.float64)
    trace_xyz = np.asarray(trace_xyz, dtype=np.float32)
    valid = valid_channels(recorded)
    if valid.sum() < 2:
        raise ValueError("Need at least 2 channels with non-zero features to align a probe.")
    feats = np.arange(recorded.shape[1]) if feature_indices is None else np.asarray(feature_indices)
    recorded_std = channel_model.standardize(recorded)
    timings = {}

    if predicted_trace_std is None:
        report(progress, 0.05, f"Predicting features along the trace ({len(trace_xyz)} samples)")
        predicted_trace_std = channel_model.predict_std(trace_xyz, pid)
        timings["predict_trace"] = time.time() - t0
    predicted_trace_std = np.asarray(predicted_trace_std, dtype=np.float64)

    report(progress, 0.55, "Building the channel x trace cost matrix")
    feature_cost = cost_matrix(recorded_std[valid][:, feats], predicted_trace_std[:, feats])
    c = feature_cost if cost is None else np.asarray(cost, dtype=np.float64)

    report(progress, 0.65, "Dynamic time warping")
    t1 = time.time()
    finite = c[np.isfinite(c)]
    if dtw_penalty_scale == "median":
        scale = float(np.median(np.nan_to_num(finite)))
    elif dtw_penalty_scale == "q10_q90":
        q10, q90 = np.nanpercentile(finite, [10.0, 90.0])
        scale = float(max(q90 - q10, 1e-8))
    else:
        raise ValueError(f"unknown dtw_penalty_scale {dtw_penalty_scale!r}")
    j_start, j_end, path, total = dynamic_time_warping(
        c,
        lam_d=0.0,
        lam_u=dtw_vertical_penalty * scale,
        lam_l=dtw_horizontal_penalty * scale,
        open_begin=True,
    )
    rigid = (j_end - j_start + 1) < int(min_overlap_fraction * int(valid.sum()))
    if rigid:
        j_start, j_end, path = rigid_assignment(
            recorded_std[valid][:, feats], predicted_trace_std[:, feats]
        )
    timings["dtw"] = time.time() - t1

    channel_to_trace = path_to_channel_map(path, valid, len(trace_xyz))
    channel_xyz = trace_xyz[channel_to_trace]
    predicted_std = predicted_trace_std[channel_to_trace]

    report(progress, 0.85, "Scoring alignment confidence")
    p_good = channel_model.confidence(recorded, channel_xyz, predicted_std)
    p_good[~valid] = np.nan
    timings["total"] = time.time() - t0
    report(progress, 1.0, "Histology-based alignment done")
    return AlignmentResult(
        method=method,
        pid=str(pid),
        channel_xyz=channel_xyz,
        recorded=recorded,
        predicted_std=predicted_std,
        recorded_std=recorded_std,
        p_good=p_good,
        valid=valid,
        trace_xyz=trace_xyz,
        feature_names=list(channel_model.features),
        cost_matrix=c.astype(np.float32),
        path=np.asarray(path, dtype=int),
        diagnostics=dict(
            predicted_trace_std=predicted_trace_std.astype(np.float32),
            feature_cost=feature_cost.astype(np.float32),
            channel_to_trace=channel_to_trace,
            j_start=int(j_start),
            j_end=int(j_end),
            total_cost=float(total),
            rigid_fallback=bool(rigid),
            dtw_penalty_scale=dtw_penalty_scale,
        ),
        timings=timings,
    )
