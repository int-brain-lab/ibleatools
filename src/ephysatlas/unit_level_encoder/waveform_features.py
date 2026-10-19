"""Waveform features of a unit's mean multi-channel waveform.

The unit-level model describes a unit by the waveform features of the channel-level feature list
(``ephysatlas.spatial_encoder.utils.WAVEFORM_FEATURES``) that are defined for a single mean
waveform, with the same definitions (``ephysatlas.features.ModelSpikeShapeFeatures``):

- the three slopes and ``tip_val`` come straight from ``ibldsp.waveforms.compute_spike_features``;
- ``spike_width_secs`` (trough minus peak time), ``predepolarisation_width_secs`` (peak minus tip
  time), ``spike_amplitude`` (trough minus peak value) and ``peak_to_trough_ratio_log``
  (``log|peak_val / trough_val|``) reparametrise its peak / trough / tip landmarks;
- ``spatial_spread_um`` is ``ibldsp.waveforms.compute_spatial_spread``: the amplitude-weighted mean
  distance of the channels from the peak channel, which needs each channel's probe position.

The across-spike standard deviations (``*_std``) have no meaning for a mean waveform, and
``slowness_s_per_m`` is not used. ``polarity`` comes last: it is categorical (+1 / -1), and the
continuous features are everything before it.
"""

from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import ibldsp.waveforms


FEATURE_NAMES = (
    "depolarisation_slope",
    "recovery_slope",
    "repolarisation_slope",
    "spatial_spread_um",
    "tip_val",
    "spike_width_secs",
    "predepolarisation_width_secs",
    "spike_amplitude",
    "peak_to_trough_ratio_log",
    "polarity",
)

# ibldsp's recovery point: a fixed 0.16 ms after the trough.
RECOVERY_DURATION_MS = 0.16

_REQUIRED_IBLDSP_COLUMNS = (
    "depolarisation_slope",
    "recovery_slope",
    "repolarisation_slope",
    "spatial_spread",
    "tip_time_idx",
    "tip_val",
    "trough_time_idx",
    "trough_val",
    "peak_time_idx",
    "peak_val",
    "invert_sign_peak",
)


def _frame_to_features(df: pd.DataFrame, fs: float) -> np.ndarray:
    missing = [c for c in _REQUIRED_IBLDSP_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            "ibldsp.waveforms did not return expected "
            f"columns {missing}. Returned columns={list(df.columns)}"
        )

    def col(name):
        return pd.to_numeric(df[name], errors="coerce").to_numpy(np.float64)

    peak_val, trough_val = col("peak_val"), col("trough_val")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_log = np.log(np.abs(peak_val / trough_val))
    return np.column_stack(
        [
            col("depolarisation_slope"),
            col("recovery_slope"),
            col("repolarisation_slope"),
            col("spatial_spread"),
            col("tip_val"),
            (col("trough_time_idx") - col("peak_time_idx")) / fs,
            (col("peak_time_idx") - col("tip_time_idx")) / fs,
            trough_val - peak_val,
            ratio_log,
            -col("invert_sign_peak"),
        ]
    ).astype(np.float32)


def _geometry_3d(channel_xy_um: np.ndarray) -> np.ndarray:
    """``[N, C, 2]`` probe positions as the ``[N, C, 3]`` geometry ibldsp expects (z = 0)."""
    xy = np.asarray(channel_xy_um, np.float64)
    return np.concatenate([xy, np.zeros(xy.shape[:-1] + (1,))], axis=-1)


def _spatial_spread(w: np.ndarray, xy: np.ndarray, peak_channel: int) -> float:
    """ibldsp's spatial spread for one ``[C, T]`` waveform: sum(|a_c| d_c) / sum(|a_c|)."""
    weights = np.abs(w[np.arange(len(w)), np.argmax(np.abs(w), axis=1)])
    dist = np.sqrt(np.sum((xy - xy[peak_channel]) ** 2, axis=1))
    return float(np.nansum(dist * weights) / np.sum(weights))


def _fallback_one(w: np.ndarray, xy: np.ndarray, fs: float) -> np.ndarray:
    """Landmarks of one ``[C, T]`` waveform when ibldsp cannot define its feature row.

    Used only for pathological averaged/decoded waveforms (for example when ibldsp's pre-tip
    search window is all NaN). It follows ibldsp's conventions: the peak is the largest absolute
    deflection of the peak channel, the trough the largest opposite deflection after it, the tip
    the largest opposite deflection before it, and the recovery point 0.16 ms after the trough.
    """
    w = np.asarray(w, np.float64)
    channel = int(np.nanargmax(np.nanmax(np.abs(w), axis=1)))
    raw = w[channel]
    n = len(raw)

    peak = int(np.nanargmax(np.abs(raw)))
    # The peak made negative, as ibldsp does.
    oriented = raw * (-1.0 if raw[peak] > 0 else 1.0)
    trough = peak + int(np.nanargmax(oriented[peak:]))
    tip = int(np.nanargmax(oriented[: peak + 1]))
    recovery = min(trough + int(round(RECOVERY_DURATION_MS * fs / 1000)), n - 1)

    peak_val, trough_val = raw[peak], raw[trough]
    tip_val, recovery_val = raw[tip], raw[recovery]
    dt = 1.0 / fs

    def slope(v1, v0, i1, i0):
        return (v1 - v0) / max((i1 - i0) * dt, dt)

    eps = 1e-6 * max(abs(peak_val), 1e-12)
    return np.asarray(
        [
            slope(peak_val, tip_val, peak, tip),
            slope(recovery_val, trough_val, recovery, trough),
            slope(trough_val, peak_val, trough, peak),
            _spatial_spread(w, xy, channel),
            tip_val,
            (trough - peak) * dt,
            (peak - tip) * dt,
            trough_val - peak_val,
            np.log(abs(peak_val) / max(abs(trough_val), eps)),
            1.0 if peak_val > 0 else -1.0,
        ],
        dtype=np.float32,
    )


def _compute_chunk_or_split(
    x: np.ndarray,
    xy: np.ndarray,
    indices: np.ndarray,
    out: np.ndarray,
    fallback_mask: np.ndarray,
    fs: float,
):
    """Use ibldsp in batches; recursively isolate only failing waveforms."""
    if len(indices) == 0:
        return

    try:
        arr = x[indices].transpose(0, 2, 1)
        df = ibldsp.waveforms.compute_spike_features(
            arr, fs=fs, recovery_duration_ms=RECOVERY_DURATION_MS
        )
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df = ibldsp.waveforms.compute_spatial_spread(arr, df, _geometry_3d(xy[indices]))
        feat = _frame_to_features(df, fs)

        # ibldsp can sometimes return a row but leave an undefined slope as
        # inf/nan. Keep exact finite rows and fallback only the affected rows.
        good = np.isfinite(feat).all(axis=1)
        out[indices[good]] = feat[good]
        for idx in indices[~good]:
            out[idx] = _fallback_one(x[idx], xy[idx], fs)
            fallback_mask[idx] = True
        return

    except (ValueError, FloatingPointError, IndexError):
        # ibldsp recovery_point can return index == waveform_length for
        # truncated 128-sample average/decoded waveforms. Treat that as an
        # undefined ibldsp feature row and isolate/fallback only that unit.
        if len(indices) == 1:
            idx = int(indices[0])
            out[idx] = _fallback_one(x[idx], xy[idx], fs)
            fallback_mask[idx] = True
            return

        mid = len(indices) // 2
        _compute_chunk_or_split(x, xy, indices[:mid], out, fallback_mask, fs)
        _compute_chunk_or_split(x, xy, indices[mid:], out, fallback_mask, fs)


def extract_generated_waveform_features(
    waveforms: np.ndarray,
    channel_xy_um: np.ndarray,
    sampling_rate_hz: float = 30_000.0,
    *,
    chunk_size: int = 4096,
    return_report: bool = False,
):
    """Compute the unit waveform features (:data:`FEATURE_NAMES`) with the IBL ibldsp implementation.

    The primary path is ``ibldsp.waveforms.compute_spike_features`` followed by
    ``ibldsp.waveforms.compute_spatial_spread`` on ``[N, T, C]``. A single pathological
    averaged/decoded waveform is not allowed to abort the whole evaluation: failed/non-finite rows
    are isolated recursively and only those rows use a narrow deterministic fallback.

    Args:
        waveforms: ``[N, C, T]`` multi-channel waveforms.
        channel_xy_um: Probe position (x, y) of each channel in micrometres, ``[N, C, 2]``, or one
            ``[C, 2]`` layout shared by all waveforms (for decoded waveforms, see
            :func:`modal_channel_layout`). NaN marks a padding channel.
        sampling_rate_hz: Sampling rate of the waveforms.
        chunk_size: Waveforms per ibldsp call.
        return_report: Also return a dict of extraction diagnostics.

    Returns:
        ``(features [N, F], FEATURE_NAMES)``, or ``(features, FEATURE_NAMES, report)``.
    """
    x = np.asarray(waveforms, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected [N,C,T], got {x.shape}")
    if not np.isfinite(x).all():
        bad = np.flatnonzero(~np.isfinite(x).all(axis=(1, 2)))
        raise ValueError(
            f"Non-finite waveform input for {len(bad)} units; "
            f"examples={bad[:10].tolist()}"
        )
    xy = np.asarray(channel_xy_um, dtype=np.float64)
    if xy.ndim == 2:
        xy = np.broadcast_to(xy, (len(x),) + xy.shape)
    if xy.shape != (x.shape[0], x.shape[1], 2):
        raise ValueError(
            f"channel_xy_um must be [N, C, 2] or [C, 2] for waveforms {x.shape}, got {xy.shape}"
        )

    fs = float(sampling_rate_hz)
    out = np.full((len(x), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    fallback_mask = np.zeros(len(x), dtype=bool)

    for start in range(0, len(x), int(chunk_size)):
        ids = np.arange(start, min(start + int(chunk_size), len(x)), dtype=np.int64)
        _compute_chunk_or_split(x, xy, ids, out, fallback_mask, fs)

    if not np.isfinite(out).all():
        bad = np.flatnonzero(~np.isfinite(out).all(axis=1))
        raise RuntimeError(
            f"Waveform feature extraction still produced non-finite values for "
            f"{len(bad)} rows after fallback; examples={bad[:10].tolist()}"
        )

    report = {
        "n_waveforms": int(len(x)),
        "n_exact_ibldsp": int((~fallback_mask).sum()),
        "n_fallback": int(fallback_mask.sum()),
        "fallback_fraction": float(fallback_mask.mean()) if len(x) else 0.0,
        "fallback_indices_first20": np.flatnonzero(fallback_mask)[:20].tolist(),
        "sampling_rate_hz": fs,
        "primary_method": "ibldsp.waveforms.compute_spike_features + compute_spatial_spread",
    }

    if return_report:
        return out, FEATURE_NAMES, report
    return out, FEATURE_NAMES


extract_ibl_waveform_features = extract_generated_waveform_features


def modal_channel_layout(
    channel_xy_um: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """The most common channel layout of a set of units, relative to its first channel.

    A decoded or sampled waveform belongs to no recorded unit, so it has no probe positions of its
    own; its ``spatial_spread_um`` is computed on the layout most training units share. Only
    distances between channels enter the spatial spread, so the layout's origin is irrelevant.

    Args:
        channel_xy_um: ``[N, C, 2]`` channel positions (NaN for padding channels).
        mask: Units to take the mode over (for example the training split). All units if None.

    Returns:
        ``[C, 2]`` positions, the first channel at the origin.
    """
    xy = np.asarray(channel_xy_um, np.float64)
    if mask is not None:
        xy = xy[np.asarray(mask, bool)]
    complete = np.isfinite(xy).all(axis=(1, 2))
    if not complete.any():
        raise ValueError("No unit has a complete channel layout")
    relative = np.round(xy[complete] - xy[complete, :1], 3)
    counts = collections.Counter(map(bytes, relative))
    layout = np.frombuffer(counts.most_common(1)[0][0], dtype=np.float64)
    return layout.reshape(xy.shape[1], 2).copy()
