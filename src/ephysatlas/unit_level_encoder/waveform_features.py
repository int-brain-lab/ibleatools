from __future__ import annotations

import numpy as np
import pandas as pd
import ibldsp.waveforms


FEATURE_NAMES = (
    "depolarisation_slope",
    "recovery_slope",
    "recovery_time_secs",
    "repolarisation_slope",
    "tip_time_secs",
    "tip_val",
    "through_time_secs",
    "trough_val",
    "peak_time_secs",
    "peak_val",
    "polarity",
)


_REQUIRED_IBLDSP_COLUMNS = (
    "depolarisation_slope",
    "recovery_slope",
    "recovery_time_idx",
    "repolarisation_slope",
    "tip_time_idx",
    "tip_val",
    "trough_time_idx",
    "trough_val",
    "peak_time_idx",
    "peak_val",
    "invert_sign_peak",
)


def _frame_to_features(
    df: pd.DataFrame,
    fs: float,
    trough_offset_samples: int,
) -> np.ndarray:
    missing = [c for c in _REQUIRED_IBLDSP_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            "ibldsp.waveforms.compute_spike_features did not return expected "
            f"columns {missing}. Returned columns={list(df.columns)}"
        )

    off = float(trough_offset_samples)
    return np.column_stack(
        [
            pd.to_numeric(df["depolarisation_slope"], errors="coerce"),
            pd.to_numeric(df["recovery_slope"], errors="coerce"),
            (pd.to_numeric(df["recovery_time_idx"], errors="coerce") - off) / fs,
            pd.to_numeric(df["repolarisation_slope"], errors="coerce"),
            (pd.to_numeric(df["tip_time_idx"], errors="coerce") - off) / fs,
            pd.to_numeric(df["tip_val"], errors="coerce"),
            (pd.to_numeric(df["trough_time_idx"], errors="coerce") - off) / fs,
            pd.to_numeric(df["trough_val"], errors="coerce"),
            (pd.to_numeric(df["peak_time_idx"], errors="coerce") - off) / fs,
            pd.to_numeric(df["peak_val"], errors="coerce"),
            -pd.to_numeric(df["invert_sign_peak"], errors="coerce"),
        ]
    ).astype(np.float32)


def _dominant_trace(w: np.ndarray) -> np.ndarray:
    # Match the usual waveform-feature intent: select the trace with largest
    # peak-to-peak excursion rather than allowing zero-padded channels to matter.
    ptp = np.ptp(w, axis=1)
    ch = int(np.nanargmax(ptp))
    return np.asarray(w[ch], dtype=np.float32)


def _fallback_one(
    w: np.ndarray,
    fs: float,
    trough_offset_samples: int,
) -> np.ndarray:
    """Narrow fallback for pathological averaged/decoded waveforms.

    This is used only when ibldsp itself cannot define a feature row (for example
    because its pre-tip search window is all NaN).  It uses the same waveform
    amplitude convention and the same 42-sample time reference as the IBL
    aggregation.  The exact ibldsp path remains the primary path.
    """
    raw = _dominant_trace(np.asarray(w, np.float32))

    # Orient landmark search so the dominant event is negative, matching the
    # conventional trough-centred representation used by the IBL feature code.
    polarity = -1.0 if abs(float(np.nanmin(raw))) >= abs(float(np.nanmax(raw))) else 1.0
    t = raw if polarity < 0 else -raw

    trough = int(np.nanargmin(t))
    tip = int(np.nanargmax(t[: trough + 1])) if trough > 0 else 0
    peak = (
        trough + int(np.nanargmax(t[trough:]))
        if trough < len(t) - 1
        else trough
    )

    # A robust recovery landmark.  This is only a fallback for cases where the
    # exact ibldsp routine cannot return a row.
    target = t[trough] + 0.8 * (t[peak] - t[trough])
    after = np.flatnonzero(t[trough:] >= target)
    recovery = trough + int(after[0]) if len(after) else peak

    sign_back = 1.0 if polarity < 0 else -1.0
    trough_val = float(sign_back * t[trough])
    tip_val = float(sign_back * t[tip])
    peak_val = float(sign_back * t[peak])
    recovery_val = float(sign_back * t[recovery])

    dt = 1.0 / fs
    dep_dt = max((trough - tip) * dt, dt)
    rep_dt = max((peak - trough) * dt, dt)
    rec_dt = max((recovery - trough) * dt, dt)

    return np.asarray(
        [
            (trough_val - tip_val) / dep_dt,
            (recovery_val - trough_val) / rec_dt,
            (recovery - trough_offset_samples) / fs,
            (peak_val - trough_val) / rep_dt,
            (tip - trough_offset_samples) / fs,
            tip_val,
            (trough - trough_offset_samples) / fs,
            trough_val,
            (peak - trough_offset_samples) / fs,
            peak_val,
            polarity,
        ],
        dtype=np.float32,
    )


def _compute_chunk_or_split(
    x: np.ndarray,
    indices: np.ndarray,
    out: np.ndarray,
    fallback_mask: np.ndarray,
    fs: float,
    trough_offset_samples: int,
):
    """Use ibldsp in batches; recursively isolate only failing waveforms."""
    if len(indices) == 0:
        return

    try:
        df = ibldsp.waveforms.compute_spike_features(
            x[indices].transpose(0, 2, 1),
            fs=fs,
        )
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        feat = _frame_to_features(df, fs, trough_offset_samples)

        # ibldsp can sometimes return a row but leave an undefined slope as
        # inf/nan. Keep exact finite rows and fallback only the affected rows.
        good = np.isfinite(feat).all(axis=1)
        out[indices[good]] = feat[good]
        bad_idx = indices[~good]
        for idx in bad_idx:
            out[idx] = _fallback_one(
                x[idx], fs, trough_offset_samples
            )
            fallback_mask[idx] = True
        return

    except (ValueError, FloatingPointError, IndexError):
        # ibldsp recovery_point can return index == waveform_length for
        # truncated 128-sample average/decoded waveforms. Treat that as an
        # undefined ibldsp feature row and isolate/fallback only that unit.
        if len(indices) == 1:
            idx = int(indices[0])
            out[idx] = _fallback_one(
                x[idx], fs, trough_offset_samples
            )
            fallback_mask[idx] = True
            return

        mid = len(indices) // 2
        _compute_chunk_or_split(
            x, indices[:mid], out, fallback_mask,
            fs, trough_offset_samples,
        )
        _compute_chunk_or_split(
            x, indices[mid:], out, fallback_mask,
            fs, trough_offset_samples,
        )


def extract_generated_waveform_features(
    waveforms: np.ndarray,
    sampling_rate_hz: float = 30_000.0,
    *,
    trough_offset_samples: int = 42,
    chunk_size: int = 4096,
    return_report: bool = False,
):
    """Compute the 11 waveform features using the IBL ibldsp implementation.

    The primary path is exactly
        ibldsp.waveforms.compute_spike_features([N,T,C], fs=...)

    A single pathological averaged/decoded waveform is not allowed to abort the
    whole evaluation.  Failed/non-finite rows are isolated recursively and only
    those rows use a narrow deterministic fallback.

    Timing follows the ibleatools aggregation convention:
        time_seconds = (feature_index - trough_offset_samples) / fs

    Returns
    -------
    features, FEATURE_NAMES
    or, if return_report=True:
    features, FEATURE_NAMES, report
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

    fs = float(sampling_rate_hz)
    out = np.full((len(x), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    fallback_mask = np.zeros(len(x), dtype=bool)

    for start in range(0, len(x), int(chunk_size)):
        ids = np.arange(start, min(start + int(chunk_size), len(x)), dtype=np.int64)
        _compute_chunk_or_split(
            x, ids, out, fallback_mask, fs, int(trough_offset_samples)
        )

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
        "trough_offset_samples": int(trough_offset_samples),
        "time_definition": "(feature_index - trough_offset_samples) / fs",
        "primary_method": "ibldsp.waveforms.compute_spike_features",
    }

    if return_report:
        return out, FEATURE_NAMES, report
    return out, FEATURE_NAMES


extract_ibl_waveform_features = extract_generated_waveform_features
