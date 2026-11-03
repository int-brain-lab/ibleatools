"""Channel features of a recording from its raw LF and AP bands, one snippet at a time.

The release features (``raw_ephys_features_denoised.pqt``) are computed from five snippets of
5 s of AP and LF per insertion, aggregated per channel and denoised along the probe. This module
runs the same pipeline on any sequence of snippets, so that a recording being acquired can be
localized as its snippets arrive:

1. :func:`snippet_features` -- one snippet: bad channels detected on the AP band, both bands
   destriped, and the ``lf``, ``csd`` and ``ap`` feature families of
   :func:`ephysatlas.feature_computation.compute_features_from_raw` (no spike detection);
2. :func:`aggregate_snippets` -- per channel, the median over the snippets (labels: mode), then
   the release's log transforms, TV denoising along the probe and NaN fill (the steps of
   :func:`ephysatlas.aggregation.denoise_raw_features_data` but its alpha-outlier filter, see
   :func:`aggregate_snippets`);
3. :func:`feature_matrix` -- the ``[C, F]`` matrix in the models' feature order and row order
   (top of the probe first) that :class:`~ephysatlas.alignment.EphysOnlyLocalizer` takes.

:func:`reference_features` runs steps 1-2 on SpikeGLX files for a list of snippet times (in
parallel processes if asked): the reference the online pipelines are compared with.
Features that need spike detection or waveforms are NaN in the matrix.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.fft

logger = logging.getLogger(__name__)

# Snippet length of the release (``snippets_df.pqt``: duration_ap = duration_lf = 5 s).
SNIPPET_S = 5.0
# LF snippets start a few samples late to compensate for the LF filter latency relative to AP
# (``feature_calculators.spikeglx_like.LF_LATENCY_SAMPLES``).
LF_LATENCY_SAMPLES = 3
# Feature families computed per snippet: everything but the spike-detection based "waveforms".
FEATURE_FAMILIES = ("lf", "csd", "ap")


def snippet_samples(fs: float, duration_s: float = SNIPPET_S) -> int:
    """Samples in a snippet of ``duration_s`` at ``fs``, rounded up to a fast FFT length (as the
    release)."""
    return int(scipy.fft.next_fast_len(int(fs * duration_s), real=True))


def read_snippet(sr_ap, sr_lf, t_start: float, duration_s: float = SNIPPET_S):
    """``(raw_ap, raw_lf)`` ``[C, ns]`` in volts from ``spikeglx.Reader``-like readers, sync
    channels left out, exactly as the release read its snippets."""
    fs_ap, fs_lf = float(sr_ap.fs), float(sr_lf.fs)
    n0_ap = int(fs_ap * t_start)
    n0_lf = int(fs_lf * t_start) + LF_LATENCY_SAMPLES
    raw_ap = sr_ap[
        slice(n0_ap, n0_ap + snippet_samples(fs_ap, duration_s)),
        : sr_ap.nc - sr_ap.nsync,
    ]
    raw_lf = sr_lf[
        slice(n0_lf, n0_lf + snippet_samples(fs_lf, duration_s)),
        : sr_lf.nc - sr_lf.nsync,
    ]
    return np.ascontiguousarray(raw_ap.T), np.ascontiguousarray(raw_lf.T)


def snippet_features(
    raw_ap: np.ndarray,
    raw_lf: np.ndarray,
    fs_ap: float,
    fs_lf: float,
    geometry: dict,
    channel_labels: Optional[np.ndarray] = None,
    neuropixel_version: int = 1,
) -> pd.DataFrame:
    """Raw (untransformed) features of one snippet, one row per channel.

    Args:
        raw_ap, raw_lf: ``[C, ns]`` AP and LF snippets in volts.
        fs_ap, fs_lf: Sampling rates (Hz).
        geometry: ibldsp channel geometry (``x``, ``y``, ``sample_shift``, ...), e.g. a
            ``spikeglx.Reader``'s ``geometry``.
        channel_labels: ``[C]`` bad-channel labels (0 good); detected on the AP snippet with
            ``ibldsp.voltage.detect_bad_channels`` when None, as the release did for streamed data.
        neuropixel_version: Probe version passed to the destriping.

    Returns:
        pd.DataFrame: ``channel``, the ``lf``, ``csd`` and ``ap`` features and ``channel_labels``.
    """
    import ibldsp.voltage

    from ephysatlas.feature_computation import compute_features_from_raw

    if channel_labels is None:
        channel_labels, _ = ibldsp.voltage.detect_bad_channels(raw_ap, fs=fs_ap)
    return compute_features_from_raw(
        raw_ap=raw_ap,
        raw_lf=raw_lf,
        fs_ap=fs_ap,
        fs_lf=fs_lf,
        geometry=geometry,
        channel_labels=np.asarray(channel_labels),
        neuropixel_version=neuropixel_version,
        features_to_compute=list(FEATURE_FAMILIES),
        output_dir=None,
    )


def aggregate_snippets(
    snippets: Sequence[pd.DataFrame], geometry: dict, name: str = "recording"
) -> pd.DataFrame:
    """Aggregate snippet features as the release: median per channel, then denoise.

    The release's alpha-outlier filter (``ephysatlas.data.outlier_treatment``) is left out: it
    acts on ``alpha_mean``/``alpha_std``, which need spike detection, and without them it falls
    back to ``psd_alpha``, whose dB values all pass its ``>= 1e3 x median`` test, so it would
    replace every channel's ``psd_alpha`` by the median.

    Args:
        snippets: Outputs of :func:`snippet_features`.
        geometry: The channel geometry (``x`` lateral and ``y`` axial positions, µm).
        name: The ``pid`` level of the result's index.

    Returns:
        pd.DataFrame: Indexed by ``(pid, channel)``; transformed (rms in dB) and denoised features,
        ``channel_labels``, ``axial_um`` and ``lateral_um``.
    """
    from ephysatlas.aggregation import aggregate_raw_features
    from ephysatlas.data import replace_nan
    from ephysatlas.features import DEFAULT_FAC, denoise_dataframe

    if not len(snippets):
        raise ValueError("no snippet to aggregate")
    df = pd.concat(list(snippets), ignore_index=True)
    df["pid"] = str(name)
    agg = aggregate_raw_features(df).reset_index()
    channels = pd.DataFrame(
        {
            "channel": np.arange(len(geometry["x"])),
            "axial_um": np.asarray(geometry["y"], dtype=float),
            "lateral_um": np.asarray(geometry["x"], dtype=float),
        }
    )
    agg = agg.merge(channels, on="channel", how="left").set_index(["pid", "channel"])
    denoised = denoise_dataframe(
        agg, fac=DEFAULT_FAC, channel_labels=agg["channel_labels"].to_numpy()
    )
    return replace_nan(denoised, columns=denoised.columns)


def feature_matrix(
    table: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
    n_channels: int = 384,
) -> np.ndarray:
    """``[C, F]`` features in row order (top of the probe first: row ``r`` is channel
    ``C - 1 - r``), as :class:`~ephysatlas.alignment.data.ChannelDataset` lays out the release.

    Args:
        table: Per-channel features with a ``channel`` column or index level.
        feature_names: Column order; default the models' 43 features. Missing columns are NaN.
        n_channels: Number of channel rows.
    """
    from ephysatlas.spatial_encoder.utils import FEATURE_LIST

    names = list(FEATURE_LIST if feature_names is None else feature_names)
    d = table.reset_index() if "channel" in (table.index.names or []) else table
    channel = d["channel"].to_numpy(dtype=int)
    out = np.full((int(n_channels), len(names)), np.nan, dtype=np.float32)
    for k, name in enumerate(names):
        if name in d.columns:
            out[channel, k] = d[name].to_numpy(dtype=np.float32)
    out[np.isinf(out)] = 0.0
    return out[::-1].copy()


def _snippet_from_files(ap_file: str, lf_file: str, t_start: float, duration_s: float):
    """Worker: read one snippet from SpikeGLX files and compute its features."""
    import time

    import spikeglx

    t0 = time.time()
    sr_ap, sr_lf = spikeglx.Reader(ap_file), spikeglx.Reader(lf_file)
    raw_ap, raw_lf = read_snippet(sr_ap, sr_lf, t_start, duration_s)
    df = snippet_features(
        raw_ap, raw_lf, float(sr_ap.fs), float(sr_lf.fs), sr_ap.geometry
    )
    df.insert(0, "t_start", float(t_start))
    return df, time.time() - t0


def reference_features(
    ap_file,
    lf_file,
    t_starts: Sequence[float],
    *,
    duration_s: float = SNIPPET_S,
    n_workers: int = 1,
    name: str = "recording",
) -> dict:
    """Features of a SpikeGLX recording from snippets starting at ``t_starts`` (s, file time).

    Args:
        ap_file, lf_file: The ``.ap.bin`` / ``.lf.bin`` (or ``.cbin``) files, with their ``.meta``.
        t_starts: Snippet start times.
        duration_s: Snippet length.
        n_workers: Processes computing snippets in parallel (1: in this process).
        name: The ``pid`` level of the aggregated table's index.

    Returns:
        dict: ``snippets`` (one :func:`snippet_features` table per snippet, with ``t_start``),
        ``table`` (:func:`aggregate_snippets` of all of them), ``seconds`` (compute time per
        snippet) and ``geometry``.
    """
    import spikeglx

    ap_file, lf_file = str(ap_file), str(lf_file)
    geometry = spikeglx.Reader(ap_file).geometry
    t_starts = [float(t) for t in t_starts]
    if n_workers <= 1:
        out = [_snippet_from_files(ap_file, lf_file, t, duration_s) for t in t_starts]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(n_workers)) as pool:
            jobs = [
                pool.submit(_snippet_from_files, ap_file, lf_file, t, duration_s)
                for t in t_starts
            ]
            out = [job.result() for job in jobs]
    snippets = [df for df, _ in out]
    table = aggregate_snippets(
        [df.drop(columns="t_start") for df in snippets], geometry, name
    )
    return dict(
        snippets=snippets,
        table=table,
        seconds=np.array([s for _, s in out]),
        geometry=geometry,
    )
