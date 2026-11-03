"""Write recorded bands and converter-derived NP2 LF into snippet recording groups."""

import importlib.metadata

import h5py
import numpy as np
import spikeglx
from ibldsp.utils import WindowGenerator
from neuropixel import NP2Converter

from ephysatlas.feature_calculators.spikeglx_like import LF_LATENCY_SAMPLES

COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def _create_band(
    rec: h5py.Group,
    band: str,
    fs: float,
    gains: np.ndarray,
    first_sample: np.ndarray,
    ns: int,
) -> tuple[h5py.Group, h5py.Dataset]:
    """Create the common band layout, preserving integer samples and conversion gains."""
    group = rec.create_group(band)
    group.attrs["fs"] = float(fs)
    data = group.create_dataset(
        "data",
        shape=(len(first_sample), len(gains), ns),
        dtype=np.int16,
        chunks=(1, len(gains), min(ns, int(fs))),
        **COMPRESSION,
    )
    group.create_dataset("first_sample", data=first_sample)
    group.create_dataset("sample2volts", data=gains.astype(np.float64))
    return group, data


def _write_band(
    rec: h5py.Group,
    band: str,
    sr: spikeglx.Reader,
    order: np.ndarray,
    first_sample: np.ndarray,
    ns: int,
) -> int:
    """Copy one recorded band in AP geometry order; return its uncompressed byte count."""
    group, data = _create_band(
        rec, band, sr.fs, sr.sample2volts[order], first_sample, ns
    )
    if band == "lf":
        group.attrs.update(source="recorded", latency_samples=LF_LATENCY_SAMPLES)
    for i, n0 in enumerate(first_sample):
        data[i] = sr._raw[n0 : n0 + ns][:, order].T
    # Preserve the original metadata bytes, including its encoding and line endings.
    group.create_dataset(
        "meta", data=np.frombuffer(sr.file_meta_data.read_bytes(), dtype=np.uint8)
    )
    return data.size * data.dtype.itemsize


def _write_np2_lf(
    rec: h5py.Group,
    converter: NP2Converter,
    order: np.ndarray,
    first_sample: np.ndarray,
    ns: int,
) -> int:
    """Store exact windows of the full NP2Converter LF output without converting the file.

    Only intersecting converter chunks are read. Original chunk boundaries, tapering,
    edge trimming and int16 quantization are retained, including at the recording ends.
    The caller owns the converter's reader and closes it after writing.
    """
    gains = converter.sr.channel_conversion_sample2v["lf"][order]
    group, data = _create_band(rec, "lf", converter.fs_lf, gains, first_sample, ns)
    group.attrs.update(
        source="derived",
        method="neuropixel.NP2Converter.extract_lfp",
        ibl_neuropixel_version=importlib.metadata.version("ibl-neuropixel"),
        scipy_version=importlib.metadata.version("scipy"),
        numpy_version=importlib.metadata.version("numpy"),
        fs_ap=converter.fs_ap,
        ratio=converter.ratio,
        samples_window=converter.samples_window,
        samples_overlap=converter.samples_overlap,
        samples_taper=converter.samples_taper,
        latency_samples=LF_LATENCY_SAMPLES,
    )
    # TODO: investigate NP2Converter's ~3 kHz versus intended 1 kHz cutoff separately.
    # Preserve the upstream coefficients for reproducibility rather than changing them here.
    group.create_dataset("filter_sos", data=converter.sos_lp)
    windows = WindowGenerator(
        converter.nsamples, converter.samples_window, converter.samples_overlap
    )
    trim = 2 * converter.samples_taper
    for first, last in windows.firstlast:
        # Locate each retained chunk in the full, concatenated LF output.
        start = (first + (trim if windows.iw else 0)) // converter.ratio
        stop = (
            (last + converter.ratio - 1) // converter.ratio
            if windows.iw == windows.nwin - 1
            else (last - trim) // converter.ratio
        )
        intersecting = np.flatnonzero(
            (first_sample < stop) & (first_sample + ns > start)
        )
        if not intersecting.size:
            continue
        # Use disk channel order, as the converter does; restore AP geometry order below.
        raw = converter.sr[first:last, :].T
        lf = converter.extract_lfp(raw[: converter.napch])
        sync = converter.extract_lfp_sync(raw[converter.idxsyncch])
        # Reuse upstream trimming and integer conversion, including its rounding behavior.
        saved = converter._ind2save(
            lf, sync, windows, ratio=converter.ratio, etype="lf"
        )
        saved = saved[:, order].T
        for i in intersecting:
            left = max(int(first_sample[i]), start)
            right = min(int(first_sample[i]) + ns, stop)
            data[i, :, left - first_sample[i] : right - first_sample[i]] = saved[
                :, left - start : right - start
            ]
    return data.size * data.dtype.itemsize
