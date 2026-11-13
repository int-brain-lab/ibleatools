"""Write recorded bands and converter-derived NP2 LF into snippet recording groups."""

import h5py
import numpy as np
import spikeglx
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
    """Make the LF of each snippet with ``NP2Converter.extract_lfp`` and store it.

    The AP samples of a snippet are read with a margin on each side and filtered in one
    call. The margin is as long as the edge the converter discards from each of its
    chunks, and is dropped after filtering. The caller owns the converter's reader and
    closes it after writing.
    """
    gains = converter.sr.channel_conversion_sample2v["lf"][order]
    group, data = _create_band(rec, "lf", converter.fs_lf, gains, first_sample, ns)
    group.attrs.update(
        source="derived",
        method="neuropixel.NP2Converter.extract_lfp",
        latency_samples=LF_LATENCY_SAMPLES,
    )
    # TODO: investigate NP2Converter's ~3 kHz versus intended 1 kHz cutoff separately.
    # Preserve the upstream coefficients for reproducibility rather than changing them here.
    group.create_dataset("filter_sos", data=converter.sos_lp)
    margin = 2 * converter.samples_taper // converter.ratio  # in LF samples
    for i, n0 in enumerate(first_sample.tolist()):
        # LF sample k is AP sample k * ratio, so the read starts on that grid. A read
        # past the end of the recording stops at its last sample.
        start = max(n0 - margin, 0)
        stop = n0 + ns + margin
        # The converter's reader gives volts in disk channel order.
        ap = converter.sr[
            start * converter.ratio : stop * converter.ratio, : converter.napch
        ].T
        lf = converter.extract_lfp(ap)[order, n0 - start : n0 - start + ns]
        # To integers as the converter does: divide by the gain and truncate.
        data[i] = (lf / gains[:, np.newaxis]).astype(np.int16)
    return data.size * data.dtype.itemsize
