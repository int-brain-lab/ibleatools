"""Pack raw AP and recorded or converter-derived LF snippets into one small HDF5 file.

Partner labs run :func:`extract_snippets` on their raw recordings (hundreds of GB each) and
send us the output, with the raw samples and their metadata kept together.

File layout (``n`` snippets, ``c`` channels, sync channel removed)::

    <output>.h5
      attrs   created
      recordings/<name>/      one group per recording
        attrs   ibleatools_version, ibl_neuropixel_version, created, source_ap_file,
                source_lf_file, probe_has_lf_band, neuropixel_version, duration,
                snippet_method, probe_model, referencing_scheme; snippet_seed and
                snippet_margin only when the snippet times were chosen automatically
                (snippet_method "stratified", not "explicit")
        t_start                 float64 (n,)       snippet start in the recording, s
        geometry/<key>          (c,)               every key of spikeglx.Reader.geometry
        ap/                     attrs fs
          data                  int16 (n, c, ns)   gzip level 4 + shuffle
          first_sample          int64 (n,)         first sample read from the original file
          sample2volts          float64 (c,)
          meta                  uint8 (n_bytes,)   the original .ap.meta file, exact bytes
        lf/                     always present; as ap/, plus attrs source and latency_samples
                                source="recorded" for NP1/3A, "derived" for NP2
                                derived LF has no original .meta; filter_sos and conversion
                                attributes record the NP2Converter processing instead

Channel rows follow the order of ``spikeglx.Reader`` (``sr[...]`` and ``sr.geometry``).
"""

import argparse
import datetime
import importlib.metadata
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
import scipy.fft
import spikeglx
from neuropixel import NP2Converter

from ephysatlas import __version__ as ibleatools_version
from ephysatlas._snippet_bands import _write_band, _write_np2_lf
from ephysatlas.feature_calculators import (
    SnippetWindow,
    check_snippet_overlap,
    get_snippet_start_times,
)
from ephysatlas.feature_calculators.spikeglx_like import LF_LATENCY_SAMPLES

LOGGER = logging.getLogger(__name__)

# Seed of the stratified draw in get_snippet_start_times (its default).
SNIPPET_SEED = 0


def _check_output_accepts(output_file: Path, name: str) -> None:
    """Raise ValueError if an existing ``output_file`` can't take the recording ``name``.

    Does nothing if ``output_file`` doesn't exist. Used before reading the recording and
    again before it is added, because the file can change in between.
    """
    if not output_file.exists():
        return
    not_snippet_file = ValueError(
        f"{output_file} is not a snippet file; choose another -o"
    )
    if not h5py.is_hdf5(output_file):
        raise not_snippet_file
    with h5py.File(output_file) as f:
        if not isinstance(f.get("recordings"), h5py.Group):
            raise not_snippet_file
        if name in f.get("recordings", {}):
            stored = f[f"recordings/{name}"].attrs
            raise ValueError(
                f"{output_file} already holds recording {name!r}, extracted from "
                f"{stored['source_ap_file']} on {stored['created']}. If this is a different "
                "recording, give it another name with --name (name=... in Python)."
            )


def _plan_reads(
    bands: Mapping[str, tuple[float, int]], t_starts: np.ndarray, duration: float
) -> dict[str, tuple[np.ndarray, int]]:
    """Calculate band offsets and FFT lengths, rejecting windows outside the data."""
    reads = {}
    for band, (fs, available_samples) in bands.items():
        ns = scipy.fft.next_fast_len(int(fs * duration), real=True)
        latency = LF_LATENCY_SAMPLES if band == "lf" else 0
        first_sample = np.array(
            [int(fs * t) + latency for t in t_starts], dtype=np.int64
        )
        outside = (first_sample < 0) | (first_sample + ns > available_samples)
        if outside.any():
            i = np.flatnonzero(outside)[0]
            raise ValueError(
                f"The snippet at t = {t_starts[i]:g} s needs {band.upper()} "
                f"samples {first_sample[i]} to {first_sample[i] + ns}, but the data has "
                f"{available_samples} samples ({available_samples / fs:.3f} s)."
            )
        reads[band] = (first_sample, ns)

    return reads


def extract_snippets(
    ap_file: str | Path,
    output_file: str | Path,
    name: str | None = None,
    t_starts: Sequence[float] | None = None,
    n_snippets: int = 5,
    duration: float = 5.0,
    margin: float = 300.0,
) -> Path:
    """Write raw AP and recorded or derived LF snippets of one SpikeGLX probe recording.

    AP and NP1/3A LF samples are copied unchanged. NP2 LF matches the int16 output of
    ``neuropixel.NP2Converter`` on the original recording, including chunk boundaries,
    tapering and quantization; only the chunks needed for these snippets are processed.
    All LF windows start ``LF_LATENCY_SAMPLES`` late, matching the direct SpikeGLX feature
    calculator. ``first_sample`` records the exact source-band sample indices.

    Args:
        ap_file: The SpikeGLX ``.ap.bin`` or ``.ap.cbin`` file. Its ``.meta`` file must be next
            to it. NP1/3A also require matching LF data and metadata next to it.
        output_file: ``.h5`` file to write; its folder is created. If it exists, it must be a
            snippet file, and the recording is added to it.
        name: Name of the recording inside the file, stored under ``recordings/<name>``. If
            None, the AP file name without ``.ap.bin`` / ``.ap.cbin``. It may not be empty
            or ``.``, and may not contain ``/``.
        t_starts: Snippet start times in seconds. If None, ``n_snippets`` start times are
            chosen by ``get_snippet_start_times`` over the length of the data in the files.
        n_snippets: Number of snippets when ``t_starts`` is None.
        duration: Length of each snippet in seconds, for both bands.
        margin: Seconds skipped at each end of the recording by automatic times.

    Returns:
        Path of the written file.

    Raises:
        ValueError: There are no snippets to extract, ``ap_file`` is not a SpikeGLX AP data
            file, ``name`` is not a valid recording name, ``output_file`` exists but is not a
            snippet file or already holds a recording called ``name``,
            the probe type is not supported, the ``.meta`` file has no channel map, the
            recording is too short for automatic times, explicit snippets overlap or reach
            outside the data.
        FileNotFoundError: The AP file or a required NP1/3A LF file does not exist.
    """
    ap_file, output_file = Path(ap_file), Path(output_file)
    if ".ap." not in ap_file.name or ap_file.suffix not in (".bin", ".cbin"):
        raise ValueError(f"pass the SpikeGLX .ap.bin or .ap.cbin file, got {ap_file}")
    if name is None:
        # Everything before the last ".ap.": rec_g0_t0.imec0.ap.cbin -> rec_g0_t0.imec0
        name = ap_file.name.rsplit(".ap.", 1)[0]
    if name in ("", ".") or "/" in name:
        raise ValueError(
            f"Recording name {name!r} is not valid: it may not be empty or '.', "
            "and may not contain '/'"
        )
    if not ap_file.exists():
        raise FileNotFoundError(f"AP file not found: {ap_file}")
    automatic_times = t_starts is None
    if (len(t_starts) if t_starts is not None else n_snippets) < 1:
        raise ValueError(
            "Extract at least one snippet: give t_starts or n_snippets >= 1"
        )
    # An existing output is checked before anything is read, and again before it is changed.
    _check_output_accepts(output_file, name)

    # Probe facts come from the meta: spikeglx.Reader fails with unclear errors on
    # probe types it doesn't know.
    ibl_neuropixel_version = importlib.metadata.version("ibl-neuropixel")
    md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
    version = spikeglx._get_neuropixel_major_version_from_meta(md)
    if version not in (1, 2, 2.4):
        raise ValueError(
            f"Probe type imDatPrb_type={md.get('imDatPrb_type'):g} (part number "
            f"{md.get('imDatPrb_pn')}) is not supported by ibl-neuropixel "
            f"{ibl_neuropixel_version}. Please report this probe type to the ibleatools "
            "maintainers."
        )
    if "snsGeomMap" not in md and "snsShankMap" not in md:
        raise ValueError(
            f"{ap_file.with_suffix('.meta')} has no channel map (snsGeomMap or snsShankMap), "
            "so the electrode positions are unknown. Use the .meta file SpikeGLX wrote "
            "for this recording."
        )
    neuropixel_version = int(version)
    probe_has_lf_band = version == 1

    lf_file = None
    if probe_has_lf_band:
        lf_base = ap_file.with_name(ap_file.name.replace(".ap.", ".lf."))
        other_suffix = {".bin": ".cbin", ".cbin": ".bin"}[ap_file.suffix]
        lf_candidates = (lf_base, lf_base.with_suffix(other_suffix))
        lf_file = next((p for p in lf_candidates if p.exists()), None)
        if lf_file is None:
            raise FileNotFoundError(
                f"No LF file found next to {ap_file}; NP1/3A extraction requires a matching "
                ".lf.bin or .lf.cbin file with its .meta file."
            )

    # The recording is always written to its own temporary file first, then renamed or
    # copied into the output, so a failure never leaves a half-written output.
    partial = output_file.with_name(f"{output_file.name}.{name}.partial")
    raw_bytes = 0
    readers = {}
    converter = None
    try:
        readers["ap"] = spikeglx.Reader(ap_file)
        if probe_has_lf_band:
            readers["lf"] = spikeglx.Reader(lf_file)
        else:
            # Always derive NP2 LF from the original AP, regardless of adjacent LF files.
            converter = NP2Converter(ap_file, compress=False)
        sr_ap = readers["ap"]
        order = sr_ap.raw_channel_order[: sr_ap.nc - sr_ap.nsync]
        bands = {band: (sr.fs, sr.ns) for band, sr in readers.items()}
        if converter is not None:
            # The full converter keeps every ratio-th sample, including a partial last bin.
            bands["lf"] = (
                converter.fs_lf,
                (converter.sr.ns + converter.ratio - 1) // converter.ratio,
            )
        # The length of the data, not fileTimeSecs: the meta can claim more than the file
        # holds, and spikeglx.Reader corrects ns for that.
        length = min(ns / fs for fs, ns in bands.values())
        if automatic_times:
            try:
                t_starts = get_snippet_start_times(
                    length, n_snippets, duration, margin=margin
                )
            except ValueError as error:
                raise ValueError(
                    f"{error}. Pass a smaller margin "
                    "(margin=... in Python, --margin on the command line)."
                ) from error
        else:
            t_starts = np.asarray(t_starts, dtype=np.float64)
            check_snippet_overlap(
                [SnippetWindow(t, duration, duration) for t in t_starts]
            )

        # Same offsets and sample counts as the SpikeGLX feature calculator. Every read
        # is checked against the data before any file is created.
        reads = _plan_reads(bands, t_starts, duration)

        LOGGER.info(
            "Extracting snippets from %s (imDatPrb_pn %s, imDatPrb_type %s, probe version %s, "
            "LF file %s) at t = %s s",
            ap_file,
            md.get("imDatPrb_pn"),
            md.get("imDatPrb_type"),
            sr_ap.version,
            lf_file,
            t_starts.tolist(),
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(partial, "w") as f:
            created = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            )
            f.attrs["created"] = created
            rec = f.create_group(f"recordings/{name}")
            rec.attrs["ibleatools_version"] = ibleatools_version
            # The geometry, row order and gains stored below are this version's reading of the meta.
            rec.attrs["ibl_neuropixel_version"] = ibl_neuropixel_version
            rec.attrs["created"] = created
            rec.attrs["source_ap_file"] = str(ap_file.resolve())
            rec.attrs["source_lf_file"] = str(lf_file.resolve()) if lf_file else ""
            rec.attrs["probe_has_lf_band"] = probe_has_lf_band
            rec.attrs["neuropixel_version"] = neuropixel_version
            rec.attrs["duration"] = float(duration)
            rec.attrs["snippet_method"] = (
                "stratified" if automatic_times else "explicit"
            )
            if automatic_times:
                rec.attrs["snippet_seed"] = SNIPPET_SEED
                rec.attrs["snippet_margin"] = float(margin)
            rec.attrs["probe_model"] = spikeglx.get_probe_model(sr_ap.meta) or ""
            rec.attrs["referencing_scheme"] = (
                spikeglx.get_referencing_scheme(sr_ap.meta) or ""
            )
            rec.create_dataset("t_start", data=t_starts)
            geometry = rec.create_group("geometry")
            for key, values in sr_ap.geometry.items():
                geometry.create_dataset(key, data=np.asarray(values))

            for band, sr in readers.items():
                first_sample, ns = reads[band]
                raw_bytes += _write_band(rec, band, sr, order, first_sample, ns)
            if converter is not None:
                raw_bytes += _write_np2_lf(rec, converter, order, *reads["lf"])

        size = partial.stat().st_size
        LOGGER.info(
            "Recording %s: %.1f MB, %.2fx smaller than the raw data",
            name,
            size / 1e6,
            raw_bytes / size,
        )
        # Checked again because the output can change while the snippets are being written.
        if not output_file.exists():
            partial.rename(output_file)
        else:
            _check_output_accepts(output_file, name)
            with h5py.File(partial) as src, h5py.File(output_file, "a") as dst:
                src.copy(src[f"recordings/{name}"], dst["recordings"], name)
    finally:
        # Gone after a rename already; removed after a copy and after any failure.
        partial.unlink(missing_ok=True)
        for sr in readers.values():
            sr.close()
        if converter is not None:
            converter.sr.close()

    with h5py.File(output_file) as f:
        n_recordings = len(f["recordings"])
    LOGGER.info(
        "Added recording %s to %s (%d recordings, %.1f MB)",
        name,
        output_file,
        n_recordings,
        output_file.stat().st_size / 1e6,
    )
    return output_file


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point of ``ibleatools-snippets``.

    Args:
        argv: Command line arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if extraction failed. The error is logged without a traceback.
    """
    parser = argparse.ArgumentParser(
        prog="ibleatools-snippets",
        description="Pack AP and LF snippets of one SpikeGLX probe recording into HDF5; "
        "NP2 LF is derived using ibl-neuropixel's NP2Converter.",
    )
    parser.add_argument(
        "ap_file", type=Path, help="the .ap.bin or .ap.cbin file of the probe"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="the .h5 file to write; if it exists, the recording is added to it",
    )
    parser.add_argument(
        "--name",
        help="name of the recording inside the file; default: the AP file name without "
        ".ap.bin/.ap.cbin",
    )
    parser.add_argument(
        "--t-starts",
        type=float,
        nargs="+",
        metavar="T",
        help="snippet start times in seconds; default: spread over the recording",
    )
    parser.add_argument(
        "--n-snippets",
        type=int,
        default=5,
        help="number of snippets when --t-starts is not given",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, help="snippet length in seconds"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=300.0,
        help="seconds skipped at each end when the start times are automatic",
    )
    args = parser.parse_args(argv)

    # Handler on the package logger, added once, so repeated calls do not duplicate messages
    # and the root logger is left alone.
    package_logger = logging.getLogger("ephysatlas")
    if not package_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)

    try:
        extract_snippets(
            args.ap_file,
            args.output,
            name=args.name,
            t_starts=args.t_starts,
            n_snippets=args.n_snippets,
            duration=args.duration,
            margin=args.margin,
        )
    except Exception as err:
        LOGGER.error("%s", err)
        return 1
    return 0
