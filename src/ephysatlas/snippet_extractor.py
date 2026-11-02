"""Pack raw AP and recorded or converter-derived LF snippets into one small HDF5 file.

Partner labs run :func:`extract_snippets` on their raw recordings (hundreds of GB each) and
send us the output, which the snippet-file feature calculator reads back.

File layout (``n`` snippets, ``c`` channels, sync channel removed)::

    <output>.h5
      attrs   created
      recordings/<name>/      one group per recording
        attrs   ibleatools_version, ibl_neuropixel_version, iblatlas_version, created,
                source_ap_file, source_lf_file, probe_has_lf_band, neuropixel_version,
                duration, snippet_method, probe_model, referencing_scheme; snippet_seed
                and snippet_margin only when the snippet times were chosen automatically
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
                                derived LF has no original .meta; the method attribute
                                and filter_sos record the NP2Converter filter instead
        histology/raw/<name>    uint8 (n_bytes,)   original histology bytes, attrs source_path
        histology/channels/     ``Histology.channels`` (x, y, z, atlas_id, acronym), from
                                channel_locations; only when given
        histology/xyz_picks     ``Histology.xyz_picks``, from xyz_picks; only when given
        trajectory              group, attrs x, y, z, depth, theta, phi; only when given

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
from ephysatlas._snippet_histology import (
    ChannelLocations as ChannelLocations,
    Histology as Histology,
    _write_model,
    read_histology,
)
from ephysatlas.feature_calculators import (
    SnippetWindow,
    check_snippet_overlap,
    get_snippet_start_times,
)
from ephysatlas.feature_calculators.spikeglx_like import LF_LATENCY_SAMPLES

LOGGER = logging.getLogger(__name__)

TRAJECTORY_KEYS = ("x", "y", "z", "depth", "theta", "phi")
# Limit on the total size of the raw histology files of one recording.
MAX_HISTOLOGY_BYTES = 100 * 1024**2
# Seed of the stratified draw in get_snippet_start_times (its default).
SNIPPET_SEED = 0


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
    channel_locations: str | Path | None = None,
    xyz_picks: str | Path | None = None,
    histology_raw: Sequence[str | Path] | None = None,
    t_starts: Sequence[float] | None = None,
    n_snippets: int = 5,
    duration: float = 5.0,
    margin: float = 300.0,
    trajectory: Mapping[str, float] | None = None,
) -> Path:
    """Write raw AP and recorded or derived LF snippets of one SpikeGLX probe recording.

    AP and NP1/3A LF samples are copied unchanged. NP2 LF is made from the AP samples
    around each snippet with ``neuropixel.NP2Converter.extract_lfp``, and converted to
    int16 as the converter does.
    All LF windows start ``LF_LATENCY_SAMPLES`` late, matching the direct SpikeGLX feature
    calculator. ``first_sample`` records the exact source-band sample indices.

    Args:
        ap_file: The SpikeGLX ``.ap.bin`` or ``.ap.cbin`` file. Its ``.meta`` file must be next
            to it. NP1/3A also require the LF file of the same name and extension
            (``.lf.bin`` or ``.lf.cbin``) with its ``.meta`` file next to it.
        output_file: ``.h5`` file to write; its folder is created. If it exists, it must be a
            snippet file, and the recording is added to it.
        name: Name of the recording inside the file, stored under ``recordings/<name>``. If
            None, the AP file name without ``.ap.bin`` / ``.ap.cbin``.
        channel_locations: ``channel_locations.json`` of this probe, as the IBL alignment
            GUI writes it. Stored as ``histology/channels`` (``read_histology``) and raw.
        xyz_picks: ``xyz_picks.json`` of this probe. Stored as ``histology/xyz_picks`` and raw.
        histology_raw: Other histology files to copy into the output as raw bytes, e.g.
            track CSVs of this probe. Each is stored under its file name. The total is
            limited to 100 MB including the two json inputs, which keeps image
            volumes out.
        t_starts: Snippet start times in seconds. If None, ``n_snippets`` start times are
            chosen by ``get_snippet_start_times`` over the length of the data in the files.
        n_snippets: Number of snippets when ``t_starts`` is None.
        duration: Length of each snippet in seconds, for both bands.
        margin: Seconds skipped at each end of the recording by automatic times.
        trajectory: Planned probe insertion with the keys ``TRAJECTORY_KEYS``: x, y, z in um
            relative to bregma, depth in um along the probe, theta and phi in degrees (IBL
            Alyx units). Other keys are ignored.

    Returns:
        Path of the written file.

    Raises:
        ValueError: ``ap_file`` is not a SpikeGLX AP data file, ``output_file`` already
            holds a recording called ``name``, the probe type is not supported, the
            ``.meta`` file has no channel map, the recording is too short for automatic
            times, explicit snippets overlap or reach outside the data, the histology
            files are over 100 MB, a trajectory value is not a number, or the
            histology json files don't fit the recording or ``Histology``.
        FileNotFoundError: The ``.meta`` file of ``ap_file``, a required NP1/3A LF file
            (both raised by ``spikeglx``), or a histology file does not exist.
        IsADirectoryError: A ``histology_raw`` path is a folder instead of a file.
        KeyError: ``trajectory`` lacks one of ``TRAJECTORY_KEYS``.
        IndexError: iblatlas doesn't know a ``brain_region_id`` of ``channel_locations``.
    """
    ap_file, output_file = Path(ap_file), Path(output_file)
    if ".ap." not in ap_file.name or ap_file.suffix not in (".bin", ".cbin"):
        raise ValueError(f"pass the SpikeGLX .ap.bin or .ap.cbin file, got {ap_file}")
    if name is None:
        # Everything before the last ".ap.": rec_g0_t0.imec0.ap.cbin -> rec_g0_t0.imec0
        name = ap_file.name.rsplit(".ap.", 1)[0]
    automatic_times = t_starts is None
    # A stored recording is never replaced, so a name can be used once per output file.
    if output_file.exists():
        with h5py.File(output_file) as f:
            if name in f["recordings"]:
                stored = f[f"recordings/{name}"].attrs
                raise ValueError(
                    f"{output_file} already holds recording {name!r}, extracted from "
                    f"{stored['source_ap_file']} on {stored['created']}. If this is a "
                    "different recording, give it another name with --name (name=... in "
                    "Python)."
                )

    # Check the total size and convert the trajectory before opening the recording.
    # The size limit keeps image volumes out of the file.
    # The histology json files are also stored as raw bytes.
    histology_raw = [
        Path(p)
        for p in (*(histology_raw or ()), *filter(None, (channel_locations, xyz_picks)))
    ]
    histology_bytes = sum(path.stat().st_size for path in histology_raw)
    if histology_bytes > MAX_HISTOLOGY_BYTES:
        raise ValueError(
            f"The histology files total {histology_bytes / 1024**2:.0f} MB, over the "
            f"{MAX_HISTOLOGY_BYTES // 1024**2} MB limit. Leave out image volumes and pass "
            "only the small channel location, pick and track files."
        )
    if trajectory is not None:
        trajectory = {key: float(trajectory[key]) for key in TRAJECTORY_KEYS}

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

    # NP1/3A record LF in a file of its own, next to the AP file. spikeglx.Reader raises
    # FileNotFoundError below if it is missing.
    lf_file = None
    if probe_has_lf_band:
        lf_file = ap_file.with_name(ap_file.name.replace(".ap.", ".lf."))

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
        # Converted on the Reader's geometry, so the channels follow the stored data rows.
        histology = read_histology(channel_locations, xyz_picks, sr_ap.geometry)

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
            t_starts = get_snippet_start_times(
                length, n_snippets, duration, margin=margin
            )
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
            rec.attrs["iblatlas_version"] = importlib.metadata.version("iblatlas")
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

            if histology_raw:
                group = rec.create_group("histology")
                for path in histology_raw:
                    dataset = group.create_dataset(
                        f"raw/{path.name}",
                        data=np.frombuffer(path.read_bytes(), dtype=np.uint8),
                    )
                    dataset.attrs["source_path"] = str(path.resolve())
                _write_model(group, histology)
            if trajectory is not None:
                rec.create_group("trajectory").attrs.update(trajectory)
        size = partial.stat().st_size
        LOGGER.info(
            "Recording %s: %.1f MB, %.2fx smaller than the raw data",
            name,
            size / 1e6,
            raw_bytes / size,
        )
        if not output_file.exists():
            partial.rename(output_file)
        else:
            # h5py refuses to copy onto a name that already exists.
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
        "--channel-locations", type=Path, help="channel_locations.json of this probe"
    )
    parser.add_argument("--xyz-picks", type=Path, help="xyz_picks.json of this probe")
    parser.add_argument(
        "--histology-raw",
        type=Path,
        nargs="+",
        metavar="PATH",
        help="other histology files of this probe, copied as they are",
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
    parser.add_argument(
        "--trajectory",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "DEPTH", "THETA", "PHI"),
        help="planned insertion, as in IBL Alyx trajectories: x y z in um relative to "
        "bregma (x: ML, right positive; y: AP, anterior positive; z: DV, dorsal "
        "positive), depth in um along the probe, theta (from straight up) and phi "
        "(from the +x axis) in degrees. Your own conventions may differ; convert "
        "them first",
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

    trajectory = (
        dict(zip(TRAJECTORY_KEYS, args.trajectory)) if args.trajectory else None
    )
    try:
        extract_snippets(
            args.ap_file,
            args.output,
            name=args.name,
            channel_locations=args.channel_locations,
            xyz_picks=args.xyz_picks,
            histology_raw=args.histology_raw,
            t_starts=args.t_starts,
            n_snippets=args.n_snippets,
            duration=args.duration,
            margin=args.margin,
            trajectory=trajectory,
        )
    except Exception as err:
        LOGGER.error("%s", err)
        return 1
    return 0
