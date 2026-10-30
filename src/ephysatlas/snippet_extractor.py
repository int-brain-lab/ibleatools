"""Extract short raw AP and LF snippets from a SpikeGLX recording into one small HDF5 file.

Partner labs run :func:`extract_snippets` on their raw recordings (hundreds of GB each) and
send us the output, which the snippet-file feature calculator reads back.

File layout, format version 1 (``n`` snippets, ``c`` channels, sync channel removed)::

    <output>.h5
      attrs   format_version, ibleatools_version, ibl_neuropixel_version, created,
              source_ap_file, source_lf_file, probe_has_lf_band, neuropixel_version, duration,
              snippet_method, probe_model, referencing_scheme; snippet_seed and snippet_margin only when the snippet times
              were chosen automatically (snippet_method "stratified", not "explicit")
      t_start                 float64 (n,)       snippet start in the recording, s
      geometry/<key>          (c,)               every key of spikeglx.Reader.geometry
      ap/                     attrs fs
        data                  int16 (n, c, ns)   gzip level 4 + shuffle
        first_sample          int64 (n,)         first sample read from the original file
        sample2volts          float64 (c,)
        meta                  uint8 (n_bytes,)   the original .ap.meta file, exact bytes
      lf/                     as ap/, plus attrs source="recorded" and latency_samples;
                              only when the probe records LF and the LF file was found
      histology/<name>        uint8 (n_bytes,)   a histology file as stored on disk, attrs
                              source_path; a folder's files are <folder name>/<relative path>;
                              only when histology was given
      trajectory              group, attrs x, y, z, depth, theta, phi as floats; only when given

Channel rows follow the order of ``spikeglx.Reader`` (``sr[...]`` and ``sr.geometry``).
"""

import argparse
import datetime
import importlib.metadata
import logging
import numbers
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
import scipy.fft
import spikeglx

from ephysatlas import __version__ as ibleatools_version
from ephysatlas.feature_calculators import (
    SnippetWindow,
    check_snippet_overlap,
    get_snippet_start_times,
)
from ephysatlas.feature_calculators.spikeglx_like import LF_LATENCY_SAMPLES

LOGGER = logging.getLogger(__name__)

FORMAT_VERSION = 1
MAX_HISTOLOGY_BYTES = 100 * 1024**2
TRAJECTORY_KEYS = ("x", "y", "z", "depth", "theta", "phi")
# Seed of the stratified draw in get_snippet_start_times (its default).
SNIPPET_SEED = 0
COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def extract_snippets(
    ap_file: str | Path,
    output_file: str | Path,
    histology: Sequence[str | Path] | None = None,
    t_starts: Sequence[float] | None = None,
    n_snippets: int = 5,
    duration: float = 5.0,
    margin: float = 300.0,
    trajectory: Mapping[str, float] | None = None,
) -> Path:
    """Write raw AP and LF snippets of one SpikeGLX probe recording to an HDF5 file.

    The samples read are the ones ``SpikeGlxLikeFeatureCalculator.load_raw_snippet`` reads for
    the same window, stored as int16 with the gains to convert them to volts. The LF band is
    read ``LF_LATENCY_SAMPLES`` late, and ``first_sample`` records where it was read.

    Args:
        ap_file: The SpikeGLX ``.ap.bin`` or ``.ap.cbin`` file. Its ``.meta`` file must be next
            to it, and the LF file is looked for next to it as well.
        output_file: ``.h5`` file to write. It must not exist; its folder is created.
        histology: Files or folders to copy into the output as raw bytes, e.g.
            ``channel_locations.json``, ``xyz_picks.json`` or track CSVs of this probe. A folder
            is copied with everything under it. The total is limited to
            ``MAX_HISTOLOGY_BYTES``, which keeps image volumes out.
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
        ValueError: There are no snippets to extract, ``ap_file`` is not a SpikeGLX AP data
            file, the probe type is not
            supported, the ``.meta`` file has no channel map, the recording is too short for
            automatic times, explicit snippets overlap or reach outside the data, two
            histology files get the same name, the histology is over ``MAX_HISTOLOGY_BYTES``,
            or ``trajectory`` lacks a key or has a value that is not a number.
        FileNotFoundError: ``ap_file`` or a ``histology`` path doesn't exist.
        FileExistsError: ``output_file`` already exists.
    """
    ap_file, output_file = Path(ap_file), Path(output_file)
    if ".ap." not in ap_file.name or ap_file.suffix not in (".bin", ".cbin"):
        raise ValueError(f"pass the SpikeGLX .ap.bin or .ap.cbin file, got {ap_file}")
    if not ap_file.exists():
        raise FileNotFoundError(f"AP file not found: {ap_file}")
    exists_message = f"{output_file} already exists; choose another -o or delete it"
    if output_file.exists():
        raise FileExistsError(exists_message)
    automatic_times = t_starts is None
    if (len(t_starts) if t_starts is not None else n_snippets) < 1:
        raise ValueError(
            "Extract at least one snippet: give t_starts or n_snippets >= 1"
        )

    # Histology and trajectory are checked before the recording is opened and before any
    # file or folder is created.
    histology_files = {}  # name stored under histology/ -> resolved source path
    for path in map(Path, histology or ()):
        if not path.exists():
            raise FileNotFoundError(f"Histology path not found: {path}")
        if path.is_dir():
            # path.name is empty for "." and "..", so those use the resolved name.
            folder_name = path.name or path.resolve().name
            members = {
                f"{folder_name}/{p.relative_to(path).as_posix()}": p
                for p in sorted(path.rglob("*"))
                if p.is_file()
            }
            if not members:
                LOGGER.warning("Histology folder %s contains no files", path)
        else:
            members = {path.name: path}
        for name, member in members.items():
            member = member.resolve()
            # Only a path given twice is skipped. A file that is also inside a listed folder
            # is stored twice, under two names.
            if name in histology_files and histology_files[name] != member:
                raise ValueError(
                    f"Two histology files would both be stored as histology/{name}: "
                    f"{histology_files[name]} and {member}. Pass only one of them."
                )
            histology_files[name] = member
    histology_sizes = {name: p.stat().st_size for name, p in histology_files.items()}
    histology_total = sum(histology_sizes.values())
    if histology_files:
        LOGGER.info(
            "Storing %d histology files (%.2f MB): %s",
            len(histology_files),
            histology_total / 1024**2,
            ", ".join(histology_files),
        )
    if histology_total > MAX_HISTOLOGY_BYTES:
        largest = sorted(histology_sizes, key=histology_sizes.get, reverse=True)[:3]
        raise ValueError(
            f"The histology files total {histology_total / 1024**2:.0f} MB, "
            f"over the {MAX_HISTOLOGY_BYTES // 1024**2} MB limit. Largest: "
            + ", ".join(
                f"{name} ({histology_sizes[name] / 1024**2:.0f} MB)" for name in largest
            )
            + ". Leave out image volumes and pass only the small channel location, "
            "pick and track files."
        )
    if trajectory is not None:
        missing = [key for key in TRAJECTORY_KEYS if key not in trajectory]
        if missing:
            raise ValueError(
                f"trajectory is missing {missing}, it needs all of {list(TRAJECTORY_KEYS)}"
            )
        # bool is an int subclass, so True would pass as a number.
        not_numbers = [
            key
            for key in TRAJECTORY_KEYS
            if isinstance(trajectory[key], bool)
            or not isinstance(trajectory[key], numbers.Real)
        ]
        if not_numbers:
            raise ValueError(
                "trajectory values must be numbers, got "
                + ", ".join(f"{key}={trajectory[key]!r}" for key in not_numbers)
            )
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

    readers = {"ap": spikeglx.Reader(ap_file)}
    lf_base = ap_file.with_name(ap_file.name.replace(".ap.", ".lf."))
    other_suffix = {".bin": ".cbin", ".cbin": ".bin"}[ap_file.suffix]
    lf_candidates = (lf_base, lf_base.with_suffix(other_suffix))
    lf_file = next((p for p in lf_candidates if p.exists()), None)
    if not probe_has_lf_band:
        # NP2 never records LF, so an .lf file next to it was derived by someone else.
        if lf_file is not None:
            LOGGER.info("%s was ignored: this probe records no LF band", lf_file)
        lf_file = None
    elif lf_file is None:
        LOGGER.warning("No LF file found next to %s, storing AP only", ap_file)
    else:
        readers["lf"] = spikeglx.Reader(lf_file)
    sr_ap = readers["ap"]

    # AP and LF files share the channel order on disk, but an LF meta can lack the
    # geometry keys, so the AP order is used for both bands. It also drops the sync
    # channel.
    order = sr_ap.raw_channel_order[: sr_ap.nc - sr_ap.nsync]
    n_channels = order.size

    partial = output_file.with_name(output_file.name + ".partial")
    raw_bytes = 0
    try:
        # The length of the data, not fileTimeSecs: the meta can claim more than the file
        # holds, and spikeglx.Reader corrects ns for that.
        length = min(sr.ns / sr.fs for sr in readers.values())
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
        reads = {}
        for band, sr in readers.items():
            ns = scipy.fft.next_fast_len(int(sr.fs * duration), real=True)
            latency = LF_LATENCY_SAMPLES if band == "lf" else 0
            first_sample = np.array(
                [int(sr.fs * t) + latency for t in t_starts], dtype=np.int64
            )
            outside = (first_sample < 0) | (first_sample + ns > sr.ns)
            if outside.any():
                i = np.flatnonzero(outside)[0]
                raise ValueError(
                    f"The snippet at t = {t_starts[i]:g} s needs {band.upper()} "
                    f"samples {first_sample[i]} to {first_sample[i] + ns}, but the data has "
                    f"{sr.ns} samples ({sr.ns / sr.fs:.3f} s)."
                )
            reads[band] = (first_sample, ns)

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
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["ibleatools_version"] = ibleatools_version
            # The geometry, row order and gains stored below are this version's reading of the meta.
            f.attrs["ibl_neuropixel_version"] = ibl_neuropixel_version
            f.attrs["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            )
            f.attrs["source_ap_file"] = str(ap_file.resolve())
            f.attrs["source_lf_file"] = str(lf_file.resolve()) if lf_file else ""
            f.attrs["probe_has_lf_band"] = probe_has_lf_band
            f.attrs["neuropixel_version"] = neuropixel_version
            f.attrs["duration"] = float(duration)
            f.attrs["snippet_method"] = "stratified" if automatic_times else "explicit"
            if automatic_times:
                f.attrs["snippet_seed"] = SNIPPET_SEED
                f.attrs["snippet_margin"] = float(margin)
            f.attrs["probe_model"] = spikeglx.get_probe_model(sr_ap.meta) or ""
            f.attrs["referencing_scheme"] = (
                spikeglx.get_referencing_scheme(sr_ap.meta) or ""
            )
            f.create_dataset("t_start", data=t_starts)
            geometry = f.create_group("geometry")
            for key, values in sr_ap.geometry.items():
                geometry.create_dataset(key, data=np.asarray(values))

            for band, sr in readers.items():
                fs = float(sr.fs)
                first_sample, ns = reads[band]

                group = f.create_group(band)
                group.attrs["fs"] = fs
                if band == "lf":
                    group.attrs["source"] = "recorded"
                    group.attrs["latency_samples"] = LF_LATENCY_SAMPLES
                data = group.create_dataset(
                    "data",
                    shape=(len(t_starts), n_channels, ns),
                    dtype=np.int16,
                    chunks=(1, n_channels, min(ns, int(fs))),
                    **COMPRESSION,
                )
                for i, n0 in enumerate(first_sample):
                    data[i] = sr._raw[n0 : n0 + ns][:, order].T
                raw_bytes += data.size * data.dtype.itemsize
                group.create_dataset("first_sample", data=first_sample)
                group.create_dataset(
                    "sample2volts", data=sr.sample2volts[order].astype(np.float64)
                )
                group.create_dataset(
                    "meta",
                    # Exact file bytes: a decode would fail on a meta in another encoding.
                    data=np.frombuffer(sr.file_meta_data.read_bytes(), dtype=np.uint8),
                )

            if histology_files:
                group = f.create_group("histology")
                for name, path in histology_files.items():
                    dataset = group.create_dataset(
                        name, data=np.frombuffer(path.read_bytes(), dtype=np.uint8)
                    )
                    dataset.attrs["source_path"] = str(path)
            if trajectory is not None:
                f.create_group("trajectory").attrs.update(trajectory)
        # Checked again because the file can appear while the snippets are being written.
        if output_file.exists():
            raise FileExistsError(exists_message)
        partial.rename(output_file)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        for sr in readers.values():
            sr.close()

    size = output_file.stat().st_size
    LOGGER.info(
        "Wrote %s: %.1f MB, %.2fx smaller than the raw data",
        output_file,
        size / 1e6,
        raw_bytes / size,
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
        description="Extract raw AP and LF snippets of one SpikeGLX probe recording into an HDF5 file.",
    )
    parser.add_argument(
        "ap_file", type=Path, help="the .ap.bin or .ap.cbin file of the probe"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="the .h5 file to write; it must not exist",
    )
    parser.add_argument(
        "--histology",
        type=Path,
        nargs="+",
        metavar="PATH",
        help="histology files or folders of this probe, copied as they are",
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
            histology=args.histology,
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
