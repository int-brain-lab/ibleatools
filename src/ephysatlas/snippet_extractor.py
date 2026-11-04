"""Pack raw AP and recorded or converter-derived LF snippets into one small HDF5 file.

Partner labs run :func:`extract_snippets` on their raw recordings (hundreds of GB each) and
send us the output, which the snippet-file feature calculator reads back.

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
        histology/              only when histology was given
          raw/<name>            uint8 (n_bytes,)   a histology file as stored on disk, attrs
                                source_path; a folder's files are <folder name>/<relative
                                path>; the given channel_locations and xyz_picks files are
                                stored here too
          channels/             the channel table converted from channel_locations; attrs
                                source, atlas, bregma_ccf_um, iblatlas_version,
                                ibleatools_version
            x, y, z             float64 (c,)       metres, IBL coordinates relative to bregma
            atlas_id            int64 (c,)         Allen CCF 2017 region id, 0 is void
            acronym             string (c,)        from iblatlas
          xyz_picks             float64 (k, 3)     the traced track in metres, IBL coordinates,
                                                   attrs source; only when xyz_picks was given
        trajectory              group, attrs x, y, z, depth, theta, phi as floats; only when
                                given

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
from neuropixel import NP2Converter

from ephysatlas import __version__ as ibleatools_version
from ephysatlas._snippet_bands import _write_band, _write_np2_lf
from ephysatlas._snippet_histology import (
    _collect_histology_files,
    _read_channel_locations,
    _read_xyz_picks,
    _write_channel_table,
    add_channel_locations as add_channel_locations,
)
from ephysatlas.feature_calculators import (
    SnippetWindow,
    check_snippet_overlap,
    get_snippet_start_times,
)
from ephysatlas.feature_calculators.spikeglx_like import LF_LATENCY_SAMPLES

LOGGER = logging.getLogger(__name__)

TRAJECTORY_KEYS = ("x", "y", "z", "depth", "theta", "phi")
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
        channel_locations: IBL ``channel_locations.json`` of this probe. It is converted to
            the channel table ``histology/channels`` (see ``_read_channel_locations``) and
            also stored as raw bytes. It needs a single-shank recording and an entry for
            every channel.
        xyz_picks: IBL ``xyz_picks.json`` of this probe, the traced track in um. It is stored
            in metres as ``histology/xyz_picks`` and also as raw bytes.
        histology_raw: Other histology files or folders to copy into the output as raw bytes,
            e.g. track CSVs of this probe. A folder is copied with everything under it.
            Together with the two json files, the total is limited to
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
            file, ``name`` is not a valid recording name, ``output_file`` exists but is not a
            snippet file or already holds a recording called ``name``,
            the probe type is not supported, the ``.meta`` file has no channel map, the
            recording is too short for automatic times, explicit snippets overlap or reach
            outside the data, two histology files get the same name, the histology is over
            ``MAX_HISTOLOGY_BYTES``, ``channel_locations`` is given for a multi-shank
            recording, doesn't fit the recording or is not shaped as expected,
            ``xyz_picks`` holds fewer than 2 points of 3 values or is not shaped as
            expected, ``trajectory`` lacks a key or has a value that is not a number, or
            the channel order of the ``.meta`` file and of ``spikeglx.Reader`` differ.
        FileNotFoundError: The AP file, a required NP1/3A LF file, or a histology path
            does not exist.
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

    # Histology and trajectory are checked before the recording is opened and before any
    # file or folder is created. The given json files are stored as raw bytes as well, so
    # the converted histology can be rebuilt from them.
    histology_paths = list(histology_raw or ())
    histology_paths.extend(p for p in (channel_locations, xyz_picks) if p is not None)
    histology_files = _collect_histology_files(histology_paths)
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

    # The histology is converted on the geometry from the meta, the same geometry and row
    # order as spikeglx.Reader's, so a file that doesn't fit the recording stops the run
    # before the recording is opened.
    channel_table = None
    if channel_locations is not None:
        channel_locations = Path(channel_locations)
        meta_geometry = spikeglx.geometry_from_meta(md)
        # IBL aligns each shank on its own, so a json describes one shank. A multi-shank
        # recording is refused before the json is read, with the clearer message.
        if np.unique(meta_geometry["shank"]).size > 1:
            raise ValueError(
                "channel_locations needs a single-shank recording; pass the file with "
                "--histology-raw instead"
            )
        channel_table = _read_channel_locations(channel_locations, meta_geometry)
        LOGGER.info(
            "Converted %s to the channel table of %d channels",
            channel_locations.name,
            meta_geometry["ind"].size,
        )
    picks = None
    if xyz_picks is not None:
        xyz_picks = Path(xyz_picks)
        picks = _read_xyz_picks(xyz_picks)

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
        if channel_table is not None and not np.array_equal(
            meta_geometry["ind"], sr_ap.geometry["ind"]
        ):
            raise ValueError(
                "The channel order from the meta file and from spikeglx.Reader differ "
                f"(ibl-neuropixel {ibl_neuropixel_version}), so the channel table can't be "
                "matched to the data rows. Please report this to the ibleatools maintainers."
            )

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

            if histology_files:
                group = rec.create_group("histology")
                for hist_name, path in histology_files.items():
                    dataset = group.create_dataset(
                        f"raw/{hist_name}",
                        data=np.frombuffer(path.read_bytes(), dtype=np.uint8),
                    )
                    dataset.attrs["source_path"] = str(path)
                if channel_table is not None:
                    _write_channel_table(group, channel_table, channel_locations.name)
                if picks is not None:
                    dataset = group.create_dataset("xyz_picks", data=picks / 1e6)
                    dataset.attrs["source"] = xyz_picks.name
            if trajectory is not None:
                rec.create_group("trajectory").attrs.update(trajectory)
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
        "--channel-locations",
        type=Path,
        metavar="FILE",
        help="IBL channel_locations.json of this probe, converted to the channel table; "
        "see the how-to for the format",
    )
    parser.add_argument(
        "--xyz-picks",
        type=Path,
        metavar="FILE",
        help="IBL xyz_picks.json of this probe, the traced track",
    )
    parser.add_argument(
        "--histology-raw",
        type=Path,
        nargs="+",
        metavar="PATH",
        help="other histology files or folders of this probe, copied as they are",
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
