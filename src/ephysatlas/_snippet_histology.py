"""Histology conversion and storage for HDF5 snippet recordings.

The extractor validates these small inputs before opening the raw recording. Channel
tables follow data-row order; raw files are retained so conversions can be repeated.
"""

import contextlib
import importlib.metadata
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM
from iblatlas.regions import BrainRegions

from ephysatlas import __version__ as ibleatools_version

# Keep histology messages under the public extractor's logger.
LOGGER = logging.getLogger("ephysatlas.snippet_extractor")
MAX_HISTOLOGY_BYTES = 100 * 1024**2
BREGMA_CCF_UM = ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"]
CHANNEL_LOCATION_KEYS = ("x", "y", "z", "brain_region_id", "axial", "lateral")


def _rows_by_index(
    ind: np.ndarray, geometry_ind: np.ndarray, source: str
) -> np.ndarray:
    """Match histology entries to the data rows of a recording by channel index.

    Args:
        ind: Index in the recorded file of the channel of each histology entry.
        geometry_ind: ``geometry["ind"]`` of the recording, the channel index of each data row.
        source: Name of the histology input, used in the error messages.

    Returns:
        For each data row, the position in ``ind`` of that row's channel, so that
        ``ind[rows]`` equals ``geometry_ind``.

    Raises:
        ValueError: ``ind`` has duplicates, lacks a channel of the recording, or has channels
            the recording doesn't have.
    """
    ind = np.asarray(ind)
    values, counts = np.unique(ind, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"{source} has more than one entry for channels "
            f"{values[counts > 1].tolist()}; pass the file with --histology-raw instead"
        )
    missing = np.setdiff1d(geometry_ind, ind)
    if missing.size:
        raise ValueError(
            f"{source} has no entry for channels {missing.tolist()} of the recording, but "
            "needs one for every channel; pass the file with --histology-raw instead"
        )
    extra = np.setdiff1d(ind, geometry_ind)
    if extra.size:
        raise ValueError(
            f"{source} has entries for channels {extra.tolist()}, which the recording "
            "doesn't have; pass the file with --histology-raw instead"
        )
    sorter = np.argsort(ind)
    return sorter[np.searchsorted(ind, geometry_ind, sorter=sorter)]


def _acronyms(atlas_id: np.ndarray, source: str) -> np.ndarray:
    """Look up the Allen acronym of each region id with iblatlas.

    Args:
        atlas_id: Allen CCF 2017 region ids. Negative ids (left hemisphere) are accepted.
        source: Name of the histology input, used in the error message.

    Returns:
        One acronym per id.

    Raises:
        ValueError: iblatlas doesn't know some of the ids.
    """
    regions = BrainRegions()
    # id2acronym drops unknown ids without an error, which would shift the acronyms
    # of the following rows.
    unknown = np.setdiff1d(atlas_id, regions.id)
    if unknown.size:
        raise ValueError(
            f"{source} has brain_region_id values that iblatlas "
            f"{importlib.metadata.version('iblatlas')} doesn't know: {unknown.tolist()}"
        )
    return regions.id2acronym(atlas_id)


@contextlib.contextmanager
def _wrong_json_shape(file_name: str):
    """Turn the errors that a wrongly shaped json raises into a ``ValueError``.

    Args:
        file_name: Name of the json file, used in the message.

    Raises:
        ValueError: The block raised an ``AttributeError``, ``TypeError`` or ``KeyError``.
            The message names the file. A ``ValueError`` from the block passes through.
    """
    try:
        yield
    except (AttributeError, TypeError, KeyError) as error:
        raise ValueError(
            f"{file_name} doesn't have the expected structure ({type(error).__name__}: "
            f"{error}). Check the format in the how-to guide, or pass the file with "
            "--histology-raw instead"
        ) from error


def _read_channel_locations(
    json_file: Path, geometry: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Convert an IBL ``channel_locations.json`` into the channel table of a recording.

    Entries are matched to the data rows by channel index, not by position on the probe,
    because labs write the probe positions in different conventions (offset, mirrored).
    The positions are checked once the rows are matched.

    Args:
        json_file: The ``channel_locations.json``. Each ``channel_<i>`` entry has ``x, y, z``
            (um, IBL coordinates relative to bregma), ``brain_region_id``, ``axial`` and
            ``lateral`` (um on the probe), and optionally ``brain_region`` and
            ``original_channel_idx``. ``origin.bregma`` must be ``BREGMA_CCF_UM``.
        geometry: The recording's geometry from ``spikeglx.geometry_from_meta``, one value
            per data row.

    Returns:
        ``x, y, z`` in metres, ``atlas_id`` and ``acronym``, in the row order of ``geometry``.

    Raises:
        ValueError: The file is not valid JSON, ``origin.bregma`` is missing or different, an
            entry lacks a key, the entries don't match the recording's channels one to one,
            the positions don't line up with the geometry, or iblatlas doesn't know a
            ``brain_region_id``.
    """
    try:
        content = json.loads(json_file.read_bytes())
    except ValueError as error:  # also raised for bytes that aren't text
        raise ValueError(
            f"{json_file.name} is not a valid JSON file: {error}"
        ) from error
    with _wrong_json_shape(json_file.name):
        bregma = content.get("origin", {}).get("bregma")
    if bregma is None or not np.array_equal(bregma, BREGMA_CCF_UM):
        raise ValueError(
            f"{json_file.name} must have origin.bregma = {BREGMA_CCF_UM.tolist()} (the IBL "
            f"bregma in Allen CCF um), but has {bregma}"
        )

    entries = {key: e for key, e in content.items() if key.startswith("channel_")}
    for key, entry in entries.items():
        with _wrong_json_shape(json_file.name):
            missing = [k for k in CHANNEL_LOCATION_KEYS if k not in entry]
        if missing:
            raise ValueError(
                f"Entry {key} of {json_file.name} has no {', '.join(map(repr, missing))}"
            )
    # The channel's index in the recorded file: original_channel_idx when the entry has
    # it, otherwise the i of channel_<i>.
    keys = list(entries)
    with _wrong_json_shape(json_file.name):
        try:
            index = [
                int(
                    entries[key].get(
                        "original_channel_idx", key.removeprefix("channel_")
                    )
                )
                for key in keys
            ]
        except ValueError as error:  # int() of something that is not a number
            raise ValueError(
                f"{json_file.name} has a channel index that is not a number ({error}). "
                "Check the format in the how-to guide, or pass the file with "
                "--histology-raw instead"
            ) from error
    rows = _rows_by_index(index, geometry["ind"], json_file.name)
    ordered = [entries[keys[row]] for row in rows]  # the entries in data-row order
    with _wrong_json_shape(json_file.name):
        values = {
            key: np.array([e[key] for e in ordered], dtype=np.float64)
            for key in ("x", "y", "z", "axial", "lateral")
        }

    # Up to 1 um, axial must follow geometry y up to an offset, and lateral must follow
    # geometry x up to an offset or mirrored. This catches a file whose channel order
    # doesn't match the recording.
    axial, lateral = values["axial"], values["lateral"]
    x, y = geometry["x"], geometry["y"]
    if not (
        np.ptp((axial - axial.min()) - (y - y.min())) <= 1.0
        and (np.ptp(lateral - x) <= 1.0 or np.ptp(lateral + x) <= 1.0)
    ):
        raise ValueError(
            f"the channels of {json_file.name} don't line up with the recording's probe "
            "geometry"
        )

    with _wrong_json_shape(json_file.name):
        atlas_id = np.array([e["brain_region_id"] for e in ordered], dtype=np.int64)
    table = {key: values[key] / 1e6 for key in ("x", "y", "z")}
    table["atlas_id"] = atlas_id
    table["acronym"] = _acronyms(atlas_id, json_file.name)
    # An entry without brain_region counts as agreeing.
    n = sum(e.get("brain_region", a) != a for e, a in zip(ordered, table["acronym"]))
    if n > 0:
        LOGGER.warning(
            "%d channels of %s have a brain_region that differs from iblatlas's acronym "
            "for their brain_region_id; iblatlas's is stored",
            n,
            json_file.name,
        )
    return table


def _write_channel_table(
    histology: h5py.Group, table: Mapping[str, np.ndarray], source: str
) -> None:
    """Write a recording's channel table to ``channels/`` in its histology group.

    Args:
        histology: The recording's ``histology`` group.
        table: ``x, y, z`` (metres), ``atlas_id`` and ``acronym``, one value per data row,
            already in the row order of the recording's geometry.
        source: Where the table came from, stored as the ``source`` attribute.
    """
    channels = histology.create_group("channels")
    channels.attrs.update(
        {
            "source": source,
            "atlas": "Allen CCF 2017",
            "bregma_ccf_um": BREGMA_CCF_UM,
            "iblatlas_version": importlib.metadata.version("iblatlas"),
            "ibleatools_version": ibleatools_version,
        }
    )
    for key in ("x", "y", "z"):
        channels.create_dataset(key, data=np.asarray(table[key], dtype=np.float64))
    channels.create_dataset(
        "atlas_id", data=np.asarray(table["atlas_id"], dtype=np.int64)
    )
    channels.create_dataset(
        "acronym",
        data=np.asarray(table["acronym"], dtype=object),
        dtype=h5py.string_dtype(),
    )


def add_channel_locations(
    h5_file: str | Path,
    recording: str,
    channels: pd.DataFrame | Mapping[str, ArrayLike],
    source: str,
) -> None:
    """Add a channel table to a recording of a snippet file, e.g. one we received.

    The rows of ``channels`` are matched to the data rows by channel index, so their order
    doesn't matter. Nothing is written if a check fails.

    Args:
        h5_file: HDF5 snippet file, changed in place.
        recording: Name of the recording in the file.
        channels: One row per channel of the recording, with ``ind`` (the channel's index in
            the recorded file), ``x``, ``y``, ``z`` (metres, IBL coordinates relative to
            bregma) and ``atlas_id`` (Allen CCF 2017 region id).
        source: Where the table came from, stored as the ``source`` attribute.

    Raises:
        ValueError: The file is not a snippet file, the recording is not
            in it, it already has a channel table, the channels don't match the recording's
            channels one to one, or iblatlas doesn't know an ``atlas_id``.
    """
    h5_file = Path(h5_file)
    if not h5py.is_hdf5(h5_file):
        raise ValueError(f"{h5_file} is not a snippet file")
    with h5py.File(h5_file, "r+") as f:
        if not isinstance(f.get("recordings"), h5py.Group):
            raise ValueError(f"{h5_file} is not a snippet file")
        recordings = list(f.get("recordings", {}))
        if recording not in recordings:
            raise ValueError(
                f"{h5_file} has no recording {recording!r}; it holds {recordings}"
            )
        group = f[f"recordings/{recording}"]
        if "channels" in group.get("histology", {}):
            raise ValueError(
                f"recording {recording!r} of {h5_file} already has a channel table"
            )
        channels = pd.DataFrame(channels)
        rows = _rows_by_index(channels["ind"], group["geometry/ind"][()], source)
        channels = channels.iloc[rows]  # now in the row order of the geometry
        table = {key: channels[key].to_numpy() for key in ("x", "y", "z", "atlas_id")}
        table["acronym"] = _acronyms(table["atlas_id"], source)
        _write_channel_table(group.require_group("histology"), table, source)
    LOGGER.info("Added a channel table from %s to %s", source, h5_file)


def _collect_histology_files(paths: Sequence[str | Path]) -> dict[str, Path]:
    """Resolve raw histology paths, checking stored names and the total size limit."""
    histology_files = {}  # name stored under histology/raw/ -> resolved source path
    for path in map(Path, paths):
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
        for hist_name, member in members.items():
            member = member.resolve()
            # Only a path given twice is skipped. A file that is also inside a listed folder
            # is stored twice, under two names.
            if hist_name in histology_files and histology_files[hist_name] != member:
                raise ValueError(
                    f"Two histology files would both be stored as histology/raw/{hist_name}: "
                    f"{histology_files[hist_name]} and {member}. Pass only one of them."
                )
            histology_files[hist_name] = member
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
    return histology_files


def _read_xyz_picks(xyz_picks: Path) -> np.ndarray:
    """Read at least two traced track points in micrometres from IBL JSON."""
    try:
        content = json.loads(xyz_picks.read_bytes())
    except ValueError as error:  # also raised for bytes that aren't text
        raise ValueError(
            f"{xyz_picks.name} is not a valid JSON file: {error}"
        ) from error
    with _wrong_json_shape(xyz_picks.name):
        picks = np.asarray(content.get("xyz_picks"), dtype=float)
    if picks.ndim != 2 or picks.shape[0] < 2 or picks.shape[1] != 3:
        raise ValueError(
            f"{xyz_picks.name} must hold xyz_picks as a list of at least 2 [x, y, z] "
            f"points, got an array of shape {picks.shape}"
        )

    return picks
