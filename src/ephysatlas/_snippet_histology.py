"""The ``Histology`` contract and conversion of snippet recordings' histology."""

import json
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pydantic
from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM, AllenAtlas
from iblatlas.regions import BrainRegions

# Fields are documented by their docstrings; NaN and inf are rejected.
CONTRACT = pydantic.ConfigDict(allow_inf_nan=False, use_attribute_docstrings=True)
METRES = {"unit": "m"}  # a field's unit, written as the "unit" attribute of its dataset
Metres = Annotated[list[float], pydantic.Field(json_schema_extra=METRES)]


class ChannelLocations(pydantic.BaseModel):
    """Brain location of every channel of a recording, stored in ``histology/channels/``.

    One value per data row, in the row order of the recording's ``geometry``. These are the
    columns of IBL's ``brainbox.io.one.load_channel_locations`` and of
    ``ephysatlas.features.ChannelDataFrameSchema``.
    """

    model_config = CONTRACT

    x: Metres
    """Medio-lateral position from bregma, right positive."""
    y: Metres
    """Antero-posterior position from bregma, anterior positive."""
    z: Metres
    """Dorso-ventral position from bregma, dorsal positive."""
    atlas_id: list[int]
    """Allen CCF 2017 region id, as in iblatlas ``BrainRegions``; 0 is void."""
    acronym: list[str]
    """Acronym of ``atlas_id`` in iblatlas ``BrainRegions``."""


class Histology(pydantic.BaseModel):
    """Structured histology of one recording, stored in ``recordings/<name>/histology/``.

    Positions are in IBL coordinates, as in the IBL alignment GUI files: metres from bregma,
    with bregma at ``iblatlas.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"]`` in the Allen
    CCF. Each field that is set is one HDF5 dataset (a nested model is a group), with its
    description, and unit if any, as attributes. The original files are kept in ``raw/``.
    """

    model_config = CONTRACT

    channels: ChannelLocations | None = None
    """Brain location of every channel, from ``channel_locations.json``, and from the
    alignment for channels missing from it."""
    xyz_picks: list[tuple[float, float, float]] | None = pydantic.Field(
        None, min_length=2, json_schema_extra=METRES
    )
    """Traced track of the probe, points of x, y, z, from ``xyz_picks.json``."""


def read_histology(
    channel_locations: str | Path | None,
    xyz_picks: str | Path | None,
    geometry: dict[str, np.ndarray],
    alignment: str | Path | None = None,
    channel_index: str | Path | None = None,
) -> Histology:
    """Convert a recording's ``channel_locations.json`` and ``xyz_picks.json`` (in um).

    ``channel_locations.json`` is the IBL alignment GUI's output. ``xyz_picks.json`` is the
    traced probe track, in the format the GUI reads as its input. Channels missing from
    ``channel_locations.json``, e.g. dropped by the spike sorter, are computed from the
    alignment as the GUI does (ibllib's ``EphysAlignment``, ``AllenAtlas(25)``).

    Args:
        channel_locations: ``channel_locations.json``, or None. Entry ``channel_<i>`` is the
            channel with index ``original_channel_idx`` in the recorded file if the entry has
            it, otherwise index ``i``. With ``channel_index``, that number is a row of the
            spike sorter's channels instead.
        xyz_picks: ``xyz_picks.json``, or None.
        geometry: The Reader's geometry: ``ind``, the recorded channel index of each data
            row, and the row's ``x``, ``y`` (um) and ``shank``.
        alignment: The GUI's ``prev_alignments.json``, or None. With ``xyz_picks``, its
            latest alignment gives the channels missing from ``channel_locations``.
        channel_index: The spike sorter's ``channels.rawInd.npy``, or None: the recorded
            channel index of each of its channel rows.

    Raises:
        ValueError: ``origin.bregma`` is not iblatlas's bregma, the entries are not one per
            channel of the recording (without ``alignment``) or include a channel twice or
            one the recording doesn't have, their ``lateral``/``axial`` don't map onto the
            geometry's ``x``/``y`` by a lateral mirror and one shift per shank, the
            alignment doesn't reproduce the entries, or a value breaks the ``Histology``
            contract.
        IndexError: iblatlas doesn't know a ``brain_region_id`` (``BrainRegions.get``).
    """
    channels = picks = None
    if channel_locations is not None:
        content = json.loads(Path(channel_locations).read_text())
        bregma = ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"]
        if not np.array_equal(content["origin"]["bregma"], bregma):
            raise ValueError(
                f"{channel_locations} must have origin.bregma = {bregma.tolist()}, the IBL "
                "bregma in Allen CCF um"
            )
        keys = [key for key in content if key.startswith("channel_")]
        index = [
            int(content[key].get("original_channel_idx", key.removeprefix("channel_")))
            for key in keys
        ]
        if channel_index is not None:
            # Numbers of the spike sorter's channels -> recorded channel indices
            index = np.load(channel_index)[index].tolist()
        recorded = geometry["ind"].tolist()
        # With the alignment, the channels without an entry are computed below.
        complete = sorted(index) == sorted(recorded)
        subset = len(set(index)) == len(index) and set(index) <= set(recorded)
        if not (complete or (subset and alignment is not None)):
            raise ValueError(
                f"{channel_locations} must have one entry per channel of the recording "
                f"({len(recorded)} channels), not {len(index)} entries for "
                f"{len(set(index))} channels. Channels the spike sorter dropped are computed "
                "when you also pass prev_alignments.json and channels.rawInd.npy "
                "(--alignment and --channel-index)"
            )
        entry_of = dict(zip(index, (content[key] for key in keys)))
        kept = np.isin(geometry["ind"], index)  # data rows that have an entry
        entries = [entry_of[i] for i in geometry["ind"][kept].tolist()]
        # The json's positions are in the spike sorter's layout, which can differ from the
        # geometry by a lateral mirror and a shift per shank (e.g. mirrored, 20 um lower).
        # With the right pairing every entry lands within 1 um; two channels paired wrongly
        # miss by 15 um or more, the closest site spacing. A general affine map would also
        # fit wrong pairings, e.g. every entry off by one channel on a two-column NP2 shank.
        pos = np.array([[e["lateral"], e["axial"]] for e in entries])
        dst = np.c_[geometry["x"], geometry["y"]][kept]
        shank = geometry["shank"][kept]
        for mirror in (1, -1):
            diff = dst - pos * (mirror, 1)
            shift = {s: np.median(diff[shank == s], axis=0) for s in np.unique(shank)}
            if np.abs(diff - [shift[s] for s in shank]).max() <= 1:  # um
                break
        else:
            raise ValueError(
                f"{channel_locations}: the entries' lateral/axial positions don't match the "
                "recording's channel positions, so they are paired with the wrong channels. "
                "If they are numbered by the spike sorter's channels, pass its "
                "channels.rawInd.npy (--channel-index)"
            )
        if alignment is not None:
            from ibllib.pipes.ephys_alignment import EphysAlignment

            # The latest alignment: the GUI adds one per save and writes the json from it
            alignments = json.loads(Path(alignment).read_text())
            feature, track = np.array(alignments[max(alignments)])
            ea = EphysAlignment(
                np.array(json.loads(Path(xyz_picks).read_text())["xyz_picks"]) / 1e6,
                track_prev=track,
                feature_prev=feature,
                brain_atlas=AllenAtlas(25),  # the GUI's atlas
            )
            # Every channel's depth in the sorter's layout, which the alignment uses
            depth = geometry["y"] - [shift[s][1] for s in geometry["shank"]]
            xyz = ea.get_channel_locations(feature, track, depths=depth / 1e6) * 1e6
            region = ea.get_brain_locations(xyz / 1e6).id
            miss = np.abs(xyz[kept] - [[e[a] for a in "xyz"] for e in entries]).max()
            ids = [e["brain_region_id"] for e in entries]
            if miss > 1 or region[kept].tolist() != ids:  # um
                raise ValueError(
                    f"The latest alignment in {alignment} doesn't reproduce the entries of "
                    f"{channel_locations}. Pass the prev_alignments.json saved with it"
                )
            for row in np.flatnonzero(~kept):
                entry_of[recorded[row]] = {
                    **dict(zip("xyz", xyz[row])),
                    "brain_region_id": int(region[row]),
                }
        rows = [entry_of[i] for i in recorded]  # in data-row order
        atlas_id = [entry["brain_region_id"] for entry in rows]
        channels = ChannelLocations(
            **{axis: [entry[axis] / 1e6 for entry in rows] for axis in "xyz"},
            atlas_id=atlas_id,
            acronym=BrainRegions().get(np.asarray(atlas_id)).acronym.tolist(),
        )
    if xyz_picks is not None:
        content = json.loads(Path(xyz_picks).read_text())
        picks = (np.asarray(content["xyz_picks"], dtype=float) / 1e6).tolist()
    return Histology(channels=channels, xyz_picks=picks)


def _write_model(group: h5py.Group, model: pydantic.BaseModel) -> None:
    """Write each field of ``model`` that is set as a dataset of ``group``.

    A nested model becomes a subgroup. The field's description, and unit if any, become
    attributes of the dataset.
    """
    for key, field in type(model).model_fields.items():
        value = getattr(model, key)
        if isinstance(value, pydantic.BaseModel):
            _write_model(group.create_group(key), value)
        elif value is not None:
            # Strings are stored as variable-length UTF-8; read them with .asstr().
            dtype = h5py.string_dtype() if field.annotation == list[str] else None
            dataset = group.create_dataset(key, data=value, dtype=dtype)
            dataset.attrs.update(
                description=field.description, **(field.json_schema_extra or {})
            )
