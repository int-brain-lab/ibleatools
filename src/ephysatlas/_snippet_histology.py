"""The ``Histology`` contract and conversion of snippet recordings' histology."""

import json
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pydantic
from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM
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
    """Brain location of every channel, from ``channel_locations.json``."""
    xyz_picks: list[tuple[float, float, float]] | None = pydantic.Field(
        None, min_length=2, json_schema_extra=METRES
    )
    """Traced track of the probe, points of x, y, z, from ``xyz_picks.json``."""


def read_histology(
    channel_locations: str | Path | None,
    xyz_picks: str | Path | None,
    geometry: dict[str, np.ndarray],
) -> Histology:
    """Convert a recording's ``channel_locations.json`` and ``xyz_picks.json`` (in um).

    ``channel_locations.json`` is the IBL alignment GUI's output. ``xyz_picks.json`` is the
    traced probe track, in the format the GUI reads as its input.

    Args:
        channel_locations: ``channel_locations.json``, or None. Entry ``channel_<i>`` is the
            channel with index ``original_channel_idx`` in the recorded file if the entry has
            it, otherwise index ``i``.
        xyz_picks: ``xyz_picks.json``, or None.
        geometry: The Reader's geometry: ``ind``, the recorded channel index of each data
            row, and the row's ``x``, ``y`` (um) and ``shank``.

    Raises:
        ValueError: ``origin.bregma`` is not iblatlas's bregma, the entries are not one per
            channel of the recording, their ``lateral``/``axial`` don't map onto the
            geometry's ``x``/``y`` by one affine transform, or a value breaks the
            ``Histology`` contract.
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
        if sorted(index) != sorted(geometry["ind"].tolist()):
            raise ValueError(
                f"{channel_locations} must have one entry per channel of the recording "
                f"({len(geometry['ind'])} channels), not {len(index)} entries for channels "
                f"{sorted(set(index))}"
            )
        entry_of = dict(zip(index, (content[key] for key in keys)))
        rows = [entry_of[i] for i in geometry["ind"].tolist()]  # in data-row order
        # The json's positions are in the spike sorter's layout. With the right pairing, one
        # affine map, shifted per shank, takes them onto the geometry (e.g. mirrored, 20 um
        # lower). Two channels paired wrongly miss by 15 um or more, the closest site spacing.
        shanks = geometry["shank"][:, None] == np.unique(geometry["shank"])
        src = np.c_[[e["lateral"] for e in rows], [e["axial"] for e in rows], shanks]
        dst = np.c_[geometry["x"], geometry["y"]]
        if np.abs(src @ np.linalg.lstsq(src, dst, rcond=None)[0] - dst).max() > 1:  # um
            raise ValueError(
                f"{channel_locations}: the entries' lateral/axial positions don't match the "
                "recording's channel positions, so they are paired with the wrong channels"
            )
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
