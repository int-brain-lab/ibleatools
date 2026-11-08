"""Probe and trace geometry for the alignment methods.

Conventions (as in :func:`ephysatlas.spatial_encoder.utils.LoadInsertionData`): positions are in
metres, IBL coordinates; a probe's channel arrays are in *row order*, top of the probe first --
row ``r`` holds Neuropixels 1.0 channel ``383 - r``. Traces are ordered top to bottom (tip).
"""

from __future__ import annotations

import numpy as np

N_CHANNELS_NP1 = 384


def np1_axial_um_in_row_order(n_channels: int = N_CHANNELS_NP1) -> np.ndarray:
    """``[n_channels]`` physical axial position (µm from the tip) of each channel row.

    Neuropixels 1.0 has two sites per axial level, so channel-index arithmetic cannot stand in
    for depth. Rows are top first, like the channel feature arrays.
    """
    import neuropixel

    header = neuropixel.trace_header(version=1)
    axial = np.asarray(header["y"], dtype=float)[:n_channels]
    return axial[::-1].copy()


def valid_xyz_mask(xyz_m: np.ndarray) -> np.ndarray:
    """Rows with finite coordinates that are not the all-zero placeholder."""
    xyz_m = np.asarray(xyz_m, dtype=float)
    return np.isfinite(xyz_m).all(axis=1) & ~np.all(xyz_m == 0.0, axis=1)


def infer_metres(xyz: np.ndarray) -> np.ndarray:
    """Coordinates in metres, converting from µm when the magnitudes say so."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.size and np.nanmax(np.abs(xyz)) > 0.1:
        xyz = xyz / 1e6
    return xyz.astype(np.float32)


def region_ids(brain_atlas, xyz_m: np.ndarray, mapping: str = "Cosmos") -> np.ndarray:
    """Allen id of the region of each position in ``mapping`` (0, void, outside the brain), for
    ``brain_atlas.regions.get`` or ``id2acronym``."""
    from ephysatlas.spatial_encoder.utils import region_ids_from_xyz

    xyz_m = np.atleast_2d(np.asarray(xyz_m, dtype=np.float32))
    return np.asarray(
        region_ids_from_xyz(brain_atlas, xyz_m, mapping=mapping, mode="clip")
    ).reshape(-1)


def histology_track(xyz_m: np.ndarray, brain_atlas) -> np.ndarray:
    """``[L, 3]`` the IBL histology track through ``xyz_m`` (m), top to bottom.

    This is the track of the alignment GUI (:class:`ibllib.pipes.ephys_alignment.EphysAlignment`):
    lines fitted to the first and last quarter of the points (at least 4) extend it up to the top
    of the atlas and down to 200 µm below the brain, sampled every 10 µm. ``xyz_m`` (histology
    picks, or channel positions) is ordered top first, as Alyx stores the picks.
    """
    from ibllib.pipes.ephys_alignment import EphysAlignment

    track = EphysAlignment(np.asarray(xyz_m, dtype=float), brain_atlas=brain_atlas)
    return track.xyz_samples[::-1].astype(np.float32)
