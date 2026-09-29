"""Probe and trace geometry for the alignment methods.

Conventions (as in :func:`ephysatlas.spatial_encoder.utils.LoadInsertionData`): positions are in
metres, IBL coordinates; a probe's channel arrays are in *row order*, top of the probe first --
row ``r`` holds Neuropixels 1.0 channel ``383 - r``. Traces are ordered top (brain surface) to
bottom (tip).
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


def resample_curve(xyz_m: np.ndarray, step_um: float = 10.0) -> np.ndarray:
    """Resample a polyline at a constant arc-length step (duplicate points removed)."""
    xyz = np.asarray(xyz_m, dtype=float)
    seg_um = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1e6
    s = np.r_[0.0, np.cumsum(seg_um)]
    keep = np.r_[True, np.diff(s) > 1e-6]
    xyz, s = xyz[keep], s[keep]
    if len(xyz) < 2 or s[-1] <= 0:
        return xyz.astype(np.float32)
    sq = np.arange(0.0, s[-1] + 0.5 * step_um, step_um)
    out = np.column_stack([np.interp(sq, s, xyz[:, d]) for d in range(3)])
    return out.astype(np.float32)


def order_top_to_bottom(xyz_m: np.ndarray) -> np.ndarray:
    """The trace ordered from its dorsal end (largest z) to its ventral end."""
    xyz = np.asarray(xyz_m)
    return xyz[::-1].copy() if xyz[0, 2] < xyz[-1, 2] else xyz.copy()


def region_ids(brain_atlas, xyz_m: np.ndarray, mapping: str = "Cosmos") -> np.ndarray:
    """Region of each position in ``mapping``, as an index into ``brain_atlas.regions`` (0, void,
    outside the brain).

    These are the indices ``brain_atlas._label2rgb`` colours; comparing them compares regions. For
    Allen region ids (e.g. for ``regions.get``) use ``brain_atlas.regions.id[...]`` of the result,
    or ``brain_atlas.get_labels``.
    """
    from ephysatlas.spatial_encoder.utils import region_ids_from_xyz

    xyz_m = np.atleast_2d(np.asarray(xyz_m, dtype=np.float32))
    return np.asarray(
        region_ids_from_xyz(brain_atlas, xyz_m, mapping=mapping, mode="clip")
    ).reshape(-1)


def extend_trace_to_brain(
    xyz_m: np.ndarray,
    brain_atlas,
    *,
    n_edge: int = 100,
    max_extra: int = 4096,
    mapping: str = "Allen",
) -> np.ndarray:
    """Extend a trace linearly at both ends, one mean edge step at a time, until it leaves the brain.

    The step at each end is the mean non-zero step over its ``n_edge`` last samples, so the
    extension keeps the trace's sampling and direction there.
    """
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if xyz.shape[0] < 2:
        return xyz.astype(np.float32)
    n_edge = int(min(n_edge, xyz.shape[0]))

    def outside(point: np.ndarray) -> bool:
        return bool(np.any(region_ids(brain_atlas, point[None, :], mapping) == 0))

    def edge_step(edge: np.ndarray) -> np.ndarray:
        d = edge[1:] - edge[:-1]
        nz = np.linalg.norm(d, axis=1) > 0
        if np.any(nz):
            return d[nz].mean(axis=0)
        return (edge[-1] - edge[0]) / max(1, edge.shape[0] - 1)

    step_top = edge_step(xyz[:n_edge])
    step_bottom = edge_step(xyz[-n_edge:])
    if np.linalg.norm(step_top) < 1e-12:
        step_top = step_bottom.copy()
    if np.linalg.norm(step_bottom) < 1e-12:
        step_bottom = step_top.copy()
    if np.linalg.norm(step_top) < 1e-12:
        return xyz.astype(np.float32)

    before, cur = [], xyz[0].copy()
    for _ in range(max_extra):
        cur = cur - step_top
        if outside(cur):
            break
        before.append(cur.copy())
    after, cur = [], xyz[-1].copy()
    for _ in range(max_extra):
        cur = cur + step_bottom
        if outside(cur):
            break
        after.append(cur.copy())
    return np.concatenate(
        [np.asarray(before[::-1]).reshape(-1, 3), xyz, np.asarray(after).reshape(-1, 3)],
        axis=0,
    ).astype(np.float32)
