"""
Ephys edges: local gradient of an encoding volume.

The encoding volume (see :func:`ephysatlas.data.download_encoding_volume`) is
z-scored, rotated onto its principal components and divided by their standard
deviations, so Euclidean distance between two voxels is their Mahalanobis distance
in feature space. The edge strength of a voxel is built from the spatial gradient of
these whitened PCs: how fast the multivariate ephys signature changes around it. The
gradient magnitude of each PC is aggregated across PCs by the median (robust to a PC
that changes alone), the mean, or the root of the sum of squares (``"rss"``, the
Mahalanobis norm of the gradient).

Functions
---------
load_whitened_pcs
    Whitened PCA of an encoding volume file
edge_strength
    Gradient magnitude of the whitened PCs, aggregated across PCs, per mm
save_edges_volume
    Write an edges volume with the encoding volume's conventions
compute_edges_volume
    Encoding volume file -> edges volume file, next to it

Examples
--------
>>> from pathlib import Path
>>> from ephysatlas.edges import compute_edges_volume
>>> volume_file = Path("encoding_volumes/ea_active/2026_W39/brainwide_ephys_atlas_50um.npz")
>>> edges = compute_edges_volume(volume_file)  # (AP, ML, DV), NaN outside the mask
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage

import iblatlas.atlas

# void and fluid are not tissue; root is kept: it contains fiber tracts
DEFAULT_EXCLUDE_ACRONYMS = ("void", "void_fluid")
DEFAULT_SIGMA_UM = 100
DEFAULT_ERODE_UM = 100
DEFAULT_VARIANCE_THRESHOLD = 0.95
DEFAULT_AGGREGATE = "median"
EDGES_UNITS = {
    "rss": "Mahalanobis distance per mm",
    "median": "median across PCs of the whitened PC gradient magnitude, per mm",
    "mean": "mean across PCs of the whitened PC gradient magnitude, per mm",
}


def load_whitened_pcs(
    volume_file,
    atlas=None,
    exclude_acronyms=DEFAULT_EXCLUDE_ACRONYMS,
    variance_threshold=DEFAULT_VARIANCE_THRESHOLD,
):
    """Whitened PCA of an encoding volume.

    PCs are computed on the voxels inside the atlas mask, outside the excluded
    regions and with data (all-zero voxels are the volume's "no probe coverage"
    sentinel). Each PC is signed so that its largest-magnitude loading is positive.

    Parameters
    ----------
    volume_file : str or Path
        Encoding volume ``.npz`` (``brainwide_ephys_atlas_{res_um}um.npz``).
    atlas : iblatlas.atlas.BrainAtlas, optional
        Atlas at the volume's resolution. Defaults to ``AllenAtlas(res_um)``.
    exclude_acronyms : iterable of str
        Regions left out of the PCA and of the valid mask.
    variance_threshold : float
        Fraction of the variance the retained PCs must explain.

    Returns
    -------
    dict
        pcs : (AP, ML, DV, k) float32 whitened PCs, NaN outside `valid`
        valid : (AP, ML, DV) bool mask of the voxels used
        explained : (k,) explained variance ratio
        loadings : pandas.DataFrame (k, n_features)
        feature_names : (n_features,) array
        project : callable mapping raw (n, n_features) values to whitened PCs
        res_um : int
    """
    archive = np.load(volume_file, allow_pickle=True)
    res_um = int(archive["res_um"][0])
    atlas = iblatlas.atlas.AllenAtlas(res_um=res_um) if atlas is None else atlas
    # stored as (ML, AP, DV, feature), iblatlas order is (AP, ML, DV)
    vol = np.transpose(archive["ephys_atlas_vol"], (1, 0, 2, 3)).astype(np.float32)
    excluded = np.where(np.isin(atlas.regions.acronym, list(exclude_acronyms)))[0]
    valid = atlas.mask() & ~np.isin(atlas.label, excluded) & ~np.all(vol == 0, axis=-1)
    mean, std = archive["mean_per_feature"], archive["std_per_feature"]
    xz = (vol[valid] - mean) / std
    centre = xz.mean(axis=0)
    xz -= centre
    eigvals, eigvecs = np.linalg.eigh(np.cov(xz, rowvar=False))
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    explained = eigvals / eigvals.sum()
    k = int(np.searchsorted(np.cumsum(explained), variance_threshold) + 1)
    # PC signs are arbitrary: orient each so its largest-|loading| feature loads positively
    eigvecs = eigvecs[:, :k] * np.sign(
        eigvecs[np.argmax(np.abs(eigvecs[:, :k]), axis=0), np.arange(k)]
    )
    pcs = np.full(vol.shape[:3] + (k,), np.nan, dtype=np.float32)
    pcs[valid] = (xz @ eigvecs) / np.sqrt(eigvals[:k])
    loadings = pd.DataFrame(
        eigvecs.T,
        columns=archive["feature_names"],
        index=[f"PC{i + 1}" for i in range(k)],
    )

    def project(raw):
        return (((raw - mean) / std - centre) @ eigvecs) / np.sqrt(eigvals[:k])

    return {
        "pcs": pcs,
        "valid": valid,
        "explained": explained[:k],
        "loadings": loadings,
        "feature_names": archive["feature_names"],
        "project": project,
        "res_um": res_um,
    }


def edge_strength(
    pcs,
    valid,
    res_um,
    sigma_um=DEFAULT_SIGMA_UM,
    erode_um=DEFAULT_ERODE_UM,
    aggregate=DEFAULT_AGGREGATE,
):
    """Gradient magnitude of whitened PCs, aggregated across PCs, per mm.

    Each PC is Gaussian-smoothed (NaN-aware: filtered data divided by the filtered
    mask, which avoids a bias at the mask border), then its gradient magnitude (root
    of the summed squared derivatives over the three axes) is taken. The smoothing
    removes the blocky grid of the interpolated volume. The per-PC magnitudes are then
    aggregated across PCs: ``"median"`` keeps voxels where many PCs change together,
    ``"mean"`` averages them, ``"rss"`` is the root of the sum of squares, i.e. the
    gradient norm in Mahalanobis distance (PCs are whitened, so this one is a distance
    per mm and grows with the number of PCs).

    Parameters
    ----------
    pcs : np.ndarray
        (AP, ML, DV, k) whitened PCs, e.g. ``load_whitened_pcs(...)["pcs"]``.
    valid : np.ndarray
        (AP, ML, DV) bool mask of voxels with data.
    res_um : float
        Voxel size, sets the sigma in voxels and the gradient spacing.
    sigma_um : float
        Gaussian sigma in um (0 = unsmoothed).
    erode_um : float
        Voxels this close to the mask border are dropped: their gradient is a mask
        artefact.
    aggregate : {"median", "mean", "rss"}
        How the per-PC gradient magnitudes are combined across PCs.

    Returns
    -------
    np.ndarray
        (AP, ML, DV) float32 edge strength (units: `EDGES_UNITS` [aggregate]), NaN
        outside the eroded mask.
    """
    if aggregate not in EDGES_UNITS:
        raise ValueError(
            f"aggregate must be one of {list(EDGES_UNITS)}, got {aggregate!r}"
        )
    sigma = sigma_um / res_um
    weight = np.maximum(
        scipy.ndimage.gaussian_filter(valid.astype(np.float32), sigma), 1e-3
    )
    inner = scipy.ndimage.binary_erosion(valid, iterations=round(erode_um / res_um))
    magnitudes = np.empty((inner.sum(), pcs.shape[-1]), dtype=np.float32)
    for c in range(pcs.shape[-1]):
        smooth = (
            scipy.ndimage.gaussian_filter(np.nan_to_num(pcs[..., c]), sigma) / weight
        )
        grad_sq = sum(g**2 for g in np.gradient(smooth, res_um / 1e3))
        magnitudes[:, c] = np.sqrt(grad_sq)[inner]
    reduce = {
        "median": lambda m: np.median(m, axis=1),
        "mean": lambda m: np.mean(m, axis=1),
        "rss": lambda m: np.sqrt((m**2).sum(axis=1)),
    }[aggregate]
    edges = np.full(valid.shape, np.nan, dtype=np.float32)
    edges[inner] = reduce(magnitudes)
    return edges


def save_edges_volume(file, edges, res_um, units, **metadata):
    """Save an edges volume with the encoding volume's conventions.

    Same layout as ``brainwide_ephys_atlas_{res_um}um.npz``: ``(ML, AP, DV)`` axis
    order, ``res_um`` and ``grid_shape`` arrays. Values are stored as float16.

    Parameters
    ----------
    file : str or Path
        Output ``.npz`` file.
    edges : np.ndarray
        (AP, ML, DV) edge strength, NaN outside the mask.
    res_um : int
        Voxel size in um.
    units : str
        Units of `edges`, e.g. ``EDGES_UNITS["median"]``.
    **metadata
        Extra scalars stored alongside (vintage, sigma_um, n_pcs...).
    """
    stored = np.transpose(edges, (1, 0, 2))
    np.savez_compressed(
        file,
        ephys_edges_vol=stored.astype(np.float16),
        res_um=np.array([res_um]),
        grid_shape=np.array(stored.shape, dtype=np.int32),
        axis_order="ML,AP,DV",
        units=units,
        **metadata,
    )


def compute_edges_volume(
    volume_file,
    out_file=None,
    atlas=None,
    sigma_um=DEFAULT_SIGMA_UM,
    erode_um=DEFAULT_ERODE_UM,
    variance_threshold=DEFAULT_VARIANCE_THRESHOLD,
    exclude_acronyms=DEFAULT_EXCLUDE_ACRONYMS,
    aggregate=DEFAULT_AGGREGATE,
):
    """Encoding volume file -> edges volume file, next to it.

    Parameters
    ----------
    volume_file : str or Path
        Encoding volume ``.npz``.
    out_file : str or Path, optional
        Output file. Defaults to ``brainwide_ephys_edges_{res_um}um.npz`` in the
        folder of `volume_file`.
    atlas, sigma_um, erode_um, variance_threshold, exclude_acronyms, aggregate
        See :func:`load_whitened_pcs` and :func:`edge_strength`.

    Returns
    -------
    np.ndarray
        (AP, ML, DV) float32 edge strength (`EDGES_UNITS` [aggregate]), NaN outside the
        mask.
    """
    volume_file = Path(volume_file)
    pca = load_whitened_pcs(volume_file, atlas, exclude_acronyms, variance_threshold)
    edges = edge_strength(
        pca["pcs"], pca["valid"], pca["res_um"], sigma_um, erode_um, aggregate
    )
    out_file = (
        volume_file.with_name(f"brainwide_ephys_edges_{pca['res_um']}um.npz")
        if out_file is None
        else Path(out_file)
    )
    save_edges_volume(
        out_file,
        edges,
        pca["res_um"],
        EDGES_UNITS[aggregate],
        aggregate=aggregate,
        source=volume_file.name,
        sigma_um=sigma_um,
        erode_um=erode_um,
        n_pcs=pca["pcs"].shape[-1],
        variance_threshold=variance_threshold,
        exclude_acronyms=",".join(exclude_acronyms),
    )
    return edges
