"""
Module designed to work on the anatomy of the brain prior to the encoding/decoding analysis.
The EcondingAtlas is a version of the Allen Atlas relabeled to account for void labels inside of the skull compared to outside.

    ea = EncodingAtlas()
    from atlasview import atlasview
    av = atlasview.view(atlas=ea)
"""

import functools

import numpy as np
import scipy.spatial
import scipy.sparse

from iblutil.numerical import ismember
import iblatlas.regions
from iblatlas.atlas import AllenAtlas

NEW_VOID = {
    "id": 2_000,
    "name": "void_fluid",
    "acronym": "void_fluid",
    "rgb": [100, 40, 40],
    "level": 0,
    "parent": np.nan,
}


class ClassifierRegions(iblatlas.regions.BrainRegions):
    def __init__(self):
        super().__init__()

    def add_new_region(self, new_region):
        """
        Adds a new region to the brain regions object
        The region will be lateralized and added at the end of the regions list
        :param new_region: dictionary with keys 'id', 'name', 'acronym', 'rgb', 'level', 'parent'
        :return: indices of the new regions
        """
        assert new_region["id"] not in self.id, "Region ID already exists"
        order = np.max(self.order) + 1
        nr = len(self.id)
        # first update the properties of the region object, appending the new region
        self.id = np.append(self.id, [new_region["id"], -new_region["id"]])
        self.name = np.append(self.name, [new_region["name"], new_region["name"]])
        self.acronym = np.append(
            self.acronym, [new_region["acronym"], new_region["acronym"]]
        )
        self.rgb = np.append(
            self.rgb,
            np.tile(np.array(new_region["rgb"])[np.newaxis, :], [2, 1]),
            axis=0,
        ).astype(np.uint8)
        self.level = np.append(
            self.level, [new_region["level"], new_region["level"]]
        ).astype(np.uint16)
        self.parent = np.append(
            self.parent, [new_region["parent"], new_region["parent"]]
        )
        self.order = np.append(self.order, [order, order])
        # then need to to update the mappings and append to them as well
        for k in self.mappings:
            # if the mappign is lateralized, we need to add lateralized indices, otherwise we keep only the first
            is_lateralized = np.any(self.id[self.mappings[k]] < 0)
            inds = [nr, nr + 1] if is_lateralized else [nr, nr]
            self.mappings[k] = np.append(self.mappings[k], inds)
        return nr, nr + 1


class ClassifierAtlas(AllenAtlas):
    """
    The Encoding Atlas is a version of the Allen Atlas where the regions labels volume is reworked:
    -   voids inside the skull are relabeled to a new region, this is done computing the convex hull of the non-void
    labels and then splitting the void in two regions: one below the convex hull and one above
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compute_surface()
        self.regions = ClassifierRegions()
        self.assign_voids_inside_skull()

    def assign_voids_inside_skull(self):
        """
        Identifies and relabels void voxels that are inside the skull.

        This method creates a mask of the brain's convex hull and identifies all voxels
        below this hull. Any voxels that were previously labeled as void (0) and are
        below the convex hull are relabeled as the new 'void_fluid' region.

        The process involves:
        1. Finding non-NaN points on the convex top surface
        2. Creating a 3D mask of the convex hull
        3. Using cumulative sum to identify all voxels below the hull
        4. Adding the new void_fluid region to the brain regions
        5. Relabeling appropriate voxels with the new region ID

        Parameters:
            None

        Returns:
            None: The method modifies the atlas label volume in-place.
        """
        # we create a mask of the convex hull and label all of the voxels below the hull to True
        i0, i1 = np.where(~np.isnan(self.convex_top))
        i2 = self.bc.z2i(self.convex_top[i0, i1])
        mask_hull = np.zeros_like(self.label, dtype=bool)
        mask_hull[i0, i1, i2] = True
        mask_hull = np.cumsum(mask_hull, axis=2)
        # then the new voids samples are the void voxels below the convex hull
        ivoids = self.regions.add_new_region(NEW_VOID)
        # so far I am not lateralizing those voids
        self.label[np.logical_and(self.label == 0, mask_hull)] = ivoids[0]

    def compute_surface(self):
        """
        Here we compute the convex hull of the surface of the brain
        All voids below the surface are re-assigned to void_fluid new region
        :return:
        """
        super().compute_surface()
        # note that ideally we should rather take the points within the convex hull of the brain seen from the top
        iok = np.where(~np.isnan(self.top))
        yxz = np.c_[np.c_[iok], self.top[iok]]
        # computes the convex hull of the surface and interpolate over the brain
        ch = scipy.spatial.ConvexHull(yxz)
        z = scipy.interpolate.griddata(
            points=yxz[ch.vertices[1:], :2],
            values=yxz[ch.vertices[1:], 2],
            xi=yxz[:, :2],
            method="linear",
        )
        # the output is the convex surface of the brain - note that we
        self.convex_top = np.zeros_like(self.top) * np.nan
        self.convex_top[iok] = z


@functools.lru_cache(maxsize=32)
def regions_transition_matrix(ba=None, mapping=None):
    """
    Computes transition matrices between brain regions based on vertical adjacency.

    This function calculates how brain regions are spatially connected to each other
    by analyzing transitions between regions along the dorsal-ventral axis. It creates
    a matrix where each element (i,j) represents the number of voxel transitions
    from region i to region j when moving along the dorsal-ventral direction.

    Parameters
    ----------
    ba : ClassifierAtlas, optional
        Brain atlas object to use for the computation. If None, a new ClassifierAtlas
        instance will be created.
    mapping : str, optional
        The region mapping to use (e.g., 'Allen', 'Cosmos', 'Beryl').
        Defaults to 'Cosmos' if None.

    Returns
    -------
    state_transitions : numpy.ndarray
        A square matrix where element (i,j) represents the number of voxel
        transitions from region i to region j along the dorsal-ventral axis.
    voxel_occurences : numpy.ndarray
        A vector containing the count of voxels for each region in the mapping.
    region_ids : numpy.ndarray
        The region IDs corresponding to the rows/columns in the transition matrix.
    """
    ba = ba if ba is not None else ClassifierAtlas()
    ba.compute_surface()
    mapping = mapping if mapping is not None else "Cosmos"
    # str_mapping = 'Allen'
    volume = ba.regions.mappings["Cosmos"][ba.label]  # ap, ml, dv
    mask = ba.mask()
    volume[~mask] = -1

    # getting the unique set of regions for the given mapping
    aids_unique = np.unique(ba.regions.id[ba.regions.mappings[mapping]])
    _, ir_unique = ismember(aids_unique, ba.regions.id)

    up = volume[:, :, :-1].flatten()
    lo = volume[:, :, 1:].flatten()
    iok = np.logical_and(up >= 0, lo >= 0)
    _, icc_up = ismember(up[iok], ir_unique)
    _, icc_lo = ismember(lo[iok], ir_unique)

    # here we count the number of voxel from each reagion
    state_transitions = np.array(
        scipy.sparse.coo_matrix(  # (data, (i, j))
            (np.ones_like(icc_lo), (icc_up, icc_lo)),
            shape=(ir_unique.size, ir_unique.size),
        ).todense()
    )
    voxel_occurences = np.array(
        scipy.sparse.coo_matrix(
            (np.ones_like(icc_lo), (icc_up, icc_up * 0)), shape=(ir_unique.size, 1)
        ).todense()
    )

    return state_transitions, voxel_occurences.squeeze(), ba.regions.id[ir_unique]
