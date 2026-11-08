"""Folding positions onto one hemisphere, about the plane of symmetry of the Allen CCF.

The models live in the left hemisphere: every position is reflected about the atlas midline
before a context lookup, a neighbour search or a comparison with a prediction. The midline is the
``midline_x`` of iblatlas's AllenAtlas, 44 µm left of bregma (iblatlas#41); reflecting about bregma
(x = 0) would move every right-hemisphere position by 88 µm.
"""

import numpy as np
from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM, ALLEN_CCF_MIDLINE_ML_UM

# The atlas midline in IBL coordinates (metres, relative to bregma).
MIDLINE_X = (
    ALLEN_CCF_MIDLINE_ML_UM - ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"][0]
) * 1e-6


def mirror_xyz_to_hemisphere(
    xyz: np.ndarray, sign: float = -1.0, midline_x: float = MIDLINE_X
) -> np.ndarray:
    """A copy of ``xyz`` (``[..., 3]``) with every position folded onto one hemisphere by
    reflection about the atlas midline: the left one (``sign`` < 0) or the right one.

    ``midline_x`` is in the units of ``xyz``: metres by default, ``MIDLINE_X * 1e6`` for µm.
    """
    out = np.array(xyz, copy=True)
    side = 1.0 if float(sign) >= 0 else -1.0
    out[..., 0] = midline_x + side * np.abs(out[..., 0] - midline_x)
    return out


def mirror_xyz_to_left(xyz: np.ndarray, midline_x: float = MIDLINE_X) -> np.ndarray:
    """A copy of ``xyz`` folded onto the left hemisphere, where the models live (see
    :func:`mirror_xyz_to_hemisphere`)."""
    return mirror_xyz_to_hemisphere(xyz, -1.0, midline_x)
