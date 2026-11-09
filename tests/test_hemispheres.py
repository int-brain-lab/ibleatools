import unittest

import numpy as np

from ephysatlas.hemispheres import (
    MIDLINE_X,
    mirror_xyz_to_hemisphere,
    mirror_xyz_to_left,
)


class TestHemispheres(unittest.TestCase):
    def test_midline_is_the_atlas_one(self):
        from iblatlas.atlas import AllenAtlas

        self.assertAlmostEqual(MIDLINE_X, AllenAtlas(res_um=50).midline_x, places=9)
        self.assertAlmostEqual(MIDLINE_X, -44e-6, places=9)

    def test_fold_about_the_midline(self):
        xyz = np.array(
            [
                [MIDLINE_X + 1e-3, 1e-3, -2e-3],
                [MIDLINE_X - 1e-3, 0, 0],
                [MIDLINE_X, 0, 0],
            ]
        )
        left = mirror_xyz_to_left(xyz)
        np.testing.assert_allclose(
            left[:, 0], [MIDLINE_X - 1e-3, MIDLINE_X - 1e-3, MIDLINE_X]
        )
        np.testing.assert_array_equal(left[:, 1:], xyz[:, 1:])
        np.testing.assert_allclose(mirror_xyz_to_left(left), left)  # idempotent
        right = mirror_xyz_to_hemisphere(xyz, 1.0)
        np.testing.assert_allclose(
            right[:, 0], [MIDLINE_X + 1e-3, MIDLINE_X + 1e-3, MIDLINE_X]
        )
        # input untouched
        np.testing.assert_array_equal(xyz[0], [MIDLINE_X + 1e-3, 1e-3, -2e-3])

    def test_any_shape_and_units(self):
        rng = np.random.default_rng(0)
        xyz_um = rng.normal(0, 2000, size=(4, 7, 3))
        left = mirror_xyz_to_left(xyz_um, midline_x=MIDLINE_X * 1e6)
        self.assertTrue(np.all(left[..., 0] <= MIDLINE_X * 1e6 + 1e-9))
        flat = mirror_xyz_to_left(xyz_um.reshape(-1, 3), midline_x=MIDLINE_X * 1e6)
        np.testing.assert_allclose(left.reshape(-1, 3), flat)

    def test_trajectory_centre_folded_about_the_midline(self):
        """A trajectory centre between the midline and bregma is on the right: it is folded about
        the midline like the trace points, so the parameters describe the trace."""
        from ephysatlas.alignment.ephys_only import (
            AnnealConfig,
            TraceParams,
            left_x_um,
            trace_points,
        )

        midline_um = MIDLINE_X * 1e6
        self.assertAlmostEqual(left_x_um(midline_um + 20.0), midline_um - 20.0)
        self.assertAlmostEqual(left_x_um(midline_um - 20.0), midline_um - 20.0)
        cfg = AnnealConfig()
        depths = np.arange(0.0, 3000.0, 100.0)

        def trace(x_um):
            params = TraceParams(
                x=x_um,
                y=-2000.0,
                z=-3000.0,
                theta=15.0,
                phi=30.0,
                length=3000.0,
                bends=np.zeros((cfg.n_sections - 1, 2), dtype=np.float32),
            )
            return trace_points(params, depths, "negative", cfg)[0]

        np.testing.assert_allclose(
            trace(midline_um + 20.0), trace(midline_um - 20.0), atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
