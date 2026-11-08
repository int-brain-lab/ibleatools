"""The channel model's optional ways of predicting, without the released model or the atlas."""

import unittest
from types import SimpleNamespace

import numpy as np

from ephysatlas.hemispheres import MIDLINE_X
from ephysatlas.models.encoder_inpainting import SpatialEncoder
from ephysatlas.spatial_encoder.utils import ContextAtlasManager


def _manager(values):
    """A context manager on a ``[X, Z, Y]`` grid: 1 atlas index per mm of distance to the midline."""
    manager = object.__new__(ContextAtlasManager)
    manager.bc = SimpleNamespace(
        xyz2i=lambda xyz, round=True, mode="raise": np.abs(xyz - [MIDLINE_X, 0, 0]) * 1e3
    )
    manager.cell_pca = values[None].astype(np.float32)
    manager.gene_pca = (2 * values[None]).astype(np.float32)
    return manager


class TestTrilinearContext(unittest.TestCase):
    """Voxels sit every 8 atlas indices, here 8 mm: two along x, of value 1 and 3."""

    def setUp(self):
        self.values = np.ones((2, 2, 2))
        self.values[1] = 3.0

    def sample(self, x_voxels):
        xyz = np.array([[MIDLINE_X + x_voxels * 8e-3, 0.0, 0.0]])  # right of the midline: mirrored
        pack = _manager(self.values).sample_context_trilinear_m(xyz)
        return float(pack["cell_pc"][0, 0]), float(pack["gene_pc"][0, 0])

    def test_a_position_between_two_voxels_takes_their_mean(self):
        self.assertAlmostEqual(self.sample(0.5)[0], 2.0)

    def test_every_pc_is_interpolated(self):
        self.assertAlmostEqual(self.sample(0.25)[1], 3.0)

    def test_empty_voxels_do_not_pull_the_context_to_zero(self):
        self.values[1] = 0.0
        self.assertAlmostEqual(self.sample(0.5)[0], 1.0)

    def test_a_position_among_empty_voxels_stays_empty(self):
        self.values[:] = 0.0
        self.assertEqual(self.sample(0.5), (0.0, 0.0))


class TestReleaseOptions(unittest.TestCase):
    def encoder(self, **prediction):
        encoder = object.__new__(SpatialEncoder)
        encoder.config = dict(prediction=prediction) if prediction else {}
        return encoder

    def test_a_release_without_options_predicts_as_it_was_published(self):
        self.assertEqual(self.encoder().prediction_options(), (False, 0))

    def test_a_release_gives_its_options(self):
        encoder = self.encoder(trilinear_context=True, neighbour_draws=4)
        self.assertEqual(encoder.prediction_options(), (True, 4))

    def test_the_caller_overrides_the_release(self):
        encoder = self.encoder(trilinear_context=True, neighbour_draws=4)
        self.assertEqual(encoder.prediction_options(False, 0), (False, 0))
        self.assertEqual(encoder.prediction_options(None, 2), (True, 2))


class TestNeighbourDraws(unittest.TestCase):
    def setUp(self):
        n = 40
        self.encoder = object.__new__(SpatialEncoder)
        self.encoder._bank = dict(
            xyz=np.column_stack([np.arange(n) * 1e-5, np.zeros(n), np.zeros(n)]).astype(
                np.float32
            ),
            feat=np.arange(n, dtype=np.float32)[:, None],
            pid=np.array(["other"] * 20 + ["own"] * 20),
        )
        self.encoder.config = dict(
            neighbourhood=dict(m_max=4, radius_um=5000.0, allow_same_probe=False)
        )
        self.query = np.zeros((3, 3), dtype=np.float32)

    def draw(self, draws=3):
        pids = np.full(3, "own")
        return list(self.encoder._neighbour_draws(self.query, pids, draws))

    def test_draws_are_reproducible_and_differ(self):
        first, second = self.draw(), self.draw()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a[0], b[0])
        self.assertFalse(np.array_equal(first[0][0], first[1][0]))

    def test_draws_keep_shape_and_leave_the_own_probe_out(self):
        for e_n, p_n, mask in self.draw():
            self.assertEqual(e_n.shape, (3, 4, 1))
            self.assertEqual(p_n.shape, (3, 4, 3))
            self.assertTrue(mask.all())
            self.assertLess(
                e_n.max(), 20
            )  # channels 20.. belong to the query's own probe


if __name__ == "__main__":
    unittest.main()
