"""
Unit tests for ephysatlas.edges module.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ephysatlas.edges import (
    compute_edges_volume,
    load_whitened_pcs,
    mahalanobis_gradient,
)

RES_UM = 50
SHAPE = (24, 20, 16)


def _fake_atlas(shape=SHAPE):
    """Atlas stand-in: everything is brain, region 0 is 'root', region 1 is 'void'."""
    return SimpleNamespace(
        regions=SimpleNamespace(acronym=np.array(["root", "void"])),
        label=np.zeros(shape, dtype=int),
        mask=lambda: np.ones(shape, dtype=bool),
    )


def _write_volume(file, n_features=4, seed=0):
    """Encoding volume with a step in feature 0 along AP, in the (ML, AP, DV) layout."""
    rng = np.random.default_rng(seed)
    vol = rng.normal(size=SHAPE + (n_features,)).astype(np.float32) * 0.05 + 1
    vol[SHAPE[0] // 2 :, ..., 0] += 2
    np.savez(
        file,
        ephys_atlas_vol=np.transpose(vol, (1, 0, 2, 3)).astype(np.float16),
        res_um=np.array([RES_UM]),
        mean_per_feature=np.ones(n_features),
        std_per_feature=np.ones(n_features),
        feature_names=np.array([f"f{i}" for i in range(n_features)]),
    )


class TestMahalanobisGradient(unittest.TestCase):
    def test_peaks_at_step_and_scales_with_height(self):
        pcs = np.zeros(SHAPE + (1,), dtype=np.float32)
        pcs[SHAPE[0] // 2 :, ..., 0] = 1
        valid = np.ones(SHAPE, dtype=bool)
        edges = mahalanobis_gradient(pcs, valid, RES_UM, sigma_um=100, erode_um=100)
        profile = np.nanmean(edges, axis=(1, 2))
        self.assertLessEqual(abs(int(np.nanargmax(profile)) - SHAPE[0] // 2), 1)
        # smoothed unit step: peak slope is 1 / (sigma sqrt(2 pi)) per voxel
        expected = 1 / (2 * np.sqrt(2 * np.pi)) * 1e3 / RES_UM
        self.assertAlmostEqual(np.nanmax(profile), expected, delta=0.1 * expected)
        double = mahalanobis_gradient(2 * pcs, valid, RES_UM, 100, 100)
        np.testing.assert_allclose(double, 2 * edges, rtol=1e-5, equal_nan=True)

    def test_nan_border_and_uniform_volume(self):
        pcs = np.ones(SHAPE + (2,), dtype=np.float32)
        valid = np.ones(SHAPE, dtype=bool)
        edges = mahalanobis_gradient(pcs, valid, RES_UM, sigma_um=100, erode_um=100)
        self.assertEqual(edges.shape, SHAPE)
        self.assertTrue(np.isnan(edges[0]).all() and np.isnan(edges[:, :, -1]).all())
        self.assertTrue(np.isfinite(edges[2:-2, 2:-2, 2:-2]).all())
        np.testing.assert_allclose(edges[2:-2, 2:-2, 2:-2], 0, atol=1e-4)

    def test_invalid_voxels_do_not_bias_the_border(self):
        pcs = np.ones(SHAPE + (1,), dtype=np.float32)
        valid = np.ones(SHAPE, dtype=bool)
        valid[:, :, SHAPE[2] // 2 :] = False
        pcs[~valid] = np.nan
        edges = mahalanobis_gradient(pcs, valid, RES_UM, sigma_um=100, erode_um=100)
        # constant data inside the mask: no edge, even next to the NaN exterior
        np.testing.assert_allclose(np.nanmax(edges), 0, atol=1e-3)


class TestEdgesVolume(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.volume_file = self.tmp.joinpath("brainwide_ephys_atlas_50um.npz")
        _write_volume(self.volume_file)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pcs_are_whitened(self):
        pca = load_whitened_pcs(self.volume_file, _fake_atlas())
        pcs = pca["pcs"][pca["valid"]]
        np.testing.assert_allclose(
            np.cov(pcs, rowvar=False), np.eye(pcs.shape[1]), atol=1e-3
        )
        self.assertEqual(pca["res_um"], RES_UM)
        self.assertEqual(pca["valid"].sum(), np.prod(SHAPE))

    def test_compute_writes_next_to_volume(self):
        edges = compute_edges_volume(self.volume_file, atlas=_fake_atlas())
        out = self.tmp.joinpath("brainwide_ephys_edges_50um.npz")
        self.assertTrue(out.exists())
        archive = np.load(out)
        stored = archive["ephys_edges_vol"]
        # same (ML, AP, DV) convention as the encoding volume
        self.assertEqual(stored.shape, (SHAPE[1], SHAPE[0], SHAPE[2]))
        np.testing.assert_array_equal(archive["grid_shape"], stored.shape)
        self.assertEqual(int(archive["res_um"][0]), RES_UM)
        np.testing.assert_allclose(
            np.transpose(stored, (1, 0, 2)).astype(np.float32),
            edges,
            rtol=1e-2,
            equal_nan=True,
        )
        profile = np.nanmean(edges, axis=(1, 2))
        self.assertLessEqual(abs(int(np.nanargmax(profile)) - SHAPE[0] // 2), 1)


if __name__ == "__main__":
    unittest.main()
