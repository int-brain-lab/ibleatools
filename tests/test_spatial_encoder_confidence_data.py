"""Tests for the probe-confidence training data of the channel-level model.

The synthetic samples are seeded by their index, so the dataset builds each one once and serves
it from memory afterwards, and the sample generator shares one Allen atlas instead of building one
per sample. The generator itself is replaced by a fake here (the real one needs the
interpolation model and the atlas).

In its own file with a setUpModule guard: it imports torch, which segfaults on macOS arm64 if
xgboost is already imported in the same process.
"""

import platform
import sys
import unittest
from unittest import mock

import numpy as np


def setUpModule():
    if "xgboost" in sys.modules and sys.platform == "darwin" and platform.machine() == "arm64":
        raise RuntimeError(
            "xgboost is already imported in this process; loading torch as well segfaults on "
            "macOS arm64. Run this file in its own pytest process."
        )


def _fake_sample(*, probe_idx, bank, cfg, rng, ctx_manager, base_model, handles):
    """Random arrays from the sample's own generator, like the real one (C=6, F_e=3, F_ctx=4)."""
    _fake_sample.calls += 1
    return (
        rng.normal(size=(6, 3)).astype(np.float32),
        rng.normal(size=(6, 4)).astype(np.float32),
        rng.normal(size=(6, 3)).astype(np.float32),
        rng.integers(0, 2, size=6).astype(np.int64),
        np.ones(6, dtype=bool),
    )


class TestConfidenceSampleCache(unittest.TestCase):
    def _dataset(self, cache):
        from ephysatlas.spatial_encoder.model import (
            ProbeConfidenceTrainConfig,
            SyntheticProbeConfidenceDataset,
        )

        cfg = ProbeConfidenceTrainConfig(samples_per_probe=2, cache_samples=cache)
        bank = [{} for _ in range(3)]
        return SyntheticProbeConfidenceDataset(
            bank, cfg, ctx_manager=None, base_model=None, handles=None, seed=7
        )

    def setUp(self):
        _fake_sample.calls = 0
        patcher = mock.patch(
            "ephysatlas.spatial_encoder.model._build_shift_based_synthetic_probe_sample",
            _fake_sample,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cached_samples_equal_rebuilt_ones_and_are_built_once(self):
        cached, rebuilt = self._dataset(True), self._dataset(False)
        self.assertEqual(len(cached), 6)
        for _epoch in range(3):
            for i in range(len(cached)):
                a, b = cached[i], rebuilt[i]
                for key in ("rec", "ctx", "pred", "labels", "valid"):
                    self.assertTrue(bool((a[key] == b[key]).all()), key)
        # 6 builds for the cached dataset (first epoch only), 18 for the uncached one.
        self.assertEqual(_fake_sample.calls, 6 + 18)

    def test_editing_a_sample_does_not_alter_the_cache(self):
        ds = self._dataset(True)
        first = ds[0]
        first["rec"].zero_()
        self.assertFalse(bool((ds[0]["rec"] == 0).all()))


class TestSharedAtlas(unittest.TestCase):
    def test_atlas_is_built_once_per_process(self):
        from ephysatlas.spatial_encoder import utils

        built = []

        class FakeAtlas:
            def __init__(self):
                built.append(self)

        utils._shared_allen_atlas.cache_clear()
        self.addCleanup(utils._shared_allen_atlas.cache_clear)
        with mock.patch.object(utils, "AllenAtlas", FakeAtlas):
            first = utils._shared_allen_atlas()
            second = utils._shared_allen_atlas()
        self.assertIs(first, second)
        self.assertEqual(len(built), 1)


if __name__ == "__main__":
    unittest.main()
