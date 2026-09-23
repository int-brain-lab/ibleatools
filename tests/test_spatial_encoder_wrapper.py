"""Tests for the SpatialEncoder release accessors (the channel-level interpolation model).

Self-contained: a tiny random-init encoder and probe-confidence model are written to a temp
directory in the published layout, with the training-data statistics, the split and a neighbour
bank. ``predict`` itself needs the Allen atlas and the real context volumes and is exercised by the
publication pipeline; these tests cover what a release exposes beside it.

In its own file with a setUpModule guard: the wrapper pulls in torch, which segfaults on macOS
arm64 if xgboost is already imported in the same process.
"""

import json
import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ephysatlas import model_registry

from tests._model_fixtures import write_checksums

FEATURES = ["rms_lf", "psd_delta", "rms_ap", "spike_width_secs"]
CONF_ARCHITECTURE = dict(
    f_ctx=100, f_e=4, d_model=16, nhead=2, depth=1, mlp_ratio=2.0, drop=0.0
)


def setUpModule():
    if (
        "xgboost" in sys.modules
        and sys.platform == "darwin"
        and platform.machine() == "arm64"
    ):
        raise RuntimeError(
            "xgboost is already imported in this process; loading torch as well segfaults on "
            "macOS arm64. Run this file in its own pytest process."
        )


def make_spatial_model_dir(path_models: Path, *, stats_offset: float = 0.0) -> Path:
    """A tiny spatial-encoder release: weights, confidence model, stats, split, bank, manifest."""
    import torch

    from ephysatlas.spatial_encoder.model import (
        NeighborInpaintingModel,
        ProbeSequenceConfidenceTransformer,
    )

    torch.manual_seed(0)
    path_model = Path(path_models).joinpath("2026_W39_encoder")
    path_model.mkdir(parents=True)
    f_e = len(FEATURES)
    e_mean = torch.arange(f_e, dtype=torch.float32)
    e_std = torch.full((f_e,), 2.0)
    ctx_mean = torch.zeros(100)
    ctx_std = torch.ones(100)
    architecture = dict(
        f_ctx=100, f_ephys=f_e, f_out=f_e, d_model=16, nhead=2, depth=1, drop=0.0
    )
    model = NeighborInpaintingModel(
        e_mean=e_mean, e_std=e_std, ctx_mean=ctx_mean, ctx_std=ctx_std, **architecture
    )
    torch.save(
        {"model_state": model.state_dict(), "architecture": architecture},
        path_model.joinpath(model_registry.ENCODER_WEIGHTS_FILE),
    )
    conf = ProbeSequenceConfidenceTransformer(**CONF_ARCHITECTURE)
    torch.save(
        {"model_state": conf.state_dict(), "architecture": CONF_ARCHITECTURE},
        path_model.joinpath(model_registry.ENCODER_CONFIDENCE_FILE),
    )
    stats = path_model.joinpath(model_registry.ENCODER_STATS_FILE)
    stats.parent.mkdir()
    np.savez(
        stats,
        e_mean=e_mean.numpy() + stats_offset,
        e_std=e_std.numpy(),
        ctx_mean=ctx_mean.numpy(),
        ctx_std=ctx_std.numpy(),
        rec_ephys_low_pctl=np.full(f_e, -5.0, np.float32),
        rec_ephys_high_pctl=np.full(f_e, 5.0, np.float32),
    )
    split = {"train_pids": ["a", "b"], "validation_pids": ["c"], "test_pids": ["d"]}
    path_model.joinpath("split.json").write_text(json.dumps(split))
    np.savez(
        path_model.joinpath(model_registry.ENCODER_BANK_FILE),
        xyz=np.zeros((3, 3), np.float32),
        feat=np.zeros((3, f_e), np.float32),
        pid=np.array(["a", "a", "b"]),
    )
    index = {
        "task": "spatial-encoding",
        "model_class": "NeighborInpaintingModel",
        "vintage": "2026_W39",
        "artifacts": {
            "weights": model_registry.ENCODER_WEIGHTS_FILE,
            "confidence": model_registry.ENCODER_CONFIDENCE_FILE,
            "neighbor_bank": model_registry.ENCODER_BANK_FILE,
            "stats": model_registry.ENCODER_STATS_FILE,
            "split": "split.json",
        },
        "inputs": {"index": ["pid", "channel"], "columns": ["x", "y", "z"]},
        "outputs": {
            "kind": "continuous",
            "columns": FEATURES,
            "feature_order_sha256": model_registry.feature_order_sha256(FEATURES),
        },
        "config": {
            "architecture": architecture,
            "neighbourhood": {"radius_um": 500.0, "m_max": 8},
        },
    }
    path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).write_text(
        json.dumps(index)
    )
    write_checksums(path_model)
    return path_model


class TestSpatialEncoderAccessors(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _encoder(self, **kwargs):
        from ephysatlas import load_pretrained

        return load_pretrained(make_spatial_model_dir(self.tmp, **kwargs), device="cpu")

    def test_features_are_the_ordered_outputs(self):
        self.assertEqual(self._encoder().features, FEATURES)

    def test_split_is_read_from_the_release(self):
        split = self._encoder().split()
        self.assertEqual(split["test_pids"], ["d"])

    def test_stats_include_clipping_thresholds_and_agree_with_the_weights(self):
        stats = self._encoder().preprocessing_stats()
        for key in (
            "e_mean",
            "e_std",
            "ctx_mean",
            "ctx_std",
            "rec_ephys_low_pctl",
            "rec_ephys_high_pctl",
        ):
            self.assertIn(key, stats)
        np.testing.assert_allclose(stats["e_mean"], np.arange(len(FEATURES)))

    def test_stats_disagreeing_with_the_weights_are_refused(self):
        with self.assertRaises(ValueError):
            self._encoder(stats_offset=1.0).preprocessing_stats()

    def test_confidence_model_is_rebuilt_from_its_architecture(self):
        from ephysatlas.spatial_encoder.model import ProbeSequenceConfidenceTransformer

        model = self._encoder().load_confidence_model()
        self.assertIsInstance(model, ProbeSequenceConfidenceTransformer)
        self.assertFalse(model.training)

    def test_neighbor_bank_is_a_copy_of_the_published_bank(self):
        encoder = self._encoder()
        bank = encoder.neighbor_bank()
        self.assertEqual(bank["feat"].shape, (3, len(FEATURES)))
        bank["feat"][:] = 1.0
        self.assertEqual(float(encoder.neighbor_bank()["feat"].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
