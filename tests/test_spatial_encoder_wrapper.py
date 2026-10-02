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


def make_spatial_model_dir(
    path_models: Path, *, stats_offset: float = 0.0, bank: dict = None
) -> Path:
    """A tiny spatial-encoder release: weights, confidence model, stats, split, bank, manifest.

    ``bank`` (``xyz``, ``feat``, ``pid``) replaces the default three channels at the origin.
    """
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
    if bank is None:
        bank = dict(
            xyz=np.zeros((3, 3), np.float32),
            feat=np.zeros((3, f_e), np.float32),
            pid=np.array(["a", "a", "b"]),
        )
    np.savez(path_model.joinpath(model_registry.ENCODER_BANK_FILE), **bank)
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


class _FakeContextManager:
    """Context that depends on the sign of x, unlike the real (self-mirroring) manager: a path
    that reached it with an unmirrored right-hemisphere position would get another context."""

    def __init__(self):
        rng = np.random.default_rng(1)
        self.weights = rng.normal(size=(3, 100)) * 1e3

    def sample_context_numpy_m(self, xyz_m, mode="clip"):
        ctx = np.asarray(xyz_m, np.float64) @ self.weights + 1.0
        return {"cell_pc": ctx[:, :50], "gene_pc": ctx[:, 50:]}


class TestSpatialEncoderPredictHemispheres(unittest.TestCase):
    """The model lives in the left hemisphere (x -> -|x|): a right-hemisphere position must be
    predicted exactly as its left mirror, for the context, the neighbours and the position input."""

    # Left-hemisphere channels around (-2, -2, -3) mm, as the published bank stores them.
    CENTRE = np.array([-2e-3, -2e-3, -3e-3])

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        n = 40
        bank = dict(
            xyz=(self.CENTRE + rng.uniform(-2e-4, 2e-4, size=(n, 3))).astype(np.float32),
            feat=rng.normal(size=(n, len(FEATURES))).astype(np.float32),
            pid=np.array(["a", "b"] * (n // 2)),
        )
        self.path_model = make_spatial_model_dir(self.tmp, bank=bank)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _predict(self, df):
        from unittest import mock

        from ephysatlas import load_pretrained
        from ephysatlas.models.encoder_inpainting import SpatialEncoder

        fake = _FakeContextManager()
        with mock.patch.object(SpatialEncoder, "_context_manager", lambda self: fake):
            return load_pretrained(self.path_model, device="cpu").predict(df)

    def _positions(self, x_sign):
        import pandas as pd

        rng = np.random.default_rng(2)
        xyz = self.CENTRE + rng.uniform(-1e-4, 1e-4, size=(6, 3))
        xyz[:, 0] = x_sign * np.abs(xyz[:, 0])
        index = pd.MultiIndex.from_tuples(
            [("query", c) for c in range(len(xyz))], names=["pid", "channel"]
        )
        return pd.DataFrame(xyz, index=index, columns=["x", "y", "z"])

    def test_right_hemisphere_predicts_as_its_mirror(self):
        from ephysatlas import load_pretrained

        left, right = self._positions(-1.0), self._positions(+1.0)
        # Guard: the left queries do have neighbours, so the comparison is not vacuous.
        encoder = load_pretrained(self.path_model, device="cpu")
        _, _, mask = encoder._neighbours(
            left.to_numpy(np.float32), np.array(["query"] * len(left))
        )
        self.assertTrue(mask.all(axis=1).all())

        np.testing.assert_allclose(
            self._predict(right).to_numpy(), self._predict(left).to_numpy(), rtol=1e-6, atol=1e-6
        )

    def test_output_keeps_the_input_index_and_order(self):
        import pandas as pd

        left, right = self._positions(-1.0), self._positions(+1.0)
        right.index = pd.MultiIndex.from_tuples(
            [("query_r", c) for c in range(len(right))], names=["pid", "channel"]
        )
        mixed = pd.concat([left, right]).sample(frac=1.0, random_state=3)
        out = self._predict(mixed)
        self.assertTrue(out.index.equals(mixed.index))
        self.assertEqual(list(out.columns), [f"pred_{f}" for f in FEATURES])
        # Row by row, each prediction is the one of its own (mirrored) position.
        expected = self._predict(left)
        for channel in range(len(left)):
            for pid in ("query", "query_r"):
                np.testing.assert_allclose(
                    out.loc[(pid, channel)].to_numpy(),
                    expected.loc[("query", channel)].to_numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_neighbor_bank_is_built_in_the_left_hemisphere(self):
        import pandas as pd

        from ephysatlas import load_pretrained
        from ephysatlas.models.encoder_inpainting import build_neighbor_bank

        encoder = load_pretrained(self.path_model, device="cpu")
        df = pd.concat([self._positions(-1.0), self._positions(+1.0)])
        for i, feature in enumerate(FEATURES):
            df[feature] = float(i)
        out = self.tmp.joinpath("bank")
        out.mkdir()
        build_neighbor_bank(out, df, encoder.index, model=encoder.model)
        with np.load(out.joinpath(model_registry.ENCODER_BANK_FILE)) as bank:
            np.testing.assert_allclose(bank["xyz"][:, 0], -np.abs(df["x"].to_numpy()), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
