"""Tests for the UnitEncoder serving wrapper (the unit-level encoder family).

Self-contained: a tiny random-init release (autoencoder, latent scaler, K=3 GMM, context-weight
net, kNN bank) is written to a temp directory in the published layout and loaded through
``load_pretrained``. Context sampling is replaced by a deterministic stand-in, so neither the Allen
atlas download nor IBL data is needed. This verifies the wrapper's mechanics -- dispatch, the
predict / mixture_weights / encode / assign contract, determinism and the selftest -- not learned
quality.

In its own file with a setUpModule guard: the wrapper pulls in torch, which segfaults on macOS
arm64 if xgboost is already imported in the same process. The guard is macOS arm64-only -- Linux
CI runs the whole suite in a single `unittest discover` process and does not segfault.
"""

import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ephysatlas import model_registry

from tests._model_fixtures import (
    FakeContextManager,
    UNIT_FEATURES,
    make_unit_model_dir,
    synthetic_units,
    unit_positions,
    write_checksums,
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


class TestUnitEncoderWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.path_model = make_unit_model_dir(cls.tmp)
        cls.positions = unit_positions()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _encoder(self, path_model=None):
        from ephysatlas import load_pretrained

        encoder = load_pretrained(path_model or self.path_model, device="cpu")
        encoder._ctx_manager = FakeContextManager()
        return encoder

    def test_load_pretrained_dispatches_to_the_unit_encoder(self):
        from ephysatlas.models.unit_encoder import UnitEncoder

        self.assertIsInstance(self._encoder(), UnitEncoder)

    def test_predict_returns_one_prefixed_column_per_feature_indexed_like_input(self):
        out = self._encoder().predict(self.positions)
        self.assertEqual(list(out.columns), [f"pred_{f}" for f in UNIT_FEATURES])
        self.assertTrue(out.index.equals(self.positions.index))
        self.assertTrue(np.isfinite(out.to_numpy()).all())

    def test_predict_mixes_component_expectations_with_local_weights(self):
        encoder = self._encoder()
        weights = encoder.mixture_weights(self.positions)
        np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, rtol=1e-5)
        expected = weights.to_numpy(np.float64) @ encoder.component_features.astype(
            np.float64
        )
        np.testing.assert_allclose(
            encoder.predict(self.positions).to_numpy(), expected, rtol=1e-5
        )

    def test_bundle_refuses_data_described_by_other_features(self):
        from types import SimpleNamespace

        encoder = self._encoder()
        waveform, acg, stpc = synthetic_units(encoder.cfg, n=4)
        data = SimpleNamespace(
            waveforms=waveform,
            acgs=acg,
            stpc=stpc,
            waveform_feature_names=["peak_val", "polarity"],
        )
        with self.assertRaisesRegex(ValueError, "peak_val"):
            encoder.bundle(data)

    def test_predict_varies_with_position_and_is_deterministic(self):
        encoder = self._encoder()
        first = encoder.predict(self.positions).to_numpy()
        np.testing.assert_array_equal(first, encoder.predict(self.positions).to_numpy())
        self.assertGreater(np.ptp(first, axis=0).max(), 0)

    def test_predict_is_symmetric_across_hemispheres(self):
        mirrored = self.positions.copy()
        mirrored["x"] = -mirrored["x"]
        encoder = self._encoder()
        np.testing.assert_allclose(
            encoder.predict(self.positions).to_numpy(),
            encoder.predict(mirrored).to_numpy(),
        )

    def test_predict_names_missing_coordinates(self):
        with self.assertRaises(KeyError) as ctx:
            self._encoder().predict(self.positions.drop(columns="z"))
        self.assertIn("z", str(ctx.exception))

    def test_predict_refuses_a_reordered_feature_list(self):
        encoder = self._encoder()
        encoder.outputs = {**encoder.outputs, "columns": list(reversed(UNIT_FEATURES))}
        with self.assertRaises(ValueError):
            encoder.predict(self.positions)

    def test_encode_returns_one_standardized_latent_per_unit(self):
        encoder = self._encoder()
        waveform, acg, stpc = synthetic_units(encoder.cfg)
        z = encoder.encode(waveform, acg, stpc)
        self.assertEqual(z.shape, (len(waveform), encoder.cfg.latent_dim()))
        raw = encoder.encode(waveform, acg, stpc, standardize=False)
        np.testing.assert_allclose(
            z, encoder.latent_scaler.transform(raw), rtol=1e-5, atol=1e-6
        )

    def test_encode_requires_every_modality_the_model_uses(self):
        encoder = self._encoder()
        waveform, acg, _ = synthetic_units(encoder.cfg)
        with self.assertRaises(ValueError):
            encoder.encode(waveform, acg)

    def test_assign_and_expected_features_cover_every_unit(self):
        encoder = self._encoder()
        z = encoder.encode(*synthetic_units(encoder.cfg))
        labels = encoder.assign(z)
        self.assertTrue(set(labels.tolist()) <= set(range(encoder.cfg.gmm_components)))
        self.assertEqual(
            encoder.expected_features(z).shape, (len(z), len(UNIT_FEATURES))
        )
        self.assertEqual(
            encoder.components()["means"].shape[0], encoder.cfg.gmm_components
        )

    def test_split_is_read_from_the_release(self):
        self.assertEqual(self._encoder().split()["test"], ["c"])


class TestUnitEncoderSelftest(unittest.TestCase):
    """selftest reproduces the golden outputs shipped with a release, and catches corruption."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = make_unit_model_dir(self.tmp, checksums=False)
        from ephysatlas.models.unit_encoder import UnitEncoder

        # Goldens are written before checksums, as publication does, so bypass load_pretrained.
        encoder = UnitEncoder(self.path_model, device="cpu")
        encoder._ctx_manager = FakeContextManager()
        example = self.path_model.joinpath("example")
        example.mkdir()
        positions = unit_positions()
        positions.to_parquet(example.joinpath("positions_sample.parquet"))
        encoder.predict(positions).to_parquet(
            example.joinpath("expected_predictions.parquet")
        )
        waveform, acg, stpc = synthetic_units(encoder.cfg)
        np.savez(
            example.joinpath("units_sample.npz"), waveform=waveform, acg=acg, stpc=stpc
        )
        np.save(
            example.joinpath("expected_latents.npy"),
            encoder.encode(waveform, acg, stpc),
        )
        write_checksums(self.path_model)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _encoder(self):
        # Constructed directly: load_pretrained would reject the tampered directories below at
        # checksum verification, before selftest -- the check under test here -- ever runs.
        from ephysatlas.models.unit_encoder import UnitEncoder

        encoder = UnitEncoder(self.path_model, device="cpu")
        encoder._ctx_manager = FakeContextManager()
        return encoder

    def test_selftest_passes_on_an_intact_release(self):
        self.assertTrue(
            model_registry.verify_checksums(self.path_model, missing_ok=False)
        )
        self.assertTrue(TestUnitEncoderWrapper._encoder(self).selftest())

    def test_selftest_catches_a_changed_golden_prediction(self):
        path = self.path_model.joinpath("example", "expected_predictions.parquet")
        import pandas as pd

        golden = pd.read_parquet(path)
        golden.iloc[0, 0] += 1.0 + abs(golden.iloc[0, 0])
        golden.to_parquet(path)
        with self.assertRaises(AssertionError):
            self._encoder().selftest()

    def test_selftest_without_golden_files_raises(self):
        shutil.rmtree(self.path_model.joinpath("example"))
        with self.assertRaises(FileNotFoundError):
            self._encoder().selftest()


if __name__ == "__main__":
    unittest.main()
