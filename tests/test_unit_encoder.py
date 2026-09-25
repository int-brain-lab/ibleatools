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

    def test_sample_needs_a_release_with_the_context_local_readout(self):
        encoder = self._encoder()
        self.assertFalse(encoder.knn_decoder.has_context_readout)
        with self.assertRaisesRegex(RuntimeError, "context-local member readout"):
            encoder.sample(self.positions)


class TestUnitEncoderContextLocalReadout(unittest.TestCase):
    """A release whose kNN bank carries the context-local member readout."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.path_model = make_unit_model_dir(cls.tmp, readout=True)
        cls.positions = unit_positions(n=30)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _encoder(self):
        return TestUnitEncoderWrapper._encoder(self)

    def _context(self, encoder):
        xyz = encoder._coordinates(self.positions)
        return encoder.context_model.transform.transform(encoder._raw_context(xyz))

    def test_bank_round_trips_the_readout_and_components_are_member_means(self):
        encoder = self._encoder()
        knn = encoder.knn_decoder
        self.assertTrue(knn.has_context_readout)
        self.assertEqual(len(knn.labels_train), len(knn.z_train))
        self.assertEqual(knn.key_train.shape, (len(knn.z_train), 2))
        np.testing.assert_allclose(
            encoder.component_features,
            knn.member_means(encoder.cfg.gmm_components),
            rtol=1e-6,
        )

    def test_predict_mixes_context_local_member_means(self):
        from ephysatlas.unit_level_encoder.pipeline import readout_settings

        encoder = self._encoder()
        weights = encoder.mixture_weights(self.positions).to_numpy(np.float64)
        expected = encoder.knn_decoder.context_local_means(
            self._context(encoder), weights, **readout_settings(encoder.cfg)
        )
        out = encoder.predict(self.positions).to_numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        global_mix = weights @ encoder.component_features.astype(np.float64)
        self.assertGreater(np.abs(out - global_mix).max(), 1e-3)
        # Infinite shrinkage falls back to the global member means.
        encoder.cfg.readout_shrinkage = 1e9
        np.testing.assert_allclose(
            encoder.predict(self.positions).to_numpy(), global_mix, rtol=1e-4, atol=1e-5
        )

    def test_sample_draws_real_members_averaging_to_predict(self):
        encoder = self._encoder()
        knn = encoder.knn_decoder
        positions = self.positions.iloc[:3]
        draws = encoder.sample(positions, n_samples=4000, seed=1)
        self.assertEqual(list(draws.columns), ["sample", "component", *UNIT_FEATURES])
        self.assertTrue(draws.index.equals(positions.index.repeat(4000)))
        features = draws.loc[:, UNIT_FEATURES].to_numpy()
        rows = np.flatnonzero(np.isin(knn.feature_train[:, 0], features[:, 0]))
        self.assertEqual(
            len(np.unique(features[:, 0])), len(np.unique(knn.feature_train[rows, 0]))
        )
        lookup = dict(zip(knn.feature_train[:, 0].tolist(), knn.labels_train.tolist()))
        self.assertEqual(
            draws["component"].tolist(), [lookup[v] for v in features[:, 0].tolist()]
        )
        means = features.reshape(3, 4000, -1).mean(axis=1)
        np.testing.assert_allclose(
            means, encoder.predict(positions).to_numpy(), atol=0.1
        )
        again = encoder.sample(positions, n_samples=4000, seed=1)
        self.assertTrue(draws.equals(again))

    def test_positions_without_context_get_the_global_member_means(self):
        encoder = self._encoder()
        n_context = encoder.cfg.n_cell_pcs + encoder.cfg.n_gene_pcs
        encoder._raw_context = lambda xyz: np.zeros((len(xyz), n_context), np.float32)
        weights = encoder.mixture_weights(self.positions).to_numpy(np.float64)
        np.testing.assert_allclose(
            encoder.predict(self.positions).to_numpy(),
            weights @ encoder.component_features.astype(np.float64),
            rtol=1e-5,
            atol=1e-6,
        )
        # Draws come from all of the drawn component's members, not only local ones.
        draws = encoder.sample(self.positions.iloc[:1], n_samples=6000, seed=2)
        np.testing.assert_allclose(
            draws.loc[:, UNIT_FEATURES].to_numpy().mean(axis=0),
            encoder.predict(self.positions.iloc[:1]).to_numpy()[0],
            atol=0.1,
        )


class TestContextLocalReadoutMath(unittest.TestCase):
    """EmpiricalKNNDecoder's readout against its formula, computed by hand."""

    def test_means_and_draws_follow_the_shrunk_local_member_means(self):
        from ephysatlas.unit_level_encoder.knn_decoder import EmpiricalKNNDecoder

        rng = np.random.default_rng(0)
        n, n_components, neighbours, shrinkage = 300, 3, 40, 3.0
        features = rng.normal(size=(n, 2))
        labels = rng.integers(0, n_components, n)
        context = rng.normal(size=(n, 4))
        knn = EmpiricalKNNDecoder.from_bank(rng.normal(size=(n, 3)), features, k=5)
        knn.set_context_readout(labels, np.eye(2, 4), np.zeros(2), context)
        # Queries in and beyond the data: the farthest are shrunk toward the global means.
        query = rng.normal(size=(6, 4)) * np.linspace(0.5, 4.0, 6)[:, None]
        weights = rng.dirichlet(np.ones(n_components), size=len(query))
        got = knn.context_local_means(
            query,
            weights,
            neighbours=neighbours,
            shrinkage=shrinkage,
            off_data_quantile=0.9,
        )

        def kernel_scale(key):
            dist = np.linalg.norm(context[:, :2] - key, axis=1)
            return np.median(np.sort(dist)[:neighbours])

        h_ref = np.quantile([kernel_scale(key) for key in context[:, :2]], 0.9)
        expected = np.zeros((len(query), 2))
        for i in range(len(query)):
            dist = np.linalg.norm(context[:, :2] - query[i, :2], axis=1)
            near = np.argsort(dist)[:neighbours]
            kernel = np.exp(-0.5 * (dist[near] / np.median(dist[near])) ** 2)
            kernel /= kernel.sum()
            in_data = min(1.0, (h_ref / np.median(dist[near])) ** 2)
            n_total = 1.0 / np.sum(kernel**2)
            for k in range(n_components):
                member = labels[near] == k
                weight_k = kernel[member].sum()
                local = kernel[member] @ features[near][member] / max(weight_k, 1e-300)
                n_k = weight_k * n_total
                mean_k = features[labels == k].mean(axis=0)
                shrunk = (n_k * local + shrinkage * mean_k) / (n_k + shrinkage)
                expected[i] += weights[i, k] * (
                    in_data * shrunk + (1.0 - in_data) * mean_k
                )
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)

        for i in (0, len(query) - 1):  # in and far beyond the data
            rows = knn.sample_context_local(
                query[i : i + 1],
                weights[i : i + 1],
                40000,
                rng,
                neighbours=neighbours,
                shrinkage=shrinkage,
                off_data_quantile=0.9,
            )
            np.testing.assert_allclose(
                features[rows[0]].mean(axis=0), got[i], atol=0.03
            )

    def test_far_from_the_data_the_readout_tends_to_the_global_member_means(self):
        from ephysatlas.unit_level_encoder.knn_decoder import EmpiricalKNNDecoder

        rng = np.random.default_rng(3)
        n, n_components = 400, 3
        features = rng.normal(size=(n, 2))
        labels = rng.integers(0, n_components, n)
        context = rng.normal(size=(n, 4))
        knn = EmpiricalKNNDecoder.from_bank(rng.normal(size=(n, 3)), features, k=5)
        knn.set_context_readout(labels, np.eye(2, 4), np.zeros(2), context)
        weights = rng.dirichlet(np.ones(n_components), size=3)
        global_mix = weights @ knn.member_means(n_components).astype(np.float64)
        settings = {"neighbours": 50, "shrinkage": 2.0}
        errors, errors_without = [], []
        for distance in (3.0, 30.0, 300.0):
            query = np.zeros((3, 4))
            query[:, 0] = distance
            without = knn.context_local_means(query, weights, **settings)
            errors_without.append(np.abs(without - global_mix).max())
            shrunk = knn.context_local_means(
                query, weights, off_data_quantile=0.99, **settings
            )
            errors.append(np.abs(shrunk - global_mix).max())
        # Without the off-data rule the edge of the data speaks for any far query; with it the
        # readout tends to the global member means as the query moves away, and is unchanged
        # within the range of the data (distance 3 of a standard normal cloud).
        self.assertGreater(min(errors_without), 0.1)
        self.assertEqual(errors[0], errors_without[0])
        self.assertTrue(errors[0] > 10 * errors[1] > 10 * errors[2])
        self.assertLess(errors[2], 1e-3)

    def test_a_bank_saved_without_the_readout_loads_without_it(self):
        from ephysatlas.unit_level_encoder.knn_decoder import EmpiricalKNNDecoder

        rng = np.random.default_rng(1)
        knn = EmpiricalKNNDecoder.from_bank(
            rng.normal(size=(30, 3)), rng.normal(size=(30, 2)), k=4
        )
        with tempfile.TemporaryDirectory() as tmp:
            loaded = EmpiricalKNNDecoder.load_bank(
                knn.save_bank(Path(tmp) / "bank.npz")
            )
        self.assertFalse(loaded.has_context_readout)
        with self.assertRaisesRegex(RuntimeError, "context-local member readout"):
            loaded.context_local_means(
                np.zeros((1, 4)), np.ones((1, 2)) / 2, neighbours=5, shrinkage=1.0
            )


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
