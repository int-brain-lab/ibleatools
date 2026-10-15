"""Tests for the UnitEncoder serving wrapper (the unit-level encoder family).

Self-contained: a tiny release (random-init autoencoder, latent scaler, hierarchical mixture
fitted on random latents, random-init context networks, members readout) is written to a temp
directory in the published layout and loaded through ``load_pretrained``. Context sampling is
replaced by a deterministic stand-in, so neither the Allen atlas download nor IBL data is needed.
This verifies the wrapper's mechanics -- dispatch, the predict / mixture_weights / sample / encode /
assign contract, the top-down hierarchy, determinism and the selftest -- not learned quality.

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

    def test_predict_mixes_subtype_member_means_with_local_weights(self):
        encoder = self._encoder()
        weights = encoder.mixture_weights(self.positions)
        np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, rtol=1e-5)
        expected = weights.to_numpy(np.float64) @ encoder.readout.means.astype(
            np.float64
        )
        np.testing.assert_allclose(
            encoder.predict(self.positions).to_numpy(), expected, rtol=1e-5, atol=1e-6
        )

    def test_subtype_weights_add_up_to_the_type_weights(self):
        encoder = self._encoder()
        types = encoder.mixture_weights(self.positions, level="type").to_numpy(
            np.float64
        )
        subtypes = encoder.mixture_weights(self.positions).to_numpy(np.float64)
        parent = encoder.mixture.parent
        summed = np.stack(
            [subtypes[:, parent == k].sum(axis=1) for k in range(types.shape[1])], 1
        )
        np.testing.assert_allclose(summed, types, rtol=1e-5, atol=1e-6)
        self.assertEqual(
            list(encoder.mixture_weights(self.positions).columns),
            [f"subtype_{label}" for label in encoder.subtype_labels],
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

    def test_assign_is_top_down_and_components_describe_the_hierarchy(self):
        encoder = self._encoder()
        z = encoder.encode(*synthetic_units(encoder.cfg))
        types, subtypes = encoder.assign(z)
        parent = encoder.mixture.parent
        np.testing.assert_array_equal(parent[subtypes], types)
        p_type, _ = encoder.posterior(z)
        np.testing.assert_array_equal(types, p_type.argmax(axis=1))
        components = encoder.components()
        self.assertEqual(components["types"]["means"].shape[0], encoder.cfg.n_types)
        self.assertEqual(len(components["subtypes"]["means"]), len(parent))
        self.assertTrue(np.all(np.diff(parent) >= 0))

    def test_split_is_read_from_the_release(self):
        self.assertEqual(self._encoder().split()["test"], ["c"])

    def test_sample_draws_training_members_averaging_to_predict(self):
        encoder = self._encoder()
        positions = self.positions.iloc[:3]
        draws = encoder.sample(positions, n_samples=4000, seed=1)
        self.assertEqual(
            list(draws.columns), ["sample", "type", "subtype", *UNIT_FEATURES]
        )
        self.assertTrue(draws.index.equals(positions.index.repeat(4000)))
        np.testing.assert_array_equal(
            encoder.mixture.parent[draws["subtype"].to_numpy()],
            draws["type"].to_numpy(),
        )
        features = draws.loc[:, UNIT_FEATURES].to_numpy()
        self.assertTrue(np.isin(features[:, 0], encoder.readout.features[:, 0]).all())
        means = features.reshape(3, 4000, -1).mean(axis=1)
        np.testing.assert_allclose(
            means, encoder.predict(positions).to_numpy(), atol=0.1
        )
        self.assertTrue(draws.equals(encoder.sample(positions, n_samples=4000, seed=1)))


class TestHierarchicalMixture(unittest.TestCase):
    """The top-down hierarchy and the members readout, on well-separated synthetic data."""

    @classmethod
    def setUpClass(cls):
        from ephysatlas.unit_level_encoder.hierarchy import fit_hierarchical_mixture
        from tests._model_fixtures import unit_test_config

        rng = np.random.default_rng(0)
        cls.cfg = unit_test_config()
        centres = rng.normal(size=(3, 4)) * 8.0
        cls.z = centres[np.arange(240) % 3] + rng.normal(size=(240, 4))
        cls.mixture = fit_hierarchical_mixture(cls.z, cls.cfg)

    def test_subtypes_nest_in_their_types(self):
        m = self.mixture
        self.assertEqual(m.n_types, 3)
        # 80 units per type: 3 sub-types of at least 8 units each.
        self.assertEqual(m.n_subtypes, 9)
        np.testing.assert_allclose(m.subtypes.weights_.sum(), 1.0)
        for k in range(m.n_types):
            np.testing.assert_allclose(m.within_weights[m.parent == k].sum(), 1.0)
            np.testing.assert_allclose(
                m.subtypes.weights_[m.parent == k].sum(), m.types.weights_[k]
            )

    def test_posteriors_are_top_down(self):
        m = self.mixture
        p_type, p_subtype = m.posterior(self.z)
        np.testing.assert_allclose(p_type, m.types.predict_proba(self.z), atol=1e-8)
        summed = np.stack(
            [p_subtype[:, m.parent == k].sum(axis=1) for k in range(m.n_types)], 1
        )
        np.testing.assert_allclose(summed, p_type, atol=1e-10)
        types, subtypes = m.assign(self.z)
        np.testing.assert_array_equal(m.parent[subtypes], types)

    def test_marginal_log_densities_use_the_chosen_dimensions(self):
        from scipy.stats import multivariate_normal

        from ephysatlas.unit_level_encoder.hierarchy import gaussian_log_prob

        m, dims = self.mixture, np.array([0, 2])
        got = gaussian_log_prob(m.types.means_, m.types.covariances_, self.z[:5], dims)
        for k in range(m.n_types):
            expected = multivariate_normal(
                m.types.means_[k, dims], m.types.covariances_[k][np.ix_(dims, dims)]
            ).logpdf(self.z[:5, dims])
            np.testing.assert_allclose(got[:, k], expected, rtol=1e-10)

    def test_members_readout_means_and_draws(self):
        from ephysatlas.unit_level_encoder.hierarchy import MemberReadout

        rng = np.random.default_rng(1)
        _, p_subtype = self.mixture.posterior(self.z)
        features = rng.normal(size=(len(self.z), 2))
        readout = MemberReadout.fit(p_subtype, features, ["a", "b"])
        expected = (p_subtype.T @ features) / p_subtype.sum(axis=0)[:, None]
        np.testing.assert_allclose(readout.means, expected, rtol=1e-5, atol=1e-6)
        weights = rng.dirichlet(np.ones(self.mixture.n_subtypes), size=2)
        subtypes, rows = readout.draw(weights, 40000, rng)
        for i in range(2):
            freq = np.bincount(subtypes[i], minlength=self.mixture.n_subtypes) / 40000
            np.testing.assert_allclose(freq, weights[i], atol=0.01)
            np.testing.assert_allclose(
                readout.features[rows[i]].mean(axis=0),
                readout.expected(weights[i : i + 1])[0],
                atol=0.03,
            )
        with tempfile.TemporaryDirectory() as tmp:
            loaded = MemberReadout.load(readout.save(Path(tmp) / "members.npz"))
        np.testing.assert_array_equal(loaded.rows, readout.rows)
        np.testing.assert_allclose(loaded.means, readout.means)

    def test_mixture_round_trips(self):
        from ephysatlas.unit_level_encoder.hierarchy import HierarchicalMixture

        with tempfile.TemporaryDirectory() as tmp:
            loaded = HierarchicalMixture.load(self.mixture.save(tmp))
            np.testing.assert_allclose(
                loaded.posterior(self.z)[1], self.mixture.posterior(self.z)[1]
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
