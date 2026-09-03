"""Tests for the UnitEncoder serving wrapper (the unit-level encoder family).

Self-contained: a tiny random-init autoencoder + PT-GMM is saved to a temp directory and the
wrapper is exercised over synthetic units. This verifies the wrapper's mechanics -- loading both
checkpoints, the encode/reconstruct/components/assign contract the figures consume, output shapes
and determinism -- without training or the 13 GB cells download.

In its own file with a setUpModule guard: the wrapper pulls in torch, which segfaults on macOS
arm64 if xgboost is already imported in the same process (see the spatial-encoder parity test).
"""

import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np


def setUpModule():
    if "xgboost" in sys.modules:
        raise RuntimeError(
            "xgboost is already imported in this process; loading torch as well segfaults on "
            "macOS arm64. Run this file in its own pytest process."
        )


class TestUnitEncoderWrapper(unittest.TestCase):
    """The encode/reconstruct/components/assign contract, against a random-init model."""

    N_UNITS = 12
    LATENT = 32  # ConvEncoder output dim = cfg.shared_latent_dim
    N_COMPONENTS = 4
    CONTEXT_DIM = 14

    @classmethod
    def setUpClass(cls):
        import joblib
        import torch
        from sklearn.preprocessing import StandardScaler

        from ephysatlas.unit_level_encoder.config import Config
        from ephysatlas.unit_level_encoder.gmm_models import PointTransformerGMM
        from ephysatlas.unit_level_encoder.model import MultimodalAutoencoder

        torch.manual_seed(0)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.path_model = cls.tmp.joinpath("2026_W26_unit")
        cls.path_model.mkdir(parents=True)

        cfg = Config(device="cpu")
        cfg.shared_latent_dim = cls.LATENT
        cfg.gmm_components = cls.N_COMPONENTS

        # ---- autoencoder checkpoint (self-describing config, as train_autoencoder writes) ----
        ae = MultimodalAutoencoder(cfg).eval()
        torch.save(
            {"model_state_dict": ae.state_dict(), "config": asdict(cfg)},
            cls.path_model.joinpath("autoencoder.pt"),
        )

        # ---- PT-GMM checkpoint (carries its own dims, as fit_point_transformer_gmm writes) ----
        gmm = PointTransformerGMM(
            cls.LATENT, cls.CONTEXT_DIM, cls.N_COMPONENTS, cfg
        ).eval()
        torch.save(
            {
                "model_state_dict": gmm.state_dict(),
                "config": asdict(cfg),
                "latent_dim": cls.LATENT,
                "context_dim": cls.CONTEXT_DIM,
                "n_components": cls.N_COMPONENTS,
            },
            cls.path_model.joinpath("point_transformer_gmm.pt"),
        )

        # ---- latent scaler (fit on random latents, as the pipeline ships) ----
        scaler = StandardScaler().fit(np.random.RandomState(0).randn(64, cls.LATENT))
        joblib.dump(scaler, cls.path_model.joinpath("shared_latent_scaler.joblib"))

        cls.index = {
            "task": "unit-encoding",
            "model_class": "MultimodalAutoencoder",
            "artifacts": {
                "autoencoder": "autoencoder.pt",
                "pt_gmm": "point_transformer_gmm.pt",
                "scaler": "shared_latent_scaler.joblib",
            },
            "outputs": {"kind": "latent", "latent_dim": cls.LATENT},
        }

        rng = np.random.RandomState(1)
        cls.waveform = rng.randn(cls.N_UNITS, 20, 128).astype(np.float32)
        cls.acg = np.abs(rng.randn(cls.N_UNITS, 10, 201)).astype(np.float32)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _encoder(self):
        from ephysatlas.models.unit_encoder import UnitEncoder

        return UnitEncoder(self.path_model, index=self.index)

    def test_encode_returns_one_latent_vector_per_unit(self):
        z = self._encoder().encode(self.waveform, self.acg)
        self.assertEqual(z.shape, (self.N_UNITS, self.LATENT))
        self.assertTrue(np.isfinite(z).all())

    def test_encode_is_deterministic(self):
        enc = self._encoder()
        np.testing.assert_array_equal(
            enc.encode(self.waveform, self.acg), enc.encode(self.waveform, self.acg)
        )

    def test_reconstruct_returns_both_modalities_at_input_shape(self):
        rec = self._encoder().reconstruct(self.waveform, self.acg)
        self.assertEqual(rec["waveform"].shape, self.waveform.shape)
        self.assertEqual(rec["acg"].shape, self.acg.shape)

    def test_components_are_the_gmm_putative_cell_types(self):
        means, log_var = self._encoder().components()
        self.assertEqual(means.shape, (self.N_COMPONENTS, self.LATENT))
        self.assertEqual(log_var.shape, (self.N_COMPONENTS, self.LATENT))

    def test_assign_maps_each_unit_to_a_component(self):
        enc = self._encoder()
        z = enc.encode(self.waveform, self.acg, standardize=True)
        comp = enc.assign(z)
        self.assertEqual(comp.shape, (self.N_UNITS,))
        self.assertTrue(((comp >= 0) & (comp < self.N_COMPONENTS)).all())

    def test_latents_reads_a_cached_atlas_and_standardises(self):
        # The atlas dataset is fetched from S3 on first use, then cached under <cache>/arrays.
        # With the cache pre-populated, latents() must read it (no download) and encode it.
        cache = self.tmp.joinpath("atlas_cache")
        arrays = cache.joinpath("arrays")
        arrays.mkdir(parents=True)
        n = 6
        rng = np.random.RandomState(2)
        np.save(
            arrays.joinpath("waveforms.npy"), rng.randn(n, 20, 128).astype(np.float32)
        )
        np.save(
            arrays.joinpath("acgs.npy"),
            np.abs(rng.randn(n, 10, 201)).astype(np.float32),
        )
        np.save(
            arrays.joinpath("ctx.npy"),
            rng.randn(n, self.CONTEXT_DIM).astype(np.float32),
        )
        np.save(arrays.joinpath("xyz.npy"), rng.randn(n, 3).astype(np.float32))
        np.save(
            arrays.joinpath("pids.npy"),
            np.array([f"pid{i % 2}" for i in range(n)], dtype=object),
            allow_pickle=True,
        )
        z = self._encoder().latents(cache_dir=cache)
        self.assertEqual(z.shape, (n, self.LATENT))
        self.assertTrue(np.isfinite(z).all())


if __name__ == "__main__":
    unittest.main()
