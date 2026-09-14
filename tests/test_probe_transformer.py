"""Tests for the ProbeTransformerClassifier serving wrapper (channel-region transformer family).

Self-contained: two tiny random-init ProbeTransformers (one per synthetic seed) are saved to a
temp directory with per-seed scalers and a root manifest, and the wrapper is exercised over a
synthetic multi-probe features table. This verifies the ensemble/global contract and the output
shape without training. torch-heavy, so it carries the macOS arm64 segfault guard.
"""

import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def setUpModule():
    # torch and xgboost segfault together on macOS arm64; run this file in its own process there.
    if (
        "xgboost" in sys.modules
        and sys.platform == "darwin"
        and platform.machine() == "arm64"
    ):
        raise RuntimeError(
            "xgboost is already imported in this process; loading torch as well segfaults on "
            "macOS arm64. Run this file in its own pytest process."
        )


CLASSES = [315, 549, 997]  # real Cosmos ids: Isocortex, TH, root
ACRONYMS = ["Isocortex", "TH", "root"]
FEATURES = ["rms_ap", "rms_lf", "psd_delta"]


class TestProbeTransformerWrapper(unittest.TestCase):
    """The ensemble/global predict contract, against two random-init seed models."""

    N_SEEDS = 2

    @classmethod
    def setUpClass(cls):
        import torch

        from ephysatlas.models.probe_transformer import ProbeTransformer

        cls.tmp = Path(tempfile.mkdtemp())
        cls.path_model = cls.tmp.joinpath("2026_W37_transformer")
        seeds = []
        for i in range(cls.N_SEEDS):
            seed_dir = cls.path_model.joinpath(f"SEED{i:02d}")
            seed_dir.mkdir(parents=True)
            # Distinct weights per seed, so the ensemble is a real average.
            torch.manual_seed(i)
            model = ProbeTransformer(
                n_features=len(FEATURES),
                n_classes=len(CLASSES),
                d_model=8,
                n_heads=2,
                n_layers=1,
                dropout=0.0,
            )
            torch.save(model.state_dict(), seed_dir.joinpath("best_model.pt"))
            np.savez(
                seed_dir.joinpath("feature_scaler.npz"),
                mean=np.zeros(len(FEATURES), np.float32),
                std=np.ones(len(FEATURES), np.float32),
            )
            seeds.append(seed_dir.name)
        cls.manifest = {
            "model_class": "ProbeTransformer",
            "config": {
                "classes": CLASSES,
                "class_acronyms": ACRONYMS,
                "region_map": "Cosmos",
                "model_config": {
                    "d_model": 8,
                    "n_heads": 2,
                    "n_transformer_layers": 1,
                    "dropout": 0.0,
                    "n_features": len(FEATURES),
                    "n_classes": len(CLASSES),
                    "n_merfish": 0,
                    "temperature": 1.0,
                    "model_type": "transformer",
                },
            },
            "inputs": {"features": FEATURES, "position_column": "axial_um"},
            "artifacts": {
                "weights": "best_model.pt",
                "scaler": "feature_scaler.npz",
                "seeds": seeds,
            },
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _features(self) -> pd.DataFrame:
        """A tiny two-probe features table, indexed by (pid, channel) with an axial_um column."""
        rng = np.random.default_rng(0)
        index, rows = [], []
        for pid in ("pid_a", "pid_b"):
            for ch in range(5):
                index.append((pid, ch))
                rows.append(rng.normal(size=len(FEATURES)))
        df = pd.DataFrame(
            rows,
            columns=FEATURES,
            index=pd.MultiIndex.from_tuples(index, names=["pid", "channel"]),
        )
        df["axial_um"] = np.tile(np.arange(5) * 20.0, 2)
        return df

    def _wrapper(self):
        from ephysatlas.models.probe_transformer import ProbeTransformerClassifier

        return ProbeTransformerClassifier(self.path_model, index=self.manifest)

    def test_ensemble_predict_shape_and_columns(self):
        out = self._wrapper().predict(self._features(), estimator="ensemble")
        self.assertEqual(len(out), 10)
        for col in (
            "predicted_acronym",
            "predicted_atlas_id",
            "prediction_probability",
            "seed_agreement",
        ):
            self.assertIn(col, out.columns)
        # The per-class probabilities are a proper distribution, and the winner is one of them.
        probs = out[[f"p_{a}" for a in ACRONYMS]].to_numpy()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
        self.assertTrue(
            ((out["seed_agreement"] >= 0) & (out["seed_agreement"] <= 1)).all()
        )
        self.assertTrue(set(out["predicted_atlas_id"]).issubset(CLASSES))

    def test_global_uses_first_seed_only(self):
        out = self._wrapper().predict(self._features(), estimator="global")
        # One model has nothing to agree with, so seed_agreement is NaN (not a false 1.0).
        self.assertTrue(out["seed_agreement"].isna().all())


if __name__ == "__main__":
    unittest.main()
