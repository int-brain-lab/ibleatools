"""The region-classifier training output IS the publish-ready release layout.

``training/train_region_classifier.py`` writes ``model.ubj`` at the root, ``folds/FOLD0k/model.ubj``
(weights only), ``split.json`` and the manifest directly -- no ``meta.yaml`` at any level (change
sets A/D/F, mirroring the other two families). This exercises the training core on a tiny synthetic
feature table (real Cosmos region ids, UUID pids so ``write_split`` is happy) in seconds on CPU: it
does not check learned quality, only that the trainer emits a valid, loadable, meta-free release.

This is an xgboost test (no torch), so it runs in the normal xgboost group.
"""

import importlib.util
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import ephysatlas.model_registry as model_registry

# Real Cosmos ids (so the manifest's acronym lookup resolves) and a few feature-column names.
CLASSES = [315, 549, 997]  # Isocortex, TH, root
FEATURES = ["rms_ap", "rms_lf", "psd_delta", "psd_theta", "spike_count"]


def _load_trainer():
    """Import ``training/train_region_classifier.py`` by path (training/ is not on sys.path)."""
    path = Path(__file__).resolve().parents[1] / "training" / "train_region_classifier.py"
    spec = importlib.util.spec_from_file_location("train_region_classifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_features(n_probes: int = 6, per_class: int = 3, seed: int = 0) -> pd.DataFrame:
    """A tiny feature table: every probe carries all classes, so every fold sees them all.

    Index is ``(pid, channel)`` with **UUID** pids (``write_split`` -> ``hash_uuids`` requires
    them). Each probe has ``per_class`` channels of each Cosmos id; features are random.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_probes):
        pid = str(uuid.uuid4())
        labels = np.repeat(CLASSES, per_class)
        n = len(labels)
        idx = pd.MultiIndex.from_arrays(
            [[pid] * n, np.arange(n)], names=["pid", "channel"]
        )
        data = {f: rng.normal(size=n) for f in FEATURES}
        # A weak signal so the classifier is not degenerate, but the test asserts only structure.
        for k, cls in enumerate(CLASSES):
            data[FEATURES[0]] = np.where(labels == cls, data[FEATURES[0]] + k, data[FEATURES[0]])
        data["Cosmos_id"] = labels.astype(float)
        frames.append(pd.DataFrame(data, index=idx))
    return pd.concat(frames)


class TestTrainRegionClassifier(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.trainer = _load_trainer()
        self.df = _synthetic_features()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_emits_a_loadable_canonical_meta_free_release(self):
        import ephysatlas.regionclassifier as regionclassifier

        path_model = self.trainer.train_region_classifier(
            self.df,
            FEATURES,
            self.tmp,
            vintage="2025_W39",
            n_folds=3,
            device="cpu",
        )

        # The model dir is named <vintage>_<region_map>_<hash>.
        self.assertTrue(path_model.name.startswith("2025_W39_Cosmos_"))

        # Canonical weights at the root and per fold; the fold and root are weights-only.
        self.assertTrue(path_model.joinpath("model.ubj").exists())
        self.assertFalse(path_model.joinpath("meta.yaml").exists())
        fold_dirs = sorted(path_model.joinpath("folds").glob("FOLD*"))
        self.assertEqual(len(fold_dirs), 3)
        for fold in fold_dirs:
            self.assertTrue(fold.joinpath("model.ubj").exists())
            self.assertFalse(fold.joinpath("meta.yaml").exists())

        # The split and predictions were published.
        self.assertTrue(path_model.joinpath(model_registry.MODEL_SPLIT_FILE).exists())
        self.assertTrue(path_model.joinpath("predictions.pqt").exists())

        # The manifest is the single source of truth: right task, lists the folds, and validates.
        index = model_registry.read_manifest(path_model)
        self.assertEqual(index["task"], model_registry.TASK_REGION_CLASSIFICATION)
        self.assertEqual(index["method"], "xgboost")
        self.assertEqual(index["artifacts"]["folds"], ["FOLD00", "FOLD01", "FOLD02"])
        self.assertEqual(index["config"]["classes"], CLASSES)
        self.assertTrue(model_registry.validate_artifacts(path_model, index))

        # The directory loads through the public wrapper and predicts over all folds.
        out = regionclassifier.RegionClassifier(path_model).predict(
            self.df.loc[:, FEATURES], estimator="ensemble"
        )
        self.assertEqual(len(out), len(self.df))
        self.assertIn("predicted_acronym", out.columns)
        self.assertFalse(np.isnan(out["fold_agreement"]).all())


class TestSelectPresentFeatures(unittest.TestCase):
    """Optional schema columns a vintage did not compute are dropped, in order, not fatal."""

    def test_keeps_present_in_order_and_reports_missing(self):
        trainer = _load_trainer()
        resolved = ["rms_ap", "rms_lf_no_car", "psd_delta", "outside"]
        available = ["psd_delta", "rms_ap", "outside", "x", "y", "z"]  # no rms_lf_no_car
        present, missing = trainer.select_present_features(resolved, available)
        # Order follows the resolved (schema) order, not the table's, since it is hashed.
        self.assertEqual(present, ["rms_ap", "psd_delta", "outside"])
        self.assertEqual(missing, ["rms_lf_no_car"])

    def test_all_present_drops_nothing(self):
        trainer = _load_trainer()
        present, missing = trainer.select_present_features(FEATURES, FEATURES + ["extra"])
        self.assertEqual(present, FEATURES)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
