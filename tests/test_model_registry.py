"""Tests for ephysatlas.model_registry and the public RegionClassifier wrapper.

All offline: a small synthetic model is trained, saved through ``save_model``, and packaged
with ``build_model_index``, so the whole save -> manifest -> load -> predict round trip is
exercised without touching S3 or the Hugging Face Hub.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import ephysatlas.model_registry as model_registry
import ephysatlas.regionclassifier as regionclassifier

# Real Cosmos region ids, so the acronym lookup resolves.
CLASSES = [315, 549, 997]  # Isocortex, TH, root
# A handful of real feature names, two of which carry a transform in the schema.
FEATURES = ["rms_ap", "rms_lf", "psd_delta", "psd_theta", "spike_count"]


def _make_model_dir(path_models: Path, n_folds: int = 2) -> Path:
    """Train and save a tiny synthetic model with folds, and build its manifest."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(90, len(FEATURES)))
    y = rng.integers(0, len(CLASSES), size=90)
    meta_template = dict(
        RANDOM_SEED=42,
        VINTAGE="2026_W32",
        REGION_MAP="Cosmos",
        FEATURES=FEATURES,
        CLASSES=CLASSES,
        ACCURACY=0.5,
        TRAINING=dict(training_size=10, testing_size=2),
    )

    def _fit():
        clf = XGBClassifier(n_estimators=2, max_depth=2)
        clf.fit(x, y)
        return clf

    path_model = regionclassifier.save_model(
        path_models, _fit(), dict(meta_template), identifier="test"
    )
    for i in range(n_folds):
        regionclassifier.save_model(
            path_models,
            _fit(),
            dict(meta_template),
            subfolder=f"FOLD{i:02d}",
            identifier="test",
        )
    # save_model writes folds beside the global model; move them under folds/ as the
    # training scripts do, so the fold-discovery path is what production sees.
    folds = path_model.joinpath("folds")
    folds.mkdir(exist_ok=True)
    for i in range(n_folds):
        shutil.move(str(path_model.joinpath(f"FOLD{i:02d}")), str(folds.joinpath(f"FOLD{i:02d}")))
    model_registry.build_model_index(path_model)
    return path_model


class TestClassAcronyms(unittest.TestCase):
    def test_all_classifier_classes_resolve(self):
        # id 2000 (void_fluid) exists only in ClassifierRegions; plain BrainRegions drops it.
        classes = [0, 315, 549, 997, 2000]
        acronyms = model_registry.class_acronyms(classes, "Cosmos")
        self.assertEqual(len(acronyms), len(classes))
        self.assertIn("void_fluid", acronyms)

    def test_unmappable_id_raises_rather_than_misaligning(self):
        with self.assertRaises(ValueError):
            model_registry.class_acronyms([315, 987654321], "Cosmos")


class TestResolveModel(unittest.TestCase):
    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            model_registry.resolve_model("whatever", source="not-a-source")

    def test_s3_without_one_raises(self):
        with self.assertRaises(ValueError):
            model_registry.resolve_model("some-model", source="s3", one=None)

    def test_hf_needs_a_repo_id_for_a_bare_model_name(self):
        # A slashless name is an S3 folder, not a hub repo: there is no implicit default.
        with self.assertRaises(ValueError):
            model_registry.resolve_model("2024_W50_Cosmos_something", source="hf")

    def test_hf_reads_repo_id_off_an_owner_name_model_id(self):
        source = model_registry.HFModelSource()
        self.assertEqual(source._resolve_repo_id("org/repo"), "org/repo")
        self.assertEqual(
            model_registry.HFModelSource(repo_id="a/b")._resolve_repo_id("ignored"), "a/b"
        )


class TestBuildModelIndexDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unknown_model_class_cannot_infer_task(self):
        meta_file = self.path_model.joinpath("meta.yaml")
        meta_file.write_text(
            meta_file.read_text().replace(
                "MODEL_CLASS: xgboost.sklearn.XGBClassifier", "MODEL_CLASS: some.other.Model"
            )
        )
        with self.assertRaises(ValueError):
            model_registry.build_model_index(self.path_model)

    def test_unregistered_task_raises(self):
        with self.assertRaises(ValueError):
            model_registry.build_model_index(self.path_model, task="spatial-encoding")


class TestLoadModelDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_xgboost_dispatch(self):
        classifier, info = regionclassifier.load_model(self.path_model)
        self.assertIsInstance(classifier, XGBClassifier)
        self.assertEqual(info["MODEL_CLASS"], "xgboost.sklearn.XGBClassifier")

    def test_unknown_model_class_in_manifest_raises(self):
        index_file = self.path_model.joinpath(model_registry.MODEL_INDEX_FILE)
        index = json.loads(index_file.read_text())
        index["model_class"] = "some.other.Model"
        index_file.write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            regionclassifier.load_model(self.path_model)

    def test_manifest_takes_precedence_over_meta_yaml(self):
        # meta.yaml is the training artifact; the manifest is the publication contract and
        # wins when both are present.
        meta_file = self.path_model.joinpath("meta.yaml")
        meta_file.write_text(
            meta_file.read_text().replace(
                "MODEL_CLASS: xgboost.sklearn.XGBClassifier", "MODEL_CLASS: some.other.Model"
            )
        )
        classifier, _ = regionclassifier.load_model(self.path_model)
        self.assertIsInstance(classifier, XGBClassifier)

    def test_legacy_model_without_manifest_falls_back_to_meta_yaml(self):
        self.path_model.joinpath(model_registry.MODEL_INDEX_FILE).unlink()
        classifier, info = regionclassifier.load_model(self.path_model)
        self.assertIsInstance(classifier, XGBClassifier)
        self.assertEqual(info["MODEL_CLASS"], "xgboost.sklearn.XGBClassifier")


class TestRegionClassifier(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)
        rng = np.random.default_rng(1)
        index = pd.MultiIndex.from_product(
            [["pid-a", "pid-b"], range(5)], names=["pid", "channel"]
        )
        self.df = pd.DataFrame(
            rng.normal(size=(len(index), len(FEATURES))), index=index, columns=FEATURES
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_manifest_core_and_task_block(self):
        index = json.loads(
            self.path_model.joinpath(model_registry.MODEL_INDEX_FILE).read_text()
        )
        # shared core: identifies the family and where the weights live
        self.assertEqual(index["task"], model_registry.TASK_REGION_CLASSIFICATION)
        self.assertEqual(index["model_class"], "xgboost.sklearn.XGBClassifier")
        self.assertEqual(index["artifacts"]["weights"], "model.ubj")
        self.assertEqual(index["artifacts"]["folds"], ["FOLD00", "FOLD01"])
        self.assertEqual(index["training"]["random_seed"], 42)
        self.assertIsNotNone(index["environment"]["xgboost"])
        # task-specific config
        config = index["config"]
        self.assertEqual(config["features"], FEATURES)
        self.assertEqual(config["classes"], CLASSES)
        self.assertEqual(len(config["class_acronyms"]), len(CLASSES))
        self.assertEqual(config["region_map"], "Cosmos")

    def test_predict_returns_acronyms_and_agreement(self):
        out = regionclassifier.RegionClassifier(self.path_model).predict(self.df)
        self.assertEqual(len(out), len(self.df))
        for column in ["acronym", "atlas_id", "probability", "fold_agreement"]:
            self.assertIn(column, out.columns)
        self.assertTrue(set(out["acronym"]).issubset({"Isocortex", "TH", "root"}))
        self.assertTrue(((out["probability"] >= 0) & (out["probability"] <= 1)).all())
        self.assertTrue(((out["fold_agreement"] > 0) & (out["fold_agreement"] <= 1)).all())
        # index is preserved so results can be joined straight back onto the input
        pd.testing.assert_index_equal(out.index, self.df.index)

    def test_missing_features_raise_and_name_them(self):
        clf = regionclassifier.RegionClassifier(self.path_model)
        with self.assertRaises(KeyError) as ctx:
            clf.predict(self.df.drop(columns=["rms_ap", "psd_theta"]))
        message = str(ctx.exception)
        self.assertIn("rms_ap", message)
        self.assertIn("psd_theta", message)

    def test_selftest_round_trip(self):
        clf = regionclassifier.RegionClassifier(self.path_model)
        example = self.path_model.joinpath("example")
        example.mkdir()
        self.df.to_parquet(example.joinpath("features_sample.parquet"))
        clf.predict(self.df).to_parquet(example.joinpath("expected_predictions.parquet"))
        self.assertTrue(clf.selftest())

    def test_selftest_without_example_raises(self):
        with self.assertRaises(FileNotFoundError):
            regionclassifier.RegionClassifier(self.path_model).selftest()


if __name__ == "__main__":
    unittest.main()
