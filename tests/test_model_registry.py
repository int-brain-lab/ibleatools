"""Tests for ephysatlas.model_registry and the public RegionClassifier wrapper.

All offline: a small synthetic model is trained, saved through ``save_model``, and packaged
with ``build_model_index``, so the whole save -> manifest -> load -> predict round trip is
exercised without touching S3 or the Hugging Face Hub.
"""

import json
import logging
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
    # Balanced and deterministic, so every held-out block still contains all classes and each
    # fold model emits the full class vector.
    y = np.tile(np.arange(len(CLASSES)), 90 // len(CLASSES))
    meta_template = dict(
        RANDOM_SEED=42,
        VINTAGE="2026_W32",
        REGION_MAP="Cosmos",
        FEATURES=FEATURES,
        CLASSES=CLASSES,
        ACCURACY=0.5,
        TRAINING=dict(training_size=10, testing_size=2),
    )

    def _fit(keep=None):
        """Fit on all rows, or on everything outside one held-out block (a real fold)."""
        mask = np.ones(len(y), bool) if keep is None else keep
        clf = XGBClassifier(n_estimators=2, max_depth=2)
        clf.fit(x[mask], y[mask])
        return clf

    # The global model sees every row; each fold holds one block out. That is what makes the
    # ensemble and the global model genuinely different estimators, as in production.
    path_model = regionclassifier.save_model(
        path_models, _fit(), dict(meta_template), identifier="test"
    )
    block = len(y) // n_folds
    for i in range(n_folds):
        keep = np.ones(len(y), bool)
        keep[i * block : (i + 1) * block] = False
        regionclassifier.save_model(
            path_models,
            _fit(keep),
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


class TestLoadPretrained(unittest.TestCase):
    """The single public entry point -- the only API a published model card should name."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_local_directory_needs_no_network_or_credentials(self):
        import ephysatlas

        model = ephysatlas.load_pretrained(self.path_model)
        self.assertIsInstance(model, regionclassifier.RegionClassifier)
        self.assertEqual(model.index["task"], model_registry.TASK_REGION_CLASSIFICATION)

    def test_exposed_at_package_top_level(self):
        # The card names `from ephysatlas import load_pretrained`; keep that import stable
        # even if the modules underneath are reorganised.
        import ephysatlas

        self.assertTrue(callable(ephysatlas.load_pretrained))

    def test_unknown_task_raises_actionably(self):
        index_file = self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE)
        index = json.loads(index_file.read_text())
        index["task"] = "some-future-task"
        index_file.write_text(json.dumps(index))
        import ephysatlas

        with self.assertRaises(ValueError) as ctx:
            ephysatlas.load_pretrained(self.path_model)
        self.assertIn("some-future-task", str(ctx.exception))

    def test_unpinned_hub_id_warns_about_moving_main(self):
        import ephysatlas

        # Resolution will fail (no network / no such repo); we only care that the warning
        # about an unpinned revision is emitted before the attempt.
        with self.assertLogs("ephysatlas.models", level="WARNING") as logs:
            with self.assertRaises(ValueError):
                ephysatlas.load_pretrained("org/does-not-exist", source="hf")
        self.assertIn("no revision pinned", "\n".join(logs.output))

    def test_local_directory_does_not_warn_about_revision(self):
        import ephysatlas

        logger = logging.getLogger("ephysatlas.models")
        with self.assertLogs(logger, level="INFO") as logs:
            ephysatlas.load_pretrained(self.path_model)
        self.assertNotIn("no revision pinned", "\n".join(logs.output))

    def test_manifestless_model_defaults_to_region_classification(self):
        self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).unlink()
        import ephysatlas

        model = ephysatlas.load_pretrained(self.path_model)
        self.assertIsInstance(model, regionclassifier.RegionClassifier)


class TestUpload(unittest.TestCase):
    """The upload sequence, with HfApi stubbed -- nothing here touches the network."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = self.tmp.joinpath("2026_W32_Cosmos_test")
        self.path_model.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, revision, tag_raises=False):
        calls = []

        class FakeApi:
            def __init__(self, token=None):
                calls.append(("HfApi", token))

            def create_repo(self, **kw):
                calls.append(("create_repo", kw["repo_id"], kw["private"], kw["exist_ok"]))

            def upload_folder(self, **kw):
                # uploading must target main (no revision=), so main is never left empty
                calls.append(
                    ("upload_folder", kw["repo_id"], kw.get("revision"), kw.get("ignore_patterns"))
                )
                return "commit"

            def create_tag(self, **kw):
                calls.append(("create_tag", kw["tag"]))
                if tag_raises:
                    raise RuntimeError("tag exists")

        import huggingface_hub

        original = huggingface_hub.HfApi
        huggingface_hub.HfApi = FakeApi
        try:
            source = model_registry.HFModelSource(repo_id="org/repo", token="t")
            source.upload(self.path_model, revision=revision)
        finally:
            huggingface_hub.HfApi = original
        return calls

    def test_creates_repo_then_uploads_to_main_then_tags(self):
        calls = self._run("2026_W32")
        names = [c[0] for c in calls]
        # create_repo must come first: the original bug was tagging a repo that did not exist
        self.assertEqual(names, ["HfApi", "create_repo", "upload_folder", "create_tag"])
        # private by default, so nothing goes world-readable as a side effect of a script
        self.assertEqual(calls[1][1:], ("org/repo", True, True))
        # revision must be None on upload_folder, i.e. the push lands on main
        self.assertIsNone(calls[2][2])
        self.assertEqual(calls[3][1], "2026_W32")

    def test_predictions_are_never_uploaded(self):
        ignore = self._run("2026_W32")[2][3]
        self.assertIn("predictions.pqt", ignore)

    def test_no_revision_skips_tagging(self):
        self.assertEqual([c[0] for c in self._run(None)][-1], "upload_folder")

    def test_existing_tag_warns_rather_than_moving_it(self):
        # Tags are immutable citations; a failure to create one must not abort the publish.
        calls = self._run("2026_W32", tag_raises=True)
        self.assertIn("create_tag", [c[0] for c in calls])

    def test_upload_without_repo_id_raises(self):
        with self.assertRaises(ValueError):
            model_registry.HFModelSource().upload(self.path_model)


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
        index_file = self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE)
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
        self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).unlink()
        classifier, info = regionclassifier.load_model(self.path_model)
        self.assertIsInstance(classifier, XGBClassifier)
        self.assertEqual(info["MODEL_CLASS"], "xgboost.sklearn.XGBClassifier")

    def test_manifest_alone_is_sufficient(self):
        # Families that never go through save_model (e.g. the torch spatial encoder) ship no
        # meta.yaml at all, so the manifest must be enough on its own.
        self.path_model.joinpath("meta.yaml").unlink()
        classifier, info = regionclassifier.load_model(self.path_model)
        self.assertIsInstance(classifier, XGBClassifier)
        # model_info is still meta-shaped, so infer_regions-style callers keep working
        self.assertEqual(info["FEATURES"], FEATURES)
        self.assertEqual(info["CLASSES"], CLASSES)
        self.assertEqual(info["MODEL_CLASS"], "xgboost.sklearn.XGBClassifier")

    def test_neither_manifest_nor_meta_raises(self):
        self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).unlink()
        self.path_model.joinpath("meta.yaml").unlink()
        with self.assertRaises(FileNotFoundError):
            regionclassifier.load_model(self.path_model)

    def test_weights_filename_comes_from_the_manifest(self):
        # The loader must honour artifacts.weights rather than hardcoding model.ubj.
        path = self.path_model
        path.joinpath("model.ubj").rename(path.joinpath("weights.ubj"))
        index_file = path.joinpath(model_registry.MODEL_MANIFEST_FILE)
        index = json.loads(index_file.read_text())
        index["artifacts"]["weights"] = "weights.ubj"
        index_file.write_text(json.dumps(index))
        classifier, _ = regionclassifier.load_model(path)
        self.assertIsInstance(classifier, XGBClassifier)


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
            self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).read_text()
        )
        # shared core: identifies the family and where the weights live
        self.assertEqual(index["task"], model_registry.TASK_REGION_CLASSIFICATION)
        self.assertEqual(index["model_class"], "xgboost.sklearn.XGBClassifier")
        self.assertEqual(index["artifacts"]["weights"], "model.ubj")
        self.assertEqual(index["artifacts"]["folds"], ["FOLD00", "FOLD01"])
        self.assertEqual(index["training"]["random_seed"], 42)
        self.assertIsNotNone(index["environment"]["xgboost"])
        # granularity, and the inputs/outputs blocks
        self.assertEqual(index["granularity"], "channel")
        self.assertEqual(index["inputs"]["index"], ["pid", "channel"])
        self.assertEqual(index["inputs"]["features"], FEATURES)
        self.assertEqual(index["outputs"]["kind"], "categorical")
        self.assertIn("predicted_acronym", index["outputs"]["columns"])
        # task-specific config
        config = index["config"]
        self.assertEqual(config["classes"], CLASSES)
        self.assertEqual(len(config["class_acronyms"]), len(CLASSES))
        self.assertEqual(config["region_map"], "Cosmos")

    def test_method_and_compatibility_recorded_only_when_given(self):
        # Omitted rather than guessed: an unverified probe-compatibility claim in a public
        # manifest is the silent failure the field exists to prevent.
        index = model_registry.build_model_index(self.path_model)
        self.assertNotIn("method", index)
        self.assertNotIn("compatibility", index)
        index = model_registry.build_model_index(
            self.path_model, method="xgboost", compatibility={"probe": ["NP1"]}
        )
        self.assertEqual(index["method"], "xgboost")
        self.assertEqual(index["compatibility"]["probe"], ["NP1"])

    def test_method_is_independent_of_model_class(self):
        # The point of keeping both: model_class is the implementation used for dispatch,
        # method is the stable semantic label. Two methods can share one model_class.
        index = model_registry.build_model_index(self.path_model, method="xgboost")
        self.assertEqual(index["model_class"], "xgboost.sklearn.XGBClassifier")
        self.assertEqual(index["method"], "xgboost")
        self.assertNotEqual(index["method"], index["model_class"])

    def test_predict_returns_acronyms_and_agreement(self):
        out = regionclassifier.RegionClassifier(self.path_model).predict(self.df)
        self.assertEqual(len(out), len(self.df))
        for column in [
            "predicted_acronym",
            "predicted_atlas_id",
            "prediction_probability",
            "fold_agreement",
        ]:
            self.assertIn(column, out.columns)
        self.assertTrue(
            set(out["predicted_acronym"]).issubset({"Isocortex", "TH", "root"})
        )
        p = out["prediction_probability"]
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        # 0 is reachable, and meaningful: the fold-averaged argmax can be a class that no
        # single fold ranked first (e.g. folds split between A and B while all rank C second).
        # Those rows are exactly the ones a user should distrust.
        agreement = out["fold_agreement"]
        self.assertTrue(((agreement >= 0) & (agreement <= 1)).all())
        # index is preserved so results can be joined straight back onto the input
        pd.testing.assert_index_equal(out.index, self.df.index)

    def test_global_estimator_uses_the_single_model(self):
        clf = regionclassifier.RegionClassifier(self.path_model)
        ens = clf.predict(self.df, estimator="ensemble")
        glob = clf.predict(self.df, estimator="global")
        # same schema and index in both modes, so downstream code does not branch
        self.assertEqual(list(ens.columns), list(glob.columns))
        pd.testing.assert_index_equal(ens.index, glob.index)
        # a single model has no folds to agree with -- NaN, not a fabricated 1.0
        self.assertTrue(glob["fold_agreement"].isna().all())
        self.assertTrue(ens["fold_agreement"].notna().all())

    def test_global_and_ensemble_are_actually_different_estimators(self):
        clf = regionclassifier.RegionClassifier(self.path_model)
        ens = clf.predict(self.df, estimator="ensemble")
        glob = clf.predict(self.df, estimator="global")
        self.assertFalse(
            np.allclose(
                ens["prediction_probability"].values, glob["prediction_probability"].values
            )
        )

    def test_unknown_estimator_raises(self):
        clf = regionclassifier.RegionClassifier(self.path_model)
        with self.assertRaises(ValueError):
            clf.predict(self.df, estimator="best")

    def test_global_estimator_without_root_weights_raises(self):
        self.path_model.joinpath("model.ubj").unlink()
        clf = regionclassifier.RegionClassifier(self.path_model)
        with self.assertRaises(ValueError) as ctx:
            clf.predict(self.df, estimator="global")
        self.assertIn("ensemble", str(ctx.exception))

    def test_prediction_columns_do_not_collide_with_ground_truth(self):
        # Both the channel feature table and the cluster table already carry histology
        # `acronym`/`atlas_id`. Predictions must be joinable onto them without a suffix.
        df = self.df.copy()
        df["acronym"] = "Isocortex"
        df["atlas_id"] = 315
        out = regionclassifier.RegionClassifier(self.path_model).predict(df)
        self.assertEqual(set(out.columns) & set(df.columns), set())
        joined = df.join(out)  # would raise if any column overlapped
        self.assertIn("predicted_acronym", joined.columns)
        self.assertIn("acronym", joined.columns)

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
