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
import yaml
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

    def test_every_download_route_is_verified_at_the_chokepoint(self):
        # resolve_model is the one function load_pretrained, RegionClassifier.from_pretrained
        # and download_model all pass through. Verifying here rather than in each caller is what
        # stops one documented entry point offering a guarantee the others do not.
        tmp = Path(tempfile.mkdtemp())
        try:
            path_model = _make_model_dir(tmp)
            model_registry.write_checksums(path_model)
            target = path_model.joinpath("model.ubj")
            target.write_bytes(target.read_bytes() + b"tampered")

            class FakeBackend:
                def fetch(self, model_id, revision=None, cache_dir=None):
                    return path_model

            original = model_registry.S3ModelSource
            model_registry.S3ModelSource = lambda **kw: FakeBackend()
            try:
                with self.assertRaises(ValueError) as ctx:
                    model_registry.resolve_model("whatever", source="s3", cache_dir=tmp)
            finally:
                model_registry.S3ModelSource = original
            self.assertIn("model.ubj", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


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

    def test_load_pretrained_verifies_checksums_when_the_model_ships_them(self):
        import ephysatlas

        model_registry.write_checksums(self.path_model)
        target = self.path_model.joinpath("model.ubj")
        target.write_bytes(target.read_bytes() + b"corrupted")
        with self.assertRaises(ValueError) as ctx:
            ephysatlas.load_pretrained(self.path_model)
        self.assertIn("model.ubj", str(ctx.exception))

    def test_load_pretrained_is_quiet_when_the_model_ships_no_checksums(self):
        # Every model published before this existed has no checksums.json; loading one must
        # neither fail nor nag.
        import ephysatlas

        with self.assertNoLogs("ephysatlas.model_registry", level="WARNING"):
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
        # "spatial-encoding" is a registered task now, so this needs a genuinely unknown one.
        with self.assertRaises(ValueError) as ctx:
            model_registry.build_model_index(self.path_model, task="cell-mixture")
        self.assertIn("cell-mixture", str(ctx.exception))

    def test_the_registered_tasks_are_the_ones_with_builders(self):
        self.assertEqual(
            sorted(model_registry.TASK_BUILDERS),
            [
                model_registry.TASK_REGION_CLASSIFICATION,
                model_registry.TASK_SPATIAL_ENCODING,
                model_registry.TASK_UNIT_ENCODING,
            ],
        )

    def test_split_is_recorded_as_an_artifact_only_when_present(self):
        # A model packaged without a published split must not claim one -- validate_artifacts
        # would then look for a file that was never written.
        index = model_registry.build_model_index(self.path_model)
        self.assertNotIn("split", index["artifacts"])

        import uuid

        pids = [str(uuid.uuid4()) for _ in range(4)]
        model_registry.write_split(
            self.path_model, pids, np.floor(np.arange(4) / 4 * 2), n_folds=2
        )
        index = model_registry.build_model_index(self.path_model)
        self.assertEqual(index["artifacts"]["split"], model_registry.MODEL_SPLIT_FILE)
        self.assertTrue(model_registry.validate_artifacts(self.path_model, index))


class TestWriteManifest(unittest.TestCase):
    """``write_manifest`` is the pure assembler: it builds the manifest from an in-memory meta
    dict, reading no ``meta.yaml`` off disk. ``build_model_index`` is now the thin
    file-reading wrapper around it, so the two must agree on the same inputs.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_manifest_matches_build_model_index(self):
        # build_model_index is just write_manifest fed from meta.yaml, so on the same inputs
        # they must produce the identical manifest.
        meta = yaml.safe_load(self.path_model.joinpath("meta.yaml").read_text())
        from_index = model_registry.build_model_index(self.path_model, method="xgboost")
        from_write = model_registry.write_manifest(self.path_model, meta, method="xgboost")
        self.assertEqual(from_write, from_index)

    def test_write_manifest_reads_no_meta_yaml(self):
        # The whole point: training hands write_manifest the values directly, with no
        # meta.yaml on disk at all.
        meta = yaml.safe_load(self.path_model.joinpath("meta.yaml").read_text())
        self.path_model.joinpath("meta.yaml").unlink()
        self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).unlink()
        index = model_registry.write_manifest(self.path_model, meta)
        self.assertEqual(index["task"], model_registry.TASK_REGION_CLASSIFICATION)
        self.assertEqual(index["model_class"], "xgboost.sklearn.XGBClassifier")
        # and it actually wrote the manifest file, matching the returned dict
        written = json.loads(
            self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).read_text()
        )
        self.assertEqual(written, index)


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


class TestMetaFreeFolds(unittest.TestCase):
    """A3 + D: folds ship weights only -- no per-fold ``meta.yaml``. The ensemble must still
    discover and load every fold from the manifest and report a real ``fold_agreement``,
    rather than silently dropping the meta-less folds and degrading to the single global model
    (the §8 failure mode the plan calls out).
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)
        rng = np.random.default_rng(2)
        index = pd.MultiIndex.from_product(
            [["pid-a", "pid-b"], range(5)], names=["pid", "channel"]
        )
        self.df = pd.DataFrame(
            rng.normal(size=(len(index), len(FEATURES))), index=index, columns=FEATURES
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_folds_carry_no_meta_yaml(self):
        # save_model writes meta.yaml only for the root model; folds hold model.ubj alone.
        for name in ("FOLD00", "FOLD01"):
            fold_dir = self.path_model.joinpath("folds", name)
            self.assertTrue(fold_dir.joinpath("model.ubj").exists())
            self.assertFalse(fold_dir.joinpath("meta.yaml").exists())

    def test_ensemble_loads_meta_free_folds(self):
        clf = regionclassifier.RegionClassifier(self.path_model)
        # both folds discovered by their weights, none dropped
        self.assertEqual(len(clf._fold_dirs()), 2)
        out = clf.predict(self.df, estimator="ensemble")
        # a real agreement over the folds actually consulted, not the all-NaN a
        # global-only fallback would produce
        self.assertFalse(np.isnan(out["fold_agreement"]).all())

    def test_validate_artifacts_accepts_meta_free_folds(self):
        # publish's completeness check must key on the fold weights, not a meta.yaml.
        index = model_registry.read_manifest(self.path_model)
        self.assertTrue(model_registry.validate_artifacts(self.path_model, index))


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

    # --- feature order -----------------------------------------------------------------
    # The estimator consumes the feature matrix positionally, so a reordered manifest list is
    # a silent wrong-answer bug rather than a load error. The digest makes it loud.

    def test_manifest_records_the_feature_order_digest(self):
        index = json.loads(
            self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).read_text()
        )
        self.assertEqual(
            index["inputs"]["feature_order_sha256"],
            model_registry.feature_order_sha256(FEATURES),
        )
        # The list itself must stay put; the digest is an addition, not a replacement.
        self.assertEqual(index["inputs"]["features"], FEATURES)

    def test_predict_passes_on_an_untouched_manifest(self):
        out = regionclassifier.RegionClassifier(self.path_model).predict(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_reordered_manifest_features_raise(self):
        index_file = self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE)
        index = json.loads(index_file.read_text())
        features = index["inputs"]["features"]
        features[0], features[1] = features[1], features[0]
        index_file.write_text(json.dumps(index))
        with self.assertRaises(ValueError) as ctx:
            regionclassifier.RegionClassifier(self.path_model).predict(self.df)
        message = str(ctx.exception).lower()
        self.assertIn("feature", message)
        self.assertIn("order", message)

    def test_a_missing_feature_still_raises_key_error_first(self):
        # Ordering is load-bearing: the missing-column check must run before the digest check,
        # or a caller who simply forgot a column gets a confusing integrity error.
        #
        # Both faults have to be present at once for the ordering to be observable at all --
        # with a valid manifest the digest check cannot raise, so its position says nothing.
        index_file = self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE)
        index = json.loads(index_file.read_text())
        index["inputs"]["feature_order_sha256"] = "0" * 64
        index_file.write_text(json.dumps(index))
        clf = regionclassifier.RegionClassifier(self.path_model)
        with self.assertRaises(KeyError):
            clf.predict(self.df.drop(columns=["rms_ap"]))

    def test_a_legacy_manifest_without_the_digest_is_skipped(self):
        index_file = self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE)
        index = json.loads(index_file.read_text())
        index["inputs"].pop("feature_order_sha256")
        index_file.write_text(json.dumps(index))
        out = regionclassifier.RegionClassifier(self.path_model).predict(self.df)
        self.assertEqual(len(out), len(self.df))

    def test_a_reordered_input_dataframe_is_still_fine(self):
        # This is a check on the manifest, not on the caller's frame: predict selects columns
        # by name, so the caller's column order has never mattered and must keep not mattering.
        clf = regionclassifier.RegionClassifier(self.path_model)
        expected = clf.predict(self.df)
        reordered = clf.predict(self.df[FEATURES[::-1]])
        np.testing.assert_array_equal(
            expected["predicted_acronym"].values, reordered["predicted_acronym"].values
        )

    def test_validate_feature_order_can_also_compare_against_caller_features(self):
        # Phase 3's encoder has a module-level FEATURE_LIST, so code-vs-release drift becomes
        # meaningful there. The classifier takes its list from the manifest, so this argument
        # is optional.
        digest = model_registry.feature_order_sha256(FEATURES)
        self.assertTrue(model_registry.validate_feature_order(FEATURES, digest, FEATURES))
        with self.assertRaises(ValueError):
            model_registry.validate_feature_order(FEATURES, digest, FEATURES[::-1])

    def test_the_digest_can_sit_on_any_block(self):
        # The encoder's positional list is outputs.columns, not inputs.features, so the
        # validator must not assume which block it came from.
        columns = ["rms_lf", "psd_lfp", "psd_alpha"]
        digest = model_registry.feature_order_sha256(columns)
        self.assertTrue(model_registry.validate_feature_order(columns, digest))
        with self.assertRaises(ValueError):
            model_registry.validate_feature_order(columns[::-1], digest)

    def test_no_digest_recorded_is_not_an_error(self):
        self.assertTrue(model_registry.validate_feature_order(FEATURES, None))


class TestWrapperDispatch(unittest.TestCase):
    """Wrapper dispatch is model_class-first, task-fallback.

    Task alone is not enough: the design keeps region decoding as one task with ``method``
    separating xgboost from a future transformer, so both would carry the same ``task`` while
    needing different wrappers -- a torch module has no ``predict_proba``.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)
        self.index_file = self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        import ephysatlas.models as models

        models.MODEL_WRAPPERS.clear()

    def test_model_class_beats_task(self):
        import ephysatlas
        import ephysatlas.models as models

        sentinel = object()
        models.MODEL_WRAPPERS["xgboost.sklearn.XGBClassifier"] = (
            lambda path_model, index, **kw: sentinel
        )
        self.assertIs(ephysatlas.load_pretrained(self.path_model), sentinel)

    def test_an_unregistered_model_class_falls_back_to_task(self):
        import ephysatlas
        import ephysatlas.models as models

        models.MODEL_WRAPPERS["something.else.Entirely"] = lambda *a, **k: object()
        model = ephysatlas.load_pretrained(self.path_model)
        self.assertIsInstance(model, regionclassifier.RegionClassifier)

    def test_neither_registered_raises_naming_both(self):
        index = json.loads(self.index_file.read_text())
        index["task"] = "some-future-task"
        index["model_class"] = "some.future.Model"
        self.index_file.write_text(json.dumps(index))
        import ephysatlas

        with self.assertRaises(ValueError) as ctx:
            ephysatlas.load_pretrained(self.path_model)
        message = str(ctx.exception)
        self.assertIn("some-future-task", message)
        self.assertIn("some.future.Model", message)

    def test_kwargs_reach_the_wrapper(self):
        # A torch family needs device=; the classifier ignores what it does not use.
        import ephysatlas
        import ephysatlas.models as models

        seen = {}

        def _spy(path_model, index, **kwargs):
            seen.update(kwargs)
            return object()

        models.MODEL_WRAPPERS["xgboost.sklearn.XGBClassifier"] = _spy
        ephysatlas.load_pretrained(self.path_model, device="cpu")
        self.assertEqual(seen, {"device": "cpu"})

    def test_the_loader_receives_the_whole_manifest(self):
        # The contract is f(path_model, manifest), not f(path_model, weights=str): a family whose
        # model is several files could not be expressed by the narrower signature.
        seen = {}

        def _spy(path_model, manifest=None):
            seen["manifest"] = manifest
            return "loaded"

        original = dict(regionclassifier.MODEL_LOADERS)
        regionclassifier.MODEL_LOADERS["xgboost.sklearn.XGBClassifier"] = _spy
        try:
            classifier, _ = regionclassifier.load_model(self.path_model)
        finally:
            regionclassifier.MODEL_LOADERS.clear()
            regionclassifier.MODEL_LOADERS.update(original)
        self.assertEqual(classifier, "loaded")
        self.assertIn("artifacts", seen["manifest"])

    def test_artifacts_now_come_from_the_task_builder(self):
        # Moved out of the shared core so a second family can declare different roles.
        index = model_registry.build_model_index(self.path_model)
        self.assertEqual(index["artifacts"]["weights"], "model.ubj")
        self.assertEqual(index["artifacts"]["folds"], ["FOLD00", "FOLD01"])
        blocks = model_registry._blocks_region_classification(
            {"CLASSES": CLASSES, "FEATURES": FEATURES, "REGION_MAP": "Cosmos"},
            self.path_model,
        )
        self.assertIn("artifacts", blocks)


class TestChecksums(unittest.TestCase):
    """``checksums.json`` is the only family-agnostic integrity check a model can carry.

    It answers a question the manifest cannot: are the bytes I just downloaded the bytes that
    were published? A file silently dropped by ``DEFAULT_UPLOAD_IGNORE``, or a truncated
    transfer, otherwise surfaces much later as a confusing load error.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _entries(self):
        payload = json.loads(
            self.path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE).read_text()
        )
        return payload, [e["path"] for e in payload["files"]]

    def test_write_covers_every_file_except_the_checksum_file_itself(self):
        model_registry.write_checksums(self.path_model)
        payload, paths = self._entries()
        self.assertEqual(payload["algo"], "sha256")
        for expected in [
            "model.ubj",
            "meta.yaml",
            model_registry.MODEL_MANIFEST_FILE,
            "folds/FOLD00/model.ubj",
            "folds/FOLD01/model.ubj",
        ]:
            self.assertIn(expected, paths)
        # A checksum file cannot hash itself, and the manifest *must* be covered -- it is the
        # publication contract, so tampering with it has to be detectable.
        self.assertNotIn(model_registry.MODEL_CHECKSUM_FILE, paths)
        for entry in payload["files"]:
            self.assertEqual(set(entry), {"path", "sha256", "bytes"})

    def test_recorded_digest_and_size_are_real(self):
        import hashlib

        model_registry.write_checksums(self.path_model)
        payload, _ = self._entries()
        entry = next(e for e in payload["files"] if e["path"] == "meta.yaml")
        target = self.path_model.joinpath("meta.yaml")
        self.assertEqual(entry["bytes"], target.stat().st_size)
        self.assertEqual(entry["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())

    def test_paths_are_relative_posix_and_sorted(self):
        model_registry.write_checksums(self.path_model)
        _, paths = self._entries()
        self.assertEqual(paths, sorted(paths))
        for path in paths:
            self.assertFalse(path.startswith("/"))
            self.assertNotIn("\\", path)

    def test_write_is_deterministic(self):
        first = model_registry.write_checksums(self.path_model).read_bytes()
        second = model_registry.write_checksums(self.path_model).read_bytes()
        self.assertEqual(first, second)

    def test_verify_passes_on_an_untouched_directory(self):
        model_registry.write_checksums(self.path_model)
        self.assertTrue(model_registry.verify_checksums(self.path_model))

    def test_verify_detects_a_changed_byte(self):
        model_registry.write_checksums(self.path_model)
        target = self.path_model.joinpath("model.ubj")
        target.write_bytes(target.read_bytes() + b"x")
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        self.assertIn("model.ubj", str(ctx.exception))

    def test_verify_detects_a_flipped_byte_that_does_not_change_the_size(self):
        # The size comparison runs first and short-circuits, so every append-based test above
        # would still pass with the sha256 comparison deleted. This is the one that needs it:
        # same length, one bit different -- exactly what silent corruption looks like.
        model_registry.write_checksums(self.path_model)
        target = self.path_model.joinpath("model.ubj")
        before = target.stat().st_size
        raw = bytearray(target.read_bytes())
        raw[5] ^= 0xFF
        target.write_bytes(bytes(raw))
        self.assertEqual(target.stat().st_size, before)
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        self.assertIn("sha256", str(ctx.exception))

    def test_verify_refuses_a_path_that_escapes_the_model_directory(self):
        # checksums.json travels with the download and no digest covers it, so a published
        # repository could point verification at an arbitrary local file and have the path
        # echoed back in the error. Reject lexically, before touching the filesystem.
        model_registry.write_checksums(self.path_model)
        checksum_file = self.path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE)
        payload = json.loads(checksum_file.read_text())
        payload["files"].append(
            {"path": "../../../../../../etc/hosts", "sha256": "0" * 64, "bytes": 1}
        )
        checksum_file.write_text(json.dumps(payload))
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        message = str(ctx.exception)
        self.assertIn("escapes the model directory", message)
        # The traversal must be refused rather than reported as a mismatch, which would mean
        # the file was stat'd and hashed first.
        self.assertNotIn("sha256 mismatch", message)

    def test_verify_refuses_an_absolute_path(self):
        model_registry.write_checksums(self.path_model)
        checksum_file = self.path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE)
        payload = json.loads(checksum_file.read_text())
        payload["files"].append({"path": "/etc/hosts", "sha256": "0" * 64, "bytes": 1})
        checksum_file.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            model_registry.verify_checksums(self.path_model)

    def test_a_malformed_checksum_file_says_so_plainly(self):
        # Otherwise a JSONDecodeError or KeyError escapes and reads as though the *model* were
        # corrupt, sending the reader after the wrong file.
        checksum_file = self.path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE)
        checksum_file.write_text("{ truncated")
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        self.assertIn("malformed", str(ctx.exception))

        checksum_file.write_text(json.dumps({"files": [{"path": "model.ubj"}]}))
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        self.assertIn("malformed", str(ctx.exception))

    def test_the_model_card_is_left_out_of_the_digest_set(self):
        # Covering README.md would mean a maintainer editing the card through the Hub breaks
        # load_pretrained for every user until checksums are regenerated.
        self.path_model.joinpath("README.md").write_text("# card\n")
        self.path_model.joinpath("LICENSE").write_text("CC-BY-4.0\n")
        model_registry.write_checksums(self.path_model)
        _, paths = self._entries()
        self.assertNotIn("README.md", paths)
        self.assertNotIn("LICENSE", paths)
        # ... and editing the card afterwards must not break verification.
        self.path_model.joinpath("README.md").write_text("# card\n\nnow with a tag\n")
        self.assertTrue(model_registry.verify_checksums(self.path_model))

    def test_nested_ignored_files_are_not_hashed(self):
        # The bare-filename fallback in _is_ignored exists for exactly this: a .DS_Store inside
        # a fold directory, which is routine on the macOS machines these models are packaged on.
        self.path_model.joinpath("folds", "FOLD00", ".DS_Store").write_bytes(b"junk")
        self.path_model.joinpath("folds", "FOLD01", "scratch.tmp").write_text("x")
        model_registry.write_checksums(self.path_model)
        _, paths = self._entries()
        self.assertNotIn("folds/FOLD00/.DS_Store", paths)
        self.assertNotIn("folds/FOLD01/scratch.tmp", paths)

    def test_hub_added_files_present_at_write_time_are_not_hashed(self):
        # Re-hashing a downloaded snapshot must not record entries no download can satisfy.
        self.path_model.joinpath(".gitattributes").write_text("* text=auto\n")
        cache = self.path_model.joinpath(".cache", "huggingface")
        cache.mkdir(parents=True)
        cache.joinpath("download-metadata").write_text("{}")
        model_registry.write_checksums(self.path_model)
        _, paths = self._entries()
        self.assertNotIn(".gitattributes", paths)
        self.assertEqual([p for p in paths if p.startswith(".cache/")], [])

    def test_verify_detects_a_missing_file_and_names_it(self):
        model_registry.write_checksums(self.path_model)
        self.path_model.joinpath("folds", "FOLD01", "model.ubj").unlink()
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        self.assertIn("folds/FOLD01/model.ubj", str(ctx.exception))

    def test_verify_reports_every_failure_at_once(self):
        model_registry.write_checksums(self.path_model)
        self.path_model.joinpath("model.ubj").unlink()
        self.path_model.joinpath("meta.yaml").unlink()
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        message = str(ctx.exception)
        # One traversal, one error: fixing them one round trip at a time is miserable.
        self.assertIn("model.ubj", message)
        self.assertIn("meta.yaml", message)

    def test_verify_tolerates_files_added_after_writing(self):
        # A Hub snapshot carries .gitattributes and a .cache/ tree, and selftest writes
        # example/. Verification must check the listed files, not reject a superset.
        model_registry.write_checksums(self.path_model)
        self.path_model.joinpath(".gitattributes").write_text("* text=auto\n")
        example = self.path_model.joinpath("example")
        example.mkdir()
        example.joinpath("features_sample.parquet").write_bytes(b"not really a parquet")
        self.assertTrue(model_registry.verify_checksums(self.path_model))

    def test_verify_is_silent_when_absent_but_strict_on_demand(self):
        # Legacy S3 models carry no checksums; loading them must not start failing.
        self.assertIsNone(model_registry.verify_checksums(self.path_model))
        with self.assertRaises(FileNotFoundError):
            model_registry.verify_checksums(self.path_model, missing_ok=False)

    def test_files_excluded_from_upload_are_not_hashed(self):
        # Hashing a file the upload drops would make every downloaded model fail verification.
        for name in ("predictions.pqt", ".DS_Store", "scratch.tmp"):
            self.path_model.joinpath(name).write_text("x")
        model_registry.write_checksums(self.path_model)
        _, paths = self._entries()
        for name in ("predictions.pqt", ".DS_Store", "scratch.tmp"):
            self.assertNotIn(name, paths)

    def test_the_golden_example_is_covered(self):
        # The example is load-bearing -- selftest compares against it, so silent drift in it
        # would defeat the very check that detects drift elsewhere. It is written before
        # checksums by the publish script, so it must land inside the digest set.
        # (The model card is deliberately *not* covered; see
        # test_the_model_card_is_left_out_of_the_digest_set.)
        example = self.path_model.joinpath("example")
        example.mkdir()
        example.joinpath("features_sample.parquet").write_bytes(b"sample")
        example.joinpath("expected_predictions.parquet").write_bytes(b"golden")
        model_registry.write_checksums(self.path_model)
        _, paths = self._entries()
        self.assertIn("example/features_sample.parquet", paths)
        self.assertIn("example/expected_predictions.parquet", paths)


class TestValidateArtifacts(unittest.TestCase):
    """Every path the manifest's ``artifacts`` block names must actually be on disk."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)
        self.index_file = self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rewrite(self, **artifacts):
        index = json.loads(self.index_file.read_text())
        index["artifacts"].update(artifacts)
        self.index_file.write_text(json.dumps(index))
        return index

    def test_a_freshly_packaged_model_validates(self):
        self.assertTrue(model_registry.validate_artifacts(self.path_model))

    def test_fold_names_resolve_under_the_folds_directory(self):
        # `artifacts.folds` holds bare names like "FOLD00"; the real path is folds/FOLD00.
        # A validator that joined path_model/"FOLD00" would call every good model broken.
        self.assertFalse(self.path_model.joinpath("FOLD00").exists())
        self.assertTrue(self.path_model.joinpath("folds", "FOLD00").is_dir())
        self.assertTrue(model_registry.validate_artifacts(self.path_model))

    def test_missing_weights_names_the_role_and_the_path(self):
        self.path_model.joinpath("model.ubj").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            model_registry.validate_artifacts(self.path_model)
        message = str(ctx.exception)
        self.assertIn("weights", message)
        self.assertIn("model.ubj", message)

    def test_missing_fold_names_the_role(self):
        shutil.rmtree(self.path_model.joinpath("folds", "FOLD01"))
        with self.assertRaises(FileNotFoundError) as ctx:
            model_registry.validate_artifacts(self.path_model)
        message = str(ctx.exception)
        self.assertIn("folds", message)
        self.assertIn("FOLD01", message)

    def test_a_fold_the_loader_would_drop_does_not_pass(self):
        # Existence is not enough: _fold_dirs keeps a fold only if it holds its weights, so a
        # directory without model.ubj passes a bare existence check and is then silently
        # dropped at load -- the ensemble quietly averaging fewer models than the manifest
        # advertises. That is precisely the "looks complete, fails for a stranger" case this
        # gate exists to catch.
        self.path_model.joinpath("folds", "FOLD01", "model.ubj").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            model_registry.validate_artifacts(self.path_model)
        self.assertIn("FOLD01", str(ctx.exception))

    def test_a_fold_replaced_by_a_plain_file_does_not_pass(self):
        shutil.rmtree(self.path_model.joinpath("folds", "FOLD01"))
        self.path_model.joinpath("folds", "FOLD01").write_text("not a directory")
        with self.assertRaises(FileNotFoundError):
            model_registry.validate_artifacts(self.path_model)

    def test_the_loader_warns_when_only_some_folds_survive(self):
        # Complements the gate above for models that were already published damaged: losing
        # every fold already warned, losing some did not, and that is the dangerous direction.
        self.path_model.joinpath("folds", "FOLD01", "model.ubj").unlink()
        clf = regionclassifier.RegionClassifier(self.path_model)
        with self.assertLogs("ephysatlas.regionclassifier", level="WARNING") as logs:
            clf._fold_dirs()
        joined = "\n".join(logs.output)
        self.assertIn("only 1 are loadable", joined)

    def test_a_directory_artifact_is_accepted(self):
        # The spatial encoder publishes a context directory, not a single file.
        self._rewrite(context="atlas_pca/")
        self.path_model.joinpath("atlas_pca").mkdir()
        self.assertTrue(model_registry.validate_artifacts(self.path_model))
        self.path_model.joinpath("atlas_pca").rmdir()
        with self.assertRaises(FileNotFoundError) as ctx:
            model_registry.validate_artifacts(self.path_model)
        self.assertIn("context", str(ctx.exception))

    def test_an_unexpected_artifact_shape_raises_type_error(self):
        self._rewrite(weird={"nested": 1})
        with self.assertRaises(TypeError) as ctx:
            model_registry.validate_artifacts(self.path_model)
        self.assertIn("weird", str(ctx.exception))

    def test_a_model_without_a_manifest_is_a_no_op(self):
        self.index_file.unlink()
        self.assertTrue(model_registry.validate_artifacts(self.path_model))


class TestPublishScriptPackaging(unittest.TestCase):
    """The packaging path of ``scripts/publish_model_to_hf.py``, which no test invoked.

    Everything asserted here is an *ordering* invariant of the script rather than of the
    functions it calls, so it cannot be covered by testing those functions directly. Packaging
    only -- no ``--upload``, no token, no network.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_model_dir(self.tmp)
        self.repo_root = Path(__file__).resolve().parents[1]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self):
        import subprocess
        import sys

        return subprocess.run(
            [
                sys.executable,
                str(self.repo_root.joinpath("scripts", "publish_model_to_hf.py")),
                "--model-dir",
                str(self.path_model),
            ],
            capture_output=True,
            text=True,
            cwd=str(self.repo_root),
        )

    def test_packaging_writes_checksums_that_cover_the_manifest(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, msg=result.stderr[-2000:])
        checksum_file = self.path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE)
        self.assertTrue(checksum_file.exists())
        paths = [e["path"] for e in json.loads(checksum_file.read_text())["files"]]
        # Written last, so the manifest the script just generated is inside the digest set.
        self.assertIn(model_registry.MODEL_MANIFEST_FILE, paths)
        self.assertTrue(model_registry.verify_checksums(self.path_model))

    def test_packaging_refuses_a_model_whose_artifacts_are_missing(self):
        # The pre-upload gate: better to fail here than to publish a repository that only breaks
        # once a stranger loads it.
        self.path_model.joinpath("model.ubj").unlink()
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model.ubj", result.stderr)

    def test_packaging_does_not_upload(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, msg=result.stderr[-2000:])
        self.assertIn("packaging only", result.stderr)


class TestSplit(unittest.TestCase):
    """The published split turns "was this scored on held-out probes?" into a checkable fact."""

    def setUp(self):
        import uuid

        self.tmp = Path(tempfile.mkdtemp())
        self.n_folds = 5
        self.pids = [str(uuid.uuid4()) for _ in range(10)]
        self.ifold = np.floor(np.arange(10) / 10 * self.n_folds)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, pids=None):
        model_registry.write_split(
            self.tmp,
            pids if pids is not None else self.pids,
            self.ifold,
            random_seed=12345,
            n_folds=self.n_folds,
        )
        return json.loads(
            self.tmp.joinpath(model_registry.MODEL_SPLIT_FILE).read_text()
        )

    def test_every_fold_is_reconstructed_and_partitions_the_pids(self):
        payload = self._write()
        self.assertEqual(len(payload["folds"]), self.n_folds)
        self.assertEqual(payload["random_seed"], 12345)
        test_sets = [set(f["test_pids"]) for f in payload["folds"]]
        union = set().union(*test_sets)
        self.assertEqual(union, set(self.pids))
        for i, left in enumerate(test_sets):
            for right in test_sets[i + 1:]:
                self.assertEqual(left & right, set())
        for fold in payload["folds"]:
            self.assertEqual(
                set(fold["train_pids"]), set(self.pids) - set(fold["test_pids"])
            )

    def test_per_fold_hashes_match_hash_uuids_in_the_stored_order(self):
        # These must equal what meta.yaml already records, or the two provenance records
        # disagree about the same split.
        from iblutil.numerical import hash_uuids

        payload = self._write()
        for fold in payload["folds"]:
            self.assertEqual(fold["hash_training"], hash_uuids(fold["train_pids"]))
            self.assertEqual(fold["hash_testing"], hash_uuids(fold["test_pids"]))

    def test_pid_order_is_preserved_not_sorted(self):
        # hash_uuids is order-sensitive, so sorting here would make meta.yaml's hashes
        # unreproducible from split.json.
        payload = self._write()
        self.assertEqual(payload["pids"], list(self.pids))

    def test_split_sha256_is_stable_and_covers_the_payload(self):
        first = self._write()["split_sha256"]
        self.assertEqual(self._write()["split_sha256"], first)
        swapped = list(self.pids)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self.assertNotEqual(self._write(pids=swapped)["split_sha256"], first)

    def test_a_non_uuid_pid_raises_and_names_the_offender(self):
        with self.assertRaises(ValueError) as ctx:
            self._write(pids=["not-a-uuid"] + self.pids[1:])
        self.assertIn("not-a-uuid", str(ctx.exception))

    def test_accepts_a_numpy_object_array_of_pids(self):
        # all_pids in the training script is an ndarray of objects, not a list.
        payload = self._write(pids=np.array(self.pids, dtype=object))
        self.assertEqual(payload["pids"], list(self.pids))

    def test_misaligned_pids_and_ifold_raise_rather_than_truncating(self):
        # The fold masks are built with zip(), which stops at the shorter sequence -- so a
        # caller whose ifold came from a filtered pid list would otherwise get a split.json
        # listing every pid while the folds account for only some, stamped with a valid digest.
        with self.assertRaises(ValueError) as ctx:
            model_registry.write_split(self.tmp, self.pids, self.ifold[:4], n_folds=2)
        self.assertIn("aligned", str(ctx.exception))
        with self.assertRaises(ValueError):
            model_registry.write_split(self.tmp, self.pids[:4], self.ifold, n_folds=5)

    def test_an_empty_split_raises(self):
        with self.assertRaises(ValueError):
            model_registry.write_split(self.tmp, [], np.array([]), n_folds=2)

    def test_the_self_describing_labels_are_recorded(self):
        # This file exists to be read by a stranger checking held-out status, so the fields that
        # say what the pid lists *mean* have to be right, not just present.
        payload = self._write()
        self.assertEqual(payload["split_unit"], "probe_insertion_pid")
        self.assertIn("shuffle", payload["fold_assignment"])
        self.assertEqual(
            [fold["fold"] for fold in payload["folds"]], list(range(self.n_folds))
        )


def _load_publish_script():
    """Import ``scripts/publish_model_to_hf.py`` as a module (it is a script, not a package)."""
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root.joinpath("scripts", "publish_model_to_hf.py")
    spec = importlib.util.spec_from_file_location("publish_model_to_hf", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_encoder_dir(path_models: Path) -> Path:
    """Write a minimal spatial-encoder directory (meta.yaml only) and build its manifest.

    No weights or volumes are needed: ``build_model_index`` records only the artifacts that
    exist, and the card renders purely from manifest fields -- so this exercises card
    generation without importing torch.
    """
    import yaml

    path_model = path_models.joinpath("2026_W26_encoder")
    path_model.mkdir(parents=True)
    meta = dict(
        RANDOM_SEED=42,
        VINTAGE="2026_W26",
        MODEL_CLASS="NeighborInpaintingModel",
        FEATURES=FEATURES,
        D_MODEL=128,
        NHEAD=8,
        DEPTH=2,
        DROP=0.1,
        F_CTX=100,
        RADIUS_UM=600.0,
        M_MAX=64,
        ALLOW_SAME_PROBE=False,
    )
    path_model.joinpath("meta.yaml").write_text(yaml.safe_dump(meta))
    model_registry.build_model_index(path_model, method="inpainting")
    return path_model


class TestEncoderCard(unittest.TestCase):
    """``write_card`` must render a spatial-encoder card, not raise NotImplementedError.

    The classifier template quotes region accuracy and a class count, which the encoder has
    none of; the encoder needs its own card stating the position->features inversion, the
    neighbour bank and context volumes it ships, the first-run Allen download, and that
    neighbour selection is deterministic nearest-M rather than the training-time random subset.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = _make_encoder_dir(self.tmp)
        self.index = model_registry.read_manifest(self.path_model)
        self.script = _load_publish_script()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_encoder_card_renders_without_raising(self):
        self.script.write_card(
            self.path_model, self.index, "int-brain-lab/ea-encoder-channel", "2026_W26"
        )
        self.assertTrue(self.path_model.joinpath("README.md").exists())
        self.assertTrue(self.path_model.joinpath("LICENSE").exists())

    def test_encoder_card_states_the_defining_facts(self):
        self.script.write_card(
            self.path_model, self.index, "int-brain-lab/ea-encoder-channel", "2026_W26"
        )
        card = self.path_model.joinpath("README.md").read_text().lower()
        # The inversion that defines this family: input is a position, output is features.
        self.assertIn("position", card)
        # The runtime artifacts a stranger must download alongside the weights.
        self.assertTrue("neighbor" in card or "neighbour" in card)
        self.assertIn("allen", card)  # the first-run public volume download
        self.assertIn("nearest", card)  # deterministic selection, not the random subset
        self.assertIn("load_pretrained", card)
        self.assertIn("2026_w26", card)  # the vintage
        # Must not carry the classifier's output contract.
        self.assertNotIn("predicted_acronym", card)

    def test_classifier_card_still_renders(self):
        # The task dispatch must not have broken the region-classifier path.
        path_clf = _make_model_dir(self.tmp)
        idx = model_registry.read_manifest(path_clf)
        self.script.write_card(
            path_clf, idx, "int-brain-lab/ea-decoder-channel-xgboost", "2026_W32"
        )
        self.assertIn("predicted_acronym", path_clf.joinpath("README.md").read_text())


class TestEncoderSourceReader(unittest.TestCase):
    """``_read_encoder_source`` joins channel positions to features and drops incomplete rows.

    The encoder is fed positions, so its source table must carry both the coordinates (from
    ``channels.pqt``) and the feature columns (from ``raw_ephys_features_denoised.pqt``) -- the
    bank stores standardised features keyed by position.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.script = _load_publish_script()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_source(self):
        idx = pd.MultiIndex.from_tuples(
            [("pidA", 0), ("pidA", 1), ("pidB", 0)], names=["pid", "channel"]
        )
        feats = pd.DataFrame(
            {"rms_ap": [1.0, 2.0, np.nan], "psd_delta": [0.1, 0.2, 0.3]}, index=idx
        )
        chans = pd.DataFrame(
            {"x": [0.001, 0.002, 0.003], "y": [0.0, 0.0, 0.0], "z": [0.0, 0.0, 0.0]},
            index=idx,
        )
        feats.to_parquet(self.tmp.joinpath("raw_ephys_features_denoised.pqt"))
        chans.to_parquet(self.tmp.joinpath("channels.pqt"))

    def test_join_carries_positions_and_features_and_drops_nan(self):
        self._write_source()
        df = self.script._read_encoder_source(self.tmp, ["rms_ap", "psd_delta"])
        for col in ["x", "y", "z", "rms_ap", "psd_delta"]:
            self.assertIn(col, df.columns)
        # The row whose feature is NaN (pidB, 0) is dropped, so the bank has no holes.
        self.assertEqual(len(df), 2)
        self.assertNotIn(("pidB", 0), df.index)
        # Index identity is preserved, so a caller can trace a channel back.
        self.assertEqual(list(df.index.names), ["pid", "channel"])


_W26 = Path.home().joinpath("ea_active", "2026_W26", "agg_full")
_SE_SOURCE = Path.home().joinpath("Downloads", "SE_model")
_ENCODER_FILES = ("SE_model.pth", "agea_vol_pca.npy", "merfish_vol_pca.npy", "meta.yaml")
_HAVE_ENCODER_DATA = (
    _W26.joinpath("raw_ephys_features_denoised.pqt").exists()
    and _W26.joinpath("channels.pqt").exists()
    and all(_SE_SOURCE.joinpath(n).exists() for n in _ENCODER_FILES)
)


@unittest.skipUnless(_HAVE_ENCODER_DATA, "needs local W26 features and the SE_model source")
class TestEncoderPublishPackaging(unittest.TestCase):
    """End-to-end packaging of a spatial encoder through the publish script (no upload).

    Runs the script as a subprocess -- a fresh process that loads torch but never xgboost, so it
    is safe here even though this file imports xgboost at module scope. Exercises the whole
    encoder path the unit tests cannot: build the neighbour bank, re-scan so it is recorded,
    render the encoder card, write the golden example, self-test, and checksum. Only runs where
    the real data lives.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = self.tmp.joinpath("2026_W26_encoder")
        self.path_model.mkdir()
        for name in _ENCODER_FILES:
            shutil.copy2(_SE_SOURCE.joinpath(name), self.path_model.joinpath(name))
        self.repo_root = Path(__file__).resolve().parents[1]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self):
        import subprocess
        import sys

        return subprocess.run(
            [
                sys.executable,
                str(self.repo_root.joinpath("scripts", "publish_model_to_hf.py")),
                "--model-dir",
                str(self.path_model),
                "--features",
                str(_W26),
                "--method",
                "inpainting",
                "--n-example-channels",
                "32",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.repo_root),
        )

    def test_packaging_builds_bank_card_example_and_selftests(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, msg=result.stderr[-3000:])
        # The bank a stranger cannot recompute must be shipped and recorded in the manifest.
        self.assertTrue(self.path_model.joinpath("neighbor_bank.npz").exists())
        idx = model_registry.read_manifest(self.path_model)
        self.assertEqual(idx["task"], model_registry.TASK_SPATIAL_ENCODING)
        self.assertEqual(idx["artifacts"]["neighbor_bank"], "neighbor_bank.npz")
        # The encoder card, not the classifier's.
        card = self.path_model.joinpath("README.md").read_text()
        self.assertIn("spatial encoder", card)
        self.assertNotIn("predicted_acronym", card)
        # Golden example, a passing selftest (logged), and checksums that cover it all.
        self.assertTrue(
            self.path_model.joinpath("example", "features_sample.parquet").exists()
        )
        self.assertIn("selftest passed", result.stderr)
        self.assertTrue(model_registry.verify_checksums(self.path_model, missing_ok=False))


class TestUnitEncoderDispatch(unittest.TestCase):
    """The unit-level encoder is wired into the registry as a third family.

    Assertions avoid the mutable ``MODEL_WRAPPERS`` (which TestWrapperDispatch clears) by
    checking the static ``MODEL_CLASS_TASKS`` map and the task-fallback ``TASK_WRAPPERS``.
    """

    def test_unit_class_maps_to_unit_task(self):
        self.assertEqual(
            model_registry.MODEL_CLASS_TASKS["MultimodalAutoencoder"],
            model_registry.TASK_UNIT_ENCODING,
        )
        self.assertEqual(
            model_registry.MODEL_CLASS_TASKS[
                "ephysatlas.unit_level_encoder.model.MultimodalAutoencoder"
            ],
            model_registry.TASK_UNIT_ENCODING,
        )

    def test_unit_task_resolves_to_the_unit_encoder_builder(self):
        import ephysatlas.models as models

        # No model_class -> resolve falls back to the task, which uses TASK_WRAPPERS.
        builder = models._resolve_wrapper(
            Path("/does/not/matter"), {"task": model_registry.TASK_UNIT_ENCODING}
        )
        self.assertIs(builder, models._unit_encoder)


if __name__ == "__main__":
    unittest.main()
