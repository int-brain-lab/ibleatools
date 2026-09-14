"""Tests for the load path of ephysatlas.model_registry and the public RegionClassifier wrapper.

All offline: a tiny synthetic model directory is built by :mod:`tests._model_fixtures` (which
reproduces just the parts of a published release the load path reads), so the whole
manifest -> verify -> load -> predict path is exercised without touching the Hugging Face Hub.

Only the load path is covered here; nothing in this repo *writes* a release.
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
from tests import _model_fixtures as fixtures

CLASSES = fixtures.CLASSES
FEATURES = fixtures.FEATURES


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

    def test_resolution_failure_propagates(self):
        # A resolution failure (here an unknown mapping name) must surface rather than
        # be swallowed into a None that hides the cause from the caller.
        with self.assertRaises(Exception):
            model_registry.class_acronyms([315], "NotARealMapping")


class TestResolveModel(unittest.TestCase):
    def test_hf_needs_a_repo_id_for_a_bare_model_name(self):
        # A slashless name has no implicit hub repo: resolving it must raise rather than guess.
        with self.assertRaises(ValueError):
            model_registry.resolve_model("2024_W50_Cosmos_something")

    def test_hf_reads_repo_id_off_an_owner_name_model_id(self):
        source = model_registry.HFModelSource()
        self.assertEqual(source._resolve_repo_id("org/repo"), "org/repo")
        self.assertEqual(
            model_registry.HFModelSource(repo_id="a/b")._resolve_repo_id("ignored"),
            "a/b",
        )

    def test_download_is_verified_at_the_chokepoint(self):
        # resolve_model is the one function load_pretrained, RegionClassifier.from_pretrained
        # and download_model all pass through. Verifying here rather than in each caller is what
        # stops one documented entry point offering a guarantee the others do not.
        tmp = Path(tempfile.mkdtemp())
        try:
            path_model = fixtures.make_model_dir(tmp)
            target = path_model.joinpath("model.ubj")
            target.write_bytes(
                target.read_bytes() + b"tampered"
            )  # after checksums written

            original = model_registry.HFModelSource.fetch
            model_registry.HFModelSource.fetch = (
                lambda self, model_id, revision=None, cache_dir=None: path_model
            )
            try:
                with self.assertRaises(ValueError) as ctx:
                    model_registry.resolve_model("org/whatever", cache_dir=tmp)
            finally:
                model_registry.HFModelSource.fetch = original
            self.assertIn("model.ubj", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestLoadPretrained(unittest.TestCase):
    """The single public entry point -- the only API a published model card should name."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = fixtures.make_model_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_local_directory_needs_no_network_or_credentials(self):
        import ephysatlas

        model = ephysatlas.load_pretrained(self.path_model)
        self.assertIsInstance(model, regionclassifier.RegionClassifier)
        self.assertEqual(model.index["task"], fixtures.TASK_REGION_CLASSIFICATION)

    def test_unknown_model_class_raises_actionably(self):
        # Dispatch is on model_class now, so an unloadable model is one with an unknown class.
        index_file = self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE)
        index = json.loads(index_file.read_text())
        index["model_class"] = "some.future.Model"
        index_file.write_text(json.dumps(index))
        fixtures.write_checksums(self.path_model)  # keep the checksum gate satisfied
        import ephysatlas

        with self.assertRaises(ValueError) as ctx:
            ephysatlas.load_pretrained(self.path_model)
        self.assertIn("some.future.Model", str(ctx.exception))

    def test_unpinned_hub_id_warns_about_moving_main(self):
        import ephysatlas

        # Resolution will fail (no network / no such repo); we only care that the warning
        # about an unpinned revision is emitted before the attempt.
        with self.assertLogs("ephysatlas.models", level="WARNING") as logs:
            with self.assertRaises(Exception):
                ephysatlas.load_pretrained("org/does-not-exist")
        self.assertIn("no revision pinned", "\n".join(logs.output))

    def test_local_directory_does_not_warn_about_revision(self):
        import ephysatlas

        logger = logging.getLogger("ephysatlas.models")
        with self.assertLogs(logger, level="INFO") as logs:
            ephysatlas.load_pretrained(self.path_model)
        self.assertNotIn("no revision pinned", "\n".join(logs.output))

    def test_load_pretrained_verifies_checksums(self):
        import ephysatlas

        target = self.path_model.joinpath("model.ubj")
        target.write_bytes(target.read_bytes() + b"corrupted")
        with self.assertRaises(ValueError) as ctx:
            ephysatlas.load_pretrained(self.path_model)
        self.assertIn("model.ubj", str(ctx.exception))

    def test_load_pretrained_requires_checksums(self):
        # Every published model ships checksums.json; the load path requires it, so a model
        # missing it is treated as an incomplete download.
        self.path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE).unlink()
        import ephysatlas

        with self.assertRaises(FileNotFoundError):
            ephysatlas.load_pretrained(self.path_model)


class TestLoadModelDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = fixtures.make_model_dir(self.tmp)

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

    def test_no_manifest_raises(self):
        # The manifest is mandatory. load_model itself does not check checksums, so a directory
        # with no manifest raises FileNotFoundError directly.
        self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).unlink()
        with self.assertRaises(FileNotFoundError):
            regionclassifier.load_model(self.path_model)

    def test_manifest_alone_is_sufficient(self):
        # The current layout: a manifest and nothing else. It must be enough on its own, and
        # model_info stays UPPER_CASE-shaped for infer_regions callers.
        classifier, info = regionclassifier.load_model(self.path_model)
        self.assertIsInstance(classifier, XGBClassifier)
        self.assertEqual(info["FEATURES"], FEATURES)
        self.assertEqual(info["CLASSES"], CLASSES)
        self.assertEqual(info["MODEL_CLASS"], "xgboost.sklearn.XGBClassifier")

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


class TestWeightsOnlyFolds(unittest.TestCase):
    """Folds ship weights only -- no per-fold metadata. The ensemble must still discover and
    load every fold from the manifest and report a real ``fold_agreement``, rather than silently
    dropping them and degrading to the single global model.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = fixtures.make_model_dir(self.tmp)
        rng = np.random.default_rng(2)
        index = pd.MultiIndex.from_product(
            [["pid-a", "pid-b"], range(5)], names=["pid", "channel"]
        )
        self.df = pd.DataFrame(
            rng.normal(size=(len(index), len(FEATURES))), index=index, columns=FEATURES
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ensemble_loads_weights_only_folds(self):
        clf = regionclassifier.RegionClassifier(self.path_model)
        # both folds discovered by their weights, none dropped
        self.assertEqual(len(clf._fold_dirs()), 2)
        out = clf.predict(self.df, estimator="ensemble")
        # a real agreement over the folds actually consulted, not the all-NaN a
        # global-only fallback would produce
        self.assertFalse(np.isnan(out["fold_agreement"]).all())

    def test_the_loader_warns_when_only_some_folds_survive(self):
        # A model published damaged (a fold lost its weights) must warn rather than silently
        # average fewer models than the manifest advertises.
        self.path_model.joinpath("folds", "FOLD01", "model.ubj").unlink()
        clf = regionclassifier.RegionClassifier(self.path_model)
        with self.assertLogs("ephysatlas.regionclassifier", level="WARNING") as logs:
            clf._fold_dirs()
        self.assertIn("only 1 are loadable", "\n".join(logs.output))


class TestRegionClassifier(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = fixtures.make_model_dir(self.tmp)
        rng = np.random.default_rng(1)
        index = pd.MultiIndex.from_product(
            [["pid-a", "pid-b"], range(5)], names=["pid", "channel"]
        )
        self.df = pd.DataFrame(
            rng.normal(size=(len(index), len(FEATURES))), index=index, columns=FEATURES
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

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
        # single fold ranked first. Those rows are exactly the ones a user should distrust.
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
                ens["prediction_probability"].values,
                glob["prediction_probability"].values,
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
        clf.predict(self.df).to_parquet(
            example.joinpath("expected_predictions.parquet")
        )
        self.assertTrue(clf.selftest())

    def test_selftest_without_example_raises(self):
        with self.assertRaises(FileNotFoundError):
            regionclassifier.RegionClassifier(self.path_model).selftest()

    # --- feature order -----------------------------------------------------------------
    # The estimator consumes the feature matrix positionally, so a reordered manifest list is
    # a silent wrong-answer bug rather than a load error. The digest makes it loud.

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

    def test_the_digest_can_sit_on_any_block(self):
        # The encoder's positional list is outputs.columns, not inputs.features, so the
        # validator must not assume which block it came from.
        columns = ["rms_lf", "psd_lfp", "psd_alpha"]
        digest = model_registry.feature_order_sha256(columns)
        self.assertTrue(model_registry.validate_feature_order(columns, digest))
        with self.assertRaises(ValueError):
            model_registry.validate_feature_order(columns[::-1], digest)


class TestWrapperDispatch(unittest.TestCase):
    """Wrapper dispatch is model_class-first.

    Task alone is not enough: region decoding is one task with ``method`` separating xgboost from
    a future transformer, so both would carry the same ``task`` while needing different wrappers
    -- a torch module has no ``predict_proba``.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = fixtures.make_model_dir(self.tmp)
        self.index_file = self.path_model.joinpath(model_registry.MODEL_MANIFEST_FILE)
        import ephysatlas.models as models

        # Tests here monkeypatch the module-level dispatch map; snapshot it so a mutation cannot
        # leak into other test classes.
        self._saved_wrappers = dict(models.MODEL_WRAPPERS)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        import ephysatlas.models as models

        models.MODEL_WRAPPERS.clear()
        models.MODEL_WRAPPERS.update(self._saved_wrappers)

    def _rechecksum(self):
        """Re-write checksums after editing a file on disk, so the load-path checksum gate passes
        and the test reaches the dispatch logic it is exercising."""
        fixtures.write_checksums(self.path_model)

    def test_dispatch_is_on_model_class(self):
        import ephysatlas
        import ephysatlas.models as models

        sentinel = object()
        models.MODEL_WRAPPERS["xgboost.sklearn.XGBClassifier"] = (
            lambda path_model, index, **kw: sentinel
        )
        self.assertIs(ephysatlas.load_pretrained(self.path_model), sentinel)

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


class TestVerifyChecksums(unittest.TestCase):
    """``checksums.json`` answers a question the manifest cannot: are the bytes I just downloaded
    the bytes that were published? These cover the load-path *verifier* only.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = fixtures.make_model_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_verify_passes_on_an_untouched_directory(self):
        self.assertTrue(model_registry.verify_checksums(self.path_model))

    def test_verify_detects_a_changed_byte(self):
        target = self.path_model.joinpath("model.ubj")
        target.write_bytes(target.read_bytes() + b"x")
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        self.assertIn("model.ubj", str(ctx.exception))

    def test_verify_detects_a_flipped_byte_that_does_not_change_the_size(self):
        # The size comparison runs first and short-circuits, so an append-based test would still
        # pass with the sha256 comparison deleted. This is the one that needs it: same length,
        # one bit different -- exactly what silent corruption looks like.
        target = self.path_model.joinpath("model.ubj")
        before = target.stat().st_size
        raw = bytearray(target.read_bytes())
        raw[5] ^= 0xFF
        target.write_bytes(bytes(raw))
        self.assertEqual(target.stat().st_size, before)
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        self.assertIn("hash", str(ctx.exception))

    def test_verify_refuses_a_path_that_escapes_the_model_directory(self):
        # checksums.json travels with the download and no digest covers it, so a published
        # repository could point verification at an arbitrary local file. Reject lexically.
        checksum_file = self.path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE)
        payload = json.loads(checksum_file.read_text())
        payload["files"].append(
            {"path": "../../../../../../etc/hosts", "hash": "0" * 40, "bytes": 1}
        )
        checksum_file.write_text(json.dumps(payload))
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        message = str(ctx.exception)
        self.assertIn("escapes the model directory", message)
        # Refused rather than reported as a mismatch, which would mean it was stat'd and hashed.
        self.assertNotIn("hash mismatch", message)

    def test_verify_refuses_an_absolute_path(self):
        checksum_file = self.path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE)
        payload = json.loads(checksum_file.read_text())
        payload["files"].append({"path": "/etc/hosts", "hash": "0" * 40, "bytes": 1})
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

    def test_verify_detects_a_missing_file_and_names_it(self):
        self.path_model.joinpath("folds", "FOLD01", "model.ubj").unlink()
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        self.assertIn("folds/FOLD01/model.ubj", str(ctx.exception))

    def test_verify_reports_every_failure_at_once(self):
        self.path_model.joinpath("model.ubj").unlink()
        self.path_model.joinpath("folds", "FOLD00", "model.ubj").unlink()
        with self.assertRaises(ValueError) as ctx:
            model_registry.verify_checksums(self.path_model)
        message = str(ctx.exception)
        # One traversal, one error: fixing them one round trip at a time is miserable.
        self.assertIn("model.ubj", message)
        self.assertIn("folds/FOLD00/model.ubj", message)

    def test_verify_tolerates_files_added_after_writing(self):
        # A Hub snapshot carries .gitattributes and a .cache/ tree, and selftest writes example/.
        # Verification must check the listed files, not reject a superset.
        self.path_model.joinpath(".gitattributes").write_text("* text=auto\n")
        example = self.path_model.joinpath("example")
        example.mkdir()
        example.joinpath("features_sample.parquet").write_bytes(b"not really a parquet")
        self.assertTrue(model_registry.verify_checksums(self.path_model))

    def test_verify_is_silent_when_absent_but_strict_on_demand(self):
        # The default (missing_ok=True) tolerates a directory with no checksums.json; the load
        # path opts into strictness with missing_ok=False.
        self.path_model.joinpath(model_registry.MODEL_CHECKSUM_FILE).unlink()
        self.assertIsNone(model_registry.verify_checksums(self.path_model))
        with self.assertRaises(FileNotFoundError):
            model_registry.verify_checksums(self.path_model, missing_ok=False)


class TestUnitEncoderDispatch(unittest.TestCase):
    """The unit-level encoder is wired into the load dispatch as a third family."""

    def test_unit_class_resolves_to_the_unit_encoder_builder(self):
        import ephysatlas.models as models

        # Dispatch is on model_class -- the bare class the unit manifest records.
        builder = models._resolve_wrapper(
            Path("/does/not/matter"), {"model_class": "MultimodalAutoencoder"}
        )
        self.assertIs(builder, models._unit_encoder)


class TestInferRegions(unittest.TestCase):
    """infer_regions routes through RegionClassifier: it works on the manifest/meta-free-fold
    layout, returns the per-fold arrays (not an average), and its fold-mean equals predict()."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = fixtures.make_model_dir(
            self.tmp
        )  # 2 folds, no meta.yaml anywhere
        rng = np.random.default_rng(3)
        index = pd.MultiIndex.from_product(
            [["pid-a", "pid-b"], range(5)], names=["pid", "channel"]
        )
        self.df = pd.DataFrame(
            rng.normal(size=(len(index), len(FEATURES))), index=index, columns=FEATURES
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_per_fold_probas_and_region_indices(self):
        probas, regions = regionclassifier.infer_regions(
            self.df, self.path_model, n_folds=2
        )
        # Per-fold, un-averaged: one probability array per fold.
        self.assertEqual(probas.shape, (2, len(self.df), len(CLASSES)))
        self.assertEqual(regions.shape, (2, len(self.df)))
        # predicted_region are class indices in [0, n_classes), the argmax of each fold's probas.
        self.assertTrue(((regions >= 0) & (regions < len(CLASSES))).all())
        np.testing.assert_array_equal(regions, np.argmax(probas, axis=2))

    def test_fold_mean_matches_predict(self):
        probas, _ = regionclassifier.infer_regions(self.df, self.path_model, n_folds=2)
        out = regionclassifier.RegionClassifier(self.path_model).predict(
            self.df, estimator="ensemble"
        )
        acronyms = model_registry.class_acronyms(CLASSES, "Cosmos")
        expected = np.column_stack([out[f"p_{a}"].values for a in acronyms])
        np.testing.assert_allclose(probas.mean(axis=0), expected, rtol=1e-6)

    def test_n_folds_is_ignored_with_a_warning_on_mismatch(self):
        with self.assertLogs("ephysatlas.regionclassifier", level="WARNING") as logs:
            probas, _ = regionclassifier.infer_regions(
                self.df, self.path_model, n_folds=3
            )
        # The manifest lists 2 folds; n_folds=3 is ignored, not honoured.
        self.assertEqual(probas.shape[0], 2)
        self.assertIn("n_folds", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
