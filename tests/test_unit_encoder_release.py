"""The unit-encoder training output IS the publish-ready release layout.

``ephysatlas.models.release.write_unit_release`` writes the manifest directly over the four
canonical files the trainers leave at the model-dir root (``autoencoder.pt``,
``point_transformer_gmm.pt``, ``shared_latent_scaler.joblib``, ``unconditional_gmm_train_only.joblib``),
with the manifest as the single source of truth and no ``meta.yaml`` alongside it.

:class:`TestWriteUnitRelease` locks that contract on hand-written tiny checkpoints -- it runs in
milliseconds and does not train. (The end-to-end pipeline test that actually runs the trainer lives
with the training scripts in ``paper-ephys-atlas``.)

IMPORTANT -- run this file in its own process::

    pytest tests/test_unit_encoder_release.py

``write_unit_release`` and the wrapper pull in torch, which segfaults alongside a module-scope
xgboost on macOS arm64 (same reason as ``tests/test_unit_encoder.py``). The guard in
:func:`setUpModule` fails loudly rather than segfaulting if something else already imported xgboost.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from ephysatlas.models.release import write_unit_release


def setUpModule():
    if "xgboost" in sys.modules:
        raise RuntimeError(
            "xgboost is already imported in this process; loading torch as well segfaults on "
            "macOS arm64. Run this file in its own pytest process."
        )


class TestWriteUnitRelease(unittest.TestCase):
    """``write_unit_release`` writes a meta-free manifest over the trainers' canonical files."""

    def setUp(self):
        from ephysatlas import model_registry

        self.mr = model_registry
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = self.tmp.joinpath("2026_W40_unit")
        self.path_model.mkdir(parents=True)

        # Stand in for the four files the two trainers write at the root. Their *contents* are
        # irrelevant to the layout contract; write_manifest only checks they exist on disk.
        for name in (
            model_registry.UNIT_AE_FILE,
            model_registry.UNIT_GMM_FILE,
            model_registry.UNIT_SCALER_FILE,
            model_registry.UNIT_UNCOND_GMM_FILE,
        ):
            self.path_model.joinpath(name).write_bytes(b"stub")

        self.meta = dict(
            MODEL_CLASS="MultimodalAutoencoder",
            VINTAGE="2026_W40",
            LATENT_DIM=32,
            GMM_COMPONENTS=16,
            PROJECT="ibl_neuropixel_brainwide_01",
            RANDOM_SEED=0,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_canonical_meta_free_manifest(self):
        write_unit_release(self.path_model, self.meta)

        # No meta.yaml and no pt_gmm/ subdirectory in the model output.
        self.assertFalse(self.path_model.joinpath("meta.yaml").exists())
        self.assertFalse(self.path_model.joinpath("pt_gmm").exists())

        # The manifest is the single source of truth and records all four artifact roles.
        self.assertTrue(self.path_model.joinpath(self.mr.MODEL_MANIFEST_FILE).exists())
        index = self.mr.read_manifest(self.path_model)
        self.assertEqual(index["task"], self.mr.TASK_UNIT_ENCODING)
        self.assertEqual(index["method"], "gmm")
        self.assertEqual(index["outputs"]["kind"], "latent")
        self.assertEqual(index["artifacts"]["autoencoder"], self.mr.UNIT_AE_FILE)
        self.assertEqual(index["artifacts"]["pt_gmm"], self.mr.UNIT_GMM_FILE)
        self.assertEqual(index["artifacts"]["scaler"], self.mr.UNIT_SCALER_FILE)
        # Everything the manifest names is present on disk.
        self.assertTrue(self.mr.validate_artifacts(self.path_model, index))

    def test_split_written_when_provided(self):
        import json

        split = dict(train_pids=["pidA"], validation_pids=["pidB"], test_pids=["pidC"])
        write_unit_release(self.path_model, self.meta, split_manifest=split)
        got = json.loads(self.path_model.joinpath(self.mr.MODEL_SPLIT_FILE).read_text())
        self.assertEqual(got["train_pids"], ["pidA"])


if __name__ == "__main__":
    unittest.main()
