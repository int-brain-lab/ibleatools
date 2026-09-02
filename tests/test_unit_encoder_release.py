"""The unit-encoder training output IS the publish-ready release layout.

``training/train_unit_encoder.py`` writes the four canonical files at the model-dir root and the
manifest directly -- no ``meta.yaml``, no ``pt_gmm/`` subdirectory, no ``best_*`` working names, no
``_work/`` staging (change sets A4/B1/B2/B3, mirroring the spatial encoder's Step 3).

Two tests, cheap-to-expensive:

- :class:`TestWriteUnitRelease` locks the finalize contract on hand-written tiny checkpoints -- it
  runs in milliseconds and does not train.
- :class:`TestRunUnitEncoder` runs the whole three-stage pipeline on tiny synthetic arrays with a
  shrunk Config, in seconds on CPU, and asserts the emitted directory is the canonical, meta-free,
  loadable release layout.

IMPORTANT -- run this file in its own process::

    pytest tests/test_unit_encoder_release.py

The trainer pulls in torch, which segfaults alongside a module-scope xgboost on macOS arm64 (same
reason as ``tests/test_unit_encoder.py``). The guard in :func:`setUpModule` fails loudly rather than
segfaulting if something else already imported xgboost.
"""

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def setUpModule():
    if "xgboost" in sys.modules:
        raise RuntimeError(
            "xgboost is already imported in this process; loading torch as well segfaults on "
            "macOS arm64. Run this file in its own pytest process."
        )


def _load_trainer():
    """Import ``training/train_unit_encoder.py`` by path (training/ is not on sys.path)."""
    path = Path(__file__).resolve().parents[1] / "training" / "train_unit_encoder.py"
    spec = importlib.util.spec_from_file_location("train_unit_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestWriteUnitRelease(unittest.TestCase):
    """``write_unit_release`` writes a meta-free manifest over the trainers' canonical files."""

    def setUp(self):
        from ephysatlas import model_registry

        self.mr = model_registry
        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = self.tmp.joinpath("2026_W40_unit")
        self.path_model.mkdir(parents=True)
        self.trainer = _load_trainer()

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
        self.trainer.write_unit_release(self.path_model, self.meta)

        # The scaffold is gone: no meta.yaml, no pt_gmm/ subdirectory in the model output.
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
        self.trainer.write_unit_release(self.path_model, self.meta, split_manifest=split)
        got = json.loads(self.path_model.joinpath(self.mr.MODEL_SPLIT_FILE).read_text())
        self.assertEqual(got["train_pids"], ["pidA"])


def _make_probe_units(base_x_um):
    """Positions (in metres) for one probe: two voxels ~250 um apart, 4 units each.

    voxel_size is 200 um, so x ~ base and x ~ base+250 fall in adjacent voxels, and 250 um < the
    500 um neighbour radius, so each voxel has the other as a candidate neighbour.
    """
    xs, ys, zs = [], [], []
    for voxel_offset_um in (0.0, 250.0):
        for jitter in range(4):
            xs.append((base_x_um + voxel_offset_um + jitter) * 1e-6)
            ys.append((jitter * 5.0) * 1e-6)
            zs.append(0.0)
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


class TestRunUnitEncoder(unittest.TestCase):
    """The full pipeline emits a loadable, canonical, meta-free release directory.

    Exercises prepare_data -> train_autoencoder -> fit_and_evaluate (the GMM trainer) -> manifest on
    tiny synthetic arrays with a shrunk Config, in seconds on CPU. It does not check learned
    quality; it checks the trainer wires the pieces together and its output IS the release layout.

    Synthetic data satisfies the voxel-neighborhood constraints the PT-GMM datasets need: several
    probes across the train/val/test split, and, on each probe, two voxels ~250 um apart each
    holding enough units, so every split has at least one usable (probe, voxel) example.
    """

    CONTEXT_DIM = 8

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data_dir = self.tmp.joinpath("arrays")
        self.data_dir.mkdir()
        self.out_dir = self.tmp.joinpath("model")

        rng = np.random.RandomState(0)
        # Four probes so a 0.2/0.2 split gives >=1 train, val and test probe.
        pids, xyz = [], []
        for p in range(4):
            xyz.append(_make_probe_units(base_x_um=1000.0 * (p + 1)))
            pids.extend([f"pid{p}"] * 8)
        xyz = np.concatenate(xyz, axis=0).astype(np.float32)
        n = len(pids)

        # Default model input shapes (the architecture assumes them); data is tiny in N so the
        # full (20,128)/(10,201) shapes are still fast.
        np.save(self.data_dir / "waveforms.npy", rng.randn(n, 20, 128).astype(np.float32))
        np.save(self.data_dir / "acgs.npy", np.abs(rng.randn(n, 10, 201)).astype(np.float32))
        ctx = rng.randn(n, self.CONTEXT_DIM).astype(np.float32)
        ctx[:, :3] = xyz  # prepare_data expects context to begin with x, y, z
        np.save(self.data_dir / "ctx.npy", ctx)
        np.save(self.data_dir / "xyz.npy", xyz)
        np.save(self.data_dir / "pids.npy", np.array(pids, dtype=object), allow_pickle=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _shrunk_cfg(self):
        """A Config small and fast enough to train on CPU in the test."""
        from ephysatlas.unit_level_encoder.config import Config

        cfg = Config(device="cpu")
        cfg.amp = False
        cfg.num_workers = 0
        cfg.ae_epochs = 1
        cfg.pt_epochs = 1
        cfg.ae_batch_size = 8
        cfg.validation_batch_size = 8
        cfg.pt_batch_size = 4
        cfg.gmm_components = 2
        cfg.gmm_sklearn_n_init = 1
        cfg.gmm_sklearn_max_iter = 20
        cfg.min_target_units_per_voxel = 2  # our voxels hold 4 units each
        cfg.max_neighbor_distance_um = 500.0
        return cfg

    def test_pipeline_emits_a_loadable_canonical_release(self):
        trainer = _load_trainer()
        from ephysatlas import model_registry
        from ephysatlas.models.unit_encoder import UnitEncoder

        model_dir = trainer.train_unit_encoder(
            self.data_dir,
            self.out_dir,
            vintage="2026_W40",
            cfg=self._shrunk_cfg(),
            device="cpu",
        )

        # 1. The four canonical artifacts landed at the directory root -- written there directly,
        #    not staged: no best_* working names, no pt_gmm/ subdirectory, no _work/ staging dir.
        for name in (
            model_registry.UNIT_AE_FILE,
            model_registry.UNIT_GMM_FILE,
            model_registry.UNIT_SCALER_FILE,
            model_registry.UNIT_UNCOND_GMM_FILE,
        ):
            self.assertTrue(model_dir.joinpath(name).exists(), msg=f"missing {name}")
        self.assertFalse(model_dir.joinpath("best_multimodal_autoencoder.pt").exists())
        self.assertFalse(model_dir.joinpath("pt_gmm").exists())
        self.assertFalse(model_dir.joinpath("_work").exists())

        # 2. The scaffold is gone and a valid unit-encoding manifest is the single source of truth.
        self.assertFalse(model_dir.joinpath("meta.yaml").exists())
        index = model_registry.read_manifest(model_dir)
        self.assertEqual(index["task"], model_registry.TASK_UNIT_ENCODING)
        self.assertEqual(index["outputs"]["kind"], "latent")
        self.assertTrue(model_registry.validate_artifacts(model_dir, index))

        # 3. The directory actually loads and encodes through the serving wrapper.
        enc = UnitEncoder(model_dir, device="cpu")
        z = enc.encode(
            np.load(self.data_dir / "waveforms.npy")[:5],
            np.load(self.data_dir / "acgs.npy")[:5],
        )
        self.assertEqual(z.shape[0], 5)
        self.assertTrue(np.isfinite(z).all())

    def test_split_manifest_holds_out_whole_probes(self):
        trainer = _load_trainer()

        pids = [f"pid{i}" for i in range(10) for _ in range(3)]
        split = trainer.make_split_manifest(pids, seed=1)
        train, val, test = (set(split[k]) for k in ("train_pids", "validation_pids", "test_pids"))
        # No probe appears in more than one split, and the held-out set is non-empty.
        self.assertEqual(train & val, set())
        self.assertEqual(train & test, set())
        self.assertEqual(val & test, set())
        self.assertGreaterEqual(len(test), 1)


if __name__ == "__main__":
    unittest.main()
