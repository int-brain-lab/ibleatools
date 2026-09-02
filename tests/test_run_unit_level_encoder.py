"""CPU smoke test for the from-scratch unit-encoder runner (examples/run_unit_level_encoder.py).

Exercises the whole three-stage pipeline -- prepare_data -> train_autoencoder ->
fit_and_evaluate (the GMM trainer) -> canonical staging -> manifest -- on tiny synthetic arrays
with a shrunk Config, in seconds, on CPU. It does not check learned quality; it checks that the
runner wires the pieces together and emits a valid, loadable release directory (the thing that was
missing).

Synthetic data is crafted to satisfy the voxel-neighborhood constraints the PT-GMM datasets need:
several probes across the train/val/test split, and, on each probe, two voxels ~250 um apart each
holding enough units, so every split has at least one usable (probe, voxel) example.

In its own file with a setUpModule guard: the runner pulls in torch, which segfaults on macOS
arm64 if xgboost is already imported in the same process (same reason as test_unit_encoder.py).
"""

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


def _load_runner():
    """Import examples/run_unit_level_encoder.py by path (examples/ is not on sys.path)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "examples" / "run_unit_level_encoder.py"
    spec = importlib.util.spec_from_file_location("run_unit_level_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_probe_units(rng, base_x_um):
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


class TestRunUnitLevelEncoder(unittest.TestCase):
    CONTEXT_DIM = 8

    def setUp(self):
        # A prepared-arrays directory like prepare_latest_cells_encoder_data writes.
        self.tmp = Path(tempfile.mkdtemp())
        self.data_dir = self.tmp.joinpath("arrays")
        self.data_dir.mkdir()
        self.out_dir = self.tmp.joinpath("model")

        rng = np.random.RandomState(0)
        # Four probes so a 0.2/0.2 split gives >=1 train, val and test probe.
        pids, xyz = [], []
        for p in range(4):
            xyz.append(_make_probe_units(rng, base_x_um=1000.0 * (p + 1)))
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
        runner = _load_runner()
        from ephysatlas import model_registry
        from ephysatlas.models.unit_encoder import UnitEncoder

        model_dir = runner.train_unit_encoder(
            self.data_dir,
            self.out_dir,
            vintage="2026_W40",
            cfg=self._shrunk_cfg(),
            device="cpu",
        )

        # 1. The four canonical artifacts landed at the directory root, correctly renamed.
        for name in (
            model_registry.UNIT_AE_FILE,
            model_registry.UNIT_GMM_FILE,
            model_registry.UNIT_SCALER_FILE,
            model_registry.UNIT_UNCOND_GMM_FILE,
        ):
            self.assertTrue(model_dir.joinpath(name).exists(), msg=f"missing {name}")

        # 2. A valid unit-encoding manifest was written and its artifacts validate.
        index = model_registry.read_manifest(model_dir)
        self.assertEqual(index["task"], model_registry.TASK_UNIT_ENCODING)
        self.assertEqual(index["outputs"]["kind"], "latent")
        self.assertTrue(model_registry.validate_artifacts(model_dir, index))

        # 3. The staged directory actually loads and encodes through the serving wrapper.
        enc = UnitEncoder(model_dir, device="cpu")
        z = enc.encode(
            np.load(self.data_dir / "waveforms.npy")[:5],
            np.load(self.data_dir / "acgs.npy")[:5],
        )
        self.assertEqual(z.shape[0], 5)
        self.assertTrue(np.isfinite(z).all())

    def test_split_manifest_holds_out_whole_probes(self):
        runner = _load_runner()

        pids = [f"pid{i}" for i in range(10) for _ in range(3)]
        split = runner.make_split_manifest(pids, seed=1)
        train, val, test = (set(split[k]) for k in ("train_pids", "validation_pids", "test_pids"))
        # No probe appears in more than one split, and the held-out set is non-empty.
        self.assertEqual(train & val, set())
        self.assertEqual(train & test, set())
        self.assertEqual(val & test, set())
        self.assertGreaterEqual(len(test), 1)


if __name__ == "__main__":
    unittest.main()
