"""The spatial-encoder training output IS the publish-ready release layout.

``ephysatlas.models.release.write_spatial_release`` writes the canonical files at the model-dir
root, builds the neighbour bank, and writes the manifest directly -- no ``meta.yaml``, no
``encoding_models/`` subdirectory. This locks that contract on a tiny seeded random-init model, so
it runs in seconds on CPU without the real W26 data or the full training pipeline.

IMPORTANT -- run this file in its own process::

    pytest tests/test_spatial_encoder_release.py

The finalize helper imports torch, which segfaults alongside a module-scope xgboost on macOS arm64
(same reason as ``tests/test_spatial_encoder_parity.py``). The guard in :func:`setUpModule` fails
loudly rather than segfaulting if something else already imported xgboost.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ephysatlas.models.release import write_spatial_release


def setUpModule():
    if "xgboost" in sys.modules:
        raise RuntimeError(
            "xgboost is already imported in this process; loading torch as well segfaults on "
            "macOS arm64. Run this file in its own pytest process."
        )


FEATURES = ["rms_ap", "rms_lf", "psd_delta", "psd_theta", "spike_count"]

# For the end-to-end load+predict test below (needs real context volumes and a W26 slice).
_W26 = Path.home().joinpath("ea_active", "2026_W26", "agg_full")
_SE_SOURCE = Path.home().joinpath("Downloads", "SE_model")
_CONTEXT_FILES = ("agea_vol_pca.npy", "merfish_vol_pca.npy")
_HAVE_DATA = _W26.joinpath("channels.pqt").exists() and all(
    _SE_SOURCE.joinpath(n).exists() for n in _CONTEXT_FILES + ("meta.yaml",)
)


class TestSpatialReleaseLayout(unittest.TestCase):
    """``write_spatial_release`` writes a meta-free, loadable, publish-ready directory."""

    def setUp(self):
        import torch
        from ephysatlas.spatial_encoder.model import NeighborInpaintingModel

        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = self.tmp.joinpath("2026_W26_encoder")
        self.path_model.mkdir(parents=True)

        f_ctx = 100
        f_e = len(FEATURES)
        torch.manual_seed(0)
        # Non-trivial normalisation stats, so the bank's standardisation is genuinely exercised.
        self.model = NeighborInpaintingModel(
            f_ctx=f_ctx,
            f_ephys=f_e,
            f_out=f_e,
            e_mean=torch.arange(f_e, dtype=torch.float32) * 0.1,
            e_std=torch.full((f_e,), 2.0),
            ctx_mean=torch.arange(f_ctx, dtype=torch.float32) * 0.01,
            ctx_std=torch.full((f_ctx,), 3.0),
            d_model=32,
            nhead=4,
            depth=2,
            drop=0.0,
        ).eval()

        # Context volumes must sit at the root for the manifest to record artifacts.context (the
        # trainer's ContextAtlasManager writes them there). Tiny placeholders suffice: this test
        # validates the layout, it does not call predict (which would read them).
        for name in ("agea_vol_pca.npy", "merfish_vol_pca.npy"):
            np.save(
                self.path_model.joinpath(name), np.zeros((2, 2, 2, 3), dtype=np.float32)
            )

        # A tiny in-memory training bank: positions, already-standardised features, pid strings --
        # exactly the shape build_neighbor_handles hands the trainer.
        self.bank_xyz = np.array(
            [[0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [0.003, 0.0, 0.0]], dtype=np.float32
        )
        self.bank_feat = np.arange(3 * len(FEATURES), dtype=np.float32).reshape(
            3, len(FEATURES)
        )
        self.bank_pid = np.array(["pidA", "pidA", "pidB"], dtype=str)

        self.meta = dict(
            RANDOM_SEED=0,
            VINTAGE="2026_W26",
            MODEL_CLASS="NeighborInpaintingModel",
            FEATURES=FEATURES,
            F_CTX=f_ctx,
            D_MODEL=32,
            NHEAD=4,
            DEPTH=2,
            DROP=0.0,
            RADIUS_UM=500.0,
            M_MAX=8,
            ALLOW_SAME_PROBE=False,
            N_CELL_PCS=50,
            N_GENE_PCS=50,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_canonical_meta_free_manifest(self):
        import ephysatlas.model_registry as mr

        write_spatial_release(
            self.path_model,
            self.model,
            self.meta,
            self.bank_xyz,
            self.bank_feat,
            self.bank_pid,
        )
        # No meta.yaml in the model output -- the manifest is the only record.
        self.assertFalse(self.path_model.joinpath("meta.yaml").exists())
        # Canonical filenames at the root, no encoding_models/ subdirectory, no _{vintage} suffix.
        self.assertTrue(self.path_model.joinpath("spatial_encoder.pt").exists())
        self.assertTrue(self.path_model.joinpath("neighbor_bank.npz").exists())
        self.assertFalse(self.path_model.joinpath("encoding_models").exists())
        # The manifest is the single source of truth and records the runtime artifacts.
        self.assertTrue(self.path_model.joinpath(mr.MODEL_MANIFEST_FILE).exists())
        on_disk = mr.read_manifest(self.path_model)
        self.assertEqual(on_disk["task"], mr.TASK_SPATIAL_ENCODING)
        self.assertEqual(on_disk["method"], "transformer")
        self.assertEqual(on_disk["artifacts"]["weights"], "spatial_encoder.pt")
        self.assertEqual(on_disk["artifacts"]["neighbor_bank"], "neighbor_bank.npz")
        self.assertEqual(
            sorted(on_disk["artifacts"]["context"]),
            ["agea_vol_pca.npy", "merfish_vol_pca.npy"],
        )
        # Everything the manifest names is present on disk.
        self.assertTrue(mr.validate_artifacts(self.path_model, on_disk))

    def test_split_written_when_provided(self):
        split_info = dict(p_tr_names=["pidA"], p_va_names=["pidB"], p_te_names=[])
        write_spatial_release(
            self.path_model,
            self.model,
            self.meta,
            self.bank_xyz,
            self.bank_feat,
            self.bank_pid,
            split_info=split_info,
        )
        got = json.loads(self.path_model.joinpath("split.json").read_text())
        self.assertEqual(got["p_tr_names"], ["pidA"])


@unittest.skipUnless(
    _HAVE_DATA, f"needs {_W26} and the context volumes in {_SE_SOURCE}"
)
class TestSpatialReleaseLoadsAndPredicts(unittest.TestCase):
    """The trainer's release output loads through the published wrapper and predicts.

    Stronger than the synthetic layout test above: it uses the *real* context volumes and a small
    W26 slice, so it proves the wrapped ``{"model_state", "architecture"}`` checkpoint the trainer
    writes is loadable by ``SpatialEncoder`` end to end -- weights, context, and neighbour bank all
    resolved through the manifest, with no ``meta.yaml``. A seeded random-init model is enough; the
    parity test already covers numerical correctness of ``predict``.
    """

    def setUp(self):
        import pandas as pd
        import torch
        import yaml
        from ephysatlas.spatial_encoder.model import NeighborInpaintingModel

        self.tmp = Path(tempfile.mkdtemp())
        self.path_model = self.tmp.joinpath("2026_W26_encoder")
        self.path_model.mkdir(parents=True)

        for name in _CONTEXT_FILES:
            shutil.copy2(_SE_SOURCE.joinpath(name), self.path_model.joinpath(name))

        features = [
            str(f)
            for f in yaml.safe_load(_SE_SOURCE.joinpath("meta.yaml").read_text())[
                "FEATURES"
            ]
        ]
        f_ctx, f_e = 100, len(features)
        torch.manual_seed(0)
        self.model = NeighborInpaintingModel(
            f_ctx=f_ctx,
            f_ephys=f_e,
            f_out=f_e,
            e_mean=torch.arange(f_e, dtype=torch.float32) * 0.1,
            e_std=torch.full((f_e,), 2.0),
            ctx_mean=torch.arange(f_ctx, dtype=torch.float32) * 0.01,
            ctx_std=torch.full((f_ctx,), 3.0),
            d_model=32,
            nhead=4,
            depth=2,
            drop=0.0,
        ).eval()

        feats = pd.read_parquet(_W26.joinpath("raw_ephys_features_denoised.pqt"))
        chans = pd.read_parquet(_W26.joinpath("channels.pqt"))
        df = feats.join(chans.loc[:, ["x", "y", "z"]], how="inner").dropna(
            subset=["x", "y", "z"] + features
        )
        slice_ = df.loc[sorted(df.index.get_level_values(0).unique())[:12]]
        # Build the in-memory bank the trainer would ship: positions + features standardised with
        # the model's own e_mean/e_std (as the collate does), keyed by pid string.
        e_mean = self.model.e_mean.detach().cpu().numpy()
        e_std = self.model.e_std.detach().cpu().numpy()
        raw = slice_.loc[:, features].to_numpy(dtype=np.float32)
        self.bank_xyz = slice_.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float32)
        self.bank_feat = ((raw - e_mean) / e_std).astype(np.float32)
        self.bank_pid = slice_.index.get_level_values(0).to_numpy().astype(str)
        self.query = slice_.loc[:, ["x", "y", "z"]].head(16)
        self.meta = dict(
            RANDOM_SEED=0,
            VINTAGE="2026_W26",
            MODEL_CLASS="NeighborInpaintingModel",
            FEATURES=features,
            F_CTX=f_ctx,
            D_MODEL=32,
            NHEAD=4,
            DEPTH=2,
            DROP=0.0,
            RADIUS_UM=500.0,
            M_MAX=8,
            ALLOW_SAME_PROBE=False,
            N_CELL_PCS=50,
            N_GENE_PCS=50,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_release_output_loads_and_predicts(self):
        from ephysatlas.models.encoder_inpainting import SpatialEncoder

        write_spatial_release(
            self.path_model,
            self.model,
            self.meta,
            self.bank_xyz,
            self.bank_feat,
            self.bank_pid,
        )
        # No meta.yaml: the manifest alone is the record the wrapper reads.
        self.assertFalse(self.path_model.joinpath("meta.yaml").exists())
        enc = SpatialEncoder(self.path_model)
        pred = enc.predict(self.query)
        self.assertEqual(len(pred), len(self.query))
        self.assertEqual(pred.shape[1], len(self.meta["FEATURES"]))
        # Output is indexed like the input, so a caller can join it straight back on.
        self.assertTrue(pred.index.equals(self.query.index))


if __name__ == "__main__":
    unittest.main()
