"""Differential parity tests for the published spatial encoder.

The question these answer is the one that matters for the migration: does
``SpatialEncoder.predict`` compute the *same numbers* as the training-time pipeline it replaces?
Rather than assert against hard-coded values, each test runs **his** code path and **mine** over
identical inputs and the identical model instance, and compares.

Feeding both paths the same weights is what makes the comparison independent of which checkpoint
is to hand: a seeded random-init model is sufficient to prove the surrounding pipeline agrees.

IMPORTANT -- run this file in its own process::

    pytest tests/test_spatial_encoder_parity.py

xgboost and torch bring incompatible OpenMP runtimes on macOS arm64 and segfault at the first
torch tensor copy when both are loaded. ``tests/test_model_registry.py`` imports xgboost at module
scope, so collecting it alongside this file in one process crashes the interpreter. Nothing here
imports xgboost, and the guard in :func:`setUpModule` fails loudly rather than segfaulting if
something else already has.

Skips unless the 2026_W26 feature tables and the published PCA context volumes are both present.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

W26 = Path.home().joinpath("ea_active", "2026_W26", "agg_full")
SE_SOURCE = Path.home().joinpath("Downloads", "SE_model")
CONTEXT_FILES = ("agea_vol_pca.npy", "merfish_vol_pca.npy")

HAVE_DATA = W26.joinpath("channels.pqt").exists() and all(
    SE_SOURCE.joinpath(n).exists() for n in CONTEXT_FILES
)


def setUpModule():
    if "xgboost" in sys.modules:
        raise RuntimeError(
            "xgboost is already imported in this process; loading torch as well segfaults on "
            "macOS arm64. Run this file in its own pytest process."
        )


@unittest.skipUnless(HAVE_DATA, f"needs {W26} and the PCA volumes in {SE_SOURCE}")
class TestSpatialEncoderParity(unittest.TestCase):
    """His pipeline vs mine, on his vintage."""

    # Enough insertions that some are genuinely near each other: arbitrary probes in a mouse
    # brain are mostly further apart than the 600 µm neighbourhood radius.
    N_BANK_PIDS = 24
    N_QUERY_PIDS = 6
    M_MAX = 512  # deliberately larger than any neighbourhood, so no subsetting occurs
    RADIUS_UM = 600.0

    @classmethod
    def setUpClass(cls):
        import torch
        import yaml

        import ephysatlas.model_registry as model_registry
        from ephysatlas.models.encoder_inpainting import build_neighbor_bank
        from ephysatlas.spatial_encoder.model import NeighborInpaintingModel

        cls.torch = torch
        cls.model_registry = model_registry

        meta = yaml.safe_load(SE_SOURCE.joinpath("meta.yaml").read_text())
        cls.features = [str(f) for f in meta["FEATURES"]]

        # ---- a small slice of W26: features + coordinates, indexed by (pid, channel) ----
        feats = pd.read_parquet(W26.joinpath("raw_ephys_features_denoised.pqt"))
        chans = pd.read_parquet(W26.joinpath("channels.pqt"))
        df = feats.join(chans.loc[:, ["x", "y", "z"]], how="inner")
        df = df.dropna(subset=["x", "y", "z"] + cls.features)
        pids = sorted(df.index.get_level_values(0).unique())[: cls.N_BANK_PIDS + cls.N_QUERY_PIDS]
        bank_pids = pids[: cls.N_BANK_PIDS]
        query_pids = pids[cls.N_BANK_PIDS:]
        cls.df_bank = df.loc[bank_pids]

        # Query on channels that actually have neighbours in the bank. Comparing empty
        # neighbourhoods would make the parity assertions vacuous -- both paths would trivially
        # agree on all-zero tensors.
        from ephysatlas.spatial_encoder.utils import ChannelNN

        candidates = df.loc[query_pids]
        cand_xyz = candidates.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64)
        nn = ChannelNN(cls.df_bank.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64))
        hits = nn.query_radius(cand_xyz, r_m=cls.RADIUS_UM * 1e-6, k_cap=1)
        has_neighbour = np.array([len(h) > 0 for h in hits])
        if not has_neighbour.any():
            raise unittest.SkipTest(
                f"no channel in the {cls.N_QUERY_PIDS} query insertions lies within "
                f"{cls.RADIUS_UM} µm of the {cls.N_BANK_PIDS}-insertion bank; raise N_BANK_PIDS"
            )
        # A handful of channels is plenty: parity is exact or it is not.
        cls.df_query = candidates.loc[has_neighbour].head(8)

        # ---- a model directory: real context volumes, a seeded random-init encoder ----
        cls.tmp = Path(tempfile.mkdtemp())
        cls.path_model = cls.tmp.joinpath("2026_W26_encoder_parity")
        cls.path_model.mkdir(parents=True)
        for name in CONTEXT_FILES:
            shutil.copy2(SE_SOURCE.joinpath(name), cls.path_model.joinpath(name))

        f_ctx = 100  # 50 cell PCs + 50 gene PCs
        f_e = len(cls.features)
        torch.manual_seed(0)
        # Non-trivial normalisation statistics, so a path that forgot to standardise cannot
        # accidentally agree with one that did.
        model = NeighborInpaintingModel(
            f_ctx=f_ctx, f_ephys=f_e, f_out=f_e,
            e_mean=torch.arange(f_e, dtype=torch.float32) * 0.1,
            e_std=torch.full((f_e,), 2.0),
            ctx_mean=torch.arange(f_ctx, dtype=torch.float32) * 0.01,
            ctx_std=torch.full((f_ctx,), 3.0),
            d_model=32, nhead=4, depth=2, drop=0.0,
        )
        model.eval()
        torch.save(model.state_dict(), cls.path_model.joinpath("spatial_encoder.pt"))
        yaml.safe_dump(
            {
                "VINTAGE": "2026_W26", "FEATURES": cls.features,
                "MODEL_CLASS": "NeighborInpaintingModel",
                "D_MODEL": 32, "NHEAD": 4, "DEPTH": 2, "DROP": 0.0,
                "M_MAX": cls.M_MAX, "RADIUS_UM": cls.RADIUS_UM,
            },
            cls.path_model.joinpath("meta.yaml").open("w"),
        )
        cls.index = model_registry.build_model_index(cls.path_model, method="inpainting")
        build_neighbor_bank(cls.path_model, cls.df_bank, cls.index)
        # Rebuild so artifacts.neighbor_bank is recorded now the bank exists.
        cls.index = model_registry.build_model_index(cls.path_model, method="inpainting")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _encoder(self):
        from ephysatlas.models.encoder_inpainting import SpatialEncoder

        return SpatialEncoder(self.path_model, index=self.index)

    # -- A: context ------------------------------------------------------------------------

    def test_context_matches_his_recorded_channel_path(self):
        """My per-channel context equals what his dataset construction produces."""
        from ephysatlas.spatial_encoder.utils import mirror_xyz_to_left

        encoder = self._encoder()
        xyz = self.df_query.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float32)

        # His recorded-channel path, transcribed from
        # build_channels_plus_emptyvoxels_with_neighbors.
        manager = encoder._context_manager()
        pack = manager.sample_context_numpy_m(mirror_xyz_to_left(xyz.copy()), mode="clip")
        his_raw = np.concatenate([pack["cell_pc"], pack["gene_pc"]], axis=1).astype(np.float32)
        ctx_mean = encoder.model.ctx_mean.detach().cpu().numpy()
        ctx_std = encoder.model.ctx_std.detach().cpu().numpy()
        his = his_raw.copy()
        valid = his_raw.sum(axis=1) != 0
        his[valid] = (his_raw[valid] - ctx_mean) / ctx_std

        mine = encoder._standardised_context(xyz)
        np.testing.assert_array_equal(mine, his)
        # Guard the test itself: all-zero context would make the comparison vacuous.
        self.assertTrue(np.any(mine != 0))

    def test_out_of_atlas_context_is_left_at_zero_not_standardised(self):
        """His ``_stdz_ctx`` only touches non-zero rows; mine must do the same."""
        encoder = self._encoder()
        xyz = np.zeros((3, 3), dtype=np.float32)  # origin: outside any labelled voxel
        mine = encoder._standardised_context(xyz)
        zero_rows = [i for i in range(3) if not np.any(mine[i])]
        self.assertEqual(
            len(zero_rows), 3,
            msg="a zero-context row was standardised into a large negative vector",
        )

    # -- B: neighbours and the forward pass -------------------------------------------------

    def test_neighbours_and_predictions_match_his_collate(self):
        """With no subsetting, my neighbour tensors and predictions equal his exactly."""
        from ephysatlas.spatial_encoder.utils import ChannelNN, NeighborCollate

        encoder = self._encoder()
        model = encoder.model
        torch = self.torch

        xyz = self.df_query.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float32)
        pids = self.df_query.index.get_level_values(0).to_numpy().astype(str)
        ctx = encoder._standardised_context(xyz)
        bank = encoder._neighbor_bank()

        # --- his path: the real NeighborCollate over the same bank ---
        # Its pid exclusion compares integers, so map both sides through one index space.
        pid_order = {p: i for i, p in enumerate(sorted(set(bank["pid"].tolist()) | set(pids.tolist())))}
        collate = NeighborCollate(
            ctx_manager=encoder._context_manager(),
            bank_xyz_m=bank["xyz"],
            bank_feat_stdzd=bank["feat"],
            bank_pid=np.array([pid_order[p] for p in bank["pid"]], dtype=np.int64),
            kdtree_bank=ChannelNN(bank["xyz"]),
            e_feat_dim=len(self.features),
            M_max=self.M_MAX,
            radius_um=self.RADIUS_UM,
            allow_same_probe=False,
        )
        items = [
            (
                torch.from_numpy(ctx[i]),
                torch.from_numpy(xyz[i]),
                torch.from_numpy(np.zeros(len(self.features), dtype=np.float32)),
                torch.tensor(pid_order[pids[i]]),
                torch.tensor(True),
            )
            for i in range(xyz.shape[0])
        ]
        his_ctx, his_p_q, his_e_n, his_p_n, his_mask, _, _, _ = collate(items)

        # --- mine ---
        mine_e_n, mine_p_n, mine_mask = encoder._neighbours(xyz, pids)

        # The regime must actually be the no-subsetting one, or this proves nothing.
        self.assertLess(
            int(his_mask.sum(dim=1).max()), self.M_MAX,
            msg="neighbourhood hit M_MAX; raise M_MAX so no random subsetting occurs",
        )
        self.assertGreater(int(his_mask.sum()), 0, msg="no neighbours found at all")

        np.testing.assert_array_equal(mine_mask, his_mask.numpy())
        np.testing.assert_array_equal(mine_p_n, his_p_n.numpy())
        np.testing.assert_array_equal(mine_e_n, his_e_n.numpy())

        # --- and therefore identical predictions, from the same model instance ---
        with torch.no_grad():
            _, his_mu = model(his_ctx, his_p_q, his_e_n, his_p_n, his_mask)
        his_pred = (his_mu.float() * model.e_std + model.e_mean).numpy()
        mine_pred = encoder.predict(self.df_query).to_numpy()
        np.testing.assert_allclose(mine_pred, his_pred, rtol=0, atol=0)

    # -- C: determinism ---------------------------------------------------------------------

    def test_predictions_do_not_depend_on_global_rng_state(self):
        """His collate drew a random neighbour subset; a published predict() must not."""
        encoder = self._encoder()
        np.random.seed(1)
        first = encoder.predict(self.df_query).to_numpy()
        np.random.seed(999)
        second = encoder.predict(self.df_query).to_numpy()
        np.testing.assert_array_equal(first, second)

    # -- shape and contract -----------------------------------------------------------------

    def test_output_is_indexed_like_the_input_and_joins_cleanly(self):
        out = self._encoder().predict(self.df_query)
        pd.testing.assert_index_equal(out.index, self.df_query.index)
        self.assertEqual(len(out.columns), len(self.features))
        # `pred_` prefix: the input already carries ground-truth columns of the same names.
        self.assertEqual(set(out.columns) & set(self.df_query.columns), set())
        joined = self.df_query.join(out)
        self.assertIn(f"pred_{self.features[0]}", joined.columns)
        self.assertIn(self.features[0], joined.columns)

    def test_missing_coordinates_raise_and_say_what_is_needed(self):
        encoder = self._encoder()
        with self.assertRaises(KeyError) as ctx:
            encoder.predict(self.df_query.drop(columns=["y"]))
        self.assertIn("y", str(ctx.exception))

    def test_a_missing_bank_is_an_explicit_failure(self):
        from ephysatlas.models.encoder_inpainting import SpatialEncoder

        index = dict(self.index)
        index["artifacts"] = {k: v for k, v in index["artifacts"].items() if k != "neighbor_bank"}
        with self.assertRaises(FileNotFoundError) as ctx:
            SpatialEncoder(self.path_model, index=index).predict(self.df_query)
        self.assertIn("neighbor_bank", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
