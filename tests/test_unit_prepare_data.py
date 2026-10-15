"""The unit model's prepared data is tied to the exact context volumes of its channel release.

The unit contexts are sampled from the channel-level release's frozen MERFISH/AGEA PCA volumes, so
both models share one PCA basis. These tests check that a stale copy of the volumes (another
release or vintage) left in the prepared-data directory is replaced rather than reused, and that
prepared arrays sampled from other volumes are rebuilt. A local directory stands in for the
channel release, so no Hub access is needed.

In its own file with a setUpModule guard: the pipeline pulls in torch, which segfaults on macOS
arm64 if xgboost is already imported in the same process.
"""

import json
import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ephysatlas.model_registry import ENCODER_CONTEXT_FILES


def setUpModule():
    if (
        "xgboost" in sys.modules
        and sys.platform == "darwin"
        and platform.machine() == "arm64"
    ):
        raise RuntimeError(
            "xgboost is already imported in this process; loading torch as well segfaults on "
            "macOS arm64. Run this file in its own pytest process."
        )


def _write_volumes(directory: Path, value: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(ENCODER_CONTEXT_FILES):
        np.save(directory / name, np.full((2, 2, 2, 3), value + i, np.float32))


class TestFrozenContextAtlas(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.release = self.tmp / "channel_release"
        _write_volumes(self.release, value=1.0)
        self.out_dir = self.tmp / "prepared"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ensure(self, release=None):
        from ephysatlas.unit_level_encoder.prepare_data import (
            _ensure_frozen_context_atlas,
        )

        return _ensure_frozen_context_atlas(
            out_dir=self.out_dir,
            channel_model=release or self.release,
            vintage="2026_W39",
        )

    def test_copies_the_release_volumes_and_reports_their_hashes(self):
        from ephysatlas.unit_level_encoder.prepare_data import channel_context_sha1

        context_dir, hashes = self._ensure()
        self.assertEqual(hashes, channel_context_sha1(self.release, "2026_W39"))
        for name in ENCODER_CONTEXT_FILES:
            self.assertEqual(
                (context_dir / name).read_bytes(), (self.release / name).read_bytes()
            )

    def test_stale_cached_volumes_are_replaced(self):
        # Volumes of another release already sit in the prepared-data directory.
        _write_volumes(self.out_dir / "context_atlas", value=-5.0)
        context_dir, _ = self._ensure()
        for name in ENCODER_CONTEXT_FILES:
            self.assertEqual(
                (context_dir / name).read_bytes(), (self.release / name).read_bytes()
            )

    def test_legacy_release_keeps_the_volumes_under_context(self):
        legacy = self.tmp / "legacy_release"
        _write_volumes(legacy / "context", value=1.0)
        _, hashes = self._ensure(release=legacy)
        _, expected = self._ensure()
        self.assertEqual(hashes, expected)


class TestPreparedDataFreshness(unittest.TestCase):
    """``prepare_unit_data`` rebuilds prepared arrays sampled from other context volumes."""

    REQUIRED = (
        "waveforms.npy",
        "acgs.npy",
        "stpc.npy",
        "xyz.npy",
        "pids.npy",
        "cosmos.npy",
        "allen.npy",
        "waveform_features.npy",
        "cluster_ids.npy",
        "unit_axial_um.npy",
    )

    def setUp(self):
        from ephysatlas.unit_level_encoder import Config
        from ephysatlas.unit_level_encoder.prepare_data import channel_context_sha1
        from ephysatlas.unit_level_encoder.waveform_features import FEATURE_NAMES

        self.tmp = Path(tempfile.mkdtemp())
        self.release = self.tmp / "channel_release"
        _write_volumes(self.release, value=1.0)
        self.cfg = Config(
            device="cpu",
            vintage="2026_W39",
            channel_model=str(self.release),
            data_dir=self.tmp / "data",
            prepared_data_dir=self.tmp / "prepared",
        )
        prepared = self.cfg.prepared_data_dir
        prepared.mkdir(parents=True)
        for name in self.REQUIRED:
            np.save(prepared / name, np.zeros(1, np.float32))
        np.save(prepared / "ctx.npy", np.zeros((4, 100), np.float32))
        np.save(
            prepared / "waveform_channel_xy_um.npy", np.zeros((4, 20, 2), np.float32)
        )
        (prepared / "waveform_feature_names.json").write_text("[]")
        self.manifest = {
            "context_type": "merfish_agea_pca",
            "n_cell_pcs": 50,
            "n_gene_pcs": 50,
            "context_vintage": "2026_W39",
            "context_volumes_sha1": channel_context_sha1(self.release, "2026_W39"),
            "waveform_feature_names": list(FEATURE_NAMES),
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rebuilt(self, manifest: dict) -> bool:
        """Run ``prepare_unit_data`` and report whether it rebuilt the prepared arrays."""
        from ephysatlas.unit_level_encoder import pipeline

        self.cfg.prepared_data_dir.joinpath(
            "latest_cells_encoder_manifest.json"
        ).write_text(json.dumps(manifest))
        data = SimpleNamespace(
            waveforms=np.zeros((1, 20, 128)),
            acgs=np.zeros((1, 10, 100)),
            stpc=np.zeros((1, 6)),
        )
        with (
            mock.patch.object(pipeline, "prepare_latest_cells_encoder_data") as prepare,
            mock.patch.object(pipeline, "load_prepared_data", return_value=data),
        ):
            pipeline.prepare_unit_data(self.cfg)
        return prepare.called

    def test_data_sampled_from_the_release_volumes_is_reused(self):
        self.assertFalse(self._rebuilt(self.manifest))

    def test_data_without_unit_axial_positions_is_rebuilt(self):
        # Prepared before the unit cost's axial positions were written with the unit arrays.
        self.cfg.prepared_data_dir.joinpath("unit_axial_um.npy").unlink()
        self.assertTrue(self._rebuilt(self.manifest))

    def test_data_sampled_from_other_volumes_is_rebuilt(self):
        stale = {**self.manifest, "context_volumes_sha1": {"agea_vol_pca.npy": "0"}}
        self.assertTrue(self._rebuilt(stale))

    def test_data_with_other_waveform_features_is_rebuilt(self):
        old = {**self.manifest, "waveform_feature_names": ["peak_val", "polarity"]}
        self.assertTrue(self._rebuilt(old))

    def test_data_without_a_volume_record_is_rebuilt(self):
        # Prepared before the volumes were recorded: their identity is unknown.
        legacy = {k: v for k, v in self.manifest.items() if k != "context_volumes_sha1"}
        self.assertTrue(self._rebuilt(legacy))


class TestMultichannelWaveformCache(unittest.TestCase):
    """The multichannel waveform cache is reused only for the units it was extracted for."""

    def setUp(self):
        import pandas as pd

        self.tmp = Path(tempfile.mkdtemp())
        self.agg = self.tmp / "cells_aggregates"
        self.agg.mkdir()
        rows = [
            {"pid": "p0", "cluster_id": u, "abs_channel": c}
            for u in range(3)
            for c in range(20)
        ]
        pd.DataFrame(rows).to_parquet(self.agg / "waveforms.table.pqt")
        voltage = np.random.default_rng(0).normal(size=(60, 128)).astype(np.float32)
        np.save(self.agg / "waveforms.voltage.npy", voltage)
        self.cache = self.tmp / "prepared" / "waveforms_good_multichannel_C20.npy"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _extract(self, clusters):
        import pandas as pd

        from ephysatlas.unit_level_encoder.prepare_data import (
            _build_multichannel_waveform_cache,
        )

        df_good = pd.DataFrame({"pid": ["p0"] * len(clusters), "cluster_id": clusters})
        return _build_multichannel_waveform_cache(
            cells_agg_path=self.agg,
            df_good=df_good,
            pid_col="pid",
            target_channels=20,
            cache_path=self.cache,
        )

    def test_cache_of_the_same_units_is_reused(self):
        first, _, _ = self._extract([0, 1, 2])
        again, _, info = self._extract([0, 1, 2])
        self.assertTrue(info.get("loaded_existing_cache"))
        np.testing.assert_array_equal(again, first)

    def test_cache_serves_a_subset_of_its_units(self):
        # Fewer units, e.g. after more insertions were excluded, in another order.
        first, _, _ = self._extract([0, 1, 2])
        waveforms, keep, info = self._extract([2, 0])
        self.assertTrue(info.get("subset_of_cache"))
        self.assertTrue(keep.all())
        np.testing.assert_array_equal(waveforms, first[[2, 0]])

    def test_cache_without_the_requested_units_is_extracted_again(self):
        self._extract([0, 1])
        waveforms, keep, info = self._extract([0, 2])
        self.assertFalse(info.get("loaded_existing_cache", False))
        self.assertEqual(waveforms.shape[0], 2)
        self.assertTrue(keep.all())

    def test_cache_without_a_unit_record_is_extracted_again(self):
        # Extracted before the cache recorded its units: their identity is unknown.
        self._extract([0, 1, 2])
        self.cache.with_name(self.cache.stem + "_unit_keys.npy").unlink()
        _, _, info = self._extract([0, 1, 2])
        self.assertFalse(info.get("loaded_existing_cache", False))


class TestModelSpaceFeatureCache(unittest.TestCase):
    """The model-space feature cache is reused only for the same units and feature names."""

    def setUp(self):
        from ephysatlas.unit_level_encoder import Config

        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = Config(device="cpu", prepared_data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _data(self, names, channel_xy_um=None):
        return SimpleNamespace(
            waveforms=np.zeros((3, 4, 128), np.float32),
            waveform_feature_names=list(names),
            channel_xy_um=channel_xy_um,
        )

    def test_cache_with_the_same_names_is_reused(self):
        from ephysatlas.unit_level_encoder.pipeline import (
            model_space_waveform_features,
            save_model_space_waveform_features,
        )

        cached = np.arange(6, dtype=np.float32).reshape(3, 2)
        save_model_space_waveform_features(self.cfg, cached, ["a", "b"])
        np.testing.assert_array_equal(
            model_space_waveform_features(self._data(["a", "b"]), self.cfg), cached
        )

    def test_cache_of_other_features_is_not_reused(self):
        from ephysatlas.unit_level_encoder.pipeline import (
            model_space_waveform_features,
            save_model_space_waveform_features,
        )

        save_model_space_waveform_features(
            self.cfg, np.zeros((3, 2)), ["peak_val", "polarity"]
        )
        # Recomputing needs the channel positions, which this data lacks.
        with self.assertRaisesRegex(RuntimeError, "channel positions"):
            model_space_waveform_features(self._data(["a", "b"]), self.cfg)


if __name__ == "__main__":
    unittest.main()
