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
    )

    def setUp(self):
        from ephysatlas.unit_level_encoder import Config
        from ephysatlas.unit_level_encoder.prepare_data import channel_context_sha1

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
        (prepared / "waveform_feature_names.json").write_text("[]")
        self.manifest = {
            "context_type": "merfish_agea_pca",
            "n_cell_pcs": 50,
            "n_gene_pcs": 50,
            "context_vintage": "2026_W39",
            "context_volumes_sha1": channel_context_sha1(self.release, "2026_W39"),
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

    def test_data_sampled_from_other_volumes_is_rebuilt(self):
        stale = {**self.manifest, "context_volumes_sha1": {"agea_vol_pca.npy": "0"}}
        self.assertTrue(self._rebuilt(stale))

    def test_data_without_a_volume_record_is_rebuilt(self):
        # Prepared before the volumes were recorded: their identity is unknown.
        legacy = {k: v for k, v in self.manifest.items() if k != "context_volumes_sha1"}
        self.assertTrue(self._rebuilt(legacy))


if __name__ == "__main__":
    unittest.main()
