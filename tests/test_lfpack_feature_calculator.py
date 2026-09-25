"""Unit tests for ``LFPackFeatureCalculator``.

No real lfpack archive is opened here: ``lfpack.LFPackReader`` is faked (same
convention as ``test_spikeglx_feature_calculator.py``'s ``_FakeReader``). The
backend was additionally validated end-to-end against real, already-compressed
local archives (``/Users/olivier/scratch/lfp/<pid>/lf_compressed*.h5``) during
development; that step needs real data on disk so it isn't repeated here.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ephysatlas.feature_calculators.lfpack import LFPackFeatureCalculator
from ephysatlas.feature_calculators.types import (
    CsdParams,
    FeatureComputationOptions,
    FeatureParams,
    SnippetWindow,
)

N_CH = 4
FS_LF = 250.0
NS = 5000


def _geometry() -> dict:
    # Single-column, 4-depth layout: current_source_density's order-2 finite
    # difference needs several distinct depths in one column (a 2x2 grid, two
    # depths, is not enough and degenerates numerically).
    return {
        "x": np.zeros(N_CH),
        "y": np.arange(N_CH) * 20.0,
    }


def _channels(annotated: bool = False) -> dict:
    channels = {
        "lateral_um": _geometry()["x"].copy(),
        "axial_um": _geometry()["y"].copy(),
    }
    if annotated:
        channels.update(
            x=np.linspace(0, 1e-3, N_CH),
            y=np.linspace(0, 1e-3, N_CH),
            z=np.linspace(0, 1e-3, N_CH),
            atlas_id=np.array([500, 500, 997, 997]),
            acronym=["MOs", "MOs", "root", "root"],
            labels=np.zeros(N_CH, dtype=np.int8),
        )
    return channels


class _FakeLFPackReader:
    """Minimal ``lfpack.LFPackReader``-like stub."""

    def __init__(self, annotated: bool = False, saturation: pd.DataFrame | None = None):
        rng = np.random.default_rng(0)
        self.fs = FS_LF
        self.ns = NS
        self.geometry = _geometry()
        self.channels = _channels(annotated=annotated)
        self._data = rng.standard_normal((NS, N_CH)).astype(np.float32)
        self._saturation = (
            saturation
            if saturation is not None
            else pd.DataFrame(columns=["start_sample", "stop_sample", "start_time", "stop_time"])
        )

    def read(self, nsel, csel, sync=False, bin_channels=None):
        start = nsel.start or 0
        stop = nsel.stop if nsel.stop is not None else self.ns
        data = self._data[start:stop]
        if sync:
            return data, None
        return data

    def saturation_times(self):
        return self._saturation

    @property
    def saturation_mask(self):
        mask = np.zeros(self.ns, dtype=bool)
        for _, row in self._saturation.iterrows():
            mask[int(row["start_sample"]) : int(row["stop_sample"])] = True
        return mask


class TestLFPackFeatureCalculator(unittest.TestCase):
    def _calc(self, **reader_kwargs) -> LFPackFeatureCalculator:
        calc = LFPackFeatureCalculator("fake.h5", recording="pid00", name="pid00")
        calc._reader = _FakeLFPackReader(**reader_kwargs)
        return calc

    def test_name_defaults_to_recording(self):
        calc = LFPackFeatureCalculator("fake.h5", recording="pid00")
        self.assertEqual(calc.name, "pid00")

    def test_name_falls_back_to_file_stem(self):
        calc = LFPackFeatureCalculator("/data/lf_compressed.h5")
        self.assertEqual(calc.name, "lf_compressed")

    def test_available_duration_is_lf_only(self):
        calc = self._calc()
        max_ap, max_lf = calc.available_duration()
        self.assertIsNone(max_ap)
        self.assertAlmostEqual(max_lf, NS / FS_LF)

    def test_load_geometry_fills_derived_defaults(self):
        calc = self._calc()
        geometry = calc.load_geometry()
        for key in ("x", "y", "sample_shift", "shank", "col", "row"):
            self.assertIn(key, geometry)
        np.testing.assert_array_equal(geometry["shank"], np.zeros(N_CH))

    def test_load_channel_metadata_unannotated(self):
        calc = self._calc(annotated=False)
        metadata = calc.load_channel_metadata()
        self.assertEqual(
            set(metadata.columns), {"channel", "lateral_um", "axial_um", "shank"}
        )
        np.testing.assert_array_equal(metadata["channel"], np.arange(N_CH))

    def test_load_channel_metadata_annotated(self):
        calc = self._calc(annotated=True)
        metadata = calc.load_channel_metadata()
        for key in ("x", "y", "z", "atlas_id", "acronym", "labels"):
            self.assertIn(key, metadata.columns)

    def test_load_raw_snippet_shape_and_no_ap(self):
        calc = self._calc()
        window = SnippetWindow(t_start=1.0, duration_ap=2.0, duration_lf=2.0)
        raw = calc.load_raw_snippet(window)
        self.assertIsNone(raw.raw_ap)
        self.assertIsNone(raw.fs_ap)
        self.assertEqual(raw.fs_lf, FS_LF)
        self.assertEqual(raw.raw_lf.shape, (N_CH, int(2.0 * FS_LF)))

    def test_saturation_times_passthrough(self):
        sat = pd.DataFrame(
            {
                "start_sample": [100],
                "stop_sample": [200],
                "start_time": [0.4],
                "stop_time": [0.8],
            }
        )
        calc = self._calc(saturation=sat)
        pd.testing.assert_frame_equal(calc.saturation_times(), sat)
        mask = calc.saturation_mask
        self.assertTrue(mask[100:200].all())
        self.assertFalse(mask[:100].any())

    def test_reader_opened_lazily_and_cached(self):
        calc = LFPackFeatureCalculator("fake.h5", recording="pid00")
        # Construction alone must not touch the filesystem: the reader is
        # imported and opened only on first access of `.reader`.
        self.assertIsNone(calc._reader)
        with mock.patch("lfpack.LFPackReader", return_value=_FakeLFPackReader()) as mocked:
            mocked.assert_not_called()
            first = calc.reader
            second = calc.reader
            self.assertEqual(mocked.call_count, 1)
            self.assertIs(first, second)

    def test_compute_snippet_produces_valid_feature_table(self):
        calc = self._calc(annotated=True)
        window = SnippetWindow(t_start=1.0, duration_ap=10.0, duration_lf=10.0)
        options = FeatureComputationOptions(
            features_to_compute=["lf", "csd"],
            skip_lf_destripe=True,
            feature_params=FeatureParams(csd=CsdParams(decimate=1, denoise=False)),
            include_trajectory=False,
            output_dir=None,
        )
        result = calc.compute_snippet(window, options)
        self.assertEqual(len(result.features), N_CH)
        self.assertIn("rms_lf", result.features.columns)
        self.assertIn("rms_lf_csd", result.features.columns)
        self.assertEqual(result.computed_features, ("lf", "csd"))


if __name__ == "__main__":
    unittest.main()
