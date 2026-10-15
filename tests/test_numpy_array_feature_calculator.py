"""Unit tests for ``NumpyArrayFeatureCalculator``. No I/O at all is involved."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ephysatlas.feature_calculators.numpy_array import NumpyArrayFeatureCalculator
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


def _lf() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_CH, NS)).astype(np.float32)


class TestNumpyArrayFeatureCalculator(unittest.TestCase):
    def test_available_duration_is_lf_only(self):
        calc = NumpyArrayFeatureCalculator(
            lf=_lf(), fs_lf=FS_LF, geometry=_geometry(), name="pid00"
        )
        max_ap, max_lf = calc.available_duration()
        self.assertIsNone(max_ap)
        self.assertAlmostEqual(max_lf, NS / FS_LF)

    def test_load_raw_snippet_shape_and_no_ap(self):
        calc = NumpyArrayFeatureCalculator(
            lf=_lf(), fs_lf=FS_LF, geometry=_geometry(), name="pid00"
        )
        window = SnippetWindow(t_start=1.0, duration_ap=2.0, duration_lf=2.0)
        raw = calc.load_raw_snippet(window)
        self.assertIsNone(raw.raw_ap)
        self.assertIsNone(raw.fs_ap)
        self.assertEqual(raw.fs_lf, FS_LF)
        self.assertEqual(raw.raw_lf.shape, (N_CH, int(2.0 * FS_LF)))

    def test_load_raw_snippet_honors_t0_offset(self):
        lf = _lf()
        calc = NumpyArrayFeatureCalculator(
            lf=lf, fs_lf=FS_LF, geometry=_geometry(), name="pid00", t0=5.0
        )
        # Array-relative sample 0 corresponds to session time t0=5.0.
        window = SnippetWindow(t_start=5.0, duration_ap=1.0, duration_lf=1.0)
        raw = calc.load_raw_snippet(window)
        np.testing.assert_array_equal(raw.raw_lf, lf[:, : int(FS_LF)])

    def test_load_channel_metadata_defaults_from_geometry(self):
        calc = NumpyArrayFeatureCalculator(
            lf=_lf(), fs_lf=FS_LF, geometry=_geometry(), name="pid00"
        )
        metadata = calc.load_channel_metadata()
        self.assertEqual(
            set(metadata.columns), {"channel", "axial_um", "lateral_um", "shank"}
        )
        np.testing.assert_array_equal(metadata["axial_um"], _geometry()["y"])

    def test_load_channel_metadata_uses_supplied_frame(self):
        supplied = pd.DataFrame(
            {
                "channel": np.arange(N_CH),
                "axial_um": _geometry()["y"],
                "lateral_um": _geometry()["x"],
                "shank": np.zeros(N_CH),
                "atlas_id": np.array([500, 500, 997, 997]),
            }
        )
        calc = NumpyArrayFeatureCalculator(
            lf=_lf(),
            fs_lf=FS_LF,
            geometry=_geometry(),
            name="pid00",
            channel_metadata=supplied,
        )
        pd.testing.assert_frame_equal(calc.load_channel_metadata(), supplied)

    def test_load_geometry_fills_derived_defaults(self):
        calc = NumpyArrayFeatureCalculator(
            lf=_lf(), fs_lf=FS_LF, geometry=_geometry(), name="pid00"
        )
        geometry = calc.load_geometry()
        for key in ("x", "y", "sample_shift", "shank", "col", "row"):
            self.assertIn(key, geometry)

    def test_compute_snippet_produces_valid_feature_table(self):
        calc = NumpyArrayFeatureCalculator(
            lf=_lf(), fs_lf=FS_LF, geometry=_geometry(), name="pid00"
        )
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
