"""Unit tests for multi-snippet aggregation in ``ephysatlas.feature_calculators.aggregate``."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ephysatlas.feature_calculators import (
    CsdParams,
    FeatureComputationOptions,
    FeatureParams,
    NumpyArrayFeatureCalculator,
    SnippetWindow,
    aggregate_snippets,
    check_snippet_overlap,
    compute_aggregated_features,
    get_snippet_start_times,
)
from ephysatlas.features import __features_version__

N_CH = 16
FS_LF = 250.0
DURATION = 660.0  # long enough for the default 5 min margins


def _calculator(labels: bool = True) -> NumpyArrayFeatureCalculator:
    """Small LF-only recording: 2 columns of 8 depths (the denoiser triangulates x/y), 660 s of noise.

    With ``labels=False`` the channel metadata has no bad-channel labels (and LF-only data
    has no AP to detect them from).
    """
    geometry = {
        "x": (np.arange(N_CH) % 2) * 32.0,
        "y": (np.arange(N_CH) // 2) * 20.0,
    }
    channels = pd.DataFrame(
        {
            "channel": np.arange(N_CH),
            "axial_um": geometry["y"],
            "lateral_um": geometry["x"],
            "shank": np.zeros(N_CH),
        }
    )
    if labels:
        channels["labels"] = np.zeros(N_CH)
    lf = np.random.default_rng(0).standard_normal((N_CH, int(DURATION * FS_LF)))
    return NumpyArrayFeatureCalculator(
        lf=lf.astype(np.float32),
        fs_lf=FS_LF,
        geometry=geometry,
        name="pid00",
        channel_metadata=channels,
    )


def _options(output_dir=None) -> FeatureComputationOptions:
    return FeatureComputationOptions(
        features_to_compute=["lf", "csd"],
        skip_lf_destripe=True,
        feature_params=FeatureParams(csd=CsdParams(decimate=1, denoise=False)),
        include_trajectory=False,
        output_dir=output_dir,
    )


def _windows(t_starts, duration=2.0):
    return [SnippetWindow(t, duration, duration) for t in t_starts]


class TestGetSnippetStartTimes(unittest.TestCase):
    def test_times_are_sorted_and_inside_usable_range(self):
        for method in ("stratified", "linspace"):
            t = get_snippet_start_times(100.0, 5, 5.0, margin=10.0, method=method)
            self.assertEqual(len(t), 5)
            np.testing.assert_array_equal(t, np.floor(t))  # whole seconds
            np.testing.assert_array_equal(t, np.sort(t))
            self.assertTrue(np.all(t >= 10.0))
            self.assertTrue(np.all(t <= 100.0 - 10.0 - 5.0))

    def test_default_margin_is_five_minutes(self):
        t = get_snippet_start_times(1000.0, 5, 5.0)
        self.assertTrue(np.all(t >= 300.0))
        self.assertTrue(np.all(t <= 1000.0 - 300.0 - 5.0))

    def test_stratified_is_reproducible_with_seed(self):
        t1 = get_snippet_start_times(1000.0, 5, 5.0, seed=3)
        t2 = get_snippet_start_times(1000.0, 5, 5.0, seed=3)
        np.testing.assert_array_equal(t1, t2)

    def test_too_short_recording_raises(self):
        with self.assertRaises(ValueError):
            get_snippet_start_times(20.0, 2, 5.0, margin=10.0)

    def test_overlap_raises_or_warns(self):
        # 5 snippets of 5 s in a 10 s range must overlap
        with self.assertRaises(ValueError):
            get_snippet_start_times(15.0, 5, 5.0, margin=0.0, method="linspace")
        with self.assertLogs("ephysatlas.feature_calculators.aggregate", "WARNING"):
            get_snippet_start_times(
                15.0, 5, 5.0, margin=0.0, method="linspace", on_overlap="warn"
            )


class TestCheckSnippetOverlap(unittest.TestCase):
    def test_disjoint_windows_pass(self):
        check_snippet_overlap(_windows([0.0, 2.0, 4.0]))

    def test_overlap_uses_longest_band(self):
        windows = [SnippetWindow(0.0, 1.0, 3.0), SnippetWindow(2.0, 1.0, 1.0)]
        with self.assertRaises(ValueError):
            check_snippet_overlap(windows)


class TestComputeAggregatedFeatures(unittest.TestCase):
    def test_one_row_per_channel_and_no_files_left(self):
        windows = _windows([1, 7, 13])
        # Point tempfile at a private directory, so that files other processes write to
        # the system temp dir meanwhile (e.g. parallel test workers) cannot fail the check.
        with (
            tempfile.TemporaryDirectory() as private_tmp,
            mock.patch.object(tempfile, "tempdir", private_tmp),
        ):
            df = compute_aggregated_features(_calculator(), windows, _options())
            self.assertEqual(list(Path(private_tmp).iterdir()), [])
        self.assertEqual(df.index.names, ["pid", "channel"])
        self.assertEqual(len(df), N_CH)
        self.assertEqual(df.index.get_level_values("pid").unique().tolist(), ["pid00"])
        self.assertIn("rms_lf", df.columns)
        # the windows used and the code versions are kept with the features
        self.assertEqual(
            df.attrs["snippet_windows"],
            [
                {"pid": "pid00", "t_start": t, "duration_ap": 2.0, "duration_lf": 2.0}
                for t in (1.0, 7.0, 13.0)
            ],
        )
        self.assertEqual(df.attrs["features_version"], __features_version__)
        self.assertIn("ibleatools_distribution_version", df.attrs)

    def test_output_dir_keeps_snippet_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = compute_aggregated_features(
                _calculator(), _windows([1, 7, 13]), _options(Path(tmp))
            )
            self.assertEqual(len(list((Path(tmp) / "pid00").glob("probe_*"))), 3)
            # the snippet windows and versions are saved in the parquet file with the features
            df.to_parquet(Path(tmp) / "features.pqt")
            saved = pd.read_parquet(Path(tmp) / "features.pqt")
            self.assertEqual(saved.attrs, df.attrs)

    def test_one_snippet(self):
        df = _calculator().compute_denoised_features(
            n_snippets=1, duration_ap=2.0, duration_lf=2.0, options=_options()
        )
        self.assertEqual(len(df), N_CH)
        self.assertEqual(len(df.attrs["snippet_windows"]), 1)

    def test_missing_channel_labels_warn_and_are_good(self):
        with self.assertLogs("ephysatlas.aggregation", "WARNING"):
            df = compute_aggregated_features(
                _calculator(labels=False), _windows([1, 7, 13]), _options()
            )
        self.assertEqual(len(df), N_CH)
        self.assertTrue((df["channel_labels"] == 0).all())

    def test_aggregate_snippets_matches_compute(self):
        calc, windows = _calculator(), _windows([1, 7, 13])
        with tempfile.TemporaryDirectory() as tmp:
            results = [calc.compute_snippet(w, _options(Path(tmp))) for w in windows]
            df = aggregate_snippets(results)
        expected = compute_aggregated_features(calc, windows, _options())
        pd.testing.assert_frame_equal(df, expected, check_like=True)
        self.assertEqual(df.attrs["snippet_windows"], expected.attrs["snippet_windows"])

    def test_overlapping_windows_raise(self):
        with self.assertRaises(ValueError):
            compute_aggregated_features(_calculator(), _windows([1.0, 2.0]), _options())

    def test_base_method_matches_explicit_path(self):
        calc = _calculator()
        df = calc.compute_denoised_features(
            n_snippets=3, duration_ap=2.0, duration_lf=2.0, options=_options()
        )
        t_starts = get_snippet_start_times(DURATION, 3, 2.0)
        expected = compute_aggregated_features(calc, _windows(t_starts), _options())
        # aggregate_raw_features orders columns through a set, so only the content is compared
        pd.testing.assert_frame_equal(df, expected, check_like=True)


if __name__ == "__main__":
    unittest.main()
