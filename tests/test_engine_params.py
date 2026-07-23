"""Tests for the PR1 engine enhancements.

Covers the four in-place engine changes, exercised with small fixtures / synthetic
data per the testing convention:

1. The ``scale`` flag on :func:`ephysatlas.features.csd` is forwarded to both
   ``current_source_density`` calls (the ``n=2`` and ``n=1`` paths).
2. The duck-typed ``feature_params`` plumbing in
   :func:`ephysatlas.feature_computation.compute_features_from_raw` reproduces today's
   defaults, and gates the new ``rms_lf_no_car`` LF feature behind
   ``feature_params.lf.compute_rms_no_car`` (default off).
3. The aggregation channel merge coerces the ``channel`` key to ``Int64`` and threads a
   presence-filtered ``distance_to_tip_um`` through as channel metadata, without letting
   it enter the median groupby or the denoise path.
"""

import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import neuropixel
import ibldsp.voltage

from ephysatlas import features
from ephysatlas.feature_computation import compute_features_from_raw
from ephysatlas.aggregation import get_aggregated_features_per_pid
from ephysatlas.feature_calculators import (
    CsdParams,
    FeatureComputationOptions,
    FeatureParams,
    WaveformParams,
)

FIXTURE_PATH = Path(features.__file__).parents[2].joinpath("tests", "fixtures")


def _lf_fixture():
    """Load the small destriped-LF fixture used across the test-suite."""
    return np.load(FIXTURE_PATH / "lf_destriped.npy").astype(np.float32)


def _linear_geometry(n_channels):
    """Simple linear NP1-like geometry (mirrors tests/test_feature_computation.py)."""
    return {
        "x": np.zeros(n_channels),
        "y": np.arange(n_channels) * 20,
        "sample_shift": np.zeros(n_channels),
        "shank": np.zeros(n_channels),
    }


def _feature_params(compute_rms_no_car=False, scale=True):
    """Minimal duck-typed FeatureParams stand-in.

    PR1 reads feature_params via getattr and must NOT import the feature_calculators
    package (avoids a circular import), so a SimpleNamespace is a faithful stand-in for
    the typed dataclasses that land in PR2.
    """
    lf = SimpleNamespace(
        bands=None, decay_features=True, compute_rms_no_car=compute_rms_no_car
    )
    csd = SimpleNamespace(bands=None, decimate=10, scale=scale)
    return SimpleNamespace(lf=lf, csd=csd, ap=None, waveforms=None)


class TestCsdScale(unittest.TestCase):
    """The new ``scale`` flag on features.csd."""

    def test_scale_default_is_true(self):
        # Default must stay True so existing behavior is unchanged.
        self.assertIs(inspect.signature(features.csd).parameters["scale"].default, True)

    def test_scale_forwarded_to_both_csd_calls(self):
        data = _lf_fixture()
        geometry = neuropixel.trace_header(version=1)
        with patch(
            "ibldsp.voltage.current_source_density",
            wraps=ibldsp.voltage.current_source_density,
        ) as m:
            features.csd(data, fs=2500, geometry=geometry, scale=False)
        # csd computes both the n=2 and n=1 CSD paths.
        self.assertEqual(m.call_count, 2)
        for call in m.call_args_list:
            self.assertIs(call.kwargs.get("scale"), False)


class TestTypedFeatureParams(unittest.TestCase):
    """The typed FeatureParams dataclasses drive the engine like the stub."""

    def setUp(self):
        self.lf = _lf_fixture()
        self.geometry = _linear_geometry(self.lf.shape[0])

    def _compute(self, features_to_compute, feature_params, geometry=None):
        with tempfile.TemporaryDirectory() as tmp:
            return compute_features_from_raw(
                raw_ap=None,
                raw_lf=self.lf,
                fs_lf=2500.0,
                geometry=self.geometry if geometry is None else geometry,
                neuropixel_version=1,
                features_to_compute=features_to_compute,
                output_dir=Path(tmp),
                feature_params=feature_params,
            )

    def test_csd_scale_forwarded_via_typed_feature_params(self):
        # CsdParams(scale=False) must reach both current_source_density calls. CSD
        # needs full geometry (col/row), so use the NP1 trace header like TestCsdScale.
        with patch(
            "ibldsp.voltage.current_source_density",
            wraps=ibldsp.voltage.current_source_density,
        ) as m:
            self._compute(
                ["csd"],
                FeatureParams(csd=CsdParams(scale=False)),
                geometry=neuropixel.trace_header(version=1),
            )
        self.assertEqual(m.call_count, 2)
        for call in m.call_args_list:
            self.assertIs(call.kwargs.get("scale"), False)

    def test_default_typed_params_match_none(self):
        # FeatureParams() (all sub-configs None) must reproduce feature_params=None.
        df_none = self._compute(["lf"], None)
        df_default = self._compute(["lf"], FeatureParams())
        self.assertEqual(set(df_none.columns), set(df_default.columns))
        np.testing.assert_allclose(
            df_none["rms_lf"].to_numpy(), df_default["rms_lf"].to_numpy()
        )


class TestFeatureParamsDict(unittest.TestCase):
    """Nested-dict convenience form is normalized to typed FeatureParams."""

    def test_from_dict_builds_typed_params(self):
        fp = FeatureParams.from_dict(
            {"csd": {"scale": False}, "lf": {"decay_features": False}}
        )
        self.assertIsInstance(fp.csd, CsdParams)
        self.assertIs(fp.csd.scale, False)
        self.assertIs(fp.lf.decay_features, False)

    def test_options_normalizes_dict_feature_params(self):
        opts = FeatureComputationOptions(feature_params={"csd": {"scale": False}})
        self.assertIsInstance(opts.feature_params, FeatureParams)
        self.assertIs(opts.feature_params.csd.scale, False)

    def test_options_leaves_typed_feature_params_unchanged(self):
        fp = FeatureParams(csd=CsdParams(scale=False))
        opts = FeatureComputationOptions(feature_params=fp)
        self.assertIs(opts.feature_params, fp)

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            FeatureParams.from_dict({"bogus": {}})

    def test_unknown_subparam_raises(self):
        # A mistyped sub-parameter must fail loudly, not be silently ignored.
        with self.assertRaises(TypeError):
            FeatureParams.from_dict({"csd": {"scal": False}})


class TestFeatureParamsLf(unittest.TestCase):
    """feature_params plumbing and the gated rms_lf_no_car LF feature."""

    def setUp(self):
        self.lf = _lf_fixture()
        self.geometry = _linear_geometry(self.lf.shape[0])

    def _compute_lf(self, feature_params):
        """Run only the lf family so the test stays fast (no AP/waveforms)."""
        with tempfile.TemporaryDirectory() as tmp:
            return compute_features_from_raw(
                raw_ap=None,
                raw_lf=self.lf,
                fs_lf=2500.0,
                geometry=self.geometry,
                neuropixel_version=1,
                features_to_compute=["lf"],
                output_dir=Path(tmp),
                feature_params=feature_params,
            )

    def test_default_params_reproduce_and_omit_rms_no_car(self):
        # feature_params=None and a default-equivalent stub must yield identical lf output,
        # and neither may carry rms_lf_no_car (the toggle defaults off).
        df_none = self._compute_lf(None)
        df_default = self._compute_lf(_feature_params(compute_rms_no_car=False))
        self.assertNotIn("rms_lf_no_car", df_none.columns)
        self.assertNotIn("rms_lf_no_car", df_default.columns)
        self.assertEqual(set(df_none.columns), set(df_default.columns))
        np.testing.assert_allclose(
            df_none["rms_lf"].to_numpy(), df_default["rms_lf"].to_numpy()
        )

    def test_rms_lf_no_car_enabled_by_flag(self):
        df = self._compute_lf(_feature_params(compute_rms_no_car=True))
        self.assertIn("rms_lf_no_car", df.columns)
        self.assertEqual(len(df), self.lf.shape[0])
        self.assertTrue(np.isfinite(df["rms_lf_no_car"].to_numpy()).all())


class TestAggregationChannelMerge(unittest.TestCase):
    """Int64 coercion and distance_to_tip_um pass-through in get_aggregated_features_per_pid."""

    def _make_pid_dir(self, tmp, with_distance=True):
        """Build a minimal on-disk PID layout and return its snippet DataFrame.

        Layout::

            <tmp>/channels.pqt          channel metadata (+ distance_to_tip_um)
            <tmp>/snippet_001/raw_features.pqt   per-channel raw features
        """
        probe_dir = Path(tmp)
        snippet_dir = probe_dir / "snippet_001"
        snippet_dir.mkdir(parents=True, exist_ok=True)
        # alpha_mean/alpha_std are required by outlier_treatment inside the aggregator;
        # all four columns live in ModelRawFeatures so they survive the median groupby.
        pd.DataFrame(
            {
                "channel": [0, 1, 2],
                "rms_lf": [10.0, 11.0, 12.0],
                "alpha_mean": [1.0, 2.0, 3.0],
                "alpha_std": [0.1, 0.2, 0.3],
            }
        ).to_parquet(snippet_dir / "raw_features.pqt")

        chan = {
            "channel": [0, 1, 2],
            "axial_um": [0.0, 20.0, 40.0],
            "lateral_um": [0.0, 0.0, 0.0],
        }
        if with_distance:
            chan["distance_to_tip_um"] = [104.5, 124.5, 144.5]
        pd.DataFrame(chan).to_parquet(probe_dir / "channels.pqt")

        return pd.DataFrame(
            {
                "pid": ["p1"],
                "base_level_dir": [str(probe_dir)],
                "snippet_level_dir": ["snippet_001"],
            }
        )

    def test_channel_coerced_to_int64_and_distance_passed_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            snippet_df = self._make_pid_dir(tmp, with_distance=True)
            out = get_aggregated_features_per_pid(snippet_df)
        # The defensive coercion makes the merged channel key nullable Int64.
        self.assertEqual(str(out["channel"].dtype), "Int64")
        self.assertIn("distance_to_tip_um", out.columns)
        # One row per channel, no duplication, and distance aligned by channel.
        self.assertEqual(len(out), 3)
        out_sorted = out.sort_values("channel")
        np.testing.assert_allclose(
            out_sorted["distance_to_tip_um"].to_numpy(dtype=float),
            [104.5, 124.5, 144.5],
        )

    def test_distance_absent_when_channels_lacks_it(self):
        # Presence-filtered: no distance_to_tip_um column when channels.pqt lacks it.
        with tempfile.TemporaryDirectory() as tmp:
            snippet_df = self._make_pid_dir(tmp, with_distance=False)
            out = get_aggregated_features_per_pid(snippet_df)
        self.assertNotIn("distance_to_tip_um", out.columns)

    def test_distance_to_tip_um_bypasses_median_and_denoise(self):
        # The design invariant that removes the need for a skip-list: distance_to_tip_um
        # is channel metadata (ChannelDataFrameSchema), NOT a raw feature, so it is
        # excluded from both the median groupby (keyed on ModelRawFeatures) and the
        # denoise set (voltage_features_set). rms_lf_no_car, by contrast, IS a real LF
        # feature and appears in both.
        raw_cols = set(features.ModelRawFeatures.to_schema().columns.keys())
        denoise_cols = set(
            features.voltage_features_set(
                ["raw_ap", "raw_lf", "raw_lf_csd", "waveforms"]
            )
        )
        chan_cols = set(features.ChannelDataFrameSchema.to_schema().columns.keys())
        self.assertNotIn("distance_to_tip_um", raw_cols)
        self.assertNotIn("distance_to_tip_um", denoise_cols)
        self.assertIn("distance_to_tip_um", chan_cols)
        self.assertIn("rms_lf_no_car", raw_cols)
        self.assertIn("rms_lf_no_car", denoise_cols)


class TestWaveformNjobs(unittest.TestCase):
    """Configurable dartsort ``n_jobs`` via FeatureParams.waveforms."""

    def test_from_dict_waveforms_njobs(self):
        fp = FeatureParams.from_dict({"waveforms": {"n_jobs": 1}})
        self.assertEqual(fp.waveforms.n_jobs, 1)
        self.assertEqual(WaveformParams().n_jobs, 0)  # default preserved

    @unittest.skipUnless(importlib.util.find_spec("dartsort"), "dartsort not installed")
    def test_njobs_forwarded_to_dartsort_subtract(self):
        # DartParameters.n_jobs must reach dartsort.subtract (previously the passed
        # params object was silently dropped, so no dartsort param was applied).
        import dartsort

        from ephysatlas.features import DartParameters

        captured = {}

        def fake_subtract(*args, **kwargs):
            captured["n_jobs"] = kwargs.get("n_jobs")
            raise RuntimeError("__stop__")  # short-circuit before real subtraction

        data = np.random.RandomState(0).randn(4, 3000).astype("float32")
        geometry = {
            "x": np.array([0.0, 32.0, 0.0, 32.0]),
            "y": np.array([0.0, 0.0, 20.0, 20.0]),
        }
        scratch = tempfile.mkdtemp(prefix="dart_njobs_")
        for n_jobs in (0, 1):
            captured.clear()
            with patch.object(dartsort, "subtract", fake_subtract):
                with self.assertRaises(RuntimeError):
                    features.dart_subtraction_numpy(
                        data,
                        30000.0,
                        geometry,
                        params=DartParameters(n_jobs=n_jobs),
                        scratch_dir=scratch,
                    )
            self.assertEqual(captured["n_jobs"], n_jobs)


if __name__ == "__main__":
    unittest.main()
