"""Unit waveform features and the channel positions they need.

Synthetic waveforms with known landmarks check the feature definitions (the channel-level ones of
``ephysatlas.features.ModelSpikeShapeFeatures``) and the ibldsp spatial spread; small synthetic
cluster/waveform tables check how the probe layout and each unit's channel positions are
recovered. No IBL data is needed.

In its own file with a setUpModule guard: the unit package pulls in torch, which segfaults on macOS
arm64 if xgboost is already imported in the same process.
"""

import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import neuropixel
import numpy as np
import pandas as pd

FS = 30_000.0
T = 128
PEAK, TROUGH, TIP = 42, 54, 34


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


def _bump(center, width=2.0):
    t = np.arange(T)
    return np.exp(-0.5 * ((t - center) / width) ** 2)


# A negative spike: tip +0.1 at 34, peak -1 at 42, trough (rebound) +0.3 at 54.
SHAPE = 0.1 * _bump(TIP) - _bump(PEAK) + 0.3 * _bump(TROUGH, width=4.0)


def _unit(xy, peak_channel=0, decay_um=40.0):
    """A [C, T] waveform whose amplitude decays with distance from ``peak_channel``."""
    dist = np.linalg.norm(xy - xy[peak_channel], axis=1)
    amplitude = np.exp(-dist / decay_um)
    return (amplitude[:, None] * SHAPE[None, :]).astype(np.float32), amplitude, dist


class TestFeatureNames(unittest.TestCase):
    def test_are_the_channel_waveform_features_defined_for_a_mean_waveform(self):
        from ephysatlas.spatial_encoder.utils import WAVEFORM_FEATURES
        from ephysatlas.unit_level_encoder.waveform_features import FEATURE_NAMES

        from tests._model_fixtures import UNIT_FEATURES

        self.assertEqual(list(FEATURE_NAMES), UNIT_FEATURES)
        self.assertEqual(FEATURE_NAMES[-1], "polarity")
        continuous = [f for f in FEATURE_NAMES if f != "polarity"]
        self.assertEqual(continuous, [f for f in WAVEFORM_FEATURES if f in continuous])
        for name in (
            "peak_to_trough_ratio_log",
            "spike_width_secs",
            "spatial_spread_um",
        ):
            self.assertIn(name, FEATURE_NAMES)


class TestExtractFeatures(unittest.TestCase):
    def setUp(self):
        header = neuropixel.trace_header(version=1)
        self.xy = np.c_[header["x"][100:108], header["y"][100:108]].astype(np.float64)

    def _extract(self, waveforms, xy):
        from ephysatlas.unit_level_encoder.waveform_features import (
            FEATURE_NAMES,
            extract_generated_waveform_features,
        )

        features, names, report = extract_generated_waveform_features(
            waveforms, xy, sampling_rate_hz=FS, return_report=True
        )
        self.assertEqual(tuple(names), FEATURE_NAMES)
        return dict(zip(names, features.T)), report

    def test_landmark_features(self):
        w, _, _ = _unit(self.xy, peak_channel=3)
        f, report = self._extract(w[None], self.xy)
        self.assertEqual(report["n_fallback"], 0)
        np.testing.assert_allclose(
            f["spike_width_secs"], (TROUGH - PEAK) / FS, rtol=1e-5
        )
        np.testing.assert_allclose(
            f["predepolarisation_width_secs"], (PEAK - TIP) / FS, rtol=1e-5
        )
        peak_val, trough_val = SHAPE[PEAK], SHAPE[TROUGH]
        np.testing.assert_allclose(
            f["spike_amplitude"], trough_val - peak_val, rtol=1e-4
        )
        np.testing.assert_allclose(
            f["peak_to_trough_ratio_log"], np.log(abs(peak_val / trough_val)), rtol=1e-4
        )
        np.testing.assert_array_equal(f["polarity"], -1.0)

    def test_spatial_spread_is_the_amplitude_weighted_distance_from_the_peak_channel(
        self,
    ):
        w, amplitude, dist = _unit(self.xy, peak_channel=3)
        f, _ = self._extract(w[None], self.xy)
        expected = np.sum(amplitude * dist) / np.sum(amplitude)
        np.testing.assert_allclose(f["spatial_spread_um"], expected, rtol=1e-4)

    def test_padding_channels_are_ignored(self):
        w, amplitude, dist = _unit(self.xy, peak_channel=3)
        w[-2:] = 0.0
        xy = self.xy.copy()
        xy[-2:] = np.nan
        f, _ = self._extract(w[None], xy)
        expected = np.sum(amplitude[:-2] * dist[:-2]) / np.sum(amplitude[:-2])
        np.testing.assert_allclose(f["spatial_spread_um"], expected, rtol=1e-4)

    def test_a_shared_layout_equals_per_unit_positions(self):
        w = np.stack([_unit(self.xy, peak_channel=c)[0] for c in (1, 3, 6)])
        shared, _ = self._extract(w, self.xy)
        per_unit, _ = self._extract(w, np.broadcast_to(self.xy, (3,) + self.xy.shape))
        for name in shared:
            np.testing.assert_array_equal(shared[name], per_unit[name])

    def test_fallback_follows_the_ibldsp_conventions(self):
        from ephysatlas.unit_level_encoder.waveform_features import (
            _fallback_one,
            extract_generated_waveform_features,
        )

        w, _, _ = _unit(self.xy, peak_channel=3)
        exact, _ = extract_generated_waveform_features(
            w[None], self.xy, sampling_rate_hz=FS
        )
        np.testing.assert_allclose(
            _fallback_one(w, self.xy, FS), exact[0], rtol=1e-4, atol=1e-7
        )

    def test_positions_must_match_the_waveforms(self):
        w, _, _ = _unit(self.xy)
        with self.assertRaises(ValueError):
            self._extract(w[None], self.xy[:-1])


class TestModalChannelLayout(unittest.TestCase):
    def test_most_common_layout_relative_to_its_first_channel(self):
        from ephysatlas.unit_level_encoder.waveform_features import modal_channel_layout

        a = np.array([[43.0, 20.0], [11.0, 20.0], [59.0, 40.0]])
        b = np.array([[11.0, 20.0], [59.0, 40.0], [27.0, 40.0]])
        padded = a.copy()
        padded[0] = np.nan
        layouts = np.stack([a + [0, 100], b, a + [0, 400], padded, padded])
        np.testing.assert_allclose(modal_channel_layout(layouts), a - a[0])
        np.testing.assert_allclose(
            modal_channel_layout(layouts, mask=[0, 1, 0, 0, 0]), b - b[0]
        )


class TestProbeLayouts(unittest.TestCase):
    def setUp(self):
        self.np1 = neuropixel.trace_header(version=1)
        self.np2 = neuropixel.trace_header(version=2)

    def _clusters(self, pid, header, channels, offset=(0.0, 0.0)):
        channels = np.asarray(channels)
        return pd.DataFrame(
            {
                "pid": pid,
                "channels": channels,
                "lateral_um": header["x"][channels] + offset[0],
                "axial_um": header["y"][channels] + offset[1],
            }
        )

    def test_each_insertion_gets_the_channel_map_its_clusters_fit(self):
        from ephysatlas.unit_level_encoder.prepare_data import probe_channel_xy_um

        df = pd.concat(
            [
                self._clusters("np1", self.np1, [0, 1, 5, 200]),
                # IBL tables place NP2 channels at x = 0 / 32 um: a constant offset.
                self._clusters("np2", self.np2, [0, 1, 7, 300], offset=(-27.0, 0.0)),
            ]
        )
        layouts = probe_channel_xy_um(df, "pid")
        np.testing.assert_array_equal(layouts["np1"][:, 0], self.np1["x"])
        np.testing.assert_array_equal(layouts["np2"][:, 1], self.np2["y"])

    def test_an_exact_match_wins_when_both_maps_fit(self):
        from ephysatlas.unit_level_encoder.prepare_data import probe_channel_xy_um

        # A single cluster fits either map up to an offset; only NP2 places it exactly.
        df = self._clusters("one", self.np2, [1])
        layouts = probe_channel_xy_um(df, "pid")
        np.testing.assert_array_equal(layouts["one"][:, 1], self.np2["y"])

    def test_an_unknown_probe_is_refused(self):
        from ephysatlas.unit_level_encoder.prepare_data import probe_channel_xy_um

        df = self._clusters("odd", self.np1, [0, 1, 2])
        df.loc[2, "axial_um"] += 7.0
        with self.assertRaisesRegex(RuntimeError, "odd"):
            probe_channel_xy_um(df, "pid")


class TestUnitChannelPositions(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_positions_replay_the_waveform_channel_selection(self):
        from ephysatlas.unit_level_encoder.prepare_data import _multichannel_channel_xy

        header = neuropixel.trace_header(version=1)
        layout = np.c_[header["x"], header["y"]].astype(np.float64)
        wide = [11, 10, 14, 12, 13, 15]  # six channels, stored out of order
        narrow = [101, 100]  # two channels: centre-padded
        table = pd.DataFrame(
            {
                "pid": ["p"] * len(wide) + ["p"] * len(narrow),
                "cluster_id": [4] * len(wide) + [9] * len(narrow),
                "abs_channel": wide + narrow,
            }
        )
        path = self.tmp / "waveforms.table.pqt"
        table.to_parquet(path)
        units = pd.DataFrame({"pid": ["p", "p"], "cluster_id": [9, 4]})

        xy = _multichannel_channel_xy(
            waveforms_table_path=path,
            df_units=units,
            pid_col="pid",
            target_channels=4,
            layouts={"p": layout},
        )
        self.assertEqual(xy.shape, (2, 4, 2))
        np.testing.assert_array_equal(xy[1], layout[[11, 12, 13, 14]])
        self.assertTrue(np.isnan(xy[0, [0, 3]]).all())
        np.testing.assert_array_equal(xy[0, 1:3], layout[[100, 101]])


class TestReferenceFeatures(unittest.TestCase):
    def test_features_derive_from_the_cluster_table_landmarks(self):
        from ephysatlas.unit_level_encoder.prepare_data import (
            WAVEFORM_FEATURE_NAMES,
            _extract_reference_waveform_features,
        )

        df = pd.DataFrame(
            {
                "depolarisation_slope": [1.0, 2.0],
                "recovery_slope": [3.0, np.nan],
                "repolarisation_slope": [4.0, 5.0],
                "tip_val": [0.1, 0.2],
                "peak_val": [-2.0, 1.0],
                "trough_val": [0.5, -0.25],
                "peak_time_idx": [42, 42],
                "trough_time_idx": [54, 51],
                "tip_time_idx": [30, 36],
                "invert_sign_peak": [1.0, -1.0],
            }
        )
        out, names, sources, missing = _extract_reference_waveform_features(
            df, sampling_rate_hz=FS
        )
        f = dict(zip(names, out.T))
        self.assertEqual(names, list(WAVEFORM_FEATURE_NAMES))
        np.testing.assert_allclose(f["spike_width_secs"], [12 / FS, 9 / FS], rtol=1e-6)
        np.testing.assert_allclose(
            f["predepolarisation_width_secs"], [12 / FS, 6 / FS], rtol=1e-6
        )
        np.testing.assert_allclose(f["spike_amplitude"], [2.5, -1.25])
        np.testing.assert_allclose(
            f["peak_to_trough_ratio_log"], np.log([4.0, 4.0]), rtol=1e-6
        )
        np.testing.assert_array_equal(f["polarity"], [-1.0, 1.0])
        # Left for the caller to compute from the waveforms.
        self.assertTrue(np.isnan(f["spatial_spread_um"]).all())
        self.assertEqual(missing, {"recovery_slope": 1, "spatial_spread_um": 2})


if __name__ == "__main__":
    unittest.main()
