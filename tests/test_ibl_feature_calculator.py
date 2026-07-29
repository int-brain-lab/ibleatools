"""Unit tests for the IBL OOP feature calculator (``IBLPIDFeatureCalculator``).

These exercise the calculator internals that back ``compute_features_from_pid``:
bad-channel label resolution (cbin vs snippet fallback), geometry-default
warnings, and the channel-metadata merge fan-out guard. No network or heavy DSP.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ephysatlas.feature_calculators.ibl import IBLPIDFeatureCalculator
from ephysatlas.feature_calculators.types import (
    FeatureComputationOptions,
    RawSnippet,
)

PID = "test_pid"
N_CH = 4
NSYNC = 1
FS_AP = 30000.0

# Deterministic synthetic AP data shaped (n_total_channels, n_samples).
AP_DATA = np.arange((N_CH + NSYNC) * 4000, dtype=np.float32).reshape(N_CH + NSYNC, 4000)


# NP2.0 four-shank meta-data; the first imroTbl entry has reference id 0 -> "external".
_META_NP2013 = {
    "imDatPrb_pn": "NP2013",
    "imDatPrb_type": 2013.0,
    "imroTbl": "(2013,384)(0 0 0 0 0)(1 0 0 0 1)(2 0 0 0 2)(3 0 0 0 3)",
}


def _geometry() -> dict:
    """Reader geometry with every key ``load_geometry`` would otherwise setdefault."""
    return {
        "x": np.array([0.0, 32.0, 0.0, 32.0]),
        "y": np.array([0.0, 0.0, 20.0, 20.0]),
        "sample_shift": np.zeros(N_CH),
        "shank": np.zeros(N_CH),
        "col": np.array([0, 1, 0, 1]),
        "row": np.array([0, 0, 1, 1]),
    }


class _FakeReader:
    """Minimal SpikeGLX-reader stand-in backed by an in-memory array."""

    def __init__(self, data: np.ndarray, fs: float, file_bin=None) -> None:
        self._data = np.ascontiguousarray(data.T)  # (samples, channels)
        self.fs = float(fs)
        self.ns = int(data.shape[1])
        self.nc = int(data.shape[0])
        self.nsync = NSYNC
        self.file_bin = file_bin
        self.geometry = _geometry()

    def __getitem__(self, item):
        return self._data[item]


class TestIBLFeatureCalculator(unittest.TestCase):
    def test_resolve_channel_labels_prefers_cbin(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        calc._sr_ap = _FakeReader(AP_DATA, FS_AP, file_bin="probe.ap.cbin")
        raw = RawSnippet(raw_ap=AP_DATA[:N_CH], raw_lf=None, fs_ap=FS_AP, fs_lf=None)
        channels = pd.DataFrame({"channel": np.arange(N_CH)})  # no stored labels
        expected = np.array([0, 1, 0, 1])
        with mock.patch(
            "ibldsp.voltage.detect_bad_channels_cbin", return_value=expected
        ) as m:
            labels = calc._resolve_channel_labels(raw, channels)
        m.assert_called_once_with("probe.ap.cbin")
        np.testing.assert_array_equal(labels, expected)

    def test_resolve_channel_labels_snippet_fallback_without_file_bin(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        calc._sr_ap = _FakeReader(AP_DATA, FS_AP, file_bin=None)
        raw = RawSnippet(raw_ap=AP_DATA[:N_CH], raw_lf=None, fs_ap=FS_AP, fs_lf=None)
        channels = pd.DataFrame({"channel": np.arange(N_CH)})  # no stored labels
        with (
            mock.patch("ibldsp.voltage.detect_bad_channels_cbin") as m_cbin,
            mock.patch(
                "ibldsp.voltage.detect_bad_channels",
                return_value=(np.zeros(N_CH), None),
            ) as m_snip,
        ):
            labels = calc._resolve_channel_labels(raw, channels)
        m_cbin.assert_not_called()
        m_snip.assert_called_once()
        np.testing.assert_array_equal(labels, np.zeros(N_CH))

    def test_load_geometry_warns_on_missing_key(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        reader = _FakeReader(AP_DATA, FS_AP)
        reader.geometry = {"x": np.zeros(N_CH), "y": np.arange(N_CH, dtype=float)}
        calc._sr_ap = reader
        with self.assertLogs(
            "ephysatlas.feature_calculators.spikeglx_like", level="WARNING"
        ) as cm:
            geometry = calc.load_geometry()
        self.assertTrue(any("sample_shift" in message for message in cm.output))
        np.testing.assert_array_equal(geometry["sample_shift"], np.zeros(N_CH))

    def test_merge_channel_metadata_matches_on_physical_site_not_index(self):
        # The crux of oliche's review: metadata must land on the physical site
        # (axial_um, lateral_um, shank), never on channel/rawInd. Here the metadata
        # rows are permuted and rawInd is deliberately unrelated to the positional
        # channel, so a positional/rawInd join would mis-align.
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        features = pd.DataFrame(
            {
                "channel": [0, 1, 2, 3],
                "axial_um": [0.0, 0.0, 20.0, 20.0],
                "lateral_um": [0.0, 32.0, 0.0, 32.0],
                "shank": [0, 0, 0, 0],
                "feat": [10.0, 11.0, 12.0, 13.0],
            }
        )
        # Rows in a different order; sites: (20,32),(0,32),(0,0),(20,0).
        channels = pd.DataFrame(
            {
                "axial_um": [20.0, 0.0, 0.0, 20.0],
                "lateral_um": [32.0, 32.0, 0.0, 0.0],
                "shank": [0, 0, 0, 0],
                "rawInd": [30, 31, 32, 33],
                "region": ["d", "b", "a", "c"],
            }
        )
        merged = calc._merge_channel_metadata(features, channels)
        # Row order and feature values come from the feature table unchanged.
        np.testing.assert_array_equal(merged["channel"], [0, 1, 2, 3])
        np.testing.assert_array_equal(merged["feat"], [10.0, 11.0, 12.0, 13.0])
        # Metadata lands by physical site, not by index: ch0=(0,0)->a, ch1=(0,32)->b,
        # ch2=(20,0)->c, ch3=(20,32)->d; rawInd is carried but was NOT the join key.
        self.assertEqual(list(merged["region"]), ["a", "b", "c", "d"])
        np.testing.assert_array_equal(merged["rawInd"], [32, 31, 33, 30])

    def test_merge_channel_metadata_tolerates_float_jitter(self):
        # Coordinates from the two sources may differ by sub-micron float jitter;
        # the rounded site key must still match them.
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        features = pd.DataFrame(
            {
                "channel": [0, 1],
                "axial_um": [0.0, 20.0],
                "lateral_um": [0.0, 0.0],
                "shank": [0, 0],
                "feat": [1.0, 2.0],
            }
        )
        channels = pd.DataFrame(
            {
                "axial_um": [0.3, 19.8],
                "lateral_um": [0.1, -0.2],
                "shank": [0, 0],
                "rawInd": [7, 8],
                "region": ["a", "b"],
            }
        )
        merged = calc._merge_channel_metadata(features, channels)
        self.assertEqual(list(merged["region"]), ["a", "b"])
        np.testing.assert_array_equal(merged["rawInd"], [7, 8])

    def test_merge_channel_metadata_raises_on_duplicate_site(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        features = pd.DataFrame(
            {
                "channel": [0, 1],
                "axial_um": [0.0, 20.0],
                "lateral_um": [0.0, 0.0],
                "shank": [0, 0],
                "feat": [1.0, 2.0],
            }
        )
        # Two metadata rows share the physical site (0, 0, 0) -> fans out the merge.
        channels = pd.DataFrame(
            {
                "axial_um": [0.0, 0.0, 20.0],
                "lateral_um": [0.0, 0.0, 0.0],
                "shank": [0, 0, 0],
                "rawInd": [0, 1, 2],
            }
        )
        with self.assertRaises(ValueError):
            calc._merge_channel_metadata(features, channels)

    def test_attach_physical_coordinates_handles_waveforms_subset(self):
        # features.spikes() returns one row per spiking channel in groupby order --
        # a reordered subset. Coordinates must be looked up by channel value.
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        geometry = _geometry()  # 4 channels
        features = pd.DataFrame({"channel": [3, 1, 2], "spike_count": [5, 7, 9]})
        stamped = calc._attach_physical_coordinates(features, geometry)
        np.testing.assert_array_equal(stamped["axial_um"], geometry["y"][[3, 1, 2]])
        np.testing.assert_array_equal(stamped["lateral_um"], geometry["x"][[3, 1, 2]])
        np.testing.assert_array_equal(stamped["shank"], [0, 0, 0])

    def test_attach_physical_coordinates_rejects_out_of_range_channel(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        features = pd.DataFrame({"channel": [0, 99]})  # 99 exceeds geometry size
        with self.assertRaises(ValueError):
            calc._attach_physical_coordinates(features, _geometry())

    def test_attach_physical_coordinates_raises_without_channel(self):
        # A missing 'channel' column means the engine contract is broken; fail loud
        # rather than silently dropping all channel metadata downstream.
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        features = pd.DataFrame({"feat": [1.0, 2.0]})  # no 'channel' column
        with self.assertRaises(ValueError):
            calc._attach_physical_coordinates(features, _geometry())

    def test_enrich_adds_probe_metadata(self):
        # The streamed IBL reader carries SpikeGLX meta-data too, so channels.pqt
        # from this source must gain the same two columns as the file-based source.
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        reader = _FakeReader(AP_DATA, FS_AP)
        reader.meta = _META_NP2013
        calc._sr_ap = reader
        out = calc.enrich_channel_metadata(
            pd.DataFrame({"channel": np.arange(N_CH)}),
            FeatureComputationOptions(include_trajectory=False),
        )
        self.assertTrue((out["probe_model"] == "NP2013").all())
        self.assertTrue((out["referencing_scheme"] == "external").all())

    def test_enrich_probe_metadata_na_without_reader_meta(self):
        # _FakeReader has no ``meta`` attribute at all, standing in for a reader
        # opened without SpikeGLX meta-data. This also pins the "never open a
        # reader just to read meta-data" contract: no LF reader is set, so a
        # fallback through the ``sr_lf`` property would construct a real
        # SpikeSortingLoader against the mock ONE and blow up.
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        calc._sr_ap = _FakeReader(AP_DATA, FS_AP)
        out = calc.enrich_channel_metadata(
            pd.DataFrame({"channel": np.arange(N_CH)}),
            FeatureComputationOptions(include_trajectory=False),
        )
        self.assertTrue(out["probe_model"].isna().all())
        self.assertTrue(out["referencing_scheme"].isna().all())


if __name__ == "__main__":
    unittest.main()
