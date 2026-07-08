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
from ephysatlas.feature_calculators.types import RawSnippet

PID = "test_pid"
N_CH = 4
NSYNC = 1
FS_AP = 30000.0

# Deterministic synthetic AP data shaped (n_total_channels, n_samples).
AP_DATA = np.arange((N_CH + NSYNC) * 4000, dtype=np.float32).reshape(N_CH + NSYNC, 4000)


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
            "ephysatlas.feature_calculators.ibl", level="WARNING"
        ) as cm:
            geometry = calc.load_geometry()
        self.assertTrue(any("sample_shift" in message for message in cm.output))
        np.testing.assert_array_equal(geometry["sample_shift"], np.zeros(N_CH))

    def test_merge_channel_metadata_raises_on_duplicate_channel(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        features = pd.DataFrame({"channel": [0, 1], "feat": [1.0, 2.0]})
        channels = pd.DataFrame(
            {"channel": [0, 0, 1], "axial_um": [0.0, 0.0, 20.0]}
        )  # duplicate channel 0 would fan out the left merge
        with self.assertRaises(ValueError):
            calc._merge_channel_metadata(features, channels)


if __name__ == "__main__":
    unittest.main()
