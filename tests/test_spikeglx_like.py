"""Unit tests for the shared ``SpikeGlxLikeFeatureCalculator`` parent.

These exercise the reader-contract logic that was lifted out of the two concrete
calculators: raw-snippet slicing, available durations, geometry defaults, and
bad-channel label resolution. A minimal concrete subclass injects fake readers so
no real ``spikeglx.Reader`` / network access is needed. A cross-subclass test then
checks that ``IBLPIDFeatureCalculator`` and ``SpikeGLXFileFeatureCalculator``
produce identical results for these shared methods.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
import scipy.fft

from ephysatlas.feature_calculators.ibl import IBLPIDFeatureCalculator
from ephysatlas.feature_calculators.spikeglx import SpikeGLXFileFeatureCalculator
from ephysatlas.feature_calculators.spikeglx_like import (
    LF_LATENCY_SAMPLES,
    SpikeGlxLikeFeatureCalculator,
)
from ephysatlas.feature_calculators.types import RawSnippet, SnippetWindow

N_CH = 4
NSYNC = 1
FS_AP = 30000.0
FS_LF = 2500.0
DUR_AP = 0.05
DUR_LF = 0.05

# Deterministic synthetic AP/LF data shaped (n_total_channels, n_samples).
AP_DATA = np.arange((N_CH + NSYNC) * 6000, dtype=np.float32).reshape(N_CH + NSYNC, 6000)
LF_DATA = (
    np.arange((N_CH + NSYNC) * 1000, dtype=np.float32).reshape(N_CH + NSYNC, 1000) * 0.5
)


def _geometry() -> dict:
    """Full geometry dict (nothing for load_geometry to fill)."""
    return {
        "x": np.array([0.0, 32.0, 0.0, 32.0]),
        "y": np.array([0.0, 0.0, 20.0, 20.0]),
        "sample_shift": np.zeros(N_CH),
        "shank": np.zeros(N_CH),
        "col": np.array([0, 1, 0, 1]),
        "row": np.array([0, 0, 1, 1]),
    }


class _FakeReader:
    """Minimal ``spikeglx.Reader``-like stub backed by an in-memory array."""

    def __init__(self, data, fs, file_bin=None, geometry=None):
        self._data = np.ascontiguousarray(data.T)  # (samples, channels)
        self.fs = float(fs)
        self.ns = int(data.shape[1])
        self.nc = int(data.shape[0])
        self.nsync = NSYNC
        self.file_bin = file_bin
        self.geometry = geometry if geometry is not None else _geometry()

    def __getitem__(self, item):
        return self._data[item]


class _ConcreteCalc(SpikeGlxLikeFeatureCalculator):
    """Smallest concrete calculator: readers are injected, metadata is trivial."""

    def __init__(self, sr_ap=None, sr_lf=None, name="test"):
        super().__init__(name=name)
        self._readers = {"ap": sr_ap, "lf": sr_lf}

    def _open_reader(self, band):
        return self._readers[band]

    def load_channel_metadata(self):
        geometry = self.load_geometry()
        return pd.DataFrame({"channel": np.arange(len(geometry["x"]))})


def _window():
    return SnippetWindow(t_start=0.0, duration_ap=DUR_AP, duration_lf=DUR_LF)


class TestSpikeGlxLikeParent(unittest.TestCase):
    def test_parent_is_abstract(self):
        # _open_reader and load_channel_metadata are abstract -> not instantiable.
        with self.assertRaises(TypeError):
            SpikeGlxLikeFeatureCalculator(name="x")

    def test_load_raw_snippet_ap_and_lf(self):
        calc = _ConcreteCalc(_FakeReader(AP_DATA, FS_AP), _FakeReader(LF_DATA, FS_LF))
        raw = calc.load_raw_snippet(_window())

        ns_ap = scipy.fft.next_fast_len(int(FS_AP * DUR_AP), real=True)
        ns_lf = scipy.fft.next_fast_len(int(FS_LF * DUR_LF), real=True)
        self.assertEqual(raw.raw_ap.shape, (N_CH, ns_ap))
        self.assertEqual(raw.raw_lf.shape, (N_CH, ns_lf))
        self.assertEqual(raw.fs_ap, FS_AP)
        self.assertEqual(raw.fs_lf, FS_LF)
        # AP starts at sample 0; LF starts LF_LATENCY_SAMPLES late.
        np.testing.assert_array_equal(raw.raw_ap, AP_DATA[:N_CH, 0:ns_ap])
        np.testing.assert_array_equal(
            raw.raw_lf, LF_DATA[:N_CH, LF_LATENCY_SAMPLES : LF_LATENCY_SAMPLES + ns_lf]
        )

    def test_load_raw_snippet_absent_band_returns_none(self):
        ap_only = _ConcreteCalc(_FakeReader(AP_DATA, FS_AP), None)
        raw = ap_only.load_raw_snippet(_window())
        self.assertIsNotNone(raw.raw_ap)
        self.assertIsNone(raw.raw_lf)
        self.assertIsNone(raw.fs_lf)

        lf_only = _ConcreteCalc(None, _FakeReader(LF_DATA, FS_LF))
        raw = lf_only.load_raw_snippet(_window())
        self.assertIsNone(raw.raw_ap)
        self.assertIsNotNone(raw.raw_lf)
        self.assertIsNone(raw.fs_ap)

    def test_available_duration(self):
        calc = _ConcreteCalc(_FakeReader(AP_DATA, FS_AP), _FakeReader(LF_DATA, FS_LF))
        max_ap, max_lf = calc.available_duration()
        self.assertAlmostEqual(max_ap, AP_DATA.shape[1] / FS_AP)
        self.assertAlmostEqual(max_lf, LF_DATA.shape[1] / FS_LF)

        ap_only = _ConcreteCalc(_FakeReader(AP_DATA, FS_AP), None)
        max_ap, max_lf = ap_only.available_duration()
        self.assertAlmostEqual(max_ap, AP_DATA.shape[1] / FS_AP)
        self.assertIsNone(max_lf)

    def test_load_geometry_fills_and_warns_on_missing_keys(self):
        reader = _FakeReader(
            AP_DATA,
            FS_AP,
            geometry={"x": np.zeros(N_CH), "y": np.arange(N_CH, dtype=float)},
        )
        calc = _ConcreteCalc(reader, None)
        with self.assertLogs(
            "ephysatlas.feature_calculators.spikeglx_like", level="WARNING"
        ) as cm:
            geometry = calc.load_geometry()
        for key in ("sample_shift", "shank", "col", "row"):
            self.assertIn(key, geometry)
            self.assertTrue(any(key in message for message in cm.output))
        np.testing.assert_array_equal(geometry["sample_shift"], np.zeros(N_CH))
        np.testing.assert_array_equal(geometry["shank"], np.zeros(N_CH))

    def test_resolve_channel_labels_explicit_wins(self):
        calc = _ConcreteCalc(_FakeReader(AP_DATA, FS_AP), None)
        raw = RawSnippet(raw_ap=AP_DATA[:N_CH], raw_lf=None, fs_ap=FS_AP, fs_lf=None)
        explicit = np.array([1, 0, 1, 0])
        labels = calc._resolve_channel_labels(
            raw, pd.DataFrame({"channel": np.arange(N_CH)}), channel_labels=explicit
        )
        np.testing.assert_array_equal(labels, explicit)

    def test_resolve_channel_labels_prefers_cbin(self):
        calc = _ConcreteCalc(
            _FakeReader(AP_DATA, FS_AP, file_bin="probe.ap.cbin"), None
        )
        raw = RawSnippet(raw_ap=AP_DATA[:N_CH], raw_lf=None, fs_ap=FS_AP, fs_lf=None)
        expected = np.array([0, 1, 0, 1])
        with mock.patch(
            "ibldsp.voltage.detect_bad_channels_cbin", return_value=expected
        ) as m:
            labels = calc._resolve_channel_labels(
                raw, pd.DataFrame({"channel": np.arange(N_CH)})
            )
        m.assert_called_once_with("probe.ap.cbin")
        np.testing.assert_array_equal(labels, expected)

    def test_resolve_channel_labels_snippet_fallback_without_file_bin(self):
        calc = _ConcreteCalc(_FakeReader(AP_DATA, FS_AP, file_bin=None), None)
        raw = RawSnippet(raw_ap=AP_DATA[:N_CH], raw_lf=None, fs_ap=FS_AP, fs_lf=None)
        with (
            mock.patch("ibldsp.voltage.detect_bad_channels_cbin") as m_cbin,
            mock.patch(
                "ibldsp.voltage.detect_bad_channels",
                return_value=(np.zeros(N_CH), None),
            ) as m_snip,
        ):
            labels = calc._resolve_channel_labels(
                raw, pd.DataFrame({"channel": np.arange(N_CH)})
            )
        m_cbin.assert_not_called()
        m_snip.assert_called_once()
        np.testing.assert_array_equal(labels, np.zeros(N_CH))

    def test_cross_subclass_identical_shared_methods(self):
        """IBL and SpikeGLX must agree on the lifted methods for identical readers."""
        sr_ap = _FakeReader(AP_DATA, FS_AP)
        sr_lf = _FakeReader(LF_DATA, FS_LF)

        ibl = IBLPIDFeatureCalculator(pid="p", one=mock.MagicMock())
        ibl._sr_ap, ibl._sr_lf = sr_ap, sr_lf
        sgx = SpikeGLXFileFeatureCalculator(ap_file="p.ap.bin", lf_file="p.lf.bin")
        sgx._sr_ap, sgx._sr_lf = sr_ap, sr_lf

        raw_ibl = ibl.load_raw_snippet(_window())
        raw_sgx = sgx.load_raw_snippet(_window())
        np.testing.assert_array_equal(raw_ibl.raw_ap, raw_sgx.raw_ap)
        np.testing.assert_array_equal(raw_ibl.raw_lf, raw_sgx.raw_lf)
        self.assertEqual(ibl.available_duration(), sgx.available_duration())

        geo_ibl, geo_sgx = ibl.load_geometry(), sgx.load_geometry()
        self.assertEqual(set(geo_ibl), set(geo_sgx))
        for key in geo_ibl:
            np.testing.assert_array_equal(geo_ibl[key], geo_sgx[key])


if __name__ == "__main__":
    unittest.main()
