"""Unit tests for the shared ``SpikeInterfaceFeatureCalculator`` parent.

These exercise the SpikeInterface ``BaseRecording`` reader-contract logic: raw
snippet slicing (with microvolt->volt conversion and the LF latency offset),
available durations, geometry construction from channel locations, and channel
metadata. A minimal concrete subclass injects fake recordings so no real
SpikeInterface / network access is needed.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
import scipy.fft

from ephysatlas.feature_calculators.spikeglx_like import LF_LATENCY_SAMPLES
from ephysatlas.feature_calculators.spikeinterface_like import (
    SpikeInterfaceFeatureCalculator,
)
from ephysatlas.feature_calculators.types import (
    FeatureComputationOptions,
    SnippetWindow,
)

N_CH = 4
FS_AP = 30000.0
FS_LF = 2500.0
DUR_AP = 0.05
DUR_LF = 0.05

# SpikeInterface exposes traces in microvolts, shaped (samples, channels). We keep
# the fakes' data in microvolts, shaped (channels, samples) for convenience, and
# transpose on read; expected volts are the microvolt slice * 1e-6.
AP_UV = np.arange(N_CH * 6000, dtype=np.float32).reshape(N_CH, 6000)
LF_UV = np.arange(N_CH * 1000, dtype=np.float32).reshape(N_CH, 1000) * 0.5


def _locations() -> np.ndarray:
    """(n_channels, 2) x/y site coordinates in micrometres."""
    return np.array([[0.0, 0.0], [32.0, 0.0], [0.0, 20.0], [32.0, 20.0]], dtype=float)


# Sentinel so ``locations=None`` can explicitly mean "no channel locations"
# (get_channel_locations raises), distinct from "not passed" (use the default).
_NO_ARG = object()


class _FakeRecording:
    """Minimal SpikeInterface ``BaseRecording``-like stub."""

    def __init__(
        self, uv, fs, locations=_NO_ARG, sample_shift=None, properties=None, groups=None
    ):
        self._uv = np.ascontiguousarray(uv)  # (channels, samples) in microvolts
        self._fs = float(fs)
        # _NO_ARG -> default geometry; None -> get_channel_locations raises (like
        # SpikeInterface when locations are unset); array -> use it.
        self._locations = _locations() if locations is _NO_ARG else locations
        self._sample_shift = sample_shift
        self._properties = properties or {}
        self._groups = (
            np.zeros(uv.shape[0], int) if groups is None else np.asarray(groups)
        )

    def get_channel_groups(self):
        return self._groups

    def get_sampling_frequency(self):
        return self._fs

    def get_num_frames(self):
        return int(self._uv.shape[1])

    def get_channel_locations(self):
        if self._locations is None:
            raise Exception("There are no channel locations")
        return self._locations

    def get_property_keys(self):
        keys = list(self._properties)
        if self._sample_shift is not None:
            keys.append("inter_sample_shift")
        return keys

    def get_property(self, key):
        if key == "inter_sample_shift":
            return self._sample_shift
        return self._properties.get(key)

    def get_traces(self, start_frame, end_frame, return_in_uV=False):
        traces = self._uv[:, int(start_frame) : int(end_frame)].T  # (samples, chans)
        return traces if return_in_uV else traces * 1e-6


class _ConcreteCalc(SpikeInterfaceFeatureCalculator):
    """Smallest concrete calculator: recordings are injected."""

    def __init__(self, rec_ap=None, rec_lf=None, name="test"):
        super().__init__(name=name)
        self._recs = {"ap": rec_ap, "lf": rec_lf}

    def _open_recording(self, band):
        return self._recs[band]


def _window():
    return SnippetWindow(t_start=0.0, duration_ap=DUR_AP, duration_lf=DUR_LF)


class TestSpikeInterfaceLikeParent(unittest.TestCase):
    def test_parent_is_abstract(self):
        # _open_recording is abstract -> the parent is not instantiable.
        with self.assertRaises(TypeError):
            SpikeInterfaceFeatureCalculator(name="x")

    def test_load_raw_snippet_ap_and_lf(self):
        calc = _ConcreteCalc(_FakeRecording(AP_UV, FS_AP), _FakeRecording(LF_UV, FS_LF))
        raw = calc.load_raw_snippet(_window())

        ns_ap = scipy.fft.next_fast_len(int(FS_AP * DUR_AP), real=True)
        ns_lf = scipy.fft.next_fast_len(int(FS_LF * DUR_LF), real=True)
        self.assertEqual(raw.raw_ap.shape, (N_CH, ns_ap))
        self.assertEqual(raw.raw_lf.shape, (N_CH, ns_lf))
        self.assertEqual(raw.fs_ap, FS_AP)
        self.assertEqual(raw.fs_lf, FS_LF)
        # Microvolts are converted to volts; AP starts at 0, LF starts late.
        np.testing.assert_allclose(raw.raw_ap, AP_UV[:, 0:ns_ap] * 1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            raw.raw_lf,
            LF_UV[:, LF_LATENCY_SAMPLES : LF_LATENCY_SAMPLES + ns_lf] * 1e-6,
            rtol=1e-6,
        )

    def test_load_raw_snippet_absent_band_returns_none(self):
        ap_only = _ConcreteCalc(_FakeRecording(AP_UV, FS_AP), None)
        raw = ap_only.load_raw_snippet(_window())
        self.assertIsNotNone(raw.raw_ap)
        self.assertIsNone(raw.raw_lf)
        self.assertIsNone(raw.fs_lf)

        lf_only = _ConcreteCalc(None, _FakeRecording(LF_UV, FS_LF))
        raw = lf_only.load_raw_snippet(_window())
        self.assertIsNone(raw.raw_ap)
        self.assertIsNotNone(raw.raw_lf)
        self.assertIsNone(raw.fs_ap)

    def test_available_duration(self):
        calc = _ConcreteCalc(_FakeRecording(AP_UV, FS_AP), _FakeRecording(LF_UV, FS_LF))
        max_ap, max_lf = calc.available_duration()
        self.assertAlmostEqual(max_ap, AP_UV.shape[1] / FS_AP)
        self.assertAlmostEqual(max_lf, LF_UV.shape[1] / FS_LF)

        ap_only = _ConcreteCalc(_FakeRecording(AP_UV, FS_AP), None)
        max_ap, max_lf = ap_only.available_duration()
        self.assertAlmostEqual(max_ap, AP_UV.shape[1] / FS_AP)
        self.assertIsNone(max_lf)

    def test_load_geometry_builds_from_locations(self):
        shift = np.array([0.0, 0.1, 0.2, 0.3])
        calc = _ConcreteCalc(_FakeRecording(AP_UV, FS_AP, sample_shift=shift), None)
        geometry = calc.load_geometry()
        for key in ("x", "y", "col", "row", "sample_shift", "shank"):
            self.assertIn(key, geometry)
        np.testing.assert_array_equal(geometry["x"], _locations()[:, 0])
        np.testing.assert_array_equal(geometry["y"], _locations()[:, 1])
        np.testing.assert_array_equal(geometry["sample_shift"], shift)
        np.testing.assert_array_equal(geometry["shank"], np.zeros(N_CH))

    def test_load_geometry_warns_and_zeros_missing_sample_shift(self):
        calc = _ConcreteCalc(_FakeRecording(AP_UV, FS_AP, sample_shift=None), None)
        with self.assertLogs(
            "ephysatlas.feature_calculators.spikeinterface_like", level="WARNING"
        ) as cm:
            geometry = calc.load_geometry()
        self.assertTrue(any("inter_sample_shift" in m for m in cm.output))
        np.testing.assert_array_equal(geometry["sample_shift"], np.zeros(N_CH))

    def test_load_geometry_falls_back_to_rel_x_rel_y(self):
        # No SpikeInterface channel locations -> fall back to rel_x/rel_y.
        props = {
            "rel_x": np.array([0.0, 32.0, 0.0, 32.0]),
            "rel_y": np.array([0.0, 0.0, 20.0, 20.0]),
        }
        calc = _ConcreteCalc(
            _FakeRecording(AP_UV, FS_AP, locations=None, properties=props), None
        )
        geometry = calc.load_geometry()
        np.testing.assert_array_equal(geometry["x"], props["rel_x"])
        np.testing.assert_array_equal(geometry["y"], props["rel_y"])

    def test_load_geometry_falls_back_to_probe_position(self):
        # Allen convention: probe_horizontal_position / probe_vertical_position.
        props = {
            "probe_horizontal_position": np.array([59, 59, 59, 59]),
            "probe_vertical_position": np.array([40, 80, 120, 160]),
        }
        calc = _ConcreteCalc(
            _FakeRecording(AP_UV, FS_AP, locations=None, properties=props), None
        )
        geometry = calc.load_geometry()
        np.testing.assert_array_equal(geometry["x"], props["probe_horizontal_position"])
        np.testing.assert_array_equal(geometry["y"], props["probe_vertical_position"])

    def test_load_geometry_raises_without_locations_or_properties(self):
        calc = _ConcreteCalc(
            _FakeRecording(AP_UV, FS_AP, locations=None, properties={}), None
        )
        with self.assertRaises(ValueError):
            calc.load_geometry()

    def test_load_geometry_rejects_multi_shank(self):
        # >1 channel group => multiple shanks => not implemented yet.
        rec = _FakeRecording(AP_UV, FS_AP, groups=np.array([0, 0, 1, 1]))
        calc = _ConcreteCalc(rec, None)
        with self.assertRaises(NotImplementedError):
            calc.load_geometry()

    def test_channel_labels_option_is_forwarded(self):
        # An explicit options.channel_labels must override automatic resolution
        # and reach compute_features_from_raw verbatim.
        calc = _ConcreteCalc(_FakeRecording(AP_UV, FS_AP), _FakeRecording(LF_UV, FS_LF))
        labels = np.array([1, 0, 1, 0])
        with mock.patch(
            "ephysatlas.feature_calculators.base.compute_features_from_raw",
            return_value=pd.DataFrame({"channel": np.arange(N_CH)}),
        ) as m:
            calc.compute_snippet(
                _window(),
                FeatureComputationOptions(
                    features_to_compute=["lf"], channel_labels=labels
                ),
            )
        np.testing.assert_array_equal(m.call_args.kwargs["channel_labels"], labels)

    def test_load_channel_metadata_from_geometry(self):
        calc = _ConcreteCalc(_FakeRecording(AP_UV, FS_AP), None)
        metadata = calc.load_channel_metadata()
        self.assertEqual(
            list(metadata.columns),
            ["channel", "rawInd", "axial_um", "lateral_um", "shank"],
        )
        np.testing.assert_array_equal(metadata["channel"], np.arange(N_CH))
        np.testing.assert_array_equal(metadata["axial_um"], _locations()[:, 1])
        np.testing.assert_array_equal(metadata["lateral_um"], _locations()[:, 0])


if __name__ == "__main__":
    unittest.main()
