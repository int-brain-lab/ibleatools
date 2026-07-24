"""Unit tests for ``SpikeGLXFileFeatureCalculator`` source-specific overrides.

The reader-contract behavior is covered in ``test_spikeglx_like.py``; here we test
only what this class overrides: file-based reader opening, geometry-derived channel
metadata, and ``traj_dict``-based trajectory enrichment. No real files are opened.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ephysatlas.feature_calculators.spikeglx import SpikeGLXFileFeatureCalculator
from ephysatlas.feature_calculators.types import FeatureComputationOptions

N_CH = 4
NSYNC = 1
FS_AP = 30000.0


def _geometry() -> dict:
    return {
        "x": np.array([0.0, 32.0, 0.0, 32.0]),
        "y": np.array([0.0, 0.0, 20.0, 20.0]),
        "sample_shift": np.zeros(N_CH),
        "shank": np.zeros(N_CH),
        "col": np.array([0, 1, 0, 1]),
        "row": np.array([0, 0, 1, 1]),
    }


_META_NP2013 = {
    "imDatPrb_pn": "NP2013",
    "imDatPrb_type": 2013.0,
    "imroTbl": "(2013,384)(0 0 0 0 0)(1 0 0 0 1)(2 0 0 0 2)(3 0 0 0 3)",
}


class _FakeReader:
    """Minimal ``spikeglx.Reader``-like stub."""

    def __init__(self, geometry=None, meta=None):
        self.fs = FS_AP
        self.ns = 6000
        self.nc = N_CH + NSYNC
        self.nsync = NSYNC
        self.file_bin = None
        self.geometry = geometry if geometry is not None else _geometry()
        self.meta = meta


def _channels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": np.arange(N_CH),
            "rawInd": np.arange(N_CH),
            "axial_um": _geometry()["y"].copy(),
            "lateral_um": _geometry()["x"].copy(),
        }
    )


class TestSpikeGLXFileFeatureCalculator(unittest.TestCase):
    def test_requires_at_least_one_file(self):
        with self.assertRaises(ValueError):
            SpikeGLXFileFeatureCalculator()

    def test_open_reader_none_when_file_absent(self):
        calc = SpikeGLXFileFeatureCalculator(ap_file="probe.ap.bin")
        # lf_file is None -> no LF reader, and no filesystem access.
        self.assertIsNone(calc._open_reader("lf"))
        self.assertIsNone(calc.sr_lf)

    def test_name_defaults_to_file_stem(self):
        calc = SpikeGLXFileFeatureCalculator(ap_file="/data/probe00.ap.cbin")
        self.assertEqual(calc.name, "probe00.ap")

    def test_load_channel_metadata_from_geometry(self):
        calc = SpikeGLXFileFeatureCalculator(ap_file="probe.ap.bin")
        calc._sr_ap = _FakeReader()
        metadata = calc.load_channel_metadata()
        self.assertEqual(
            list(metadata.columns),
            ["channel", "rawInd", "axial_um", "lateral_um", "shank"],
        )
        np.testing.assert_array_equal(metadata["channel"], np.arange(N_CH))
        np.testing.assert_array_equal(metadata["axial_um"], _geometry()["y"])
        np.testing.assert_array_equal(metadata["lateral_um"], _geometry()["x"])
        np.testing.assert_array_equal(metadata["shank"], _geometry()["shank"])

    def test_enrich_skipped_when_not_included(self):
        calc = SpikeGLXFileFeatureCalculator(ap_file="probe.ap.bin", traj_dict=None)
        calc._sr_ap = _FakeReader(meta=_META_NP2013)
        channels = _channels()
        out = calc.enrich_channel_metadata(
            channels, FeatureComputationOptions(include_trajectory=False)
        )
        self.assertIs(out, channels)
        self.assertEqual((out["probe_model"] == "NP2013").all(), True)
        self.assertEqual((out["referencing_scheme"] == "external").all(), True)

    def test_enrich_returns_unchanged_when_traj_missing_and_not_required(self):
        calc = SpikeGLXFileFeatureCalculator(ap_file="probe.ap.bin", traj_dict=None)
        calc._sr_ap = _FakeReader(meta=_META_NP2013)
        channels = _channels()
        out = calc.enrich_channel_metadata(
            channels,
            FeatureComputationOptions(
                include_trajectory=True, require_trajectory=False
            ),
        )
        self.assertIs(out, channels)

    def test_enrich_raises_when_traj_required_but_missing(self):
        calc = SpikeGLXFileFeatureCalculator(ap_file="probe.ap.bin", traj_dict=None)
        calc._sr_ap = _FakeReader(meta=_META_NP2013)
        with self.assertRaises(ValueError):
            calc.enrich_channel_metadata(
                _channels(),
                FeatureComputationOptions(
                    include_trajectory=True, require_trajectory=True
                ),
            )

    def test_enrich_with_traj_dict_calls_add_target_coordinates(self):
        traj = {"x": 0.0, "y": 0.0, "z": 0.0, "depth": 0.0, "theta": 0.0, "phi": 0.0}
        calc = SpikeGLXFileFeatureCalculator(ap_file="probe.ap.bin", traj_dict=traj)
        calc._sr_ap = _FakeReader(meta=_META_NP2013)
        channels = _channels()

        def _fake_add(channels=None, traj_dict=None):
            enriched = dict(channels)
            n = len(enriched["channel"])
            enriched["x_target"] = np.arange(n, dtype=float)
            enriched["y_target"] = np.arange(n, dtype=float)
            enriched["z_target"] = np.arange(n, dtype=float)
            return enriched

        with mock.patch(
            "ephysatlas.feature_calculators.spikeglx.add_target_coordinates",
            side_effect=_fake_add,
        ) as m:
            out = calc.enrich_channel_metadata(
                channels, FeatureComputationOptions(include_trajectory=True)
            )
        self.assertEqual(m.call_args.kwargs["traj_dict"], traj)
        for col in ("x_target", "y_target", "z_target"):
            self.assertIn(col, out.columns)

    def test_enrich_probe_metadata_none_without_reader_meta(self):
        calc = SpikeGLXFileFeatureCalculator(ap_file="probe.ap.bin", traj_dict=None)
        calc._sr_ap = _FakeReader(meta=None)
        out = calc.enrich_channel_metadata(
            _channels(), FeatureComputationOptions(include_trajectory=False)
        )
        self.assertTrue(out["probe_model"].isna().all())
        self.assertTrue(out["referencing_scheme"].isna().all())


if __name__ == "__main__":
    unittest.main()
