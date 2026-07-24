"""Unit tests for ``SpikeInterfaceZarrFeatureCalculator``.

The reader-contract behavior (snippet/geometry/durations) is covered in
``test_spikeinterface_like.py``; here we test only this class's overrides: requiring
at least one store, name defaulting, band absence, and that ``_open_recording``
forwards the path + ``storage_options`` to ``read_zarr``. No real zarr is opened.
"""

from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

from ephysatlas.feature_calculators import SpikeInterfaceZarrFeatureCalculator


class TestSpikeInterfaceZarr(unittest.TestCase):
    def test_requires_at_least_one_store(self):
        with self.assertRaises(ValueError):
            SpikeInterfaceZarrFeatureCalculator()

    def test_name_defaults_to_store_stem(self):
        calc = SpikeInterfaceZarrFeatureCalculator(lf_zarr="/data/sub_ProbeD-LFP.zarr")
        self.assertEqual(calc.name, "sub_ProbeD-LFP")

    def test_open_recording_none_for_absent_band(self):
        # lf-only: the AP band must resolve to None without importing spikeinterface.
        calc = SpikeInterfaceZarrFeatureCalculator(lf_zarr="/data/x-LFP.zarr")
        self.assertIsNone(calc._open_recording("ap"))
        self.assertIsNone(calc.rec_ap)

    @unittest.skipUnless(
        importlib.util.find_spec("spikeinterface"), "spikeinterface not installed"
    )
    def test_open_recording_forwards_path_and_storage_options(self):
        calc = SpikeInterfaceZarrFeatureCalculator(
            ap_zarr="s3://bucket/x-AP.zarr", storage_options={"anon": True}
        )
        with mock.patch("spikeinterface.core.read_zarr", return_value="REC") as m:
            self.assertEqual(calc.rec_ap, "REC")
        m.assert_called_once_with(
            "s3://bucket/x-AP.zarr", storage_options={"anon": True}
        )


if __name__ == "__main__":
    unittest.main()
