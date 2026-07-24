"""Unit tests for ``NwbFeatureCalculator`` and its ``NwbSource`` resolver.

The reader-contract behavior is covered in ``test_spikeinterface_like.py``; here we
test only NWB-specific concerns: requiring at least one source, name defaulting,
band resolution, and how ``NwbSource`` funnels local / remote / DANDI acquisition
into SpikeInterface. The optional heavy dependencies (spikeinterface, dandi) are
faked via ``sys.modules`` so these tests run without a ``[full]`` install.
"""

from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

from ephysatlas.feature_calculators.nwb import NwbFeatureCalculator, NwbSource


@contextmanager
def _fake_optional_modules(extractor=None, dandi_client=None):
    """Inject fake ``spikeinterface.extractors`` / ``dandi.dandiapi`` modules.

    ``NwbSource`` imports these lazily inside ``to_recording``/``_resolve_url``, so
    replacing them in ``sys.modules`` lets the resolution logic be tested without
    the real (optional) packages installed.
    """
    modules = {}
    if extractor is not None:
        si = types.ModuleType("spikeinterface")
        si_ext = types.ModuleType("spikeinterface.extractors")
        si_ext.NwbRecordingExtractor = extractor
        si.extractors = si_ext
        modules["spikeinterface"] = si
        modules["spikeinterface.extractors"] = si_ext
    if dandi_client is not None:
        dandi = types.ModuleType("dandi")
        dandi_api = types.ModuleType("dandi.dandiapi")
        dandi_api.DandiAPIClient = dandi_client
        dandi.dandiapi = dandi_api
        modules["dandi"] = dandi
        modules["dandi.dandiapi"] = dandi_api
    with mock.patch.dict(sys.modules, modules):
        yield


def _fake_dandi_client(s3_url):
    """Build a DandiAPIClient mock whose asset resolves to ``s3_url``."""
    client = mock.MagicMock()
    ctx = client.return_value.__enter__.return_value
    asset = ctx.get_dandiset.return_value.get_asset_by_path.return_value
    asset.get_content_url.return_value = s3_url
    return client


class TestNwbFeatureCalculator(unittest.TestCase):
    def test_requires_at_least_one_source(self):
        with self.assertRaises(ValueError):
            NwbFeatureCalculator()

    def test_open_recording_none_when_source_absent(self):
        # No lf_source -> _open_recording("lf") returns None without importing SI.
        calc = NwbFeatureCalculator(ap_source=NwbSource.local("probe.nwb"))
        self.assertIsNone(calc._open_recording("lf"))
        self.assertIsNone(calc.rec_lf)

    def test_name_defaults_to_local_stem(self):
        calc = NwbFeatureCalculator(ap_source=NwbSource.local("/data/probe00.nwb"))
        self.assertEqual(calc.name, "probe00")

    def test_name_defaults_to_dandi_filepath_stem(self):
        calc = NwbFeatureCalculator.from_dandi(
            "000004", ap_filepath="sub-P11/sub-P11_ses-x_ecephys.nwb"
        )
        self.assertEqual(calc.name, "sub-P11_ses-x_ecephys")


class TestNwbSourceResolution(unittest.TestCase):
    def test_local_to_recording_passes_no_stream_mode(self):
        extractor = mock.MagicMock(return_value="REC")
        source = NwbSource.local(
            "probe.nwb", electrical_series="acquisition/ElectricalSeriesAp"
        )
        with _fake_optional_modules(extractor=extractor):
            rec = source.to_recording()
        self.assertEqual(rec, "REC")
        _, kwargs = extractor.call_args
        self.assertEqual(kwargs["file_path"], "probe.nwb")
        self.assertEqual(
            kwargs["electrical_series_path"], "acquisition/ElectricalSeriesAp"
        )
        self.assertNotIn("stream_mode", kwargs)  # local reads are not streamed

    def test_remote_to_recording_uses_stream_mode(self):
        extractor = mock.MagicMock(return_value="REC")
        source = NwbSource.remote("https://s3/probe.nwb")
        with _fake_optional_modules(extractor=extractor):
            source.to_recording()
        _, kwargs = extractor.call_args
        self.assertEqual(kwargs["file_path"], "https://s3/probe.nwb")
        self.assertEqual(kwargs["stream_mode"], "remfile")

    def test_remote_zarr_defaults_to_zarr_stream_mode(self):
        extractor = mock.MagicMock(return_value="REC")
        # backend auto-detected from the .zarr extension, stream mode not overridden.
        source = NwbSource.remote("https://s3/probe.nwb.zarr", stream_mode=None)
        with _fake_optional_modules(extractor=extractor):
            source.to_recording()
        _, kwargs = extractor.call_args
        self.assertEqual(kwargs["stream_mode"], "zarr")

    def test_dandi_resolves_asset_url_then_streams(self):
        extractor = mock.MagicMock(return_value="REC")
        s3_url = "https://dandiarchive.s3/probe.nwb"
        client = _fake_dandi_client(s3_url)
        source = NwbSource.dandi(
            "000004", "sub-P11/sub-P11_ecephys.nwb", version="draft"
        )
        with _fake_optional_modules(extractor=extractor, dandi_client=client):
            source.to_recording()
        # The DANDI asset was resolved and handed to the extractor as file_path.
        ctx = client.return_value.__enter__.return_value
        ctx.get_dandiset.assert_called_once_with("000004", "draft")
        ctx.get_dandiset.return_value.get_asset_by_path.assert_called_once_with(
            "sub-P11/sub-P11_ecephys.nwb"
        )
        _, kwargs = extractor.call_args
        self.assertEqual(kwargs["file_path"], s3_url)
        self.assertEqual(kwargs["stream_mode"], "remfile")

    def test_from_dandi_single_file_two_series_builds_two_sources(self):
        calc = NwbFeatureCalculator.from_dandi(
            "000004",
            ap_filepath="sub-x/sub-x_ecephys.nwb",
            lf_filepath="sub-x/sub-x_ecephys.nwb",
            ap_electrical_series="acquisition/ElectricalSeriesAp",
            lf_electrical_series="acquisition/ElectricalSeriesLf",
        )
        self.assertEqual(
            calc.ap_source.electrical_series, "acquisition/ElectricalSeriesAp"
        )
        self.assertEqual(
            calc.lf_source.electrical_series, "acquisition/ElectricalSeriesLf"
        )
        self.assertEqual(calc.ap_source.filepath, calc.lf_source.filepath)

    def test_import_error_points_to_full_extra(self):
        # With no fake modules injected and the real packages absent, to_recording
        # raises a helpful ImportError. Skip if spikeinterface is actually installed.
        import importlib.util

        if importlib.util.find_spec("spikeinterface") is not None:
            self.skipTest("spikeinterface installed; ImportError path not exercised")
        with self.assertRaises(ImportError) as cm:
            NwbSource.local("probe.nwb").to_recording()
        self.assertIn("ibleatools[full]", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
