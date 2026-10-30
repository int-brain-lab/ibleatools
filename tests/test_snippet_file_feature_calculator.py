"""Tests for SnippetFileFeatureCalculator, which reads files written by extract_snippets."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pandas as pd
from neuropixel import NP2Converter

from ephysatlas.feature_calculators import SpikeGLXFileFeatureCalculator
from ephysatlas.feature_calculators.snippet_file import (
    SnippetFileFeatureCalculator,
    derive_lf,
)
from ephysatlas.feature_calculators.types import (
    FeatureComputationOptions,
    SnippetWindow,
)
from ephysatlas.snippet_extractor import extract_snippets
from tests._snippet_fixtures import write_spikeglx_recording


class TestSnippetFileFeatureCalculator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.np1_ap = write_spikeglx_recording(cls.tmp / "np1", "np1", 1.0)
        cls.np1_lf = cls.np1_ap.with_name(cls.np1_ap.name.replace(".ap.", ".lf."))
        cls.np2_ap = write_spikeglx_recording(
            cls.tmp / "np2", "np24_shank", 1.0, with_lf=False
        )
        cls.np1_nolf_ap = write_spikeglx_recording(
            cls.tmp / "np1_nolf", "np1", 1.0, with_lf=False
        )
        cls.np1_h5 = extract_snippets(
            cls.np1_ap, cls.tmp / "np1.h5", t_starts=[0.1, 0.5], duration=0.1
        )
        cls.np1_nolf_h5 = extract_snippets(
            cls.np1_nolf_ap, cls.tmp / "np1_nolf.h5", t_starts=[0.1, 0.5], duration=0.1
        )
        cls.np2_h5 = extract_snippets(
            cls.np2_ap, cls.tmp / "np2.h5", t_starts=[0.1, 0.5], duration=0.1
        )

        cls.np1_3snip_h5 = extract_snippets(
            cls.np1_ap, cls.tmp / "np1_3snip.h5", t_starts=[0.1, 0.4, 0.7], duration=0.1
        )
        cls.np1_traj_h5 = extract_snippets(
            cls.np1_ap,
            cls.tmp / "np1_traj.h5",
            t_starts=[0.1],
            duration=0.1,
            trajectory=dict(x=1.0, y=2.0, z=3.0, depth=4.0, theta=5.0, phi=6.0),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_ap_and_lf_match_spikeglx_calculator(self):
        calc = SnippetFileFeatureCalculator(self.np1_h5)
        ref = SpikeGLXFileFeatureCalculator(ap_file=self.np1_ap, lf_file=self.np1_lf)
        for window in calc.windows:
            ours, theirs = calc.load_raw_snippet(window), ref.load_raw_snippet(window)
            np.testing.assert_array_equal(ours.raw_ap, theirs.raw_ap)
            np.testing.assert_array_equal(ours.raw_lf, theirs.raw_lf)
            self.assertEqual((ours.fs_ap, ours.fs_lf), (theirs.fs_ap, theirs.fs_lf))

    def test_unknown_format_version_raises(self):
        copy = self.tmp / "v2.h5"
        shutil.copy(self.np1_h5, copy)
        with h5py.File(copy, "r+") as f:
            f.attrs["format_version"] = 2
        with self.assertRaisesRegex(ValueError, "v2.h5 has format version 2"):
            SnippetFileFeatureCalculator(copy)

    def test_windows_name_and_version(self):
        calc = SnippetFileFeatureCalculator(self.np1_h5)
        self.assertEqual(
            calc.windows, [SnippetWindow(0.1, 0.1, 0.1), SnippetWindow(0.5, 0.1, 0.1)]
        )
        self.assertEqual(calc.name, self.np1_h5.stem)
        self.assertEqual(calc.neuropixel_version, 1)
        self.assertEqual(
            SnippetFileFeatureCalculator(self.np2_h5).neuropixel_version, 2
        )

    def test_windows_not_in_file_raise(self):
        calc = SnippetFileFeatureCalculator(self.np1_h5)
        for window in (SnippetWindow(0.2, 0.1, 0.1), SnippetWindow(0.1, 0.1, 25.0)):
            with self.assertRaisesRegex(ValueError, "0.1"):  # lists the stored windows
                calc.load_raw_snippet(window)

    def test_geometry_and_channel_metadata(self):
        calc = SnippetFileFeatureCalculator(self.np1_h5)
        geometry = calc.load_geometry()
        channels = calc.load_channel_metadata()
        self.assertEqual(
            list(channels.columns),
            ["channel", "rawInd", "axial_um", "lateral_um", "shank"],
        )
        np.testing.assert_array_equal(channels["rawInd"], geometry["ind"])
        np.testing.assert_array_equal(channels["axial_um"], geometry["y"])
        given = pd.DataFrame({"channel": [0]})
        self.assertIs(
            SnippetFileFeatureCalculator(
                self.np1_h5, channel_metadata=given
            ).load_channel_metadata(),
            given,
        )

    def test_shank_selection(self):
        multi = self.tmp / "multi.h5"
        shutil.copy(self.np2_h5, multi)
        with h5py.File(multi, "r+") as f:
            f["geometry/shank"][...] = np.arange(96) % 2  # pretend two shanks
        with self.assertRaisesRegex(ValueError, "shank"):
            SnippetFileFeatureCalculator(multi)
        calc = SnippetFileFeatureCalculator(multi, shank=1)
        full = (
            SnippetFileFeatureCalculator(self.np2_h5)
            .load_raw_snippet(calc.windows[0])
            .raw_ap
        )
        np.testing.assert_array_equal(
            calc.load_raw_snippet(calc.windows[0]).raw_ap, full[1::2]
        )
        self.assertEqual(len(calc.load_geometry()["x"]), 48)
        self.assertTrue(calc.name.endswith("_shank1"))

    def test_exported_from_package(self):
        from ephysatlas.feature_calculators import (
            SnippetFileFeatureCalculator as exported,
        )

        self.assertIs(exported, SnippetFileFeatureCalculator)

    def test_trajectory_and_no_lf(self):
        traj = dict(x=1.0, y=2.0, z=3.0, depth=4.0, theta=5.0, phi=6.0)
        out = extract_snippets(
            self.np2_ap,
            self.tmp / "traj.h5",
            t_starts=[0.1],
            duration=0.1,
            trajectory=traj,
        )
        calc = SnippetFileFeatureCalculator(out)
        self.assertEqual(calc.trajectory, traj)
        self.assertIsNone(SnippetFileFeatureCalculator(self.np2_h5).trajectory)
        self.assertEqual(calc.load_raw_snippet(calc.windows[0]).raw_lf.shape, (96, 250))
        self.assertEqual(calc.available_duration(), (0.2, 0.2))
        self.assertEqual(
            SnippetFileFeatureCalculator(self.np1_h5).available_duration(), (0.6, 0.6)
        )
        self.assertEqual(
            SnippetFileFeatureCalculator(self.np1_nolf_h5).available_duration(),
            (0.6, None),
        )

    def test_derive_lf_matches_np2converter(self):
        conv = NP2Converter(self.np2_ap, compress=False)
        x = np.random.default_rng(0).standard_normal((96, 30000)) * 1e-4
        ours, fs_lf = derive_lf(x.copy(), 30000.0)
        theirs = conv.extract_lfp(
            x.copy()
        )  # tapers its input in place, hence the copies
        self.assertEqual(fs_lf, 2500.0)
        np.testing.assert_allclose(
            ours[:, 50:-50], theirs[:, 50:-50], rtol=1e-5, atol=1e-9
        )

    def test_np2_file_derives_lf(self):
        calc = SnippetFileFeatureCalculator(self.np2_h5)
        raw = calc.load_raw_snippet(calc.windows[0])
        expected, fs_lf = derive_lf(raw.raw_ap, raw.fs_ap)
        np.testing.assert_array_equal(raw.raw_lf, expected)
        self.assertEqual(raw.fs_lf, fs_lf)
        self.assertEqual(raw.raw_lf.shape, (96, 250))

    def test_np1_without_lf_returns_none_and_never_derives(self):
        calc = SnippetFileFeatureCalculator(self.np1_nolf_h5)
        target = "ephysatlas.feature_calculators.snippet_file"
        with mock.patch(f"{target}.derive_lf") as m:
            with self.assertLogs(target, level="WARNING") as logs:
                for window in calc.windows:
                    raw = calc.load_raw_snippet(window)
                    self.assertIsNone(raw.raw_lf)
        m.assert_not_called()
        self.assertEqual(len(logs.records), 1)  # once per calculator, not per snippet

    def test_recorded_lf_is_used_as_stored(self):
        calc = SnippetFileFeatureCalculator(self.np1_h5)
        with mock.patch("ephysatlas.feature_calculators.snippet_file.derive_lf") as m:
            calc.load_raw_snippet(calc.windows[0])
        m.assert_not_called()

    def test_labels_majority_vote_computed_once(self):
        calc = SnippetFileFeatureCalculator(self.np1_3snip_h5)
        per_snippet = [
            np.r_[0, 1, 0, np.zeros(381)],
            np.r_[0, 1, 3, np.zeros(381)],
            np.r_[0, 0, 3, np.zeros(381)],
        ]
        target = "ephysatlas.feature_calculators.snippet_file"
        with mock.patch(
            f"{target}.ibldsp.voltage.detect_bad_channels",
            side_effect=[(labels, None) for labels in per_snippet],
        ) as m:
            raw = calc.load_raw_snippet(calc.windows[0])
            first = calc._resolve_channel_labels(raw, calc.load_channel_metadata())
            second = calc._resolve_channel_labels(raw, calc.load_channel_metadata())
        np.testing.assert_array_equal(first[:3], [0, 1, 3])
        np.testing.assert_array_equal(first, second)
        self.assertEqual(m.call_count, 3)

    def test_explicit_and_table_labels_take_priority(self):
        calc = SnippetFileFeatureCalculator(self.np1_3snip_h5)
        raw = calc.load_raw_snippet(calc.windows[0])
        explicit = np.ones(384)
        table = calc.load_channel_metadata().assign(labels=np.full(384, 2))
        target = "ephysatlas.feature_calculators.snippet_file"
        with mock.patch(f"{target}.ibldsp.voltage.detect_bad_channels") as m:
            self.assertIs(calc._resolve_channel_labels(raw, table, explicit), explicit)
            np.testing.assert_array_equal(
                calc._resolve_channel_labels(raw, table), table["labels"]
            )
        m.assert_not_called()

    def test_enrich_adds_probe_metadata_and_trajectory(self):
        calc = SnippetFileFeatureCalculator(self.np1_traj_h5)
        with mock.patch(
            "ephysatlas.feature_calculators.snippet_file.add_target_coordinates",
            side_effect=lambda channels, traj_dict: {
                **channels,
                "x_target": channels["axial_um"],
            },
        ) as m:
            out = calc.enrich_channel_metadata(
                calc.load_channel_metadata(), FeatureComputationOptions()
            )
        self.assertEqual(m.call_args.kwargs["traj_dict"], calc.trajectory)
        self.assertTrue(
            {"probe_model", "referencing_scheme", "x_target"} <= set(out.columns)
        )

    def test_require_trajectory_without_one_raises(self):
        calc = SnippetFileFeatureCalculator(self.np1_h5)
        with self.assertRaises(ValueError):
            calc.enrich_channel_metadata(
                calc.load_channel_metadata(),
                FeatureComputationOptions(require_trajectory=True),
            )

    def test_compute_denoised_features_uses_stored_windows(self):
        calc = SnippetFileFeatureCalculator(self.np1_3snip_h5)
        target = "ephysatlas.feature_calculators.snippet_file"
        with mock.patch(f"{target}.compute_aggregated_features") as m:
            calc.compute_denoised_features()
        self.assertEqual(list(m.call_args.args[1]), calc.windows)
        with self.assertRaises(ValueError):
            calc.compute_denoised_features(n_snippets=7)


if __name__ == "__main__":
    unittest.main()
