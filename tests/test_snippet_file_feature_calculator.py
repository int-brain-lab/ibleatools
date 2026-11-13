"""Tests for SnippetFileFeatureCalculator, which reads files written by extract_snippets."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pandas as pd

from ephysatlas.feature_calculators import SpikeGLXFileFeatureCalculator
from ephysatlas.feature_calculators.snippet_file import (
    SnippetFileFeatureCalculator,
    list_recordings,
)
from ephysatlas.feature_calculators.types import (
    FeatureComputationOptions,
    SnippetWindow,
)
from ephysatlas.snippet_extractor import extract_snippets
from tests._snippet_fixtures import write_spikeglx_recording

REC = "recordings/rec_g0_t0.imec0"


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
        cls.np1_h5 = extract_snippets(
            cls.np1_ap, cls.tmp / "np1.h5", t_starts=[0.1, 0.5], duration=0.1
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
        self.assertEqual(calc.available_duration(), (0.6, 0.6))
        for window in calc.windows:
            ours, theirs = calc.load_raw_snippet(window), ref.load_raw_snippet(window)
            np.testing.assert_array_equal(ours.raw_ap, theirs.raw_ap)
            np.testing.assert_array_equal(ours.raw_lf, theirs.raw_lf)
            self.assertEqual((ours.fs_ap, ours.fs_lf), (theirs.fs_ap, theirs.fs_lf))

    def test_np1_features_match_spikeglx_calculator(self):
        """Packing a generated recording preserves its computed LF, CSD and AP features."""
        # One second supports the LF spectral analysis while keeping this test small.
        ap_file = write_spikeglx_recording(self.tmp / "np1_features", "np1", 1.2)
        lf_file = ap_file.with_name(ap_file.name.replace(".ap.", ".lf."))
        packed = extract_snippets(
            ap_file, self.tmp / "np1_features.h5", t_starts=[0.1], duration=1.0
        )
        options = FeatureComputationOptions(
            features_to_compute=["lf", "csd", "ap"],
            # Random noise is a transport fixture, so bypass biological channel QC.
            channel_labels=np.zeros(384),
            include_trajectory=False,
            output_dir=self.tmp / "features",
        )
        calc = SnippetFileFeatureCalculator(packed)
        reference = SpikeGLXFileFeatureCalculator(ap_file=ap_file, lf_file=lf_file)
        ours = calc.compute_snippet(calc.windows[0], options)
        theirs = reference.compute_snippet(SnippetWindow(0.1, 1.0, 1.0), options)
        ours_df = ours.features.sort_values("channel").reset_index(drop=True)
        theirs_df = theirs.features.sort_values("channel").reset_index(drop=True)
        self.assertEqual(len(theirs_df), 384)
        pd.testing.assert_frame_equal(
            ours_df[theirs_df.columns], theirs_df, check_dtype=False
        )

    def test_windows_name_and_version(self):
        calc = SnippetFileFeatureCalculator(self.np1_h5)
        self.assertEqual(
            calc.windows, [SnippetWindow(0.1, 0.1, 0.1), SnippetWindow(0.5, 0.1, 0.1)]
        )
        self.assertEqual(calc.name, f"{self.np1_h5.stem}_rec_g0_t0.imec0")
        self.assertEqual(calc.recording, "rec_g0_t0.imec0")
        self.assertEqual(calc.neuropixel_version, 1)
        self.assertEqual(
            SnippetFileFeatureCalculator(self.np2_h5).neuropixel_version, 2
        )

    def test_recording_selection(self):
        """Each recording in a collection loads the same data as its standalone file."""
        two = self.tmp / "two.h5"
        shutil.copy(self.np2_h5, two)
        with h5py.File(self.np1_h5) as src, h5py.File(two, "a") as dst:
            src.copy(src[REC], dst["recordings"], "np1")
        self.assertEqual(list_recordings(two), ["np1", "rec_g0_t0.imec0"])
        with self.assertRaisesRegex(ValueError, "np1.*rec_g0_t0.imec0"):
            SnippetFileFeatureCalculator(two)
        self.assertEqual(
            SnippetFileFeatureCalculator(two, recording="np1").neuropixel_version, 1
        )
        for name, source in (("np1", self.np1_h5), ("rec_g0_t0.imec0", self.np2_h5)):
            with self.subTest(recording=name):
                calc = SnippetFileFeatureCalculator(two, recording=name)
                reference = SnippetFileFeatureCalculator(source)
                raw = calc.load_raw_snippet(calc.windows[0])
                expected = reference.load_raw_snippet(reference.windows[0])
                np.testing.assert_array_equal(raw.raw_ap, expected.raw_ap)
                np.testing.assert_array_equal(raw.raw_lf, expected.raw_lf)
        with self.assertRaisesRegex(ValueError, "nope.*np1.*rec_g0_t0.imec0"):
            SnippetFileFeatureCalculator(two, recording="nope")
        SnippetFileFeatureCalculator(self.np1_h5)  # one recording: no argument needed

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
            f[REC]["geometry/shank"][...] = np.arange(96) % 2  # pretend two shanks
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

    def test_np2_reads_stored_lf(self):
        """Use the saved LF samples and gains without filtering or shifting them again."""
        calc = SnippetFileFeatureCalculator(self.np2_h5)
        self.assertEqual(calc.available_duration(), (0.6, 0.6))
        self.assertIsNone(calc.trajectory)
        with h5py.File(self.np2_h5, "r") as file:
            lf = file[f"{REC}/lf"]
            for index, window in enumerate(calc.windows):
                raw = calc.load_raw_snippet(window)
                expected = lf["data"][index].astype(np.float32)
                expected *= lf["sample2volts"][:][:, np.newaxis]
                np.testing.assert_array_equal(raw.raw_lf, expected)
                self.assertEqual(raw.fs_lf, float(lf.attrs["fs"]))
                self.assertEqual(raw.raw_lf.dtype, np.float32)
                self.assertEqual(raw.raw_lf.shape, (96, 250))

    def test_missing_band_raises(self):
        """Reject incomplete recordings for either probe type before reading snippets."""
        for source in (self.np1_h5, self.np2_h5):
            for band in ("ap", "lf"):
                with self.subTest(source=source.name, band=band):
                    incomplete = self.tmp / f"{source.stem}_no_{band}.h5"
                    shutil.copy(source, incomplete)
                    with h5py.File(incomplete, "r+") as file:
                        del file[f"{REC}/{band}"]
                    with self.assertRaisesRegex(ValueError, f"required band '{band}'"):
                        SnippetFileFeatureCalculator(incomplete)

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
        self.assertEqual(
            calc.trajectory, dict(x=1.0, y=2.0, z=3.0, depth=4.0, theta=5.0, phi=6.0)
        )
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
