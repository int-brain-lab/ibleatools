"""Tests for ``ephysatlas.snippet_extractor``, on tiny synthetic SpikeGLX recordings."""

import contextlib
import importlib.metadata
import io
import logging
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import scipy.fft
import spikeglx

from ephysatlas.snippet_extractor import extract_snippets, main
from tests._snippet_fixtures import FIXTURES, write_spikeglx_recording


class TestExtractNP1(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.ap_file = write_spikeglx_recording(self.tmp, "np1", n_seconds=1.0)
        self.lf_file = self.ap_file.with_name(self.ap_file.name.replace(".ap.", ".lf."))
        self.out = self.tmp / "out" / "probe.h5"  # parent folder doesn't exist yet
        self.t_starts = [0.1, 0.5]
        extract_snippets(self.ap_file, self.out, t_starts=self.t_starts, duration=0.1)

    def test_ap_matches_spikeglx_reader(self):
        # stored int16 x gains, converted as spikeglx.Reader.read does, equals the Reader's volts
        sr = spikeglx.Reader(self.ap_file)
        with h5py.File(self.out) as f:
            data, s2v, first = (
                f["ap/data"][()],
                f["ap/sample2volts"][()],
                f["ap/first_sample"][()],
            )
        self.assertEqual(data.dtype, np.int16)
        self.assertEqual(data.shape, (2, 384, 3000))
        np.testing.assert_array_equal(first, [int(sr.fs * t) for t in self.t_starts])
        for i, n0 in enumerate(first):
            volts = data[i].astype(np.float32)
            volts *= s2v[:, np.newaxis]
            np.testing.assert_array_equal(volts, sr[n0 : n0 + 3000, :384].T)

    def test_lf_is_read_three_samples_late(self):
        fs_lf = spikeglx.Reader(self.lf_file).fs
        with h5py.File(self.out) as f:
            np.testing.assert_array_equal(
                f["lf/first_sample"][()], [int(fs_lf * t) + 3 for t in self.t_starts]
            )
            self.assertEqual(f["lf/data"].shape, (2, 384, 250))
            self.assertEqual(f["lf"].attrs["source"], "recorded")
            self.assertEqual(f["lf"].attrs["latency_samples"], 3)

    def test_root_attributes_geometry_and_meta(self):
        sr = spikeglx.Reader(self.ap_file)
        with h5py.File(self.out) as f:
            self.assertEqual(f.attrs["format_version"], 1)
            self.assertTrue(f.attrs["probe_has_lf_band"])
            self.assertEqual(f.attrs["neuropixel_version"], 1)
            self.assertEqual(f.attrs["duration"], 0.1)
            self.assertEqual(f.attrs["snippet_method"], "explicit")
            # seed and margin only describe automatic times
            self.assertNotIn("snippet_seed", f.attrs)
            self.assertNotIn("snippet_margin", f.attrs)
            self.assertEqual(f.attrs["source_ap_file"], str(self.ap_file.resolve()))
            self.assertEqual(f.attrs["probe_model"], spikeglx.get_probe_model(sr.meta))
            self.assertEqual(
                f.attrs["ibl_neuropixel_version"],
                importlib.metadata.version("ibl-neuropixel"),
            )
            np.testing.assert_array_equal(f["t_start"][()], self.t_starts)
            self.assertEqual(set(f["geometry"]), set(sr.geometry))
            np.testing.assert_array_equal(f["geometry/x"][()], sr.geometry["x"])
            # exact file content: the fixture metas have CRLF line endings
            self.assertEqual(f["ap/meta"].dtype, np.uint8)
            self.assertEqual(
                f["ap/meta"][()].tobytes(),
                self.ap_file.with_suffix(".meta").read_bytes(),
            )
            self.assertEqual(
                f["lf/meta"][()].tobytes(),
                self.lf_file.with_suffix(".meta").read_bytes(),
            )

    def test_channels_are_stored_in_ap_reader_order(self):
        # The fixture's channel order is the identity, so the snsShankMap entries are permuted
        # to tell file order from the Reader's sorted order, and the gains differ per channel.
        # The LF meta has no channel map, so the LF Reader's own order is the identity.
        shank_map = spikeglx.read_meta_data(FIXTURES / "np1.ap.meta")["snsShankMap"]
        sites = re.findall(r"\(\d+:\d+:\d+:\d+\)", shank_map)
        permutation = np.random.default_rng(1).permutation(len(sites))
        gains = (125, 250, 500, 1000)
        edits = {
            "snsShankMap": shank_map[: shank_map.index(sites[0])]
            + "".join(sites[i] for i in permutation),
            "imroTbl": "(0,384)"
            + "".join(
                f"({i} 0 0 {gains[i % 4]} {gains[(i + 1) % 4]} 1)" for i in range(384)
            ),
        }
        ap = write_spikeglx_recording(self.tmp / "perm", "np1", 1.0, edits=edits)
        out = extract_snippets(
            ap, self.tmp / "perm.h5", t_starts=self.t_starts, duration=0.1
        )
        sr_ap = spikeglx.Reader(ap)
        sr_lf = spikeglx.Reader(ap.with_name(ap.name.replace(".ap.", ".lf.")))
        order = sr_ap.raw_channel_order[:384]
        self.assertFalse(np.array_equal(order, np.arange(384)))
        self.assertFalse(np.array_equal(sr_lf.raw_channel_order[:384], order))
        self.assertGreater(len(np.unique(sr_ap.sample2volts[:384])), 1)

        with h5py.File(out) as f:
            s2v = f["ap/sample2volts"][()]
            np.testing.assert_array_equal(s2v, sr_ap.sample2volts[order])
            for i, n0 in enumerate(f["ap/first_sample"][()]):
                volts = f["ap/data"][i].astype(np.float32)
                volts *= s2v[:, np.newaxis]
                np.testing.assert_array_equal(volts, sr_ap[n0 : n0 + 3000, :384].T)
            np.testing.assert_array_equal(
                f["lf/sample2volts"][()], sr_lf.sample2volts[order]
            )
            for i, n0 in enumerate(f["lf/first_sample"][()]):
                np.testing.assert_array_equal(
                    f["lf/data"][i], sr_lf._raw[n0 : n0 + 250][:, order].T
                )

    def test_refuses_existing_output(self):
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            extract_snippets(self.ap_file, self.out, t_starts=[0.1], duration=0.1)

    def test_output_created_during_the_run_is_not_replaced(self):
        # a file that appears after the start-of-run check must survive
        out = self.tmp / "race.h5"
        real_next_fast_len = scipy.fft.next_fast_len

        def create_then_compute(*args, **kwargs):
            out.write_bytes(b"theirs")
            return real_next_fast_len(*args, **kwargs)

        with mock.patch.object(scipy.fft, "next_fast_len", create_then_compute):
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                extract_snippets(self.ap_file, out, t_starts=[0.1], duration=0.1)
        self.assertEqual(out.read_bytes(), b"theirs")
        self.assertEqual(list(self.tmp.glob("race.h5.partial")), [])

    def test_meta_bytes_round_trip_with_non_ascii_text(self):
        # spikeglx.read_meta_data decodes as UTF-8, so a non-UTF-8 meta can't be read at
        # all; a UTF-8 "é" is the closest case that can
        ap = write_spikeglx_recording(self.tmp / "utf", "np1", 1.0)
        meta = ap.with_suffix(".meta")
        meta.write_bytes(
            meta.read_bytes().replace(b"fileName=D:", "fileName=D:/café".encode())
        )
        out = extract_snippets(ap, self.tmp / "utf.h5", t_starts=[0.1], duration=0.1)
        with h5py.File(out) as f:
            self.assertEqual(f["ap/meta"][()].tobytes(), meta.read_bytes())

    def test_ap_file_missing_names_the_ap_file(self):
        missing = self.tmp / "typo_g0_t0.imec0.ap.bin"
        with self.assertRaisesRegex(FileNotFoundError, "typo_g0_t0.imec0.ap.bin"):
            extract_snippets(missing, self.tmp / "x.h5", t_starts=[0.1], duration=0.1)

    def test_no_snippets_raises(self):
        for kwargs in ({"t_starts": []}, {"n_snippets": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "at least one snippet"):
                    extract_snippets(
                        self.ap_file, self.tmp / "x.h5", duration=0.1, **kwargs
                    )

    def test_interrupted_write_leaves_nothing(self):
        out = self.tmp / "interrupted.h5"
        with mock.patch.object(
            h5py.Group, "create_dataset", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                extract_snippets(self.ap_file, out, t_starts=[0.1], duration=0.1)
        self.assertEqual(list(self.tmp.glob("interrupted.h5*")), [])

    def test_calibrated_sample_rate_matches_spikeglx_loader(self):
        ap = write_spikeglx_recording(
            self.tmp / "cal", "np1", 1.0, edits={"imSampRate": "30000.110084"}
        )
        # at this time the calibrated and the nominal rate give different sample indices
        t = 0.369999
        out = extract_snippets(ap, self.tmp / "cal.h5", t_starts=[t], duration=0.1)
        sr = spikeglx.Reader(ap)
        self.assertNotEqual(int(sr.fs * t), int(30000 * t))
        with h5py.File(out) as f:
            self.assertEqual(f["ap"].attrs["fs"], sr.fs)
            self.assertEqual(f["ap/first_sample"][0], int(sr.fs * t))
            self.assertEqual(
                f["ap/data"].shape[-1],
                scipy.fft.next_fast_len(int(sr.fs * 0.1), real=True),
            )

    def test_rejects_files_that_are_not_ap(self):
        for wrong in (self.lf_file, self.ap_file.with_suffix(".meta")):
            with self.assertRaisesRegex(ValueError, r"\.ap\.bin"):
                extract_snippets(
                    wrong, self.tmp / "wrong.h5", t_starts=[0.1], duration=0.1
                )


class TestProbeTypes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_np24_stores_wideband_ap_only(self):
        ap = write_spikeglx_recording(
            self.tmp, "np24_shank", 1.0
        )  # an .lf file sits next to it
        with self.assertLogs("ephysatlas.snippet_extractor", level="INFO") as logs:
            out = extract_snippets(
                ap, self.tmp / "np2.h5", t_starts=[0.1], duration=0.1
            )
        self.assertTrue(any("ignored" in line for line in logs.output))
        with h5py.File(out) as f:
            self.assertNotIn("lf", f)
            self.assertFalse(f.attrs["probe_has_lf_band"])
            self.assertEqual(f.attrs["neuropixel_version"], 2)
            self.assertEqual(f.attrs["source_lf_file"], "")

    def test_np24_channel_order_matches_reader(self):
        ap = write_spikeglx_recording(self.tmp, "np24_shank", 1.0)
        out = extract_snippets(
            ap, self.tmp / "np2.h5", t_starts=[0.1, 0.5], duration=0.1
        )
        sr = spikeglx.Reader(ap)
        order = sr.raw_channel_order[:96]
        self.assertFalse(
            np.array_equal(order, np.arange(96))
        )  # the fixture really reorders
        with h5py.File(out) as f:
            np.testing.assert_array_equal(
                f["ap/sample2volts"][()], sr.sample2volts[order]
            )
            np.testing.assert_array_equal(f["geometry/y"][()], sr.geometry["y"])
            for i, n0 in enumerate(f["ap/first_sample"][()]):
                volts = f["ap/data"][i].astype(np.float32)
                volts *= f["ap/sample2volts"][()][:, np.newaxis]
                np.testing.assert_array_equal(volts, sr[n0 : n0 + 3000, :96].T)

    def test_np1_without_lf_file_stores_ap_only_and_warns(self):
        ap = write_spikeglx_recording(self.tmp, "np1", 1.0, with_lf=False)
        with self.assertLogs("ephysatlas.snippet_extractor", level="WARNING"):
            out = extract_snippets(
                ap, self.tmp / "nolf.h5", t_starts=[0.1], duration=0.1
            )
        with h5py.File(out) as f:
            self.assertNotIn("lf", f)
            self.assertTrue(f.attrs["probe_has_lf_band"])

    def test_unknown_probe_type_raises_clear_error(self):
        ap = write_spikeglx_recording(
            self.tmp, "np24_shank", 1.0, edits={"imDatPrb_type": "2014"}
        )
        part_number = spikeglx.read_meta_data(ap.with_suffix(".meta"))["imDatPrb_pn"]
        with self.assertRaisesRegex(ValueError, "2014") as ctx:
            extract_snippets(ap, self.tmp / "x.h5", t_starts=[0.1], duration=0.1)
        self.assertIn(str(part_number), str(ctx.exception))
        self.assertNotIn("2014.0", str(ctx.exception))
        self.assertFalse((self.tmp / "x.h5").exists())

    def test_non_numeric_probe_version_raises(self):  # 1100 = NPultra, a text version
        ap = write_spikeglx_recording(
            self.tmp, "np24_shank", 1.0, edits={"imDatPrb_type": "1100"}
        )
        with self.assertRaisesRegex(ValueError, "1100"):
            extract_snippets(ap, self.tmp / "x.h5", t_starts=[0.1], duration=0.1)

    def test_missing_channel_map_raises(self):
        ap = write_spikeglx_recording(
            self.tmp, "np24_shank", 1.0, edits={"snsGeomMap": None, "snsShankMap": None}
        )
        with self.assertRaisesRegex(ValueError, "channel map"):
            extract_snippets(ap, self.tmp / "x.h5", t_starts=[0.1], duration=0.1)


class TestSnippetTimes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_automatic_times_use_real_data_length(self):
        # the meta claims 4971 s, but the file holds 1 s; spikeglx.Reader reports the real length
        ap = write_spikeglx_recording(
            self.tmp, "np1", 1.0, edits={"fileTimeSecs": "4971.0"}
        )
        with mock.patch(
            "ephysatlas.snippet_extractor.get_snippet_start_times",
            return_value=np.array([0.1, 0.5]),
        ) as m:
            # spikeglx warns about the wrong fileTimeSecs, once per band
            with self.assertLogs("ibllib", level="WARNING") as logs:
                out = extract_snippets(
                    ap, self.tmp / "auto.h5", n_snippets=2, duration=0.1, margin=0.0
                )
        fudged = [r for r in logs.records if "do not checkout" in r.getMessage()]
        self.assertEqual(len(fudged), 2)
        args, kwargs = m.call_args
        self.assertAlmostEqual(args[0], 1.0, places=3)
        self.assertEqual(args[1:], (2, 0.1))
        self.assertEqual(kwargs, {"margin": 0.0})
        with h5py.File(out) as f:
            np.testing.assert_array_equal(f["t_start"][()], [0.1, 0.5])
            self.assertEqual(f.attrs["snippet_method"], "stratified")
            self.assertEqual(f.attrs["snippet_seed"], 0)
            self.assertEqual(f.attrs["snippet_margin"], 0.0)

    def test_automatic_times_with_the_real_helper(self):
        # get_snippet_start_times rounds to whole seconds, so a 1 s recording only has room for one
        ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        out = extract_snippets(
            ap, self.tmp / "auto.h5", n_snippets=1, duration=0.1, margin=0.0
        )
        with h5py.File(out) as f:
            np.testing.assert_array_equal(f["t_start"][()], [0.0])
            self.assertEqual(f["ap/data"].shape, (1, 384, 3000))
            self.assertEqual(f.attrs["snippet_method"], "stratified")

    def test_too_short_recording_mentions_margin(self):
        ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        with self.assertRaisesRegex(ValueError, "smaller margin"):
            extract_snippets(ap, self.tmp / "short.h5")  # default 300 s margin
        self.assertFalse((self.tmp / "short.h5").exists())

    def test_explicit_overlapping_times_raise(self):
        ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            extract_snippets(ap, self.tmp / "o.h5", t_starts=[0.1, 0.15], duration=0.1)

    def test_explicit_time_past_end_raises(self):
        ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        out = self.tmp / "new" / "end.h5"
        # The 1 s files hold 30000 AP and 2500 LF samples. Only the next_fast_len rounding
        # of the AP sample count (3003 -> 3072) pushes a read past the end; the LF read
        # and the unrounded AP read fit.
        t, duration = 0.898, 0.1001
        n0 = int(30000 * t)
        ns = int(30000 * duration)
        self.assertLessEqual(n0 + ns, 30000)
        self.assertGreater(n0 + scipy.fft.next_fast_len(ns, real=True), 30000)
        self.assertLessEqual(int(2500 * t) + 3 + int(2500 * duration), 2500)
        with self.assertRaisesRegex(ValueError, "t = 0.898 s needs AP samples"):
            extract_snippets(ap, out, t_starts=[t], duration=duration)
        # nothing was written, not even the output folder or a .partial file
        self.assertFalse(out.parent.exists())

    def test_negative_time_raises(self):
        ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        with self.assertRaisesRegex(ValueError, "negative"):
            extract_snippets(ap, self.tmp / "neg.h5", t_starts=[-0.1], duration=0.1)


class TestHistologyAndTrajectory(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        self.a, self.b = self.tmp / "a", self.tmp / "b"
        for folder, content in (
            (self.a, b'{"channel_0": {"x": 1}}'),
            (self.b, b'{"channel_0": {"x": 2}}'),
        ):
            folder.mkdir()
            (folder / "channel_locations.json").write_bytes(content)

    def test_files_and_folders_stored_byte_identical(self):
        loc = self.tmp / "channel_locations.json"
        loc.write_bytes(b'{"channel_0": {"x": 1}}')
        (self.tmp / "tracks/sub").mkdir(parents=True)
        (self.tmp / "tracks/sub/b.csv").write_bytes(b"Index,Region acronym\n0,CA1\n")
        out = extract_snippets(
            self.ap,
            self.tmp / "h.h5",
            histology=[loc, self.tmp / "tracks"],
            t_starts=[0.1],
            duration=0.1,
        )
        with h5py.File(out) as f:
            self.assertEqual(set(f["histology"]), {"channel_locations.json", "tracks"})
            self.assertEqual(f["histology/channel_locations.json"].dtype, np.uint8)
            self.assertEqual(
                f["histology/channel_locations.json"][()].tobytes(), loc.read_bytes()
            )
            self.assertEqual(
                f["histology/tracks/sub/b.csv"][()].tobytes(),
                (self.tmp / "tracks/sub/b.csv").read_bytes(),
            )
            self.assertEqual(
                f["histology/channel_locations.json"].attrs["source_path"],
                str(loc.resolve()),
            )
            self.assertEqual(
                f["histology/tracks/sub/b.csv"].attrs["source_path"],
                str((self.tmp / "tracks/sub/b.csv").resolve()),
            )

    def test_name_collision_raises(self):
        # two different files that would both be stored as histology/channel_locations.json
        out = self.tmp / "new" / "c.h5"
        with self.assertRaisesRegex(ValueError, "channel_locations.json"):
            extract_snippets(
                self.ap,
                out,
                histology=[
                    self.a / "channel_locations.json",
                    self.b / "channel_locations.json",
                ],
                t_starts=[0.1],
                duration=0.1,
            )
        self.assertFalse(out.parent.exists())

    def test_over_limit_raises_naming_the_largest_files(self):
        folder = self.tmp / "brainreg"
        folder.mkdir()
        # sparse files: no real disk use
        sizes = {
            self.tmp / "registered_atlas.tiff": 101 * 1024**2,
            folder / "a.tiff": 3 * 1024**2,
            folder / "b.tiff": 2 * 1024**2,
            folder / "c.tiff": 1024**2,
            folder / "tracks.csv": 10,
        }
        for path, size in sizes.items():
            with open(path, "wb") as fid:
                fid.truncate(size)
        out = self.tmp / "new" / "big.h5"
        with self.assertRaises(ValueError) as ctx:
            extract_snippets(
                self.ap,
                out,
                histology=[self.tmp / "registered_atlas.tiff", folder],
                t_starts=[0.1],
                duration=0.1,
            )
        message = str(ctx.exception)
        for named in ("registered_atlas.tiff", "brainreg/a.tiff", "brainreg/b.tiff"):
            self.assertIn(named, message)
        for not_named in ("c.tiff", "tracks.csv"):
            self.assertNotIn(not_named, message)
        self.assertFalse(out.parent.exists())

    def test_empty_histology_folder_warns(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertLogs("ephysatlas.snippet_extractor", level="WARNING") as logs:
            extract_snippets(
                self.ap,
                self.tmp / "e.h5",
                histology=[empty],
                t_starts=[0.1],
                duration=0.1,
            )
        self.assertIn("no files", logs.output[0])

    def test_missing_histology_path_raises(self):
        out = self.tmp / "new" / "m.h5"
        with self.assertRaises(FileNotFoundError):
            extract_snippets(
                self.ap,
                out,
                histology=[self.a / "channel_locations.json", self.tmp / "nope.json"],
                t_starts=[0.1],
                duration=0.1,
            )
        self.assertFalse(out.parent.exists())

    def test_trajectory_stored(self):
        traj = dict(x=-2243.0, y=-2000.0, z=0.0, depth=4000.0, theta=15.0, phi=180.0)
        out = extract_snippets(
            self.ap, self.tmp / "t.h5", trajectory=traj, t_starts=[0.1], duration=0.1
        )
        with h5py.File(out) as f:
            self.assertEqual(
                {k: float(v) for k, v in f["trajectory"].attrs.items()}, traj
            )
        # integers are stored as floats
        out = extract_snippets(
            self.ap,
            self.tmp / "t_int.h5",
            trajectory=dict(x=-2243, y=-2000, z=0, depth=4000, theta=15, phi=180),
            t_starts=[0.1],
            duration=0.1,
        )
        with h5py.File(out) as f:
            self.assertTrue(
                all(isinstance(v, np.floating) for v in f["trajectory"].attrs.values())
            )

    def test_bad_trajectory_raises(self):
        good = dict(x=0, y=0, z=0, depth=1, theta=0, phi=0)
        out = self.tmp / "new" / "bt.h5"
        for traj in (
            {k: v for k, v in good.items() if k != "phi"},  # phi missing
            {**good, "x": "a"},  # not numeric
            {**good, "x": "1.5"},  # float() would accept this
            {**good, "x": True},
        ):  # bool is not a number here
            with self.assertRaises(ValueError):
                extract_snippets(
                    self.ap, out, trajectory=traj, t_starts=[0.1], duration=0.1
                )
            self.assertFalse(out.parent.exists())

    def test_no_trajectory_no_group(self):
        out = extract_snippets(
            self.ap, self.tmp / "nt.h5", t_starts=[0.1], duration=0.1
        )
        with h5py.File(out) as f:
            self.assertNotIn("trajectory", f)
            self.assertNotIn("histology", f)


class TestCli(unittest.TestCase):
    def setUp(self):
        # main() configures the "ephysatlas" logger; undo it so other tests stay quiet
        package_logger = logging.getLogger("ephysatlas")
        self.addCleanup(
            setattr, package_logger, "handlers", list(package_logger.handlers)
        )
        self.addCleanup(package_logger.setLevel, package_logger.level)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_main_passes_every_option(self):
        with mock.patch("ephysatlas.snippet_extractor.extract_snippets") as m:
            rc = main(
                [
                    "rec.imec0.ap.bin",
                    "-o",
                    "out.h5",
                    "--histology",
                    "a.json",
                    "b.csv",
                    "--t-starts",
                    "1",
                    "2.5",
                    "--n-snippets",
                    "3",
                    "--duration",
                    "2",
                    "--margin",
                    "60",
                    "--trajectory",
                    "1",
                    "2",
                    "3",
                    "4000",
                    "15",
                    "180",
                ]
            )
        self.assertEqual(rc, 0)
        m.assert_called_once_with(
            Path("rec.imec0.ap.bin"),
            Path("out.h5"),
            histology=[Path("a.json"), Path("b.csv")],
            t_starts=[1.0, 2.5],
            n_snippets=3,
            duration=2.0,
            margin=60.0,
            trajectory=dict(x=1.0, y=2.0, z=3.0, depth=4000.0, theta=15.0, phi=180.0),
        )

    def test_main_defaults(self):
        with mock.patch("ephysatlas.snippet_extractor.extract_snippets") as m:
            main(["rec.imec0.ap.bin", "-o", "out.h5"])
        m.assert_called_once_with(
            Path("rec.imec0.ap.bin"),
            Path("out.h5"),
            histology=None,
            t_starts=None,
            n_snippets=5,
            duration=5.0,
            margin=300.0,
            trajectory=None,
        )

    def test_main_requires_output(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["rec.imec0.ap.bin"])

    def test_main_end_to_end_and_error_exit_code(self):
        ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        out = self.tmp / "cli.h5"
        args = [str(ap), "-o", str(out), "--t-starts", "0.1", "--duration", "0.1"]
        self.assertEqual(main(args), 0)
        self.assertTrue(out.exists())
        with self.assertLogs("ephysatlas.snippet_extractor", level="ERROR") as logs:
            self.assertEqual(main(args), 1)  # the output exists now
        self.assertIn(str(out), logs.output[0])
        self.assertIn("already exists", logs.output[0])

    def test_error_lines_show_the_level(self):
        stream = io.StringIO()
        with mock.patch("sys.stdout", stream):
            self.assertEqual(
                main([str(self.tmp / "x.ap.bin"), "-o", str(self.tmp / "o.h5")]), 1
            )
        self.assertTrue(stream.getvalue().startswith("ERROR: "), stream.getvalue())


if __name__ == "__main__":
    unittest.main()
