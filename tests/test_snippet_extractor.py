"""Tests for ``ephysatlas.snippet_extractor``, on tiny synthetic SpikeGLX recordings."""

import contextlib
import copy
import importlib.metadata
import io
import json
import logging
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import mtscomp
import numpy as np
import pandas as pd
import scipy.fft
import spikeglx
from neuropixel import NP2Converter

import ephysatlas
from ephysatlas.snippet_extractor import add_channel_locations, extract_snippets, main
from tests._snippet_fixtures import (
    FIXTURES,
    channel_locations_dict,
    write_spikeglx_recording,
)

REC = "recordings/rec_g0_t0.imec0"


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
                f[REC]["ap/data"][()],
                f[REC]["ap/sample2volts"][()],
                f[REC]["ap/first_sample"][()],
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
                f[REC]["lf/first_sample"][()],
                [int(fs_lf * t) + 3 for t in self.t_starts],
            )
            self.assertEqual(f[REC]["lf/data"].shape, (2, 384, 250))
            self.assertEqual(f[REC]["lf"].attrs["source"], "recorded")
            self.assertEqual(f[REC]["lf"].attrs["latency_samples"], 3)

    def test_name_given(self):
        out = extract_snippets(
            self.ap_file,
            self.tmp / "n.h5",
            name="probe00",
            t_starts=[0.1],
            duration=0.1,
        )
        with h5py.File(out) as f:
            self.assertEqual(list(f["recordings"]), ["probe00"])

    def test_bad_names_raise_before_reading(self):
        out = self.tmp / "new" / "bad.h5"
        for bad in ("", ".", "a/b"):
            with self.subTest(name=bad):
                with mock.patch(
                    "ephysatlas.snippet_extractor.spikeglx.Reader",
                    side_effect=AssertionError("read"),
                ):
                    with self.assertRaisesRegex(ValueError, "name"):
                        extract_snippets(
                            self.ap_file, out, name=bad, t_starts=[0.1], duration=0.1
                        )
                self.assertFalse(out.parent.exists())

    def test_root_attributes_geometry_and_meta(self):
        sr = spikeglx.Reader(self.ap_file)
        with h5py.File(self.out) as f:
            self.assertEqual(list(f["recordings"]), ["rec_g0_t0.imec0"])
            self.assertEqual(f[REC].attrs["created"], f.attrs["created"])
            self.assertTrue(f[REC].attrs["probe_has_lf_band"])
            self.assertEqual(f[REC].attrs["neuropixel_version"], 1)
            self.assertEqual(f[REC].attrs["duration"], 0.1)
            self.assertEqual(f[REC].attrs["snippet_method"], "explicit")
            self.assertNotIn("trajectory", f[REC])
            self.assertNotIn("histology", f[REC])
            # seed and margin only describe automatic times
            self.assertNotIn("snippet_seed", f[REC].attrs)
            self.assertNotIn("snippet_margin", f[REC].attrs)
            self.assertEqual(
                f[REC].attrs["source_ap_file"], str(self.ap_file.resolve())
            )
            self.assertEqual(
                f[REC].attrs["probe_model"], spikeglx.get_probe_model(sr.meta)
            )
            self.assertEqual(
                f[REC].attrs["ibl_neuropixel_version"],
                importlib.metadata.version("ibl-neuropixel"),
            )
            np.testing.assert_array_equal(f[REC]["t_start"][()], self.t_starts)
            self.assertEqual(set(f[REC]["geometry"]), set(sr.geometry))
            np.testing.assert_array_equal(f[REC]["geometry/x"][()], sr.geometry["x"])
            # exact file content: the fixture metas have CRLF line endings
            self.assertEqual(f[REC]["ap/meta"].dtype, np.uint8)
            self.assertEqual(
                f[REC]["ap/meta"][()].tobytes(),
                self.ap_file.with_suffix(".meta").read_bytes(),
            )
            self.assertEqual(
                f[REC]["lf/meta"][()].tobytes(),
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
            s2v = f[REC]["ap/sample2volts"][()]
            np.testing.assert_array_equal(s2v, sr_ap.sample2volts[order])
            for i, n0 in enumerate(f[REC]["ap/first_sample"][()]):
                volts = f[REC]["ap/data"][i].astype(np.float32)
                volts *= s2v[:, np.newaxis]
                np.testing.assert_array_equal(volts, sr_ap[n0 : n0 + 3000, :384].T)
            np.testing.assert_array_equal(
                f[REC]["lf/sample2volts"][()], sr_lf.sample2volts[order]
            )
            for i, n0 in enumerate(f[REC]["lf/first_sample"][()]):
                np.testing.assert_array_equal(
                    f[REC]["lf/data"][i], sr_lf._raw[n0 : n0 + 250][:, order].T
                )

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
            self.assertEqual(f[REC]["ap/meta"][()].tobytes(), meta.read_bytes())

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
            self.assertEqual(f[REC]["ap"].attrs["fs"], sr.fs)
            self.assertEqual(f[REC]["ap/first_sample"][0], int(sr.fs * t))
            self.assertEqual(
                f[REC]["ap/data"].shape[-1],
                scipy.fft.next_fast_len(int(sr.fs * 0.1), real=True),
            )

    def test_rejects_files_that_are_not_ap(self):
        for wrong in (self.lf_file, self.ap_file.with_suffix(".meta")):
            with self.assertRaisesRegex(ValueError, r"\.ap\.bin"):
                extract_snippets(
                    wrong, self.tmp / "wrong.h5", t_starts=[0.1], duration=0.1
                )


class TestAddToFile(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.np1_ap = write_spikeglx_recording(self.tmp / "np1", "np1", n_seconds=1.0)
        self.np2_ap = write_spikeglx_recording(
            self.tmp / "np2", "np24_shank", n_seconds=1.0
        )
        self.out = self.tmp / "out.h5"
        extract_snippets(self.np1_ap, self.out, t_starts=[0.1], duration=0.1)

    def test_second_call_adds_recording(self):
        with h5py.File(self.out) as f:
            before = {k: f[REC][k][()] for k in ("ap/data", "lf/data", "t_start")}
            before_attrs, created = dict(f[REC].attrs), f.attrs["created"]
        with self.assertLogs("ephysatlas.snippet_extractor", level="INFO") as logs:
            extract_snippets(
                self.np2_ap, self.out, name="np2", t_starts=[0.2], duration=0.1
            )
        self.assertTrue(
            any(
                "Added recording np2" in line and "2 recordings" in line
                for line in logs.output
            )
        )
        with h5py.File(self.out) as f:
            self.assertEqual(list(f["recordings"]), ["np2", "rec_g0_t0.imec0"])
            self.assertEqual(f.attrs["created"], created)
            for k, v in before.items():
                np.testing.assert_array_equal(f[REC][k][()], v)
            self.assertEqual(dict(f[REC].attrs).keys(), before_attrs.keys())
            for k, v in before_attrs.items():
                np.testing.assert_array_equal(f[REC].attrs[k], v)
            self.assertEqual(f["recordings/np2/ap/data"].compression, "gzip")
            self.assertFalse(f["recordings/np2"].attrs["probe_has_lf_band"])
        self.assertEqual(list(self.tmp.glob("*.partial")), [])

    def test_taken_name_raises_before_reading(self):
        data = self.out.read_bytes()
        # A different source with the same default name must report the stored source.
        other_ap = write_spikeglx_recording(self.tmp / "other", "np1", n_seconds=1.0)
        with mock.patch(
            "ephysatlas.snippet_extractor.spikeglx.Reader",
            side_effect=AssertionError("read"),
        ):
            with self.assertRaises(ValueError) as ctx:
                extract_snippets(other_ap, self.out, t_starts=[0.5], duration=0.1)
        for part in (
            "already holds recording 'rec_g0_t0.imec0'",
            str(self.np1_ap.resolve()),
            "--name",
        ):
            self.assertIn(part, str(ctx.exception))
        self.assertNotIn(str(other_ap.resolve()), str(ctx.exception))
        self.assertEqual(self.out.read_bytes(), data)

    def test_output_that_is_not_a_snippet_file_raises(self):
        bad = self.tmp / "bad.h5"

        def text_file():
            bad.write_bytes(b"theirs")

        def no_attrs():
            with h5py.File(bad, "w"):
                pass

        for make in (text_file, no_attrs):
            with self.subTest(make=make.__name__):
                make()
                data = bad.read_bytes()
                with self.assertRaisesRegex(ValueError, "not a snippet file"):
                    extract_snippets(
                        self.np2_ap, bad, name="np2", t_starts=[0.1], duration=0.1
                    )
                self.assertEqual(bad.read_bytes(), data)
                self.assertEqual(list(self.tmp.glob("*.partial")), [])

    def test_failure_during_extraction_leaves_file_unchanged(self):
        data = self.out.read_bytes()
        with mock.patch.object(
            h5py.Group, "create_dataset", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                extract_snippets(
                    self.np2_ap, self.out, name="np2", t_starts=[0.2], duration=0.1
                )
        self.assertEqual(self.out.read_bytes(), data)
        self.assertEqual(list(self.tmp.glob("*.partial")), [])

    def test_output_created_during_the_run(self):
        # a file that appears after the start-of-run check must survive
        out = self.tmp / "race.h5"
        real_next_fast_len = scipy.fft.next_fast_len

        def create_then_compute(*args, **kwargs):
            out.write_bytes(b"theirs")
            return real_next_fast_len(*args, **kwargs)

        with mock.patch.object(scipy.fft, "next_fast_len", create_then_compute):
            with self.assertRaisesRegex(ValueError, "not a snippet file"):
                extract_snippets(self.np1_ap, out, t_starts=[0.1], duration=0.1)
        self.assertEqual(out.read_bytes(), b"theirs")
        self.assertEqual(list(self.tmp.glob("*.partial")), [])


class TestProbeTypes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_compressed_input_matches_uncompressed(self):
        """Match packed samples and metadata for generated .bin and .cbin recordings."""
        cases = (
            ("np1", "3B2", 1, True),
            ("3a", "3A", 1, True),
            ("np24_shank", "NP2.4", 2, False),
        )
        for template, probe_version, major_version, has_lf in cases:
            with self.subTest(template=template):
                ap = write_spikeglx_recording(
                    self.tmp / template, template, 1.0, with_lf=has_lf
                )
                times = [0.12, 0.53]
                uncompressed = extract_snippets(
                    ap, self.tmp / f"{template}_bin.h5", t_starts=times, duration=0.1
                )
                with spikeglx.Reader(ap) as reader:
                    self.assertEqual(reader.version, probe_version)
                for binary in sorted(ap.parent.glob("*.bin")):
                    with spikeglx.Reader(binary) as reader:
                        # Short chunks force snippet reads across compression boundaries.
                        mtscomp.compress(
                            binary,
                            out=binary.with_suffix(".cbin"),
                            outmeta=binary.with_suffix(".ch"),
                            sample_rate=reader.fs,
                            n_channels=reader.nc,
                            dtype=np.int16,
                            chunk_duration=0.05,
                            n_threads=1,
                        )
                compressed = extract_snippets(
                    ap.with_suffix(".cbin"),
                    self.tmp / f"{template}_cbin.h5",
                    t_starts=times,
                    duration=0.1,
                )
                with (
                    h5py.File(uncompressed) as reference,
                    h5py.File(compressed) as packed,
                ):
                    expected, actual = reference[REC], packed[REC]
                    self.assertEqual(actual.attrs["neuropixel_version"], major_version)
                    self.assertIn("lf", actual)
                    np.testing.assert_array_equal(actual["t_start"][:], times)
                    for key in expected["geometry"]:
                        np.testing.assert_array_equal(
                            actual[f"geometry/{key}"][:], expected[f"geometry/{key}"][:]
                        )
                    for band in ("ap", "lf"):
                        self.assertEqual(
                            dict(actual[band].attrs), dict(expected[band].attrs)
                        )
                        self.assertEqual(set(actual[band]), set(expected[band]))
                        for key in expected[band]:
                            np.testing.assert_array_equal(
                                actual[f"{band}/{key}"][:], expected[f"{band}/{key}"][:]
                            )

    def test_np24_stores_derived_lf_without_changing_input(self):
        """NP2 always gets converter LF; an unrelated adjacent LF is left untouched."""
        ap = write_spikeglx_recording(self.tmp, "np24_shank", 1.0)
        lf = ap.with_name(ap.name.replace(".ap.", ".lf."))
        originals = {path: path.read_bytes() for path in (ap, lf)}
        out = extract_snippets(ap, self.tmp / "np2.h5", t_starts=[0.1], duration=0.1)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        with h5py.File(out) as file:
            rec = file[REC]
            self.assertFalse(rec.attrs["probe_has_lf_band"])
            self.assertEqual(rec.attrs["neuropixel_version"], 2)
            self.assertEqual(rec.attrs["source_lf_file"], "")
            self.assertEqual(rec["lf/data"].shape, (1, 96, 250))
            self.assertEqual(rec["lf"].attrs["source"], "derived")
            self.assertEqual(
                rec["lf"].attrs["method"], "neuropixel.NP2Converter.extract_lfp"
            )
            self.assertEqual(rec["lf"].attrs["latency_samples"], 3)
            np.testing.assert_array_equal(rec["lf/first_sample"][:], [253])
            self.assertNotIn(
                "meta", rec["lf"]
            )  # No original LF meta exists for this stream.

    def test_np2_lf_exactly_matches_full_converter_output(self):
        """Match all samples across chunk seams and recording ends, for one or four shanks."""
        for n_shanks in (1, 4):
            with self.subTest(shanks=n_shanks):
                # The template contains the original four-shank map; restore its full
                # channel count to represent an unsplit probe.
                edits = (
                    {
                        "NP2.4_shank": None,
                        "nSavedChans": "385",
                        "snsApLfSy": "384,0,1",
                        "snsSaveChanSubset": "0:384",
                    }
                    if n_shanks == 4
                    else None
                )
                folder = self.tmp / f"np2_{n_shanks}"
                # Exercise a partial final LF bin: the AP length is not divisible by 12.
                ap = write_spikeglx_recording(
                    folder, "np24_shank", 4.1001, edits=edits, with_lf=False
                )
                # The middle interval crosses a converter seam; the last reaches the LF end.
                times = [0.0, 1.9501, 3.9992]
                out = extract_snippets(
                    ap, self.tmp / f"np2_{n_shanks}.h5", t_starts=times, duration=0.1
                )
                converter = NP2Converter(ap, post_check=False, compress=False)
                self.addCleanup(converter.sr.close)
                if n_shanks == 1:
                    # process() refuses already-split NP2.4 input; its LF-only path still
                    # provides the reference conversion for these existing files.
                    status = converter._process_NP21(assert_shanks=False)
                else:
                    status = converter.process()
                self.assertEqual(status, 1)
                # Reassemble the converter's actual on-disk output into original channel order.
                reference = np.empty(
                    (
                        (converter.sr.ns + converter.ratio - 1) // converter.ratio,
                        converter.napch,
                    ),
                    dtype=np.int16,
                )
                for info in converter.shank_info.values():
                    channels = info["chns"]
                    recorded = channels < converter.napch
                    values = np.fromfile(info["lf_file"], dtype=np.int16).reshape(
                        -1, len(channels)
                    )
                    reference[:, channels[recorded]] = values[:, recorded]
                with spikeglx.Reader(ap) as reader, h5py.File(out) as file:
                    rec = file[REC]
                    order = reader.raw_channel_order[: converter.napch]
                    self.assertEqual(len(np.unique(rec["geometry/shank"][:])), n_shanks)
                    np.testing.assert_array_equal(
                        rec["lf/filter_sos"][:], converter.sos_lp
                    )
                    np.testing.assert_array_equal(
                        rec["lf/sample2volts"][:],
                        converter.sr.channel_conversion_sample2v["lf"][order],
                    )
                    for i, first in enumerate(rec["lf/first_sample"][:]):
                        expected = reference[first : first + 250, order].T
                        np.testing.assert_array_equal(rec["lf/data"][i], expected)
                    for i, first in enumerate(rec["ap/first_sample"][:]):
                        np.testing.assert_array_equal(
                            rec["ap/data"][i],
                            reader._raw[first : first + 3000][:, order].T,
                        )

    def test_np2_only_converts_intersecting_chunks(self):
        """A small extraction must not filter the whole recording."""
        ap = write_spikeglx_recording(self.tmp, "np24_shank", 6.1, with_lf=False)
        original = NP2Converter.extract_lfp
        with mock.patch.object(
            NP2Converter, "extract_lfp", autospec=True, side_effect=original
        ) as convert:
            extract_snippets(ap, self.tmp / "np2.h5", t_starts=[2.5, 2.8], duration=0.1)
        self.assertEqual(convert.call_count, 1)

    def test_np2_rejects_lf_window_past_end(self):
        """Validate derived LF bounds, including the existing three-sample offset."""
        ap = write_spikeglx_recording(self.tmp, "np24_shank", 1.0, with_lf=False)
        output = self.tmp / "new" / "np2.h5"
        with self.assertRaisesRegex(ValueError, "LF samples"):
            extract_snippets(ap, output, t_starts=[0.9], duration=0.1)
        self.assertFalse(output.parent.exists())

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
                f[REC]["ap/sample2volts"][()], sr.sample2volts[order]
            )
            np.testing.assert_array_equal(f[REC]["geometry/y"][()], sr.geometry["y"])
            for i, n0 in enumerate(f[REC]["ap/first_sample"][()]):
                volts = f[REC]["ap/data"][i].astype(np.float32)
                volts *= f[REC]["ap/sample2volts"][()][:, np.newaxis]
                np.testing.assert_array_equal(volts, sr[n0 : n0 + 3000, :96].T)

    def test_recorded_lf_is_required_before_opening_readers(self):
        """Reject missing NP1/3A LF before opening data or creating an output folder."""
        for template in ("np1", "3a"):
            with self.subTest(template=template):
                ap = write_spikeglx_recording(
                    self.tmp / template, template, 1.0, with_lf=False
                )
                output = self.tmp / f"{template}_output" / "snippets.h5"
                with mock.patch(
                    "ephysatlas.snippet_extractor.spikeglx.Reader"
                ) as reader:
                    with self.assertRaisesRegex(
                        FileNotFoundError, "No LF file found"
                    ) as error:
                        extract_snippets(ap, output, t_starts=[0.1], duration=0.1)
                self.assertIn(str(ap), str(error.exception))
                reader.assert_not_called()
                self.assertFalse(output.parent.exists())

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
            with self.assertLogs("ibllib", level="WARNING"):
                out = extract_snippets(
                    ap, self.tmp / "auto.h5", n_snippets=2, duration=0.1, margin=0.0
                )
        args, kwargs = m.call_args
        self.assertAlmostEqual(args[0], 1.0, places=3)
        self.assertEqual(args[1:], (2, 0.1))
        self.assertEqual(kwargs, {"margin": 0.0})
        with h5py.File(out) as f:
            np.testing.assert_array_equal(f[REC]["t_start"][()], [0.1, 0.5])
            self.assertEqual(f[REC].attrs["snippet_method"], "stratified")
            self.assertEqual(f[REC].attrs["snippet_seed"], 0)
            self.assertEqual(f[REC].attrs["snippet_margin"], 0.0)

    def test_automatic_times_with_the_real_helper(self):
        # get_snippet_start_times rounds to whole seconds, so a 1 s recording only has room for one
        ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        out = extract_snippets(
            ap, self.tmp / "auto.h5", n_snippets=1, duration=0.1, margin=0.0
        )
        with h5py.File(out) as f:
            np.testing.assert_array_equal(f[REC]["t_start"][()], [0.0])
            self.assertEqual(f[REC]["ap/data"].shape, (1, 384, 3000))
            self.assertEqual(f[REC].attrs["snippet_method"], "stratified")

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
            histology_raw=[loc, self.tmp / "tracks"],
            t_starts=[0.1],
            duration=0.1,
        )
        with h5py.File(out) as f:
            self.assertEqual(set(f[REC]["histology"]), {"raw"})
            self.assertEqual(
                set(f[REC]["histology/raw"]), {"channel_locations.json", "tracks"}
            )
            self.assertEqual(
                f[REC]["histology/raw/channel_locations.json"].dtype, np.uint8
            )
            self.assertEqual(
                f[REC]["histology/raw/channel_locations.json"][()].tobytes(),
                loc.read_bytes(),
            )
            self.assertEqual(
                f[REC]["histology/raw/tracks/sub/b.csv"][()].tobytes(),
                (self.tmp / "tracks/sub/b.csv").read_bytes(),
            )
            self.assertEqual(
                f[REC]["histology/raw/channel_locations.json"].attrs["source_path"],
                str(loc.resolve()),
            )
            self.assertEqual(
                f[REC]["histology/raw/tracks/sub/b.csv"].attrs["source_path"],
                str((self.tmp / "tracks/sub/b.csv").resolve()),
            )

    def test_name_collision_raises(self):
        # two different files that would both be stored as
        # histology/raw/channel_locations.json
        out = self.tmp / "new" / "c.h5"
        with self.assertRaisesRegex(ValueError, "histology/raw/channel_locations.json"):
            extract_snippets(
                self.ap,
                out,
                histology_raw=[
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
                histology_raw=[self.tmp / "registered_atlas.tiff", folder],
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
                histology_raw=[empty],
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
                histology_raw=[
                    self.a / "channel_locations.json",
                    self.tmp / "nope.json",
                ],
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
                {k: float(v) for k, v in f[REC]["trajectory"].attrs.items()}, traj
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
                all(
                    isinstance(v, np.floating)
                    for v in f[REC]["trajectory"].attrs.values()
                )
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


class TestChannelLocations(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        self.cl = channel_locations_dict(self.ap.with_suffix(".meta"))
        self.out = self.tmp / "new" / "cl.h5"  # parent folder doesn't exist yet

    def _write(self, d):
        path = self.tmp / "cl.json"
        path.write_text(json.dumps(d))
        return path

    def extract(self, **kw):
        return extract_snippets(self.ap, self.out, t_starts=[0.1], duration=0.1, **kw)

    def _assert_table(self, out):
        # channel_locations_dict derives each value from the channel's index in the file
        with h5py.File(out) as f:
            g, ind = f[REC]["histology/channels"], f[REC]["geometry/ind"][()]
            np.testing.assert_array_equal(g["x"][()], (1000.0 + ind) / 1e6)
            np.testing.assert_array_equal(g["y"][()], (-2000.0 + ind) / 1e6)
            np.testing.assert_array_equal(g["z"][()], (-3000.0 - ind) / 1e6)
            np.testing.assert_array_equal(
                g["atlas_id"][()], np.array([477, 672, 0])[ind % 3]
            )
            self.assertEqual(
                list(g["acronym"].asstr()[()]),
                [("STR", "CP", "void")[i % 3] for i in ind],
            )

    def test_table_from_spikeglx_convention(self):
        out = self.extract(channel_locations=self._write(self.cl))
        self._assert_table(out)
        with h5py.File(out) as f:
            g = f[REC]["histology/channels"]
            self.assertEqual(g["x"].dtype, np.float64)
            self.assertEqual(g["atlas_id"].dtype, np.int64)
            self.assertEqual(g.attrs["source"], "cl.json")
            self.assertEqual(g.attrs["atlas"], "Allen CCF 2017")
            np.testing.assert_array_equal(g.attrs["bregma_ccf_um"], [5739, 5400, 332])
            self.assertEqual(
                g.attrs["iblatlas_version"], importlib.metadata.version("iblatlas")
            )
            self.assertEqual(g.attrs["ibleatools_version"], ephysatlas.__version__)
            self.assertEqual(
                f[REC]["histology/raw/cl.json"][()].tobytes(),
                (self.tmp / "cl.json").read_bytes(),
            )
            self.assertNotIn("xyz_picks", f[REC]["histology"])

    def test_rows_follow_the_geometry_when_file_order_differs(self):
        # Permuted snsShankMap entries: the geometry's rows (sorted by position) no
        # longer follow the channel order in the file, so geometry/ind is not 0, 1, 2, ...
        shank_map = spikeglx.read_meta_data(FIXTURES / "np1.ap.meta")["snsShankMap"]
        sites = re.findall(r"\(\d+:\d+:\d+:\d+\)", shank_map)
        permutation = np.random.default_rng(1).permutation(len(sites))
        edits = {
            "snsShankMap": shank_map[: shank_map.index(sites[0])]
            + "".join(sites[i] for i in permutation)
        }
        ap = write_spikeglx_recording(self.tmp / "perm", "np1", 1.0, edits=edits)
        cl = channel_locations_dict(ap.with_suffix(".meta"))
        out = extract_snippets(
            ap,
            self.out,
            channel_locations=self._write(cl),
            t_starts=[0.1],
            duration=0.1,
        )
        with h5py.File(out) as f:
            self.assertFalse(np.array_equal(f[REC]["geometry/ind"][()], np.arange(384)))
        self._assert_table(out)

    def test_probeinterface_convention_matches_by_index(self):
        cl = channel_locations_dict(
            self.ap.with_suffix(".meta"), convention="probeinterface"
        )
        # mirrored and offset positions: a match on position would find nothing
        self.assertNotEqual(cl["channel_0"]["lateral"], self.cl["channel_0"]["lateral"])
        self.assertNotEqual(cl["channel_0"]["axial"], self.cl["channel_0"]["axial"])
        self._assert_table(self.extract(channel_locations=self._write(cl)))

    def test_key_order_does_not_matter(self):
        cl = channel_locations_dict(self.ap.with_suffix(".meta"), with_index=False)
        channels = [key for key in cl if key.startswith("channel_")]
        reordered = {key: cl[key] for key in reversed(channels)}
        reordered["origin"] = cl["origin"]
        self._assert_table(self.extract(channel_locations=self._write(reordered)))

    def test_original_channel_idx_takes_priority(self):
        # entries re-keyed channel_0..383 in reversed ind order: channel_0 now holds the
        # entry of the deepest channel, and only original_channel_idx says where it
        # belongs
        entries = [self.cl[f"channel_{ind}"] for ind in range(384)][::-1]
        rekeyed = {f"channel_{k}": entry for k, entry in enumerate(entries)}
        rekeyed["origin"] = self.cl["origin"]

        # without original_channel_idx the keys decide, which pairs the wrong rows
        without_index = copy.deepcopy(rekeyed)
        for key in without_index:
            without_index[key].pop("original_channel_idx", None)
        with self.assertRaisesRegex(ValueError, "line up"):
            self.extract(channel_locations=self._write(without_index))
        self.assertFalse(self.out.parent.exists())

        self._assert_table(self.extract(channel_locations=self._write(rekeyed)))

    def test_rejected_jsons_create_nothing(self):
        def drop_channel_191(d):
            del d["channel_191"]

        def other_bregma(d):
            d["origin"]["bregma"] = [5700, 5400, 332]

        def no_origin(d):
            del d["origin"]

        def swap_axial(d):
            a, b = d["channel_0"], d["channel_100"]
            a["axial"], b["axial"] = b["axial"], a["axial"]

        def unknown_region(d):
            d["channel_7"]["brain_region_id"] = 123456789

        def entry_without_x(d):
            del d["channel_5"]["x"]

        cases = [
            (drop_channel_191, ["191", "--histology-raw"]),
            (other_bregma, ["bregma"]),
            (no_origin, ["bregma"]),
            (swap_axial, ["line up"]),
            (unknown_region, ["123456789"]),
            (entry_without_x, ["'x'", "channel_5"]),
        ]
        texts = {}
        for change, expected in cases:
            d = copy.deepcopy(self.cl)
            change(d)
            texts[change.__name__] = (json.dumps(d), expected)
        texts["not_json"] = ("{not json", ["cl.json"])
        texts["origin_null"] = (
            json.dumps({**self.cl, "origin": None}),
            ["cl.json", "--histology-raw"],
        )
        texts["top_level_list"] = (
            json.dumps([self.cl]),
            ["cl.json", "--histology-raw"],
        )
        bad_id = copy.deepcopy(self.cl)
        bad_id["channel_3"]["brain_region_id"] = None
        texts["null_region_id"] = (json.dumps(bad_id), ["cl.json", "--histology-raw"])
        bad_key = {**self.cl, "channel_x": {**self.cl["channel_0"]}}
        del bad_key["channel_x"]["original_channel_idx"]
        texts["non_numeric_key"] = (json.dumps(bad_key), ["cl.json", "--histology-raw"])

        for case, (text, expected) in texts.items():
            with self.subTest(case=case):
                (self.tmp / "cl.json").write_text(text)
                with self.assertRaises(ValueError) as ctx:
                    self.extract(channel_locations=self.tmp / "cl.json")
                for part in expected:
                    self.assertIn(part, str(ctx.exception))
                self.assertFalse(self.out.parent.exists())

    def test_reader_row_order_differing_from_meta_raises(self):
        # a different ibl-neuropixel could order the channels differently in the Reader
        # than in geometry_from_meta, which the channel table was built on
        # spikeglx.Reader calls the same function (patched here too), so only the call
        # without return_index, the one from snippet_extractor, is permuted.
        real = spikeglx.geometry_from_meta

        def permuted(md, *args, **kwargs):
            geometry = real(md, *args, **kwargs)
            if kwargs.get("return_index"):  # spikeglx.Reader's own call is left alone
                return geometry
            order = np.arange(geometry["ind"].size)[::-1]
            return {k: v[order] for k, v in geometry.items()}

        with mock.patch(
            "ephysatlas.snippet_extractor.spikeglx.geometry_from_meta", permuted
        ):
            with self.assertRaisesRegex(ValueError, "channel order"):
                self.extract(channel_locations=self._write(self.cl))
        self.assertFalse(self.out.parent.exists())

    def test_multi_shank_recording_refuses_channel_locations(self):
        # The np24_shank fixture is one shank cut out of a four-shank recording: its
        # snsGeomMap still lists all four shanks, and NP2.4_shank selects one. Without
        # that key and with all 384 channels saved, it is a recording that was not split
        # per shank.
        ap = write_spikeglx_recording(
            self.tmp / "np24",
            "np24_shank",
            1.0,
            with_lf=False,
            edits={"NP2.4_shank": None, "nSavedChans": "385"},
        )
        meta = spikeglx.read_meta_data(ap.with_suffix(".meta"))
        self.assertEqual(np.unique(spikeglx.geometry_from_meta(meta)["shank"]).size, 4)
        with self.assertRaisesRegex(ValueError, "single-shank"):
            extract_snippets(
                ap,
                self.out,
                channel_locations=self._write(self.cl),
                t_starts=[0.1],
                duration=0.1,
            )
        self.assertFalse(self.out.parent.exists())

    def test_acronym_mismatch_warns(self):
        for key in ("channel_0", "channel_1"):
            self.cl[key]["brain_region"] = "CA1"
        with self.assertLogs("ephysatlas.snippet_extractor", level="WARNING") as logs:
            out = self.extract(channel_locations=self._write(self.cl))
        self.assertTrue(any("2 channels" in line for line in logs.output), logs.output)
        self._assert_table(out)  # the stored acronyms are iblatlas's

    def test_xyz_picks_stored_in_metres(self):
        picks = self.tmp / "picks.json"
        picks.write_text(json.dumps({"xyz_picks": [[1000, 2000, -3000]]}))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            self.extract(xyz_picks=picks)
        self.assertFalse(self.out.parent.exists())
        picks.write_text(json.dumps([[1, 2, 3], [4, 5, 6]]))  # top level is a list
        with self.assertRaisesRegex(ValueError, "picks.json"):
            self.extract(xyz_picks=picks)
        self.assertFalse(self.out.parent.exists())

        points = [[1000, 2000, -3000], [1100, 2100, -3500]]
        picks.write_text(json.dumps({"xyz_picks": points}))
        out = self.extract(xyz_picks=picks)
        with h5py.File(out) as f:
            histology = f[REC]["histology"]
            self.assertEqual(histology["xyz_picks"].dtype, np.float64)
            np.testing.assert_array_equal(
                histology["xyz_picks"][()], np.array(points) / 1e6
            )
            self.assertEqual(histology["xyz_picks"].attrs["source"], "picks.json")
            self.assertEqual(
                histology["raw/picks.json"][()].tobytes(), picks.read_bytes()
            )
            self.assertNotIn("channels", histology)


class TestAddChannelLocations(unittest.TestCase):
    """add_channel_locations writes a channel table into a received file."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        ap = write_spikeglx_recording(self.tmp, "np1", 1.0)
        self.out = extract_snippets(ap, self.tmp / "a.h5", t_starts=[0.1], duration=0.1)
        with h5py.File(self.out) as f:
            self.geom_ind = f[REC]["geometry/ind"][()]

    def _channels(self, ind=None, atlas_id=477):
        """A channel table in a shuffled order, with x derived from the channel index."""
        ind = self.geom_ind if ind is None else ind
        return pd.DataFrame(
            {"ind": ind, "x": ind / 1e6, "y": 0.0, "z": 0.0, "atlas_id": atlas_id}
        )

    def test_writes_table_matched_by_ind(self):
        order = np.random.default_rng(0).permutation(len(self.geom_ind))
        channels = self._channels(self.geom_ind[order])
        add_channel_locations(
            self.out, "rec_g0_t0.imec0", channels, source="test converter"
        )
        with h5py.File(self.out) as f:
            g = f[REC]["histology/channels"]
            np.testing.assert_array_equal(g["x"][()], self.geom_ind / 1e6)
            self.assertEqual(set(g["acronym"].asstr()[()]), {"STR"})
            self.assertEqual(g.attrs["source"], "test converter")

    def test_refusals(self):
        before = self.out.read_bytes()
        name = "rec_g0_t0.imec0"
        wrong_ind = np.append(self.geom_ind, 100000)  # one channel too many
        duplicated_ind = self.geom_ind.copy()
        duplicated_ind[1] = duplicated_ind[0]
        cases = {
            "no recording": (name.replace("rec", "nope"), self._channels(), name),
            "dropped row": (name, self._channels().iloc[1:], str(self.geom_ind[0])),
            "duplicated ind": (name, self._channels(duplicated_ind), "more than one"),
            "ind not in recording": (name, self._channels(wrong_ind), "100000"),
            "unknown atlas_id": (name, self._channels(atlas_id=123456789), "123456789"),
        }
        for label, (recording, channels, message) in cases.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, message):
                    add_channel_locations(self.out, recording, channels, source="t")
        self.assertEqual(self.out.read_bytes(), before)  # nothing was written
        with self.subTest("second call"):
            add_channel_locations(self.out, name, self._channels(), source="t")
            with self.assertRaisesRegex(ValueError, "already has a channel table"):
                add_channel_locations(self.out, name, self._channels(), source="t")


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
                    "--name",
                    "probe00",
                    "--channel-locations",
                    "cl.json",
                    "--xyz-picks",
                    "picks.json",
                    "--histology-raw",
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
            name="probe00",
            channel_locations=Path("cl.json"),
            xyz_picks=Path("picks.json"),
            histology_raw=[Path("a.json"), Path("b.csv")],
            t_starts=[1.0, 2.5],
            n_snippets=3,
            duration=2.0,
            margin=60.0,
            trajectory=dict(x=1.0, y=2.0, z=3.0, depth=4000.0, theta=15.0, phi=180.0),
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
        self.assertIn("already holds recording", logs.output[0])

    def test_error_lines_show_the_level(self):
        stream = io.StringIO()
        with mock.patch("sys.stdout", stream):
            self.assertEqual(
                main([str(self.tmp / "x.ap.bin"), "-o", str(self.tmp / "o.h5")]), 1
            )
        self.assertTrue(stream.getvalue().startswith("ERROR: "), stream.getvalue())


if __name__ == "__main__":
    unittest.main()
