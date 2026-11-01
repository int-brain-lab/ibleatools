"""Insertion quality control: the region reference, the measures, the levels, the fixture and the
loaders' filter."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ephysatlas.qc import cases, levels, measures
from ephysatlas.qc import pipeline as qp
from ephysatlas.qc.fixture import fixture_labels, write_fixture
from ephysatlas.qc.regions import (
    BIN_CHANNELS,
    RegionReference,
    bin_probe,
    parcel_labels,
)
from ephysatlas.spatial_encoder.utils import FEATURE_LIST

BANDS = {"lf": [0, 1], "ap": [2, 3], "wf": [4, 5]}
N_BINS, N_FEATURES = 38, 6


class Atlas:
    """Regions in layers 500 µm thick and columns 2 mm wide. Allen regions are the layers of a
    column, Beryl regions join two layers, Cosmos regions are the columns. Above z = 0 is outside
    the brain."""

    @staticmethod
    def get_labels(xyz, mapping="Allen", mode="clip"):
        xyz = np.asarray(xyz)
        layer = np.floor(-xyz[..., 2] / 500e-6).astype(int)
        column = np.floor((xyz[..., 0] + 4e-3) / 2e-3).astype(int)
        fine = {"Allen": layer, "Beryl": layer // 2, "Cosmos": 0 * layer}[mapping]
        ids = 1 + column * 100 + fine
        return np.where(xyz[..., 2] > 0, 0, ids)


def track(x_mm: float, top_mm: float = 0.0) -> np.ndarray:
    """``[B, 3]`` positions of a vertical probe at ``x_mm``, its top at depth ``top_mm``."""
    z = -(top_mm + 0.05 + 0.1 * np.arange(N_BINS)) * 1e-3
    return np.column_stack([np.full(N_BINS, x_mm * 1e-3), np.zeros(N_BINS), z])


def recording(xyz: np.ndarray, rng, noise: float = 0.1) -> np.ndarray:
    """``[B, F]`` what a probe at ``xyz`` records: each Allen region has its own values."""
    region = Atlas.get_labels(xyz)
    pattern = np.random.default_rng(1).normal(size=(2000, N_FEATURES)) * 2
    return pattern[region] + rng.normal(scale=noise, size=(len(xyz), N_FEATURES))


def dataset(n_columns: int = 3, per_column: int = 15, seed: int = 0):
    """Probes in columns at x = -3, -1, 1 mm: ``z``, ``xyz`` and their :class:`RegionReference`."""
    rng = np.random.default_rng(seed)
    xyz = np.array(
        [
            track(-3 + 2 * c + rng.uniform(-0.3, 0.3))
            for c in range(n_columns)
            for _ in range(per_column)
        ]
    )
    z = np.array([recording(x, rng) for x in xyz], dtype=np.float32)
    return z, xyz, RegionReference(z, parcel_labels(xyz, Atlas))


class TestRegionReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z, cls.xyz, cls.ref = dataset()

    def test_bins_are_medians_of_the_channels_with_a_signal(self):
        n = BIN_CHANNELS * 3
        features = np.ones((n, 2))
        features[0] = 99.0  # an odd channel
        labels = np.zeros(n, dtype=int)
        labels[BIN_CHANNELS : 2 * BIN_CHANNELS - 2] = 1  # most of the second bin dead
        values, positions = bin_probe(features, labels, np.zeros((n, 3)))
        self.assertEqual(values.shape, (3, 2))
        self.assertEqual(values[0, 0], 1.0)
        self.assertTrue(np.isnan(values[1]).all())
        self.assertEqual(positions.shape, (3, 3))

    def test_the_probe_is_left_out_of_its_own_reference(self):
        region = int(Atlas.get_labels(self.xyz[0])[5])
        probes, values = self.ref.tables["Allen"][region]
        full = self.ref.stats("Allen", region)
        left_out = self.ref.stats("Allen", region, exclude=[int(probes[0])])
        np.testing.assert_allclose(full[0], np.median(values, axis=0))
        np.testing.assert_allclose(left_out[0], np.median(values[1:], axis=0))

    def test_a_region_with_few_probes_falls_back_to_the_coarser_parcellation(self):
        labels = parcel_labels(self.xyz, Atlas)
        labels["Allen"][:3, 5:8] = 9999  # a small region, in only three probes
        ref = RegionReference(self.z, labels)
        self.assertIsNone(ref.stats("Allen", 9999))
        one = {name: v[0] for name, v in labels.items()}
        self.assertTrue(np.isfinite(ref.residuals(self.z[0], one, [0])[5:8]).all())

    def test_expected_gives_the_residuals(self):
        labels = parcel_labels(self.xyz[0], Atlas)
        median, spread = self.ref.expected(labels, [0])
        np.testing.assert_allclose(
            self.ref.residuals(self.z[0], labels, [0]), (self.z[0] - median) / spread
        )
        self.assertTrue((spread >= self.ref.floor).all())

    def test_bins_outside_the_brain_have_no_residual(self):
        xyz = self.xyz[0] + np.array([0.0, 0.0, 1.0e-3])  # 1 mm too shallow
        e = self.ref.residuals(self.z[0], parcel_labels(xyz, Atlas), [0])
        self.assertTrue(np.isnan(e[:5]).all())
        self.assertTrue(np.isfinite(e[20:]).all())

    def test_residuals_are_in_mads_of_the_region(self):
        labels = parcel_labels(self.xyz[0], Atlas)
        self.assertLess(
            np.nanmedian(np.abs(self.ref.residuals(self.z[0], labels, [0]))), 2.0
        )
        shifted = self.z[0] + 10.0
        self.assertGreater(
            np.nanmedian(np.abs(self.ref.residuals(shifted, labels, [0]))), 5.0
        )


class TestMeasures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z, cls.xyz, cls.ref = dataset()
        cls.rng = np.random.default_rng(3)
        expected = [cls.ref.expected(parcel_labels(x, Atlas))[0] for x in cls.xyz]
        cls.tracks = measures.TrackProfiles(np.stack(expected), BANDS)
        cls.base, _ = measures.measure_probe(
            cls.ref, Atlas, cls.z[0], cls.xyz[0], 0, BANDS, cls.tracks
        )

    def scores(self, z, xyz=None, probe=0):
        xyz = self.xyz[0] if xyz is None else xyz
        return measures.measure_probe(
            self.ref, Atlas, z, xyz, probe, BANDS, self.tracks
        )[0]

    def test_an_offset_of_a_band_raises_its_level_not_its_shape(self):
        e = np.random.default_rng(0).normal(size=(N_BINS, N_FEATURES))
        s0 = measures.band_scores(e, BANDS)
        e[:, BANDS["lf"]] += 5.0
        s = measures.band_scores(e, BANDS)
        self.assertGreater(s["lf_level"], 3 * s0["lf_level"])
        self.assertAlmostEqual(s["lf_shape"], s0["lf_shape"], places=5)
        self.assertEqual(s["ap_level"], s0["ap_level"])

    def test_noise_on_a_band_raises_its_shape(self):
        z = self.z[0].copy()
        z[:, BANDS["wf"]] += self.rng.normal(scale=2.0, size=(N_BINS, 2))
        s = self.scores(z)
        self.assertGreater(s["wf_shape"], 3 * self.base["wf_shape"])

    def test_a_band_broken_on_half_of_the_probe_raises_its_shape(self):
        z = self.z[0].copy()
        z[N_BINS // 2 :, BANDS["ap"]] += 5.0
        self.assertGreater(self.scores(z)["ap_shape"], 3 * self.base["ap_shape"])

    def test_one_noisy_feature_does_not_flag_a_band(self):
        band = {"lf": [0, 1, 2]}
        e = np.zeros((N_BINS, 3))
        e[:, 0] = 50.0
        out = measures.band_scores(e, band)
        self.assertEqual((out["lf_level"], out["lf_shape"]), (0.0, 0.0))

    def test_a_recording_follows_the_profile_of_its_regions(self):
        self.assertGreater(self.base["fit"], 0.9)
        moved = self.xyz[0] + np.array([0.0, 0.0, -1000e-6])
        self.assertLess(self.scores(self.z[0], moved)["fit"], 0.5)

    def test_an_offset_does_not_change_the_fit(self):
        z = self.z[0] + 5.0
        self.assertAlmostEqual(self.scores(z)["fit"], self.base["fit"], places=4)

    def test_each_band_counts_once_in_the_fit(self):
        a = np.arange(N_BINS, dtype=float)[:, None] * np.ones((1, 3))
        b = a.copy()
        b[:, 2] = -b[:, 2]
        out = measures.fit(a, b, {"many": [0, 1], "one": [2]})[0]
        self.assertAlmostEqual(out, 0.0, places=6)

    def test_too_few_bins_have_no_fit(self):
        expected = self.z[0].copy()
        expected[measures.MIN_SCAN_BINS - 1 :] = np.nan
        out, n = measures.fit(self.z[0], expected, BANDS)
        self.assertTrue(np.isnan(out))
        self.assertEqual(n, measures.MIN_SCAN_BINS - 1)

    def test_the_depth_scan_finds_a_moved_probe(self):
        moved = self.xyz[0] + np.array([0.0, 0.0, -500e-6])  # said to be 500 µm deeper
        s = self.scores(self.z[0], moved)
        self.assertGreater(s["depth"], 10 * max(self.base["depth"], 0.01))
        self.assertEqual(s["depth_best"], -500.0)  # it fits 500 µm shallower

    def test_the_depth_scan_reaches_two_millimetres(self):
        self.assertEqual(max(measures.SHIFTS_UM), 2000.0)
        moved = self.xyz[0] + np.array([0.0, 0.0, -1750e-6])
        self.assertEqual(self.scores(self.z[0], moved)["depth_best"], -1750.0)

    def test_the_sideways_scan_finds_a_moved_probe(self):
        moved = self.xyz[0] + np.array([1e-3, 0.0, 0.0])
        s = self.scores(self.z[0], moved)
        self.assertGreater(s["sideways"], 0.3)
        self.assertLess(s["sideways_best"], 0)

    def test_a_probe_where_it_belongs_gains_nothing_by_moving(self):
        self.assertLess(self.base["depth"], 0.1)
        self.assertLess(self.base["sideways"], 0.1)

    def test_a_move_out_of_the_brain_does_not_count(self):
        self.assertTrue(np.isnan(measures.gain((0.1, 20), (0.9, 9))))
        self.assertAlmostEqual(measures.gain((0.1, 20), (0.9, 10)), 0.8)

    def test_a_recording_follows_its_own_track_better_than_the_others(self):
        self.assertLess(
            self.base["track_rank"], 0.4
        )  # only the tracks of its own column

    def test_a_recording_on_the_wrong_track_ranks_low(self):
        moved = self.xyz[0] + np.array([0.0, 0.0, -1000e-6])
        self.assertGreater(self.scores(self.z[0], moved)["track_rank"], 0.5)

    def test_the_tracks_left_out_are_not_compared(self):
        fits = self.tracks.fits(self.z[0], [0, 1])
        self.assertTrue(np.isnan(fits[:2]).all())
        self.assertTrue(np.isfinite(fits[2:]).all())

    def test_swap_gains(self):
        a, b = (self.z[0], self.xyz[0]), (self.z[20], self.xyz[20])
        gains = measures.swap_gains(self.ref, Atlas, a, b, (0, 20), BANDS)
        self.assertLess(max(gains), 0)
        swapped_a, swapped_b = (self.z[0], self.xyz[20]), (self.z[20], self.xyz[0])
        gains = measures.swap_gains(
            self.ref, Atlas, swapped_a, swapped_b, (0, 20), BANDS
        )
        self.assertGreater(min(gains), 0)

    def test_dead_fraction(self):
        self.assertEqual(measures.dead_fraction(np.array([0, 1, 2, 3])), 0.5)


def measurements(n=400, seed=0) -> pd.DataFrame:
    """Synthetic measurements: normal around their medians, as a release of good probes."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=pd.Index([f"p{i:03d}" for i in range(n)], name="pid"))
    for test in levels.TESTS:
        df[test] = rng.normal(1.0, 0.1, size=n)
    df["track_rank"] = 0.05
    df["swap_gain"] = np.nan
    df["dead_fraction"] = 0.0
    return df


def with_problem(df, pid, **values) -> pd.DataFrame:
    df = df.copy()
    for column, v in values.items():
        df.loc[pid, column] = v
    return df


class TestLevels(unittest.TestCase):
    def setUp(self):
        self.df = measurements()
        self.cal = levels.calibrate(self.df)

    def table(self, df):
        return levels.level_table(df, self.cal)

    def test_zscores_are_mad_zscores(self):
        z = levels.zscores(self.df, self.cal)
        self.assertLess(z.median().abs().max(), 1e-9)
        mad = (z - z.median()).abs().median()
        np.testing.assert_allclose(mad, 1 / levels.MAD_TO_SD)

    def test_only_the_data_decide(self):
        for test in levels.TESTS:
            self.assertIn(test, self.df)
        self.assertEqual(set(levels.calibrate(self.df)), set(levels.TESTS))

    def test_the_ladder(self):
        """A test just beyond each limit marks the probe from that level on."""
        for test, limit in (("lf_level", "k"), ("depth", "k_position")):
            cal = self.cal[test]
            for level in levels.LEVELS:
                x = cal["median"] + (getattr(level, limit) + 0.1) * cal["sd"]
                t = self.table(with_problem(self.df, "p000", **{test: x}))
                self.assertEqual(t.loc["p000", "first_level"], level.name)
                self.assertEqual(t.loc["p000", f"{level.name}_failed"], test)

    def test_levels_are_nested(self):
        t = self.table(measurements(seed=1).assign(depth=lambda d: d["depth"] * 2))
        names = [lv.name for lv in levels.LEVELS]
        for a, b in zip(names, names[1:]):
            self.assertTrue((t[f"{b}_bad"] | ~t[f"{a}_bad"]).all())

    def test_review_is_bad_at_the_next_level_only(self):
        cal = self.cal["ap_level"]
        k = levels.LEVELS[2].k + 0.1  # L3
        t = self.table(
            with_problem(self.df, "p000", ap_level=cal["median"] + k * cal["sd"])
        )
        self.assertTrue(t.loc["p000", "L2_review"])
        self.assertFalse(t.loc["p000", "L3_review"])
        self.assertEqual(t.loc["p000", "L2_ap_label"], "review")
        self.assertEqual(t.loc["p000", "L3_ap_label"], "bad")

    def test_fixed_tests_fail_at_every_level(self):
        df = with_problem(self.df, "p000", swap_gain=0.5)
        df = with_problem(df, "p001", dead_fraction=0.5)
        t = self.table(df)
        self.assertEqual(t.loc["p000", "first_level"], "L1")
        self.assertEqual(t.loc["p000", "L1_failed"], "swap")
        self.assertEqual(t.loc["p001", "L1_failed"], "channels")

    def test_no_swap_test_without_a_partner(self):
        t = self.table(with_problem(self.df, "p000", swap_gain=np.nan))
        self.assertFalse(t.loc["p000", "L5_fail_swap"])

    def test_what_is_bad_per_component(self):
        z = {t: 0.0 for t in levels.TESTS}
        cases = {
            "lf_level": {"lf"},
            "lf_shape": {"lf"},
            "ap_level": {"ap", "waveform"},
            "ap_shape": {"ap", "waveform"},
            "wf_level": {"waveform"},
            "wf_shape": {"waveform"},
            "depth": {"alignment"},
            "sideways": {"alignment"},
        }
        for test, bad in cases.items():
            df = pd.DataFrame(index=["p"])
            df["swap_gain"], df["dead_fraction"] = np.nan, 0.0
            frame = pd.DataFrame({**z, test: 9.0}, index=["p"])
            row = levels.label(df, frame, levels.LEVELS[0]).loc["p"]
            self.assertEqual({c for c in levels.LABELS if row[c]}, bad, test)
        df = pd.DataFrame({"swap_gain": [np.nan], "dead_fraction": [0.5]}, index=["p"])
        row = levels.label(df, pd.DataFrame(z, index=["p"]), levels.LEVELS[0]).loc["p"]
        self.assertEqual({c for c in levels.LABELS if row[c]}, {"ap", "waveform"})
        df = pd.DataFrame({"track_rank": [0.95]}, index=["p"])
        row = levels.label(df, pd.DataFrame(z, index=["p"]), levels.LEVELS[0]).loc["p"]
        self.assertEqual({c for c in levels.LABELS if row[c]}, {"alignment"})

    def test_groups(self):
        df = with_problem(self.df, "p000", swap_gain=0.5)
        df = with_problem(df, "p001", dead_fraction=0.5)
        cal = self.cal["wf_shape"]
        df = with_problem(df, "p002", wf_shape=cal["median"] + 9 * cal["sd"])
        g = levels.group_failures(self.table(df), "L1")
        self.assertEqual(g.loc["p000"].to_dict()["elsewhere"], True)
        self.assertEqual(g.loc["p001"].to_dict()["channels"], True)
        self.assertEqual(g.loc["p002"].to_dict()["regions"], True)
        self.assertEqual(int(g.any(axis=1).sum()), 3)

    def test_summary_and_bad_list(self):
        t = self.table(with_problem(self.df, "p000", dead_fraction=0.5))
        s = levels.summary(t)
        self.assertEqual(list(s.index), [lv.name for lv in levels.LEVELS])
        self.assertGreaterEqual(s.loc["L1", "bad"], 1)
        bad = levels.bad_list(t, "L1")
        self.assertIn("p000", bad.index)
        self.assertEqual(bad.loc["p000", "first_level"], "L1")

    def test_detection_compares_the_same_insertions(self):
        table = self.table(with_problem(self.df, "p000", dead_fraction=0.5))
        planted = self.table(self.df.iloc[:4]).reset_index(drop=True)
        planted["source_pid"] = ["p000", "p000", "p001", "p001"]
        planted["fake"] = ["a", "b", "a", "b"]
        planted.loc[0, "L1_bad"] = (
            True  # planted in an insertion already bad: it counts
        )
        planted.loc[3, "L1_bad"] = True
        d = levels.detection(table, planted).set_index(["level", "fake"])
        self.assertEqual(d.loc[("L1", "as recorded"), "n"], 2)
        self.assertEqual(d.loc[("L1", "as recorded"), "detected"], 1)
        self.assertEqual(d.loc[("L1", "a"), "detected"], 1)
        self.assertEqual(d.loc[("L1", "b"), "percent"], 50.0)


class TestExampleFigures(unittest.TestCase):
    """Which tests the figure of an insertion shows."""

    def setUp(self):
        self.df = measurements()
        self.cal = levels.calibrate(self.df)

    def at(self, test, z):
        c = self.cal[test]
        return {test: c["median"] + z * c["sd"]}

    def shown(self, **values):
        t = levels.level_table(with_problem(self.df, "p000", **values), self.cal)
        t["first_level"] = t["first_level"].fillna("")
        return cases.tests_to_show(t.loc["p000"])

    def test_the_fixed_tests_come_first(self):
        shown = self.shown(swap_gain=0.5, **self.at("depth", 9))
        self.assertEqual(shown, ["swap", "depth"])

    def test_every_figure_shows_the_same_features(self):
        names = set(cases.FEATURES["lf"] + cases.FEATURES["ap"] + cases.FEATURES["wf"])
        from ephysatlas.spatial_encoder.utils import FEATURE_GROUPS

        known = {f for group in FEATURE_GROUPS.values() for f in group}
        self.assertLessEqual(names, known)
        for band, group in (("lf", "lf"), ("ap", "ap"), ("wf", "waveform")):
            self.assertLessEqual(set(cases.FEATURES[band]), set(FEATURE_GROUPS[group]))

    def test_a_band_that_fails_adds_no_row(self):
        shown = self.shown(**self.at("lf_level", 9), **self.at("depth", 6))
        self.assertEqual(shown, ["depth"])

    def test_the_tests_of_position_by_how_far_beyond_their_limits(self):
        shown = self.shown(track_rank=0.99, **self.at("depth", 5.2))
        self.assertEqual(shown, ["track", "depth"])

    def test_at_most_two_rows(self):
        shown = self.shown(
            swap_gain=0.5,
            dead_fraction=0.5,
            track_rank=0.95,
            **self.at("depth", 9),
        )
        self.assertEqual(len(shown), cases.MAX_ROWS)

    def test_an_insertion_no_level_marks_shows_the_test_closest_to_a_limit(self):
        t = levels.level_table(self.df, self.cal)
        t["first_level"] = t["first_level"].fillna("")
        row = t[t["first_level"].eq("")].iloc[0]
        shown = cases.tests_to_show(row)
        self.assertEqual(len(shown), 1)
        self.assertIn(shown[0], ("depth", "sideways", "track"))


class TestPlantedProblems(unittest.TestCase):
    def test_fake_sources_are_a_fixed_random_sample(self):
        pids = [f"p{i:04d}" for i in range(1000)]
        a = qp.fake_sources(pids, 50, seed=0)
        self.assertEqual(a, qp.fake_sources(pids, 50, seed=0))
        self.assertEqual(a, sorted(a))
        self.assertEqual(len(set(a)), 50)
        self.assertNotEqual(a, qp.fake_sources(pids, 50, seed=1))
        self.assertNotEqual(a, sorted(pids)[:50])

    def test_settings_have_no_model_or_auto_alignment(self):
        names = " ".join(qp.QCSettings(out_dir=Path(".")).dump())
        for word in ("model", "auto", "split"):
            self.assertNotIn(word, names)


class TestFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = measurements()
        cal = levels.calibrate(df)
        for pid, test in (("p000", "lf_level"), ("p001", "depth")):
            df.loc[pid, test] = cal[test]["median"] + 9 * cal[test]["sd"]
        df.loc["p002", "ap_shape"] = (
            cal["ap_shape"]["median"] + 4.7 * cal["ap_shape"]["sd"]
        )  # bad at L4 only
        cls.table = levels.level_table(df, cal)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_fixture(cls.table, "2099_W01", "L3", Path(tmp) / "fx.py")
            spec = importlib.util.spec_from_file_location("fx", path)
            cls.fx = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.fx)

    def test_round_trip(self):
        t, fx = self.table, self.fx
        self.assertEqual((fx.VINTAGE, fx.LEVEL), ("2099_W01", "L3"))
        self.assertEqual(fx.excluded_pids(), sorted(t.index[t["L3_bad"]]))
        self.assertEqual(
            fx.excluded_pids(include_review=True), sorted(t.index[t["L4_bad"]])
        )
        self.assertEqual(fx.excluded_pids(("lf",)), ["p000"])
        self.assertEqual(fx.excluded_pids(("alignment",)), ["p001"])
        with self.assertRaises(ValueError):
            fx.excluded_pids(("spikes",))

    def test_labels_and_checks(self):
        fx = self.fx
        self.assertEqual(
            fx.qc_labels("p399"), {**{c: "good" for c in fx.COMPONENTS}, "checks": ""}
        )
        self.assertEqual(fx.qc_labels("p000")["lf"], "bad")
        self.assertEqual(fx.qc_labels("p000")["checks"], "lf_level")
        labels = fx.qc_labels("p002")  # only to review: the tests of the next level
        self.assertEqual((labels["ap"], labels["waveform"]), ("review", "review"))
        self.assertEqual(labels["checks"], "ap_shape")

    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            fixture_labels(self.table, "L9")


class TestLoadersExcludePids(unittest.TestCase):
    """``exclude_pids`` of ``LoadInsertionData``: the QC filter's bad insertions by default, any
    list on request."""

    def load(self, **kwargs):
        from ephysatlas.fixtures import pid_qc_filter
        from ephysatlas.spatial_encoder import utils

        self.misaligned = sorted(pid_qc_filter.excluded_pids())[0]
        self.pids = sorted(
            ["kept-1", self.misaligned, "kept-2"]
        )  # read sorted, as on disk
        pids = self.pids
        index = pd.MultiIndex.from_product([pids, range(4)], names=["pid", "channel"])
        df = pd.DataFrame(1.0, index=index, columns=list(FEATURE_LIST))
        for c in ("x", "y", "z", "x_target", "y_target", "z_target"):
            df[c] = 0.0
        with (
            mock.patch.object(utils, "ONE"),
            mock.patch.object(utils, "download_tables"),
            mock.patch.object(utils, "read_features_from_disk", return_value=df),
        ):
            return list(utils.LoadInsertionData(VINTAGE="2099_W01", **kwargs)[0])

    def test_default_leaves_out_the_qc_bad_insertions(self):
        self.assertEqual(self.load(), [p for p in self.pids if p != self.misaligned])

    def test_explicit_lists(self):
        self.assertEqual(self.load(exclude_pids=[]), self.pids)
        kept = self.load(exclude_pids=["kept-2"])
        self.assertEqual(kept, [p for p in self.pids if p != "kept-2"])


if __name__ == "__main__":
    unittest.main()
