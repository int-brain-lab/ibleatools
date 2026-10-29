"""Insertion quality control: the local reference, the levels, the fixture and the loaders' filter."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ephysatlas.qc import levels
from ephysatlas.qc import pipeline as qp
from ephysatlas.qc.data import QCDataset
from ephysatlas.qc.fixture import fixture_labels, write_fixture
from ephysatlas.qc.modelfree import LocalReference, depth_scan, swap_gain
from ephysatlas.qc.pipeline import QCSettings, fake_sources, feature_sd
from ephysatlas.qc.regions import robust_component_scores
from ephysatlas.spatial_encoder.utils import FEATURE_GROUPS, FEATURE_LIST


def level_input(n=400, seed=0) -> pd.DataFrame:
    """Synthetic :func:`levels.insertion_table`: right-skewed positive measurements, as most of
    the real ones are."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=pd.Index([f"p{i:03d}" for i in range(n)], name="pid"))
    for c in levels.measurement_columns():
        df[c] = rng.lognormal(0.0, 0.5, size=n)
    df["sig_bad_channel_frac"] = 0.0
    return df


def z_table(rows: dict) -> pd.DataFrame:
    """z-scores of every test, 0 unless given: ``{pid: {test: z}}``."""
    z = pd.DataFrame(
        0.0, index=pd.Index(list(rows), name="pid"), columns=list(levels.TESTS)
    )
    for pid, values in rows.items():
        for test, v in values.items():
            z.loc[pid, test] = v
    return z


class TestLevels(unittest.TestCase):
    def setUp(self):
        self.df = level_input()
        self.df.loc["p000", "h_lc_lf"] = 1e3  # a gross LF outlier
        self.df.loc["p001", "sig_bad_channel_frac"] = 0.5  # half the channels dead
        self.cal = levels.calibrate(self.df)
        self.table = levels.level_table(self.df, self.cal)

    def test_calibrated_zscores_are_standard(self):
        z = levels.zscores(self.df, self.cal)
        self.assertEqual(list(z.columns), list(levels.TESTS))
        self.assertLess(abs(float(z["wf_signal"].median())), 0.15)
        # After the normalizing transform, about 2.3% of a normal sample is beyond 2 SD.
        self.assertLess(float(z["wf_signal"].gt(2).mean()), 0.06)

    def test_only_the_data_decide(self):
        # Eight failure cases, none of them reading an auto-alignment.
        self.assertEqual(len(levels.TESTS), 8)
        self.assertNotIn("relocation", levels.TESTS)
        self.assertEqual(sorted(self.cal), levels.measurement_columns())
        for col in levels.measurement_columns():
            self.assertNotRegex(col, levels.LEGACY_COLUMNS)
        # A split, or auto-alignment columns left in a table, change nothing.
        n = len(self.df)
        df = self.df.assign(
            split=np.where(np.arange(n) % 2, "train", "test"),
            auto_dist_um=np.linspace(0.0, 3000.0, n),
            a_lc_lf=0.0,
        )
        cal = levels.calibrate(df)
        self.assertEqual(cal, self.cal)
        table = levels.level_table(df, cal)
        for lv in levels.LEVELS:
            pd.testing.assert_series_equal(
                table[f"{lv.name}_bad"], self.table[f"{lv.name}_bad"]
            )

    def test_level_ladder(self):
        rules = {lv.name: (lv.k, lv.m, lv.k2, lv.m2) for lv in levels.LEVELS}
        self.assertEqual(rules["L1"][:3], (3.0, 1, None))
        self.assertEqual(rules["L2"][:3], (2.5, 1, None))
        for name, k2 in (("L3", 1.5), ("L4", 1.25), ("L5", 1.0)):
            self.assertEqual(rules[name], (2.5, 1, k2, 2))
        z = z_table(
            {
                "strong": {"lf_outlier": 3.2},  # one test beyond 3 SD
                "one": {"lf_outlier": 2.7},  # one beyond 2.5 SD only
                "two_mid": {"ap_outlier": 1.6, "depth": 1.6},  # two beyond 1.5 SD
                "two_low": {"ap_outlier": 1.1, "depth": 1.1},  # two beyond 1 SD
                "one_mid": {"ap_outlier": 2.0},  # one moderate sign is not enough
                "clean": {},
            }
        )
        df = pd.DataFrame({"sig_bad_channel_frac": 0.0}, index=z.index)
        bad = pd.DataFrame(
            {lv.name: levels.label(df, z, lv)["bad"] for lv in levels.LEVELS}
        )
        expected = {
            "strong": [True] * 5,
            "one": [False] + [True] * 4,
            "two_mid": [False, False, True, True, True],
            "two_low": [False] * 4 + [True],
            "one_mid": [False] * 5,
            "clean": [False] * 5,
        }
        for pid, row in expected.items():
            self.assertEqual(bad.loc[pid].tolist(), row, pid)

    def test_what_is_bad_per_component(self):
        z = z_table(
            {
                "depth": {"depth": 3.0},
                "swap": {"swap": 3.0},
                "lf_and_spike": {"lf_outlier": 3.0, "wf_outlier": 3.0},
                "lf": {"lf_outlier": 3.0},
                "ap": {"ap_outlier": 3.0},
                "wf": {"wf_signal": 3.0},
            }
        )
        df = pd.DataFrame({"sig_bad_channel_frac": 0.0}, index=z.index)
        t = levels.label(df, z, levels.LEVELS[1])
        self.assertEqual(
            t["bad_type"].to_dict(),
            {
                "depth": "alignment",
                "swap": "alignment",
                "lf_and_spike": "alignment",
                "lf": "lf",
                "ap": "ap+waveform",
                "wf": "waveform",
            },
        )

    def test_levels_are_nested(self):
        t = self.table
        names = [lv.name for lv in levels.LEVELS]
        for a, b in zip(names, names[1:]):
            self.assertTrue((~t[f"{a}_bad"] | t[f"{b}_bad"]).all(), (a, b))
        counts = [int(t[f"{n}_bad"].sum()) for n in names]
        self.assertEqual(counts, sorted(counts))

    def test_review_is_bad_at_the_next_level_only(self):
        t = self.table
        names = [lv.name for lv in levels.LEVELS]
        for a, b in zip(names, names[1:]):
            expected = ~t[f"{a}_bad"] & t[f"{b}_bad"]
            pd.testing.assert_series_equal(
                t[f"{a}_review"], expected, check_names=False
            )
            for comp in levels.LABELS:
                lab = t[f"{a}_{comp}_label"]
                self.assertTrue((lab.eq("bad") == t[f"{a}_{comp}"]).all())
                review = ~t[f"{a}_{comp}"] & t[f"{b}_{comp}"]
                self.assertTrue((lab.eq("review") == review).all())
        self.assertFalse(t[f"{names[-1]}_review"].any())

    def test_what_is_bad(self):
        t = self.table
        self.assertTrue(t.loc["p000", "L2_bad"])
        self.assertEqual(t.loc["p000", "L2_bad_type"], "lf")
        self.assertIn("lf_outlier", t.loc["p000", "L2_failed"])
        # Dead channels fail the AP signal test, whatever its z.
        self.assertTrue(t.loc["p001", "L2_fail_ap_signal"])
        self.assertEqual(t.loc["p001", "L2_bad_type"], "ap+waveform")
        # A bad insertion has at least one bad component.
        comps = t[[f"L2_{c}" for c in levels.LABELS]].any(axis=1)
        self.assertTrue((comps == t["L2_bad"]).all())

    def test_summary_and_bad_list(self):
        s = levels.summary(self.table)
        for lv in levels.LEVELS:
            row = s.loc[lv.name]
            self.assertEqual(row["bad"], int(self.table[f"{lv.name}_bad"].sum()))
            parts = [*levels.BAD_GROUPS, "several bands"]
            self.assertEqual(sum(row[k] for k in parts), row["bad"])
        bad = levels.bad_list(self.table, "L2")
        self.assertEqual(set(bad.index), set(self.table.index[self.table["L2_bad"]]))
        self.assertIn("tests_failed", bad)
        self.assertTrue(bad["n_tests_failed"].is_monotonic_decreasing)

    def test_detection_counts_problems_in_good_sources_only(self):
        t = self.table
        bad_src = t.index[t["L2_bad"]][0]
        good_src = t.index[~t["L5_bad"]][:2].tolist()
        fakes = pd.DataFrame(
            {
                "fake": ["shift"] * 3 + ["lf_noise"] * 3,
                "source_pid": [bad_src, *good_src] * 2,
                "L2_bad": [True, True, False, True, False, False],
            }
        )
        for lv in levels.LEVELS:
            if lv.name != "L2":
                fakes[f"{lv.name}_bad"] = False
        det = levels.detection(t, fakes).set_index(["level", "fake"])
        # The problem planted in the bad source is not counted.
        self.assertEqual(det.loc[("L2", "shift"), ["n", "detected"]].tolist(), [2, 1])
        self.assertEqual(det.loc[("L2", "lf_noise"), "detected"], 0)
        self.assertEqual(
            det.loc[("L2", "unmodified"), ["n", "detected"]].tolist(), [3, 1]
        )
        self.assertEqual(det.loc[("L2", "shift"), "percent"], 50.0)
        s = levels.summary(t, fakes)
        self.assertEqual(s.loc["L2", "shift %"], 50.0)
        self.assertAlmostEqual(s.loc["L2", "sources flagged %"], 33.3)


class TestTables(unittest.TestCase):
    def test_insertion_table(self):
        mf = pd.DataFrame(
            {
                "pid": ["b", "a", "b", "c"],
                "h_lc_lf": [1.0, 2.0, 3.0, 4.0],
                "a_lc_lf": 0.0,  # version 4: at the auto-aligned positions
                "auto_dist_um": 100.0,
            }
        )
        swap = pd.DataFrame(
            {"pid_a": ["a", "a"], "pid_b": ["b", "c"], "swap_gain_mf": [0.1, -0.3]}
        )
        t = levels.insertion_table(mf, swap)
        self.assertEqual(t.index.tolist(), ["a", "b", "c"])
        self.assertEqual(t.columns.tolist(), ["h_lc_lf", "swap_gain_mf"])
        self.assertEqual(
            t.loc["b", "h_lc_lf"], 3.0
        )  # the last row of a pid scored twice
        # The largest gain over the pairs a probe is in.
        np.testing.assert_allclose(t["swap_gain_mf"], [0.1, 0.1, -0.3])
        t = levels.insertion_table(mf, None)
        self.assertTrue(t["swap_gain_mf"].isna().all())

    def test_fakes_table_drops_failures(self):
        fk = pd.DataFrame(
            {
                "fake": ["x", "y", "z"],
                "error": [np.nan, "Boom", ""],
                "auto_dist_um": 1.0,
            }
        )
        out = levels.fakes_table(fk)
        self.assertEqual(out["fake"].tolist(), ["x", "z"])
        self.assertNotIn("auto_dist_um", out)


class TestPlantedProblems(unittest.TestCase):
    """The planted problems: in a random sample of all insertions, not in a model's split."""

    pids = [f"pid{i:04d}" for i in range(1000)]

    def test_fake_sources_are_a_fixed_random_sample(self):
        a = fake_sources(self.pids, 150, seed=0)
        self.assertEqual(len(a), 150)
        self.assertEqual(len(set(a)), 150)
        self.assertTrue(set(a) <= set(self.pids))
        self.assertEqual(a, sorted(a))
        self.assertEqual(a, fake_sources(self.pids, 150, seed=0))  # reproducible
        self.assertNotEqual(a, fake_sources(self.pids, 150, seed=1))
        # Spread over the whole release, not a block of it.
        idx = np.array([self.pids.index(p) for p in a])
        self.assertLess(idx.min(), 100)
        self.assertGreater(idx.max(), 900)
        self.assertEqual(fake_sources(self.pids[:10], 150), self.pids[:10])

    def test_settings_have_no_model_or_auto_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = QCSettings(out_dir=Path(tmp))
            fields = set(s.dump())
        self.assertTrue({"fake_n_sources", "fake_seed"} <= fields)
        self.assertEqual((s.fake_n_sources, s.fake_seed), (150, 0))
        for name in ("auto_alignment_dir", "channel_model"):
            self.assertNotIn(name, fields)
        self.assertEqual(qp.FAKES_CSV, "fakes5.csv")

    def test_stage_fakes_plants_in_the_random_sample(self):
        class Planted(Exception):
            pass

        ds = _dataset()
        with tempfile.TemporaryDirectory() as tmp:
            s = QCSettings(out_dir=Path(tmp), fake_n_sources=4, fake_seed=3)
            with mock.patch.object(qp, "make_fakes", side_effect=Planted) as make:
                with self.assertRaises(Planted):
                    qp.stage_fakes(ds, s, brain_atlas=None, reference=None)
        args, kwargs = make.call_args
        self.assertEqual(args[3], fake_sources(ds.pids, 4, seed=3))
        self.assertEqual(kwargs["seed"], 3)


class TestFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = level_input()
        df.loc["p000", "h_lc_lf"] = 1e3
        cls.table = levels.level_table(df, levels.calibrate(df))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_fixture(cls.table, "2099_W01", "L2", Path(tmp) / "fx.py")
            spec = importlib.util.spec_from_file_location("fx", path)
            cls.fx = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.fx)

    def test_round_trip(self):
        t, fx = self.table, self.fx
        self.assertEqual((fx.VINTAGE, fx.LEVEL), ("2099_W01", "L2"))
        self.assertEqual(fx.excluded_pids(), sorted(t.index[t["L2_bad"]]))
        self.assertEqual(
            fx.excluded_pids(include_review=True), sorted(t.index[t["L3_bad"]])
        )
        self.assertEqual(fx.excluded_pids(("lf",)), sorted(t.index[t["L2_lf"]]))
        self.assertIn("p000", fx.excluded_pids(("lf",)))
        self.assertNotIn("p000", fx.excluded_pids(("alignment",)))
        with self.assertRaises(ValueError):
            fx.excluded_pids(("spikes",))

    def test_labels_and_checks(self):
        t, fx = self.table, self.fx
        good = next(p for p in t.index if not t.loc[p, "L3_bad"])
        self.assertEqual(
            fx.qc_labels(good), {**{c: "good" for c in fx.COMPONENTS}, "checks": ""}
        )
        self.assertEqual(fx.qc_labels("p000")["lf"], "bad")
        self.assertEqual(fx.qc_labels("p000")["checks"], t.loc["p000", "L2_failed"])
        review = [p for p in t.index if t.loc[p, "L2_review"]]
        self.assertTrue(review)
        for p in review:  # only to review: the failure cases of the next level
            labels = fx.qc_labels(p)
            self.assertNotIn("bad", [labels[c] for c in fx.COMPONENTS])
            self.assertEqual(labels["checks"], t.loc[p, "L3_failed"])
        listed = fixture_labels(t, "L2")
        self.assertEqual(set(listed.index), set(fx.pid_qc_filter))


class _AllInBrain:
    """A brain atlas in which every position is inside the brain."""

    @staticmethod
    def get_labels(xyz, mode="clip"):
        return np.ones(len(xyz), dtype=int)


def _features_at(z_um: np.ndarray, rng) -> np.ndarray:
    """``[C, F]`` features that vary smoothly with depth (period of 1.5-3 mm), plus noise."""
    k = np.arange(len(FEATURE_LIST))
    period = 1500.0 + 1500.0 * k / len(k)
    f = np.sin(2 * np.pi * z_um[:, None] / period + k)
    return f + 0.05 * rng.normal(size=f.shape)


def _dataset(seed=0):
    """Six reference probes along vertical tracks 50-150 µm apart (4 mm, every 20 µm), and the
    probes of the tests: ``moved`` (recorded at one depth, placed 500 µm shallower), ``lf_off``
    (its LF band from elsewhere) and two probes ``a`` / ``b`` recorded on each other's tracks."""
    rng = np.random.default_rng(seed)
    n = 200
    z_ref = -1000.0 - 20.0 * np.arange(n)  # top first, µm
    probes = []
    for x in (-150.0, -100.0, -50.0, 50.0, 100.0, 150.0):
        probes.append(
            (np.c_[np.full(n, x), np.zeros(n), z_ref], _features_at(z_ref, rng))
        )
    z = z_ref[50:150]  # 2 mm in the middle of the reference
    probes.append(
        (np.c_[np.zeros(100), np.zeros(100), z + 500.0], _features_at(z, rng))
    )
    f = _features_at(z, rng)
    lf = [FEATURE_LIST.index(c) for c in FEATURE_GROUPS["lf"]]
    f[:, lf] = _features_at(z - 1500.0, rng)[:, lf]
    probes.append((np.c_[np.zeros(100), np.zeros(100), z], f))
    za, zb = z_ref[:60], z_ref[-60:]
    xyz = np.c_[np.zeros(60), np.zeros(60), za]
    probes.append((xyz, _features_at(zb, rng)))
    probes.append((np.c_[np.zeros(60), np.zeros(60), zb], _features_at(za, rng)))
    c = max(len(p[0]) for p in probes)
    xyz = np.full((len(probes), c, 3), np.nan)
    feats = np.full((len(probes), c, len(FEATURE_LIST)), np.nan)
    for i, (x, f) in enumerate(probes):
        xyz[i, : len(x)], feats[i, : len(f)] = x * 1e-6, f
    pids = [f"ref{i}" for i in range(6)] + ["moved", "lf_off", "a", "b"]
    return QCDataset(
        pids=np.array(pids),
        features=feats,
        human_xyz=xyz,
        planned_xyz=xyz,
        channel_labels=np.zeros(xyz.shape[:2], dtype=np.int8),
        probe_model=np.array(["NP1"] * len(pids)),
        feature_names=list(FEATURE_LIST),
    )


class TestModelFree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = _dataset()
        cls.local = LocalReference(cls.ds, feature_sd(cls.ds))
        cls.local.fit_noise(cls.ds)

    def probe(self, pid):
        i = self.ds.index(pid)
        return dict(
            features=self.ds.features[i],
            xyz=self.ds.human_xyz[i],
            labels=self.ds.channel_labels[i],
            probe=i,
        )

    def band_scores(self, pid, xyz=None):
        p = self.probe(pid)
        xyz = p["xyz"] if xyz is None else xyz
        return self.local.band_scores(p["features"], xyz, p["labels"], [p["probe"]])

    def test_noise_is_fitted(self):
        noise = self.local.noise
        self.assertEqual(noise.shape, (len(FEATURE_LIST),))
        self.assertTrue(np.isfinite(noise).all() and (noise > 0).all())

    def test_a_foreign_band_is_unusual_for_where_the_probe_is(self):
        ok = self.band_scores("lf_off")
        moved_back = self.ds.human_xyz[self.ds.index("moved")].copy()
        moved_back[:, 2] -= 500e-6
        right = self.band_scores("moved", moved_back)
        self.assertGreater(ok["lf"], 3 * right["lf"])
        self.assertLess(ok["ap"], 2 * right["ap"])

    def test_depth_scan_finds_the_offset(self):
        p = self.probe("moved")
        summary, profile = depth_scan(
            self.local, _AllInBrain(), p["features"], p["xyz"], p["labels"], p["probe"]
        )
        self.assertEqual(summary["depth_best"], 500.0)  # back down, deeper
        self.assertGreater(summary["depth_gain"], 0.0)
        self.assertEqual(set(profile), {"all", *levels.COMPONENTS})

    def test_swap_gain(self):
        a, b, ref = self.probe("a"), self.probe("b"), self.probe("ref0")
        self.assertGreater(
            swap_gain(self.local, a, b), 0.0
        )  # recorded on each other's track
        self.assertLess(swap_gain(self.local, ref, self.probe("ref5")), 0.1)

    def test_robust_band_score_ignores_one_noisy_feature(self):
        names = [f for g in FEATURE_GROUPS.values() for f in g]
        z = pd.DataFrame({f"z{i}": [0.5, 0.5] for i in range(len(names))})
        z["probe"], z["level"], z["n_core"], z["source"] = 0, "Beryl", 20, "Beryl"
        z[f"z{names.index(FEATURE_GROUPS['ap'][0])}"] = 10.0  # one AP feature far off
        self.assertAlmostEqual(
            robust_component_scores(z, names).loc[0, "beryl_ap"], 0.5
        )


class TestLoadersExcludePids(unittest.TestCase):
    """``exclude_pids`` of ``LoadInsertionData``: the misaligned list by default (the released
    models were trained without it), any list on request."""

    def load(self, **kwargs):
        from ephysatlas.fixtures import misaligned_pids
        from ephysatlas.spatial_encoder import utils

        self.misaligned = sorted(misaligned_pids)[0]
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

    def test_default_leaves_out_the_misaligned_list(self):
        self.assertEqual(self.load(), [p for p in self.pids if p != self.misaligned])

    def test_explicit_lists(self):
        self.assertEqual(self.load(exclude_pids=[]), self.pids)
        kept = self.load(exclude_pids=["kept-2"])
        self.assertEqual(kept, [p for p in self.pids if p != "kept-2"])


if __name__ == "__main__":
    unittest.main()
