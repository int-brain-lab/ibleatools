"""Tests for ephysatlas.realtime helpers that need neither recordings nor the released model."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ephysatlas.realtime.features import (
    aggregate_snippets,
    feature_matrix,
    snippet_samples,
)
from ephysatlas.realtime.plan import load_plan, save_plan


def _np1_geometry():
    import neuropixel

    return {k: np.asarray(v) for k, v in neuropixel.trace_header(version=1).items()}


class TestFeatureMatrix(unittest.TestCase):
    def test_rows_run_from_the_top_channel(self):
        table = pd.DataFrame({"channel": np.arange(4), "rms_ap": [0.0, 1.0, 2.0, 3.0]})
        x = feature_matrix(table, ["rms_ap", "spike_count"], n_channels=4)
        np.testing.assert_array_equal(x[:, 0], [3.0, 2.0, 1.0, 0.0])
        self.assertTrue(np.isnan(x[:, 1]).all())  # absent features are NaN

    def test_snippet_lengths_match_the_release(self):
        self.assertEqual(snippet_samples(30000.0), 150000)
        self.assertEqual(snippet_samples(2500.0), 12500)


class TestAggregateSnippets(unittest.TestCase):
    def test_median_then_db_and_psd_alpha_kept_per_channel(self):
        geometry = _np1_geometry()
        rng = np.random.default_rng(0)
        nc = len(geometry["x"])
        snippets = [
            pd.DataFrame(
                {
                    "channel": np.arange(nc),
                    "rms_ap": np.full(nc, 1e-5) * (1 + 0.01 * k),
                    "psd_alpha": -100.0
                    + np.linspace(0, 10, nc)
                    + rng.normal(0, 1e-3, nc),
                    "channel_labels": np.zeros(nc),
                }
            )
            for k in range(3)
        ]
        agg = aggregate_snippets(snippets, geometry, name="p")
        # rms in dB of the median snippet (TV denoising leaves a constant profile unchanged)
        np.testing.assert_allclose(agg["rms_ap"], 20 * np.log10(1.01e-5), atol=1e-3)
        # psd_alpha keeps its depth profile (the release's alpha-outlier fallback would flatten it)
        self.assertGreater(np.ptp(agg["psd_alpha"].to_numpy()), 5.0)


class TestTestRecordingFiles(unittest.TestCase):
    def test_sglx_file_names_keep_the_band(self):
        from ephysatlas.realtime.data import sglx_files

        b, m = sglx_files(Path("rec"), "143dd7cf-6a47-47a1-906d-927ad7fe9117", "lf")
        self.assertEqual(b.name, "143dd7cf_g0_t0.imec0.lf.bin")
        self.assertEqual(m.name, "143dd7cf_g0_t0.imec0.lf.meta")

    def test_meta_describes_the_window(self):
        from ephysatlas.realtime.data import _edit_meta

        meta = (
            "nSavedChans=385\nfileSizeBytes=999\nfileTimeSecs=1.0\nimSampRate=30000\n"
        )
        out = _edit_meta(
            meta, n_samples=300, n_channels=385, fs=30000.0, first_sample=9000
        )
        self.assertIn("fileSizeBytes=231000", out)
        self.assertIn("fileTimeSecs=0.01", out)
        self.assertIn("firstSample=9000", out)
        self.assertIn("nSavedChans=385", out)


class TestPlanFile(unittest.TestCase):
    def test_round_trip(self):
        xyz = np.random.default_rng(1).normal(0, 1e-3, size=(384, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_plan(
                dict(pid="abc", source="test", planned_xyz_m=xyz), Path(tmp) / "p.json"
            )
            plan = load_plan(path)
        np.testing.assert_allclose(plan["planned_xyz_m"], xyz, atol=1e-9)
        self.assertEqual(plan["pid"], "abc")
        self.assertEqual(plan["n_channels"], 384)


def _pinpoint_doc(atlas="allen_mouse", names=("Probe A",), model="NP1010"):
    """A Pinpoint v5 experiment of vertical probes with their tips 3 mm below bregma; contact c
    of the probeinterface definition sits 20 (c / 2) + 20 µm above the bottom of the contour."""
    from ephysatlas.realtime.plan import PINPOINT_ALLEN_BREGMA_MM

    bregma = list(PINPOINT_ALLEN_BREGMA_MM) if atlas == "allen_mouse" else [5.0, 0.5, 5.5]
    tip = [bregma[0], bregma[1] + 3.0, bregma[2]]  # ASR mm: 3 mm ventral of bregma
    definition = dict(
        si_units="um",
        annotations=dict(model_name=model),
        probe_planar_contour=[[0, 0], [70, 0], [70, 10000], [0, 10000]],
        contact_positions=[[16 + 32 * (c % 2), 20 * (c // 2) + 20] for c in range(960)],
    )
    probes = [
        dict(
            name=n,
            probeInterfaceIdentifier="np1",
            tipPosition=tip,
            rotation=[0.0, 0.0, np.pi / 2],  # pitch 90°: shank vertical, up = -DV in ASR
        )
        for n in names
    ]
    atlas_doc = dict(name=atlas, manifest=dict(bregma=bregma))
    return dict(
        version="5.0.0-test",
        atlas=atlas_doc,
        probes=probes,
        probeInterfaceProbes=dict(np1=definition),
    )


def _pinpoint_np2_doc():
    """A Pinpoint v5 experiment of one vertical NP2014 (four shanks 250 µm apart, 1280
    electrodes each, rows 15 µm apart) with its tip 3 mm below bregma, the tip-most row 175 µm
    above the tip."""
    doc = _pinpoint_doc(model="NP2014")
    rows, cols = np.arange(1280) // 2, np.arange(1280) % 2
    positions, shank_ids = [], []
    for s in range(4):
        positions += [[int(250 * s + 27 + 32 * c), int(15 * r)] for r, c in zip(rows, cols)]
        shank_ids += [str(s)] * 1280
    doc["probeInterfaceProbes"]["np1"] = dict(
        si_units="um",
        annotations=dict(
            model_name="NP2014", shank_tips=[[250 * s + 35, -175] for s in range(4)]
        ),
        probe_planar_contour=[[0, -175], [820, -175], [820, 10000], [0, 10000]],
        contact_positions=positions,
        shank_ids=shank_ids,
    )
    return doc


class TestPinpointPlan(unittest.TestCase):
    def _write(self, tmp, doc):
        import json

        path = Path(tmp) / "experiment.json"
        path.write_text(json.dumps(doc))
        return path

    def test_vertical_probe_at_bregma(self):
        from ephysatlas.realtime.plan import plan_from_pinpoint

        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_from_pinpoint(self._write(tmp, _pinpoint_doc()))
        xyz = plan["planned_xyz_m"]
        self.assertEqual(xyz.shape, (384, 3))
        np.testing.assert_allclose(xyz[:, :2], 0.0, atol=1e-12)
        # Row r is channel 383 - r, 20 ((383 - r) / 2) + 20 µm above the tip at DV -3 mm.
        channel = 383 - np.arange(384)
        np.testing.assert_allclose(xyz[:, 2], -3e-3 + (20 * (channel // 2) + 20) * 1e-6, atol=1e-12)
        self.assertEqual(plan["probe"], "Probe A")

    def test_mri_atlas_is_scaled_to_the_ccf(self):
        from ephysatlas.realtime.plan import PINPOINT_QIU2018_SCALE, plan_from_pinpoint

        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_from_pinpoint(self._write(tmp, _pinpoint_doc("qiu2018_mouse")))
        tip_z = plan["planned_xyz_m"][-1, 2]  # channel 0, 20 µm above the tip
        self.assertAlmostEqual(tip_z, (-3.0 + 0.02) / PINPOINT_QIU2018_SCALE[2] * 1e-3, places=12)

    def test_probe_choice(self):
        from ephysatlas.realtime.plan import pinpoint_probe_names

        doc = _pinpoint_doc(names=("Probe A", "Probe B"))
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, doc)
            self.assertEqual(pinpoint_probe_names(path), ["Probe A", "Probe B"])
            with self.assertRaisesRegex(ValueError, "2 probes"):
                load_plan(path)
            with self.assertRaisesRegex(ValueError, "no probe Probe C"):
                load_plan(path, probe="Probe C")
            self.assertEqual(load_plan(path, probe="Probe B")["probe"], "Probe B")

    def test_neuropixels_2_gives_the_shank_line(self):
        from ephysatlas.realtime.probe import ProbeLayout

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, _pinpoint_np2_doc())
            plan = load_plan(path, shank=0)
            self.assertEqual(plan["warning"], "")
            # one position per contact row of shank 0, top first, 15 µm apart
            heights = plan["heights_um"]
            self.assertEqual(len(heights), 640)
            np.testing.assert_allclose(heights[[0, -1]], [20 + 15 * 639, 20])
            xyz = plan["planned_xyz_m"]
            np.testing.assert_allclose(xyz[:, :2], 0.0, atol=1e-12)
            # tip 3 mm below bregma, the tip-most row 175 µm above it
            np.testing.assert_allclose(xyz[-1, 2], -3e-3 + 175e-6, atol=1e-12)
            saved = load_plan(save_plan(plan, Path(tmp) / "plan.json"))
            np.testing.assert_allclose(saved["heights_um"], heights)
            # a recording of shank 0, bank 0: the rows lie on that line
            layout = ProbeLayout.from_sites(
                np.zeros(384),
                27 + 32 * (np.arange(384) % 2),
                15 * (np.arange(384) // 2),
                part_number="NP2014",
                has_lf=False,
            )
            rows = layout.plan_rows(saved["planned_xyz_m"], saved["heights_um"])
            np.testing.assert_allclose(
                rows[:, 2], -3e-3 + 155e-6 + layout.row_height_um * 1e-6, atol=1e-9
            )
            other = load_plan(path, shank=2)
            self.assertIn("lateral offset (500 µm) is not applied", other["warning"])
            with self.assertRaisesRegex(ValueError, "4 shanks"):
                load_plan(path, shank=4)


class TestNoPlanPrior(unittest.TestCase):
    def test_no_centre_or_direction_term(self):
        from ephysatlas.alignment.ephys_only import (
            DepthBins,
            PlanPrior,
            ScoreConfig,
        )
        from ephysatlas.realtime.whole_brain import NoPlanPrior

        cfg = ScoreConfig()
        bins = DepthBins.from_valid(np.ones(384, dtype=bool), cfg)
        params = dict(
            centre=dict(x=(5.0, 0.0, 300.0), y=(5.0, 0.0, 500.0), z=(5.0, 0.0, 500.0)),
            axis=(5.0, 4.5),
            length=(1.0, 0.04),
            bend=(4.0, 4.2),
        )
        # A vertical plan (row r 10 r µm below the top, which is 100 µm below bregma), the
        # trajectory along it, and the same trajectory tilted by 20° and moved by a few mm.
        rows = np.arange(384) * cfg.channel_spacing_um
        planned = np.column_stack([np.zeros(384), np.zeros(384), -(rows + 100.0)]) * 1e-6
        depth = np.asarray(bins.centers_um, dtype=float)
        straight = np.column_stack([np.zeros_like(depth), np.zeros_like(depth), -(depth + 100.0)])
        theta = np.deg2rad(20.0)
        rot = np.array(
            [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
        )
        moved = straight @ rot.T + np.array([-2000.0, -3000.0, -1000.0])
        traces = np.stack([straight, moved])
        no_plan = NoPlanPrior(params, cfg)
        cost = no_plan.evaluate(traces, no_plan.reference(None, bins))
        self.assertAlmostEqual(cost[0], cost[1], places=9)
        with_plan = PlanPrior(params, cfg)
        plan_cost = with_plan.evaluate(traces, with_plan.reference(planned, bins))
        self.assertGreater(plan_cost[1], plan_cost[0] + 10.0)


def _np2_sites(electrodes, shanks):
    e = np.asarray(electrodes)
    return np.asarray(shanks, dtype=float), 27.0 + 32 * (e % 2), 15.0 * (e // 2)


class TestProbeLayout(unittest.TestCase):
    def test_a_standard_np1_is_used_as_it_is(self):
        from ephysatlas.realtime.probe import ProbeLayout, np1_heights_um

        c = np.arange(384)
        layout = ProbeLayout.from_sites(
            np.zeros(384), 11 + 16 * (c % 4), np1_heights_um() - 20, part_number="NP1000"
        )
        self.assertTrue(layout.identity)
        self.assertEqual(layout.n_rows, 384)
        v = np.random.default_rng(0).normal(size=(384, 3))
        np.testing.assert_array_equal(layout.rows(v), v[::-1].astype(np.float32))
        np.testing.assert_array_equal(layout.plan_rows(v), v)
        np.testing.assert_array_equal(layout.channel_xyz(v), v[::-1])
        sentinel = object()
        self.assertIs(layout.localizer(sentinel), sentinel)

    def test_four_shanks_one_shank_at_a_time(self):
        from ephysatlas.alignment.ephys_only import AnnealConfig
        from ephysatlas.realtime.probe import ProbeLayout, np1_heights_um

        c = np.arange(384)
        s, x, z = _np2_sites(c % 96, c // 96)  # 96 channels at the bottom of each shank
        layout = ProbeLayout.from_sites(s, x, z, part_number="NP2014", has_lf=False)
        self.assertFalse(layout.identity)
        self.assertEqual(layout.shank, 0)  # ties: the first shank
        self.assertEqual(layout.n_shanks, 4)
        np.testing.assert_array_equal(np.sort(layout.channels), np.arange(96))
        np.testing.assert_allclose(layout.row_height_um[[0, -1]], [725.0, 25.0])
        self.assertEqual(layout.neuropixel_version, 2)
        # values linear in height are resampled exactly
        h = layout.z_um + 20.0
        per_channel = np.column_stack([h, 2 * h])
        np.testing.assert_allclose(
            layout.rows(per_channel)[:, 0], layout.row_height_um, atol=1e-3
        )
        # a straight 1.0 plan is read as a line, evaluated at the rows' heights
        up = np.array([0.1, 0.2, 0.97]) / np.linalg.norm([0.1, 0.2, 0.97])
        tip = np.array([-1e-3, -2e-3, -4e-3])
        plan = tip + (np1_heights_um()[::-1, None] * 1e-6) * up
        rows = layout.plan_rows(plan)
        np.testing.assert_allclose(
            rows, tip + layout.row_height_um[:, None] * 1e-6 * up, atol=1e-12
        )
        np.testing.assert_allclose(
            layout.channel_xyz(rows), tip + h[:, None] * 1e-6 * up, atol=1e-12
        )
        third = ProbeLayout.from_sites(s, x, z, has_lf=False, shank=3)
        np.testing.assert_array_equal(np.sort(third.channels), 288 + np.arange(96))

        class Loc:
            anneal_cfg = AnnealConfig()

        loc = Loc()
        clone = layout.localizer(loc)
        f = 700.0 / 3830.0
        self.assertAlmostEqual(clone.anneal_cfg.probe_length_um, 3840.0 * f)
        self.assertAlmostEqual(clone.anneal_cfg.min_probe_length_um, 3500.0 * f)
        self.assertEqual(loc.anneal_cfg.probe_length_um, 3840.0)

    def test_gaps_and_short_shanks(self):
        from ephysatlas.realtime.probe import ProbeLayout

        e = np.r_[np.arange(64), 200 + np.arange(64)]  # two blocks 1 mm apart
        s, x, z = _np2_sites(e, np.zeros(len(e)))
        layout = ProbeLayout.from_sites(s, x, z, has_lf=False)
        rows = layout.rows(np.ones((len(e), 1)))
        in_gap = (layout.row_height_um > 20 + 15 * 31 + 40) & (
            layout.row_height_um < 20 + 15 * 100 - 40
        )
        self.assertTrue(in_gap.any())
        self.assertTrue(np.isnan(rows[in_gap]).all())
        outside = (layout.row_height_um <= 20 + 15 * 31) | (
            layout.row_height_um >= 20 + 15 * 100
        )
        self.assertTrue(np.isfinite(rows[outside]).all())
        s, x, z = _np2_sites(np.arange(16), np.zeros(16))
        with self.assertRaisesRegex(ValueError, "at least 200"):
            ProbeLayout.from_sites(s, x, z, has_lf=False)
        with self.assertRaisesRegex(ValueError, "records no channel"):
            ProbeLayout.from_sites(*_np2_sites(np.arange(384), np.zeros(384)), shank=1)

    def test_spikeglx_geom_map(self):
        from ephysatlas.realtime.probe import ProbeLayout

        s, x, z = _np2_sites(np.arange(384) + 384, np.full(384, 2))
        kv = dict(head_partNumber="NP2013", head_numShanks="4")
        for c in range(384):
            kv.update({f"ch{c}_s": s[c], f"ch{c}_x": x[c], f"ch{c}_z": z[c], f"ch{c}_u": 1})
        layout = ProbeLayout.from_geom_map(kv, 384, has_lf=False)
        self.assertEqual(
            (layout.part_number, layout.n_shanks, layout.shank), ("NP2013", 4, 2)
        )
        np.testing.assert_allclose(
            layout.row_height_um[[0, -1]], [20 + 15 * 383, 20 + 15 * 192 + 5]
        )
        default = ProbeLayout.from_geom_map({}, 384, has_lf=True)  # no site map: a 1.0
        self.assertTrue(default.identity)

    def test_lf_from_wideband(self):
        from ephysatlas.realtime.probe import lf_from_wideband

        fs = 30000.0
        t = np.arange(150000) / fs
        x = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 5000 * t)]) + 0.3
        lf, fs_lf = lf_from_wideband(x.astype(np.float32), fs)
        self.assertEqual(lf.shape, (2, 12500))
        self.assertEqual(fs_lf, 2500.0)
        mid = slice(1000, -1000)
        self.assertAlmostEqual(float(np.std(lf[0, mid])), np.sqrt(0.5), places=2)
        self.assertLess(float(np.std(lf[1, mid])), 1e-3)


if __name__ == "__main__":
    unittest.main()
