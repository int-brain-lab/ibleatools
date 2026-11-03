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


if __name__ == "__main__":
    unittest.main()
