"""Snippet extractor checks on the example datasets (skipped when they are not mounted)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import spikeglx
from scipy.fft import next_fast_len

from ephysatlas.feature_calculators import SpikeGLXFileFeatureCalculator
from ephysatlas.feature_calculators.snippet_file import SnippetFileFeatureCalculator
from ephysatlas.feature_calculators.types import (
    FeatureComputationOptions,
    SnippetWindow,
)
from ephysatlas.snippet_extractor import extract_snippets

DATA = Path("/mnt/s0/Data")
BENCH = DATA / "2026_spike_sorting_benchmarks/recordings"
NP1_BENCH = BENCH / "1a276285-8b0e-4cc9-9f0a-a3a002978724"
NP2_BENCH = BENCH / "f420a08b-9355-453d-8ed6-cadb89bae21d"
AP_NAME = "_spikeglx_ephysData_g0_t0.imec0.ap.cbin"


@unittest.skipUnless(DATA.exists(), "example datasets not mounted")
class TestSnippetExtractorRealData(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_extracts_each_example_type(self):
        cases = [
            (
                "cazettes",
                DATA
                / "2026_cazettes/2026_cazettes/Data/VF066/2025_12_04/Rec/probe00/disabled_g0_t0.imec0.ap.cbin",
                True,
                1,
            ),
            (
                "nuo_li",
                DATA
                / "2026_nuo_li/Munni/20210527_g0_imec0/20210527_g0_t0.imec0.ap.bin",
                True,
                1,
            ),
            (
                "bench_3a",
                BENCH
                / "11a5a93e-58a9-4ed0-995e-52279ec16b98/_spikeglx_ephysData_g0_t0.imec.ap.cbin",
                True,
                1,
            ),
            ("bench_np24", NP2_BENCH / AP_NAME, False, 2),
        ]
        for name, ap_file, has_lf, version in cases:
            with self.subTest(name):
                out = extract_snippets(
                    ap_file, self.tmp / f"{name}.h5", t_starts=[600.0], duration=1.0
                )
                with h5py.File(out, "r") as f:
                    n_samples = next_fast_len(int(f["ap"].attrs["fs"]), real=True)
                    self.assertEqual(
                        f["ap/data"].shape, (1, f["ap/data"].shape[1], n_samples)
                    )
                    self.assertEqual("lf" in f, has_lf)
                    self.assertEqual(int(f.attrs["neuropixel_version"]), version)

    def test_np1_features_match_spikeglx_calculator(self):
        ap_file = NP1_BENCH / AP_NAME
        lf_file = ap_file.with_name(AP_NAME.replace(".ap.", ".lf."))
        out = extract_snippets(
            ap_file, self.tmp / "np1.h5", t_starts=[600.0], duration=5.0
        )
        options = FeatureComputationOptions(
            features_to_compute=["lf", "csd", "ap"],
            channel_labels=np.zeros(384),
            include_trajectory=False,
        )
        ours_calc = SnippetFileFeatureCalculator(out)
        ref_calc = SpikeGLXFileFeatureCalculator(ap_file=ap_file, lf_file=lf_file)
        ours = ours_calc.compute_snippet(ours_calc.windows[0], options)
        ref = ref_calc.compute_snippet(SnippetWindow(600.0, 5.0, 5.0), options)
        ours_df = ours.features.sort_values("channel").reset_index(drop=True)
        ref_df = ref.features.sort_values("channel").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            ours_df[ref_df.columns], ref_df, check_dtype=False
        )

    def test_np2_derived_lf_matches_np2converter_file(self):
        out = extract_snippets(
            NP2_BENCH / AP_NAME, self.tmp / "np2.h5", t_starts=[600.0], duration=5.0
        )
        calc = SnippetFileFeatureCalculator(out)
        raw_lf = calc.load_raw_snippet(calc.windows[0]).raw_lf
        sr = spikeglx.Reader(NP2_BENCH / AP_NAME.replace(".ap.", ".lf."))
        n0, n = int(sr.fs * 600), raw_lf.shape[1]
        ref = sr[n0 : n0 + n, :96].T
        # NP2Converter stores int16, so one step of the gain is the tolerance. The edges
        # are skipped because filtering a 5 s piece differs from the continuous file there.
        np.testing.assert_allclose(
            raw_lf[:, 25:-25], ref[:, 25:-25], atol=sr.sample2volts.max()
        )


if __name__ == "__main__":
    unittest.main()
