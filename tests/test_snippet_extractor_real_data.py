"""Snippet extractor checks on the example datasets (skipped when they are not mounted)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.fft import next_fast_len

from ephysatlas.snippet_extractor import extract_snippets

DATA = Path("/mnt/s0/Data")
BENCH = DATA / "2026_spike_sorting_benchmarks/recordings"
NP2_BENCH = BENCH / "f420a08b-9355-453d-8ed6-cadb89bae21d"
VF066 = DATA / "2026_cazettes/2026_cazettes/Data/VF066/2025_12_04"
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
                    # The default name is the AP file name without .ap.bin/.ap.cbin.
                    rec = f["recordings"][ap_file.name.split(".ap.")[0]]
                    n_samples = next_fast_len(int(rec["ap"].attrs["fs"]), real=True)
                    self.assertEqual(
                        rec["ap/data"].shape, (1, rec["ap/data"].shape[1], n_samples)
                    )
                    self.assertEqual("lf" in rec, has_lf)
                    self.assertEqual(int(rec.attrs["neuropixel_version"]), version)

    def test_two_recordings_in_one_file(self):
        out = self.tmp / "vf066.h5"
        for probe in ("probe00", "probe01"):
            extract_snippets(
                VF066 / f"Rec/{probe}/disabled_g0_t0.imec{probe[-1]}.ap.cbin",
                out,
                t_starts=[600.0],
                duration=1.0,
            )
        names = ["disabled_g0_t0.imec0", "disabled_g0_t0.imec1"]
        with h5py.File(out) as file:
            self.assertEqual(list(file["recordings"]), names)
            first = file[f"recordings/{names[0]}/ap/data"][()]
            second = file[f"recordings/{names[1]}/ap/data"][()]
        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
