"""Tests for the destripe/feature split.

``compute_features_from_raw`` was split into ``destripe_ap_lf`` (the shared
destriping primitive) and ``compute_features_from_destriped`` (features computed
on already-destriped arrays). This checks:
  * ``destripe_ap_lf`` output shapes and per-band ``None`` handling,
  * ``compute_features_from_raw`` == ``destripe_ap_lf`` -> ``compute_features_from_destriped``,
  * ``rms_lf_no_car`` is gated purely by whether ``des_lf_no_car`` is supplied, and
  * ``BaseFeatureCalculator.get_destriped_snippet`` delegates to ``destripe_ap_lf``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import neuropixel

from ephysatlas.feature_calculators import DestripeOptions, SnippetWindow
from ephysatlas.feature_calculators.spikeglx import SpikeGLXFileFeatureCalculator
from ephysatlas.feature_computation import (
    compute_features_from_destriped,
    compute_features_from_raw,
    destripe_ap_lf,
)

FIXTURES = Path(__file__).parent / "fixtures"
AP_DATA = np.load(FIXTURES / "ap_destriped.npy").astype(np.float32)
LF_DATA = np.load(FIXTURES / "lf_destriped.npy").astype(np.float32)
FS_AP = 30000.0
FS_LF = 2500.0
GEOMETRY = neuropixel.trace_header(version=1)


class TestDestripeApLf(unittest.TestCase):
    def test_shapes_and_per_band_none(self):
        des_ap, des_lf = destripe_ap_lf(
            AP_DATA, LF_DATA, fs_ap=FS_AP, fs_lf=FS_LF, geometry=GEOMETRY
        )
        self.assertEqual(des_ap.shape, AP_DATA.shape)
        self.assertEqual(des_lf.shape, LF_DATA.shape)

        # An absent band yields None for that band only.
        none_ap, only_lf = destripe_ap_lf(None, LF_DATA, fs_lf=FS_LF, geometry=GEOMETRY)
        self.assertIsNone(none_ap)
        self.assertEqual(only_lf.shape, LF_DATA.shape)
        only_ap, none_lf = destripe_ap_lf(AP_DATA, None, fs_ap=FS_AP, geometry=GEOMETRY)
        self.assertIsNone(none_lf)
        self.assertEqual(only_ap.shape, AP_DATA.shape)


class TestSplitComposesToRaw(unittest.TestCase):
    def test_raw_equals_destripe_then_features(self):
        # compute_features_from_raw must equal the explicit two-step composition.
        with (
            tempfile.TemporaryDirectory() as t1,
            tempfile.TemporaryDirectory() as t2,
        ):
            df_raw = compute_features_from_raw(
                raw_ap=None,
                raw_lf=LF_DATA,
                fs_lf=FS_LF,
                geometry=GEOMETRY,
                features_to_compute=["lf"],
                output_dir=Path(t1),
            )
            _, des_lf = destripe_ap_lf(
                None, LF_DATA, fs_lf=FS_LF, geometry=GEOMETRY, lf_k_filter=False
            )
            df_split = compute_features_from_destriped(
                None,
                des_lf,
                fs_lf=FS_LF,
                geometry=GEOMETRY,
                features_to_compute=["lf"],
                output_dir=Path(t2),
            )
        pd.testing.assert_frame_equal(df_raw, df_split)


class TestRmsLfNoCarGate(unittest.TestCase):
    def setUp(self):
        _, self.des_lf = destripe_ap_lf(
            None, LF_DATA, fs_lf=FS_LF, geometry=GEOMETRY, lf_k_filter=False
        )
        _, self.des_lf_no_car = destripe_ap_lf(
            None, LF_DATA, fs_lf=FS_LF, geometry=GEOMETRY, lf_k_filter=None
        )

    def _run(self, des_lf_no_car):
        with tempfile.TemporaryDirectory() as tmp:
            return compute_features_from_destriped(
                None,
                self.des_lf,
                fs_lf=FS_LF,
                geometry=GEOMETRY,
                des_lf_no_car=des_lf_no_car,
                features_to_compute=["lf"],
                output_dir=Path(tmp),
            )

    def test_absent_without_no_car_array(self):
        self.assertNotIn("rms_lf_no_car", self._run(None).columns)

    def test_present_with_no_car_array(self):
        df = self._run(self.des_lf_no_car)
        self.assertIn("rms_lf_no_car", df.columns)
        self.assertTrue(np.isfinite(df["rms_lf_no_car"].to_numpy()).all())


# --- get_destriped_snippet delegation (BaseFeatureCalculator) ---------------
N_CH = 4
_AP = np.arange(N_CH * 3000, dtype=np.float32).reshape(N_CH, 3000)
_LF = np.arange(N_CH * 300, dtype=np.float32).reshape(N_CH, 300) * 0.5


def _geom(n: int) -> dict:
    return {
        "x": np.zeros(n),
        "y": np.arange(n) * 20.0,
        "sample_shift": np.zeros(n),
        "shank": np.zeros(n),
        "col": np.zeros(n),
        "row": np.arange(n),
    }


class _FakeReader:
    def __init__(self, data: np.ndarray, fs: float) -> None:
        self._data = np.ascontiguousarray(data.T)
        self.fs = float(fs)
        self.ns = int(data.shape[1])
        self.nc = int(data.shape[0])
        self.nsync = 0
        self.file_bin = None
        self.geometry = _geom(int(data.shape[0]))

    def __getitem__(self, item):
        return self._data[item]


class TestGetDestripedSnippetDelegates(unittest.TestCase):
    def test_delegates_to_shared_primitive_and_wraps_result(self):
        calc = SpikeGLXFileFeatureCalculator(ap_file="a.ap.bin", lf_file="b.lf.bin")
        calc._sr_ap = _FakeReader(_AP, 30000.0)
        calc._sr_lf = _FakeReader(_LF, 2500.0)
        sentinel_ap = np.ones((N_CH, 8))
        sentinel_lf = np.ones((N_CH, 4))

        with mock.patch(
            "ephysatlas.feature_calculators.base.destripe_ap_lf",
            return_value=(sentinel_ap, sentinel_lf),
        ) as m:
            snip = calc.get_destriped_snippet(
                SnippetWindow(t_start=0.0, duration_ap=0.05, duration_lf=0.05),
                DestripeOptions(lf_k_filter=None, ap_k_filter=True, nshank=2),
                channel_labels=np.zeros(N_CH),
            )

        # Delegated once, forwarding the DestripeOptions knobs.
        m.assert_called_once()
        kwargs = m.call_args.kwargs
        self.assertIs(kwargs["ap_k_filter"], True)
        self.assertIsNone(kwargs["lf_k_filter"])
        self.assertEqual(kwargs["nshank"], 2)
        # The primitive's output is wrapped into the DestripedSnippet.
        self.assertIs(snip.des_ap, sentinel_ap)
        self.assertIs(snip.des_lf, sentinel_lf)
        self.assertIsNotNone(snip.raw)
        self.assertIn("x", snip.geometry)


if __name__ == "__main__":
    unittest.main()
