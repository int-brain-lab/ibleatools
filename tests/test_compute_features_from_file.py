"""End-to-end test for the collapsed ``compute_features_from_file``.

This drives the *real* feature engine (``compute_features_from_raw``) through the
public ``compute_features_from_file`` entry point, with the ``spikeglx.Reader``
objects replaced by fake readers backed by the checked-in destriped AP/LF
fixtures (``tests/fixtures/{ap,lf}_destriped.npy``). It is network-free and
touches no filesystem for input: only ``SpikeGLXFileFeatureCalculator._open_reader``
is patched, so the real lazy ``sr_ap``/``sr_lf`` properties and geometry-derived
channel metadata are exercised. The trajectory path uses the real
``add_target_coordinates`` (pure trajectory math, no Alyx).

The fixtures are fed as the raw AP/LF snippet (the same way
``test_feature_computation.py`` uses them), so the engine destripes and computes
features for real. There are no golden feature values for this path, so we assert
the expected *schema* (one row per channel, the lf/csd/ap feature columns, merged
channel metadata + trajectory), that values are populated, the ``channels.pqt``
write under the stem-named probe dir, the snippet manifest ``.attrs``, and that
the LF result is deterministic.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ephysatlas.feature_calculators.spikeglx import SpikeGLXFileFeatureCalculator
from ephysatlas.feature_computation import compute_features_from_file
from ephysatlas.utils import get_aggregated_snippets_df

# Fake file paths; the readers are mocked so these are never opened. The AP stem
# ("probe.ap") becomes the calculator name and the probe-directory name.
AP_FILE = "/fake/probe.ap.cbin"
LF_FILE = "/fake/probe.lf.cbin"
NAME_AP = Path(AP_FILE).stem  # "probe.ap" -> the stem-based output dir name
NAME_LF = Path(LF_FILE).stem  # "probe.lf" (used when AP is absent)
FS_AP = 30000.0
FS_LF = 2500.0
# Real (network-free) trajectory dict -> real add_target_coordinates adds targets.
TRAJ = {"x": 1000, "y": 2000, "z": 3000, "depth": 4000, "theta": 0, "phi": 0}

FIXTURES = Path(__file__).parent / "fixtures"
AP_DATA = np.load(FIXTURES / "ap_destriped.npy").astype(np.float32)  # (channels, ns)
LF_DATA = np.load(FIXTURES / "lf_destriped.npy").astype(np.float32)  # (channels, ns)
N_CH = AP_DATA.shape[0]


def _geometry(n: int) -> dict:
    """Simple single-column geometry (matches test_feature_computation.py)."""
    return {
        "x": np.zeros(n),
        "y": np.arange(n) * 20.0,
        "sample_shift": np.zeros(n),
        "shank": np.zeros(n),
        "col": np.zeros(n),
        "row": np.arange(n),
    }


class _FakeReader:
    """``spikeglx.Reader``-like stub over an in-memory (channels, samples) array."""

    def __init__(self, data: np.ndarray, fs: float) -> None:
        self._data = np.ascontiguousarray(data.T)  # (samples, channels)
        self.fs = float(fs)
        self.ns = int(data.shape[1])
        self.nc = int(data.shape[0])
        self.nsync = 0
        self.file_bin = None  # forces the snippet-level bad-channel fallback
        self.geometry = _geometry(int(data.shape[0]))

    def __getitem__(self, item):
        return self._data[item]


def _run(output_dir, features, with_ap=True):
    """Call compute_features_from_file with fixture-backed fake readers."""
    ap_reader = _FakeReader(AP_DATA, FS_AP) if with_ap else None
    lf_reader = _FakeReader(LF_DATA, FS_LF)
    ap_file = AP_FILE if with_ap else None

    def _fake_open(self, band):
        return ap_reader if band == "ap" else lf_reader

    with mock.patch.object(
        SpikeGLXFileFeatureCalculator,
        "_open_reader",
        autospec=True,
        side_effect=_fake_open,
    ):
        return compute_features_from_file(
            ap_file=ap_file,
            lf_file=LF_FILE,
            t_start=0.0,
            duration_ap=0.5,
            duration_lf=0.5,
            # Pass a fresh copy: add_target_coordinates rewrites traj["theta"]/
            # traj["phi"] in place (tilt correction), so a shared dict would be
            # mutated across runs and break the determinism check.
            traj_dict=dict(TRAJ),
            features_to_compute=features,
            output_dir=output_dir,
        )


class TestComputeFeaturesFromFile(unittest.TestCase):
    def test_end_to_end_lf_csd_ap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            df = _run(out, ["lf", "csd", "ap"])

            # One row per channel, in channel order.
            self.assertEqual(len(df), N_CH)
            np.testing.assert_array_equal(df["channel"].to_numpy(), np.arange(N_CH))

            # Expected feature columns from each family.
            for col in ("rms_lf", "psd_delta", "psd_alpha", "psd_gamma", "psd_lfp"):
                self.assertIn(col, df.columns)  # lf
            for col in ("rms_ap", "cor_ratio"):
                self.assertIn(col, df.columns)  # ap
            self.assertTrue(
                any(c.endswith("_csd") for c in df.columns), "no CSD columns found"
            )

            # Channel metadata and trajectory are merged into the returned frame.
            for col in (
                "axial_um",
                "lateral_um",
                "shank",
                "x_target",
                "y_target",
                "z_target",
            ):
                self.assertIn(col, df.columns)

            # Physical-site columns (the merge keys) come from geometry: this is a
            # single-column probe (lateral_um == 0, shank == 0, axial_um == y).
            np.testing.assert_allclose(
                df["axial_um"].to_numpy(), np.arange(N_CH) * 20.0
            )
            np.testing.assert_array_equal(df["lateral_um"].to_numpy(), np.zeros(N_CH))
            np.testing.assert_array_equal(df["shank"].to_numpy(), np.zeros(N_CH))

            # Values are populated (not all-NaN).
            self.assertTrue(np.isfinite(df["rms_lf"].to_numpy(float)).all())
            self.assertTrue(np.isfinite(df["rms_ap"].to_numpy(float)).all())

            # channels.pqt written under the stem-named probe-level dir.
            channels_pqt = out / NAME_AP / "channels.pqt"
            self.assertTrue(channels_pqt.exists())
            cdf = pd.read_parquet(channels_pqt)
            for col in ("channel", "axial_um", "lateral_um", "shank", "x_target"):
                self.assertIn(col, cdf.columns)

            # Snippet manifest .attrs the aggregation layer reads. The file path
            # stamps "filename" (compute_features_from_pid stamps "pid").
            manifest = get_aggregated_snippets_df(out / NAME_AP)
            self.assertEqual(manifest["filename"].iloc[0], AP_FILE)
            self.assertEqual(float(manifest["duration_ap"].iloc[0]), 0.5)

    def test_lf_result_is_deterministic(self):
        # LF features are deterministic: identical fixtures -> identical frame.
        with (
            tempfile.TemporaryDirectory() as tmp1,
            tempfile.TemporaryDirectory() as tmp2,
        ):
            a = _run(Path(tmp1), ["lf"], with_ap=False)
            b = _run(Path(tmp2), ["lf"], with_ap=False)
        a = a.sort_values("channel").reset_index(drop=True)
        b = b.sort_values("channel").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
