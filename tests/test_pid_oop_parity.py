"""Parity tests: ``compute_features_from_pid_oop`` must match ``compute_features_from_pid``.

The procedural entry point and its OOP twin are driven with the *same* mocked
readers / channel metadata, and the shared engine ``compute_features_from_raw``
is replaced by a spy. This pins the orchestration parity deterministically, with
no network or heavy DSP:

* identical engine inputs (the only thing that sets feature values),
* an identical returned DataFrame,
* matching ``channels.pqt`` key columns, and
* matching snippet-manifest ``.attrs`` (read by the aggregation layer).

End-to-end numerical parity on a real PID is covered by
``examples/verify_pid_oop_parity.py`` (offline, not run in CI).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ephysatlas.feature_calculators.ibl import IBLPIDFeatureCalculator
from ephysatlas.feature_calculators.types import RawSnippet
from ephysatlas.feature_computation import (
    compute_features_from_pid,
    compute_features_from_pid_oop,
)
from ephysatlas.utils import get_aggregated_snippets_df

PID = "test_pid"
N_CH = 4
NSYNC = 1
FS_AP = 30000.0
FS_LF = 2500.0
DUR = 0.05

# Deterministic synthetic AP/LF data shaped (n_total_channels, n_samples).
AP_DATA = np.arange((N_CH + NSYNC) * 4000, dtype=np.float32).reshape(N_CH + NSYNC, 4000)
LF_DATA = (
    np.arange((N_CH + NSYNC) * 1000, dtype=np.float32).reshape(N_CH + NSYNC, 1000) * 0.5
)


def _geometry() -> dict:
    """Reader geometry with every key ``load_geometry`` would otherwise setdefault."""
    return {
        "x": np.array([0.0, 32.0, 0.0, 32.0]),
        "y": np.array([0.0, 0.0, 20.0, 20.0]),
        "sample_shift": np.zeros(N_CH),
        "shank": np.zeros(N_CH),
        "col": np.array([0, 1, 0, 1]),
        "row": np.array([0, 0, 1, 1]),
    }


def _channels() -> dict:
    """Complete channels dict (needs no augmentation by either path)."""
    return {
        "rawInd": np.arange(N_CH),
        "axial_um": _geometry()["y"].copy(),
        "lateral_um": _geometry()["x"].copy(),
        "labels": np.zeros(N_CH),
    }


def _canned() -> pd.DataFrame:
    """Feature table returned by the engine spy (has the ``channel`` merge key)."""
    return pd.DataFrame(
        {
            "channel": np.arange(N_CH),
            "feat_a": np.arange(N_CH, dtype=float),
            "feat_b": np.arange(N_CH, dtype=float) * 2.0,
        }
    )


def _fake_add_targets(pid=None, one=None, channels=None, traj_dict=None):
    """Deterministic stand-in for add_target_coordinates used by both paths."""
    ch = dict(channels)
    n = len(next(iter(ch.values())))
    ch["x_target"] = np.arange(n, dtype=float)
    ch["y_target"] = np.arange(n, dtype=float) + 10.0
    ch["z_target"] = np.arange(n, dtype=float) + 20.0
    return ch


class _FakeReader:
    """Minimal SpikeGLX-reader stand-in backed by an in-memory array."""

    def __init__(self, data: np.ndarray, fs: float, file_bin=None) -> None:
        self._data = np.ascontiguousarray(data.T)  # (samples, channels)
        self.fs = float(fs)
        self.ns = int(data.shape[1])
        self.nc = int(data.shape[0])
        self.nsync = NSYNC
        self.file_bin = file_bin
        self.geometry = _geometry()

    def __getitem__(self, item):
        return self._data[item]


class _FakeSSL:
    """SpikeSortingLoader stand-in returning fixed readers and channels."""

    def __init__(self, sr_ap, sr_lf, channels) -> None:
        self._sr_ap = sr_ap
        self._sr_lf = sr_lf
        self._channels = channels

    def raw_electrophysiology(self, band, stream):
        return self._sr_ap if band == "ap" else self._sr_lf

    def load_channels(self):
        return dict(self._channels)


class _EngineSpy:
    """Stand-in for compute_features_from_raw: records kwargs, writes a file."""

    def __init__(self, canned: pd.DataFrame) -> None:
        self.canned = canned
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        self.canned.to_parquet(out / "lf_features.pqt")
        return self.canned.copy()


def _run_procedural(output_dir, features):
    spy = _EngineSpy(_canned())
    readers = (_FakeReader(AP_DATA, FS_AP), _FakeReader(LF_DATA, FS_LF), _channels())
    with (
        mock.patch(
            "ephysatlas.feature_computation.load_data_from_pid", return_value=readers
        ),
        mock.patch("ephysatlas.feature_computation.compute_features_from_raw", spy),
        mock.patch(
            "ephysatlas.feature_computation.add_target_coordinates",
            side_effect=_fake_add_targets,
        ),
    ):
        df = compute_features_from_pid(
            pid=PID,
            one=mock.MagicMock(),
            t_start=0.0,
            duration_ap=DUR,
            duration_lf=DUR,
            features_to_compute=features,
            output_dir=output_dir,
        )
    return df, spy


def _run_oop(output_dir, features):
    spy = _EngineSpy(_canned())
    ssl = _FakeSSL(
        _FakeReader(AP_DATA, FS_AP), _FakeReader(LF_DATA, FS_LF), _channels()
    )
    with (
        mock.patch.object(
            IBLPIDFeatureCalculator,
            "ssl",
            new_callable=mock.PropertyMock,
            return_value=ssl,
        ),
        mock.patch(
            "ephysatlas.feature_calculators.base.compute_features_from_raw", spy
        ),
        mock.patch(
            "ephysatlas.feature_calculators.ibl.add_target_coordinates",
            side_effect=_fake_add_targets,
        ),
    ):
        df = compute_features_from_pid_oop(
            pid=PID,
            one=mock.MagicMock(),
            t_start=0.0,
            duration_ap=DUR,
            duration_lf=DUR,
            features_to_compute=features,
            output_dir=output_dir,
        )
    return df, spy


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by channel and reconcile integer merge-key dtypes for comparison."""
    df = df.sort_values("channel").reset_index(drop=True)
    for col in ("channel", "rawInd"):
        if col in df.columns:
            df[col] = df[col].astype("int64")
    return df


class TestPidOopParity(unittest.TestCase):
    def test_matches_procedural(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc_dir = Path(tmp) / "proc"
            oop_dir = Path(tmp) / "oop"
            df_proc, spy_proc = _run_procedural(proc_dir, ["lf"])
            df_oop, spy_oop = _run_oop(oop_dir, ["lf"])

            # 1. Identical engine inputs (the only thing that sets feature values).
            kp, ko = spy_proc.kwargs, spy_oop.kwargs
            np.testing.assert_array_equal(kp["raw_ap"], ko["raw_ap"])
            np.testing.assert_array_equal(kp["raw_lf"], ko["raw_lf"])
            self.assertEqual(kp["fs_ap"], ko["fs_ap"])
            self.assertEqual(kp["fs_lf"], ko["fs_lf"])
            np.testing.assert_array_equal(kp["channel_labels"], ko["channel_labels"])
            self.assertEqual(
                list(kp["features_to_compute"]), list(ko["features_to_compute"])
            )
            self.assertEqual(kp["lf_k_filter"], ko["lf_k_filter"])
            self.assertEqual(set(kp["geometry"]), set(ko["geometry"]))
            for key in kp["geometry"]:
                np.testing.assert_array_equal(kp["geometry"][key], ko["geometry"][key])
            self.assertEqual(ko["neuropixel_version"], 1)  # procedural engine default

            # 2. Returned DataFrame parity (values; integer key dtypes reconciled).
            pd.testing.assert_frame_equal(
                _sorted(df_proc), _sorted(df_oop), check_like=True, check_dtype=False
            )

            # 3. channels.pqt key columns match.
            key_cols = [
                "axial_um",
                "lateral_um",
                "x_target",
                "y_target",
                "z_target",
                "pid",
            ]
            cp = _sorted(pd.read_parquet(proc_dir / PID / "channels.pqt"))
            co = _sorted(pd.read_parquet(oop_dir / PID / "channels.pqt"))
            pd.testing.assert_frame_equal(
                cp[["channel"] + key_cols],
                co[["channel"] + key_cols],
                check_like=True,
                check_dtype=False,
            )

            # 4. Snippet manifest .attrs the aggregation layer reads.
            mp = get_aggregated_snippets_df(proc_dir / PID)
            mo = get_aggregated_snippets_df(oop_dir / PID)
            for col in (
                "pid",
                "t_start",
                "duration_ap",
                "duration_lf",
                "snippet_level_dir",
            ):
                self.assertEqual(mp[col].iloc[0], mo[col].iloc[0])

    def test_resolve_channel_labels_prefers_cbin(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        calc._sr_ap = _FakeReader(AP_DATA, FS_AP, file_bin="probe.ap.cbin")
        raw = RawSnippet(raw_ap=AP_DATA[:N_CH], raw_lf=None, fs_ap=FS_AP, fs_lf=None)
        channels = pd.DataFrame({"channel": np.arange(N_CH)})  # no stored labels
        expected = np.array([0, 1, 0, 1])
        with mock.patch(
            "ibldsp.voltage.detect_bad_channels_cbin", return_value=expected
        ) as m:
            labels = calc._resolve_channel_labels(raw, channels)
        m.assert_called_once_with("probe.ap.cbin")
        np.testing.assert_array_equal(labels, expected)

    def test_resolve_channel_labels_snippet_fallback_without_file_bin(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        calc._sr_ap = _FakeReader(AP_DATA, FS_AP, file_bin=None)
        raw = RawSnippet(raw_ap=AP_DATA[:N_CH], raw_lf=None, fs_ap=FS_AP, fs_lf=None)
        channels = pd.DataFrame({"channel": np.arange(N_CH)})  # no stored labels
        with (
            mock.patch("ibldsp.voltage.detect_bad_channels_cbin") as m_cbin,
            mock.patch(
                "ibldsp.voltage.detect_bad_channels",
                return_value=(np.zeros(N_CH), None),
            ) as m_snip,
        ):
            labels = calc._resolve_channel_labels(raw, channels)
        m_cbin.assert_not_called()
        m_snip.assert_called_once()
        np.testing.assert_array_equal(labels, np.zeros(N_CH))

    def test_load_geometry_warns_on_missing_key(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        reader = _FakeReader(AP_DATA, FS_AP)
        reader.geometry = {"x": np.zeros(N_CH), "y": np.arange(N_CH, dtype=float)}
        calc._sr_ap = reader
        with self.assertLogs(
            "ephysatlas.feature_calculators.ibl", level="WARNING"
        ) as cm:
            geometry = calc.load_geometry()
        self.assertTrue(any("sample_shift" in message for message in cm.output))
        np.testing.assert_array_equal(geometry["sample_shift"], np.zeros(N_CH))

    def test_merge_channel_metadata_raises_on_duplicate_channel(self):
        calc = IBLPIDFeatureCalculator(pid=PID, one=mock.MagicMock())
        features = pd.DataFrame({"channel": [0, 1], "feat": [1.0, 2.0]})
        channels = pd.DataFrame(
            {"channel": [0, 0, 1], "axial_um": [0.0, 0.0, 20.0]}
        )  # duplicate channel 0 would fan out the left merge
        with self.assertRaises(ValueError):
            calc._merge_channel_metadata(features, channels)


if __name__ == "__main__":
    unittest.main()
