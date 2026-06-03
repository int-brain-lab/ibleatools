import json
import tempfile
import unittest
from functools import reduce
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ephysatlas.feature_calculators import (
    BaseFeatureCalculator,
    FeatureComputationOptions,
    RawSnippet,
    SnippetWindow,
)
from ephysatlas.feature_calculators.provenance import collect_ibleatools_provenance
from ephysatlas.utils import setup_output_directory


class FakeFeatureCalculator(BaseFeatureCalculator):
    """Small calculator used to test the shared OOP workflow."""

    def __init__(self):
        super().__init__(name="fake_probe", neuropixel_version=1)

    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Return tiny AP/LF arrays for tests."""
        raw_ap = np.zeros((4, 32), dtype=np.float32)
        raw_lf = np.zeros((4, 16), dtype=np.float32)
        return RawSnippet(raw_ap=raw_ap, raw_lf=raw_lf, fs_ap=30000.0, fs_lf=2500.0)

    def load_geometry(self) -> dict[str, np.ndarray]:
        """Return minimal Neuropixels-like geometry."""
        return {
            "x": np.array([0.0, 32.0, 0.0, 32.0]),
            "y": np.array([0.0, 0.0, 20.0, 20.0]),
            "sample_shift": np.zeros(4),
            "shank": np.zeros(4),
            "row": np.array([0.0, 0.0, 1.0, 1.0]),
            "col": np.array([0.0, 1.0, 0.0, 1.0]),
        }

    def load_channel_metadata(self) -> pd.DataFrame:
        """Return channel metadata with labels to avoid bad-channel detection."""
        return pd.DataFrame(
            {
                "channel": np.arange(4),
                "rawInd": np.arange(4),
                "axial_um": np.array([0.0, 0.0, 20.0, 20.0]),
                "lateral_um": np.array([0.0, 32.0, 0.0, 32.0]),
                "labels": np.zeros(4, dtype=int),
            }
        )

    def available_duration(self) -> tuple[float | None, float | None]:
        """Return long enough fake stream durations."""
        return 100.0, 100.0


def fake_compute_features_from_raw(*args, **kwargs) -> pd.DataFrame:
    """Write small feature files while mimicking cache semantics."""
    output_dir = kwargs["output_dir"]
    features_to_compute = kwargs["features_to_compute"]
    skip_saved = kwargs.get("skip_saved_computation", False)
    frames = []

    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_name in features_to_compute:
        file_path = output_dir / f"{feature_name}_features.pqt"
        if skip_saved and file_path.exists():
            frame = pd.read_parquet(file_path)
        else:
            frame = pd.DataFrame(
                {
                    "channel": np.arange(4),
                    f"{feature_name}_value": np.arange(4, dtype=float),
                }
            )
            frame.attrs["raw_compute_marker"] = feature_name
            frame.to_parquet(file_path)
        frames.append(frame)

    return reduce(lambda left, right: left.merge(right, on="channel"), frames)


class TestFeatureCalculators(unittest.TestCase):
    def test_snippet_window_rejects_negative_times(self):
        """Test that snippet windows validate negative values."""
        with self.assertRaises(ValueError):
            SnippetWindow(t_start=-1.0)

    def test_selective_provenance_updates_only_recomputed_features(self):
        """Test that provenance is not applied to unrelated feature files."""
        calculator = FakeFeatureCalculator()
        window = SnippetWindow(t_start=0.0, duration_ap=1.0, duration_lf=1.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _, snippet_dir = setup_output_directory(
                {
                    "pid": calculator.name,
                    "t_start": window.t_start,
                    "duration_ap": window.duration_ap,
                    "duration_lf": window.duration_lf,
                    "output_dir": output_dir,
                }
            )

            ap_file = snippet_dir / "ap_features.pqt"
            old_ap = pd.DataFrame({"channel": np.arange(4), "ap_value": np.ones(4)})
            old_ap.attrs["old_provenance"] = "keep"
            old_ap.to_parquet(ap_file)

            with patch(
                "ephysatlas.feature_calculators.base.compute_features_from_raw",
                side_effect=fake_compute_features_from_raw,
            ):
                result = calculator.compute_snippet(
                    window,
                    FeatureComputationOptions(
                        features_to_compute=["lf", "csd"],
                        output_dir=output_dir,
                    ),
                )

            self.assertEqual(result.computed_features, ("lf", "csd"))
            self.assertEqual(result.cached_features, ())

            ap_after = pd.read_parquet(ap_file)
            self.assertEqual(ap_after.attrs["old_provenance"], "keep")
            self.assertNotIn("feature_calculator_class", ap_after.attrs)

            for feature_name in ("lf", "csd"):
                df_feature = pd.read_parquet(
                    snippet_dir / f"{feature_name}_features.pqt"
                )
                self.assertEqual(
                    df_feature.attrs["feature_calculator_class"],
                    "FakeFeatureCalculator",
                )
                self.assertEqual(
                    df_feature.attrs["feature_calculator_feature_name"],
                    feature_name,
                )
                self.assertIn("ibleatools_is_editable_install", df_feature.attrs)

    def test_cached_feature_keeps_existing_attrs(self):
        """Test that cached feature files are not rewritten with new provenance."""
        calculator = FakeFeatureCalculator()
        window = SnippetWindow(t_start=0.0, duration_ap=1.0, duration_lf=1.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            _, snippet_dir = setup_output_directory(
                {
                    "pid": calculator.name,
                    "t_start": window.t_start,
                    "duration_ap": window.duration_ap,
                    "duration_lf": window.duration_lf,
                    "output_dir": output_dir,
                }
            )
            lf_file = snippet_dir / "lf_features.pqt"
            old_lf = pd.DataFrame({"channel": np.arange(4), "lf_value": np.ones(4)})
            old_lf.attrs["old_provenance"] = "keep"
            old_lf.to_parquet(lf_file)

            with patch(
                "ephysatlas.feature_calculators.base.compute_features_from_raw",
                side_effect=fake_compute_features_from_raw,
            ):
                result = calculator.compute_snippet(
                    window,
                    FeatureComputationOptions(
                        features_to_compute=["lf"],
                        output_dir=output_dir,
                        skip_saved_computation=True,
                    ),
                )

            self.assertEqual(result.computed_features, ())
            self.assertEqual(result.cached_features, ("lf",))
            lf_after = pd.read_parquet(lf_file)
            self.assertEqual(lf_after.attrs["old_provenance"], "keep")
            self.assertNotIn("feature_calculator_class", lf_after.attrs)

    def test_collect_provenance_records_editable_git_state(self):
        """Test editable-install provenance parsing and git metadata collection."""
        direct_url = {
            "dir_info": {"editable": True},
            "url": "file:///tmp/example-ibleatools",
        }
        fake_dist = Mock()
        fake_dist.version = "1.2.3"
        fake_dist.read_text.return_value = json.dumps(direct_url)

        def fake_run_git(repo_path: Path, args: list[str]) -> str:
            if args == ["rev-parse", "HEAD"]:
                return "abc123"
            if args == ["branch", "--show-current"]:
                return "oop_refactor"
            if args == ["status", "--porcelain"]:
                return " M src/file.py"
            raise AssertionError(args)

        with (
            patch(
                "ephysatlas.feature_calculators.provenance.metadata.distribution",
                return_value=fake_dist,
            ),
            patch(
                "ephysatlas.feature_calculators.provenance._run_git",
                side_effect=fake_run_git,
            ),
        ):
            provenance = collect_ibleatools_provenance(
                calculator_name="FakeFeatureCalculator",
                feature_names=("lf",),
            )

        self.assertEqual(provenance["ibleatools_distribution_version"], "1.2.3")
        self.assertTrue(provenance["ibleatools_is_editable_install"])
        self.assertEqual(provenance["ibleatools_git_commit_hash"], "abc123")
        self.assertEqual(provenance["ibleatools_git_branch"], "oop_refactor")
        self.assertTrue(provenance["ibleatools_git_is_dirty"])


if __name__ == "__main__":
    unittest.main()
