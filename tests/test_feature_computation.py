import unittest
import numpy as np
from pathlib import Path
import tempfile
from types import SimpleNamespace
import pandas as pd

from iblatlas.atlas import AllenAtlas, Insertion
from ibllib.pipes.histology import interpolate_along_track

from ephysatlas.feature_computation import (
    load_data_from_files,
    add_target_coordinates,
    compute_features_from_raw,
)

import ephysatlas

# Provenance values as Alyx stores them, used by the fake ONE client below to reproduce
# the server-side "provenance__lte" filter.
ALYX_PROVENANCE = {
    "Planned": 10,
    "Micro-manipulator": 30,
    "Histology track": 50,
    "Ephys aligned histology track": 70,
}


class TestFeatureComputation(unittest.TestCase):
    def test_load_data_from_files_nonexistent_files(self):
        """Test load_data_from_files with non-existent files"""
        with self.assertRaises(RuntimeError):
            load_data_from_files("nonexistent_ap.cbin", "nonexistent_lf.cbin")

    def test_load_data_from_files_invalid_file_types(self):
        """Test load_data_from_files with invalid file types"""
        with (
            tempfile.NamedTemporaryFile(suffix=".txt") as tmp_ap,
            tempfile.NamedTemporaryFile(suffix=".txt") as tmp_lf,
        ):
            with self.assertRaises(RuntimeError):
                load_data_from_files(tmp_ap.name, tmp_lf.name)

    def test_add_target_coordinates_no_arguments(self):
        """Test add_target_coordinates with no arguments"""
        with self.assertRaises(ValueError):
            add_target_coordinates()

    def test_add_target_coordinates_only_pid(self):
        """Test add_target_coordinates with only pid provided"""
        with self.assertRaises(ValueError):
            add_target_coordinates(pid="test_pid")

    def test_add_target_coordinates_only_one(self):
        """Test add_target_coordinates with only one provided"""
        with self.assertRaises(ValueError):
            add_target_coordinates(one="test_one")

    def test_add_target_coordinates_valid_trajectory(self):
        """Test add_target_coordinates with valid trajectory dictionary"""
        channels = {"rawInd": np.arange(10), "axial_um": np.linspace(0, 1000, 10)}

        traj_dict = {
            "x": 1000,
            "y": 2000,
            "z": 3000,
            "depth": 4000,
            "theta": 0,
            "phi": 0,
        }

        result = add_target_coordinates(channels=channels, traj_dict=traj_dict)

        # Check output format
        self.assertIsInstance(result, dict)
        self.assertIn("x_target", result)
        self.assertIn("y_target", result)
        self.assertIn("z_target", result)
        self.assertEqual(len(result["x_target"]), len(channels["rawInd"]))
        self.assertEqual(len(result["y_target"]), len(channels["rawInd"]))
        self.assertEqual(len(result["z_target"]), len(channels["rawInd"]))

    def test_add_target_coordinates_without_rawInd(self):
        """Test add_target_coordinates when rawInd is not provided"""
        channels = {
            "axial_um": np.linspace(0, 1000, 384)  # 384 channels as per the code
        }

        traj_dict = {
            "x": 1000,
            "y": 2000,
            "z": 3000,
            "depth": 4000,
            "theta": 0,
            "phi": 0,
        }

        result = add_target_coordinates(channels=channels, traj_dict=traj_dict)

        # Check that rawInd was added
        self.assertIn("rawInd", result)
        self.assertEqual(len(result["rawInd"]), 384)
        self.assertTrue(np.array_equal(result["rawInd"], np.arange(384)))

    # ------------------------------------------------------------------
    # Trajectory provenance preference (Alyx mode)
    # ------------------------------------------------------------------
    # Minimal Alyx-like trajectories for one insertion, one per provenance. The
    # angles differ between them so the chosen trajectory is identifiable from the
    # coordinates it produces. The Histology track values are those Alyx holds for
    # pid 1a924329-65aa-465d-b201-c2dd898aebd0, an insertion with no
    # Micro-manipulator and no Planned trajectory.
    MICRO_MANIPULATOR_TRAJ = {
        "provenance": "Micro-manipulator",
        "x": -2243.1,
        "y": -1999.8,
        "z": -361.0,
        "depth": 4000.0,
        "theta": 15.0,
        "phi": 180.0,
    }
    PLANNED_TRAJ = {
        "provenance": "Planned",
        "x": -2200.0,
        "y": -2050.0,
        "z": -300.0,
        "depth": 4200.0,
        "theta": 10.0,
        "phi": 170.0,
    }
    HISTOLOGY_TRAJ = {
        "provenance": "Histology track",
        "x": -2514.0,
        "y": -1950.0,
        "z": -193.0,
        "depth": 6089.9,
        "theta": 11.56,
        "phi": -173.13,
    }
    EPHYS_ALIGNED_TRAJ = {
        "provenance": "Ephys aligned histology track",
        "x": -2464.0,
        "y": -2000.0,
        "z": -193.0,
        "depth": 5687.6,
        "theta": 11.48,
        "phi": -169.04,
    }

    @staticmethod
    def _target_channels():
        """Small 10-channel input shared by the trajectory preference tests."""
        return {"rawInd": np.arange(10), "axial_um": np.linspace(0, 3840, 10)}

    @staticmethod
    def _fake_one(trajs):
        """Stand in for a remote ONE client serving ``trajs`` through Alyx's REST filter."""

        def rest(*args, django=None, **kwargs):
            # Reproduce "provenance__lte,<value>" server side, so the tests cover which
            # provenances the query lets through as well as which one is then picked.
            ceiling = (
                int(django.split(",")[-1]) if django else max(ALYX_PROVENANCE.values())
            )
            return [t for t in trajs if ALYX_PROVENANCE[t["provenance"]] <= ceiling]

        return SimpleNamespace(mode="remote", alyx=SimpleNamespace(rest=rest))

    def _assert_same_targets(self, result, expected):
        """Assert two add_target_coordinates outputs hold the same target coordinates."""
        for key in ("x_target", "y_target", "z_target"):
            np.testing.assert_allclose(result[key], expected[key])

    def test_add_target_coordinates_prefers_micro_manipulator(self):
        """Micro-manipulator wins when Planned and Histology track are also available"""
        trajs = [self.HISTOLOGY_TRAJ, self.PLANNED_TRAJ, self.MICRO_MANIPULATOR_TRAJ]
        result = add_target_coordinates(
            pid="pid", one=self._fake_one(trajs), channels=self._target_channels()
        )
        expected = add_target_coordinates(
            channels=self._target_channels(),
            traj_dict=dict(self.MICRO_MANIPULATOR_TRAJ),
        )
        self._assert_same_targets(result, expected)

    def test_add_target_coordinates_falls_back_to_planned(self):
        """Without a Micro-manipulator trajectory, Planned is preferred over histology"""
        trajs = [self.HISTOLOGY_TRAJ, self.PLANNED_TRAJ]
        result = add_target_coordinates(
            pid="pid", one=self._fake_one(trajs), channels=self._target_channels()
        )
        expected = add_target_coordinates(
            channels=self._target_channels(), traj_dict=dict(self.PLANNED_TRAJ)
        )
        self._assert_same_targets(result, expected)

    def test_add_target_coordinates_falls_back_to_histology_track(self):
        """Histology track is the last resort, read in the Allen frame it is stored in

        The Ephys aligned track alongside it must be ignored: the query stops at 50.
        """
        channels = self._target_channels()
        result = add_target_coordinates(
            pid="pid",
            one=self._fake_one([self.HISTOLOGY_TRAJ, self.EPHYS_ALIGNED_TRAJ]),
            channels=channels,
        )
        # A Histology track trajectory is stored as "IBL-Allen", so the expected track is
        # a plain Allen insertion: no -5 degree pitch correction, no needles -> Allen scaling.
        txyz = np.flipud(
            Insertion.from_dict(dict(self.HISTOLOGY_TRAJ), brain_atlas=AllenAtlas()).xyz
        )
        expected = interpolate_along_track(txyz, channels["axial_um"] / 1e6)
        np.testing.assert_allclose(
            np.c_[result["x_target"], result["y_target"], result["z_target"]], expected
        )

    def test_add_target_coordinates_no_usable_trajectory(self):
        """An insertion with none of the three usable provenances raises ValueError"""
        with self.assertRaises(ValueError):
            add_target_coordinates(
                pid="pid", one=self._fake_one([]), channels=self._target_channels()
            )

    def test_compute_features_from_raw_with_destriped_files(self):
        """Test compute_features_from_raw using the available destriped data files"""
        # Load the destriped data files
        ap_data = np.load(
            Path(__file__).parent.joinpath("fixtures", "ap_destriped.npy")
        )
        lf_data = np.load(
            Path(__file__).parent.joinpath("fixtures", "lf_destriped.npy")
        )

        # Define sampling frequencies (typical values for Neuropixel)
        fs_ap = 30000.0  # 30 kHz for AP data
        fs_lf = 2500.0  # 2.5 kHz for LF data

        # Create geometry dictionary for Neuropixel 1.0
        # Using a simple linear arrangement for testing
        n_channels = ap_data.shape[0]
        geometry = {
            "x": np.zeros(n_channels),  # All channels in same column
            "y": np.arange(n_channels) * 20,  # 20 um spacing between channels
            "sample_shift": np.zeros(n_channels),
            "shank": np.zeros(n_channels),
        }

        # Test with a subset of features to avoid long computation time
        features_to_compute = ["lf", "ap"]  # Skip CSD and waveforms for faster testing

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Call the function
            result_df = compute_features_from_raw(
                raw_ap=ap_data,
                raw_lf=lf_data,
                fs_ap=fs_ap,
                fs_lf=fs_lf,
                geometry=geometry,
                features_to_compute=features_to_compute,
                output_dir=output_dir,
            )

            # Check that result is a pandas DataFrame
            self.assertIsInstance(result_df, pd.DataFrame)

            # Check that DataFrame has expected columns
            expected_columns = ["channel"]
            if "lf" in features_to_compute:
                expected_columns.extend(
                    [
                        "rms_lf",
                        "psd_delta",
                        "psd_theta",
                        "psd_alpha",
                        "psd_beta",
                        "psd_gamma",
                        "psd_lfp",
                    ]
                )
            if "ap" in features_to_compute:
                expected_columns.extend(["rms_ap", "cor_ratio"])

            for col in expected_columns:
                self.assertIn(
                    col,
                    result_df.columns,
                    f"Expected column '{col}' not found in result",
                )

            # Check that DataFrame has expected number of rows (one per channel)
            self.assertEqual(len(result_df), n_channels)

            # Check that channel column contains expected values
            self.assertTrue(
                np.array_equal(result_df["channel"].values, np.arange(n_channels))
            )

            # Check that output files were created
            if "lf" in features_to_compute:
                self.assertTrue((output_dir / "lf_features.pqt").exists())
                # Check that package version metadata is added to the lf_features.pqt file
                with open(output_dir / "lf_features.pqt", "rb") as f:
                    df = pd.read_parquet(f)
                    self.assertIn("ibleatools_version", df.attrs)
                    self.assertEqual(
                        df.attrs["ibleatools_version"], ephysatlas.__version__
                    )
            if "ap" in features_to_compute:
                self.assertTrue((output_dir / "ap_features.pqt").exists())
                # Check that package version metadata is added to the ap_features.pqt file
                with open(output_dir / "ap_features.pqt", "rb") as f:
                    df = pd.read_parquet(f)
                    self.assertIn("ibleatools_version", df.attrs)
                    self.assertEqual(
                        df.attrs["ibleatools_version"], ephysatlas.__version__
                    )

    def test_compute_features_from_raw_with_only_ap_or_lf(self):
        """Test compute_features_from_raw with only AP data or only LF data"""
        # Load the destriped data files
        ap_data = np.load(
            Path(__file__).parent.joinpath("fixtures", "ap_destriped.npy")
        )
        lf_data = np.load(
            Path(__file__).parent.joinpath("fixtures", "lf_destriped.npy")
        )

        # Define sampling frequencies (typical values for Neuropixel)
        fs_ap = 30000.0  # 30 kHz for AP data
        fs_lf = 2500.0  # 2.5 kHz for LF data

        # Test 1: Only AP data (no LF data)
        n_channels_ap = ap_data.shape[0]
        geometry_ap = {
            "x": np.zeros(n_channels_ap),
            "y": np.arange(n_channels_ap) * 20,
            "sample_shift": np.zeros(n_channels_ap),
            "shank": np.zeros(
                n_channels_ap
            ),  # Assuming all channels are on the same shank for testing
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Test with only AP data - should run without errors
            result_df_ap = compute_features_from_raw(
                raw_ap=ap_data,
                raw_lf=None,
                fs_ap=fs_ap,
                fs_lf=None,
                geometry=geometry_ap,
                features_to_compute=[
                    "ap"
                ],  # Let function determine based on available data
                output_dir=output_dir,
            )

            # Check that result is a pandas DataFrame
            self.assertIsInstance(result_df_ap, pd.DataFrame)

            # Check that DataFrame has expected columns for AP features
            self.assertIn("channel", result_df_ap.columns)
            # The function should automatically compute AP and waveforms when only AP is provided
            # But to avoid long computation, we can just check that it runs without error

        # Test 2: Only LF data (no AP data)
        n_channels_lf = lf_data.shape[0]
        geometry_lf = {
            "x": np.zeros(n_channels_lf),
            "y": np.arange(n_channels_lf) * 20,
            "col": np.zeros(n_channels_lf),
            "row": np.arange(n_channels_lf),
            "sample_shift": np.zeros(n_channels_lf),
            "shank": np.zeros(
                n_channels_lf
            ),  # Assuming all channels are on the same shank for testing
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Test with only LF data - should run without errors
            result_df_lf = compute_features_from_raw(
                raw_ap=None,
                raw_lf=lf_data,
                fs_ap=None,
                fs_lf=fs_lf,
                geometry=geometry_lf,
                features_to_compute=None,  # Let function determine based on available data
                output_dir=output_dir,
            )

            # Check that result is a pandas DataFrame
            self.assertIsInstance(result_df_lf, pd.DataFrame)

            # Check that DataFrame has expected columns for LF features
            self.assertIn("channel", result_df_lf.columns)
            # The function should automatically compute LF and CSD when only LF is provided
            # But to avoid long computation, we can just check that it runs without error


if __name__ == "__main__":
    unittest.main()
