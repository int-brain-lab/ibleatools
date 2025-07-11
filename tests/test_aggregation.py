# TODO: I have only added some basic tests. Need to add more tests. Especially for the denoise function.
# TODO: Add tests for test_aggregate_channels_data
import unittest
import pandas as pd
from pathlib import Path
import tempfile

import ephysatlas.aggregation as aggregation


class TestAggregation(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.probe_dir = Path(self.temp_dir.name)

        # Create a basic channels.pqt file
        self.channels_file = self.probe_dir / "channels.pqt"
        df_channels = pd.DataFrame(
            {
                "pid": ["p1", "p2"],
                "channel": [0, 1],
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "z": [0.0, 1.0],
                "axial_um": [0.0, 1.0],
                "lateral_um": [0.0, 1.0],
                "acronym": ["A", "B"],
                "atlas_id": [1, 2],
                "channel_labels": [0, 1],
                "labels": [0, 0],
            }
        )
        df_channels.to_parquet(self.channels_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_aggregate_all_probes(self):
        """Test aggregate_all_probes function with basic functionality"""
        # Create a simple probe directory structure
        probe1_dir = self.probe_dir / "probe1"
        probe1_dir.mkdir()

        # Create a snippet directory with metadata
        snippet_dir = probe1_dir / "snippet_001"
        snippet_dir.mkdir()

        # Create a parquet file with some metadata
        df_metadata = pd.DataFrame({"test_col": [1, 2, 3]})
        df_metadata.attrs = {"pid": "test_pid", "duration": 5.0}
        df_metadata.to_parquet(snippet_dir / "test_features.parquet")

        # Test the function
        paths = [probe1_dir]
        result = aggregation.aggregate_all_probes(paths)

        # Basic assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_aggregate_channel_labels_no_files(self):
        """Test aggregate_channel_labels when no ap_features.parquet files exist"""
        result = aggregation.aggregate_channel_labels(self.probe_dir)
        self.assertIsNone(result)

    def test_aggregate_channel_labels_missing_column(self):
        """Test aggregate_channel_labels when channel_labels column is missing"""
        ap_file = self.probe_dir / "ap_features.parquet"
        df_ap = pd.DataFrame({"other_column": [1, 2, 3]})
        df_ap.to_parquet(ap_file)

        result = aggregation.aggregate_channel_labels(self.probe_dir)
        self.assertIsNone(result)

    def test_update_channel_pqt_with_channel_labels_success(self):
        """Test updating channels.pqt with channel labels"""
        # Create ap_features.parquet with channel_labels
        ap_file = self.probe_dir / "ap_features.parquet"
        df_ap = pd.DataFrame({"channel_labels": [1, 2]})
        df_ap.to_parquet(ap_file)

        # Run the function
        aggregation.update_channel_pqt_with_channel_labels(self.probe_dir)

        # Check that the file was updated
        df_updated = pd.read_parquet(self.channels_file)
        self.assertIn("channel_labels", df_updated.columns)

    # def test_aggregate_channels_data(self):
    #     """Test aggregate_channels_data function"""
    #     # Create two channel parquet files
    #     f1 = self.probe_dir / "c1.parquet"
    #     f2 = self.probe_dir / "c2.parquet"

    #     df1 = pd.DataFrame({
    #         "pid": ["p1"],
    #         "channel": [0],
    #         "x": [0.0],
    #         "y": [0.0],
    #         "z": [0.0],
    #         "axial_um": [0.0],
    #         "lateral_um": [0.0],
    #         "acronym": ["A"],
    #         "atlas_id": [1]
    #     })
    #     df2 = pd.DataFrame({
    #         "pid": ["p2"],
    #         "channel": [1],
    #         "x": [1.0],
    #         "y": [1.0],
    #         "z": [1.0],
    #         "axial_um": [1.0],
    #         "lateral_um": [1.0],
    #         "acronym": ["B"],
    #         "atlas_id": [2]
    #     })

    #     df1.to_parquet(f1)
    #     df2.to_parquet(f2)

    #     # Test without output directory
    #     result = aggregation.aggregate_channels_data([f1, f2])
    #     self.assertIsInstance(result, pd.DataFrame)
    #     self.assertIn("pid", result.columns)
    #     self.assertIn("channel", result.columns)

    def test_aggregate_channels_data_with_output_dir(self):
        """Test aggregate_channels_data with output directory"""
        # Create a channel parquet file
        f1 = self.probe_dir / "c1.parquet"
        df1 = pd.DataFrame(
            {
                "pid": ["p1"],
                "channel": [0],
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
                "axial_um": [0.0],
                "lateral_um": [0.0],
                "acronym": ["A"],
                "atlas_id": [1],
            }
        )
        df1.to_parquet(f1)

        # Test with output directory
        output_dir = self.probe_dir / "output"
        result = aggregation.aggregate_channels_data([f1], output_dir=output_dir)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue((output_dir / "channels.parquet").exists())

    def test_get_features_from_snippets(self):
        """Test get_features_from_snippets function"""
        # Create snippet directory with parquet files
        snippet_dir = self.probe_dir / "snippet_test"
        snippet_dir.mkdir()

        # Create two parquet files with different features
        f1 = snippet_dir / "lf_features.parquet"
        f2 = snippet_dir / "ap_features.parquet"

        df1 = pd.DataFrame({"channel": [0, 1], "rms_lf": [1.0, 2.0]})
        df2 = pd.DataFrame({"channel": [0, 1], "rms_ap": [3.0, 4.0]})

        df1.to_parquet(f1)
        df2.to_parquet(f2)

        result = aggregation.get_features_from_snippets(snippet_dir)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("rms_lf", result.columns)
        self.assertIn("rms_ap", result.columns)
        self.assertEqual(len(result), 2)  # Two channels

    def test_aggregate_raw_features(self):
        """Test aggregate_raw_features function"""
        # Create snippet directory with parquet files
        snippet_dir = self.probe_dir / "snippet_raw"
        snippet_dir.mkdir()

        # Create a parquet file with features
        f1 = snippet_dir / "features.parquet"
        df1 = pd.DataFrame({"channel": [0, 1], "feature1": [1.0, 2.0]})
        df1.to_parquet(f1)

        # Create input dataframe
        input_df = pd.DataFrame({"snippet_level_dir": [snippet_dir]})

        # Test without output directory
        result = aggregation.aggregate_raw_features(input_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("feature1", result.columns)

    def test_aggregate_raw_features_with_output_dir(self):
        """Test aggregate_raw_features with output directory"""
        # Create snippet directory with parquet files
        snippet_dir = self.probe_dir / "snippet_raw_output"
        snippet_dir.mkdir()

        # Create a parquet file with features
        f1 = snippet_dir / "features.parquet"
        df1 = pd.DataFrame({"channel": [0, 1], "feature1": [1.0, 2.0]})
        df1.to_parquet(f1)

        # Create input dataframe
        input_df = pd.DataFrame({"snippet_level_dir": [snippet_dir]})

        # Test with output directory
        output_dir = self.probe_dir / "raw_output"
        result = aggregation.aggregate_raw_features(input_df, output_dir=output_dir)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue((output_dir / "raw_ephys_features.parquet").exists())


if __name__ == "__main__":
    unittest.main()
