import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

from ephysatlas.aggregation import aggregate_all_probes, produce_output_dataframes
import ephysatlas.data

FIXTURE_PATH = Path(ephysatlas.data.__file__).parents[2].joinpath("tests", "fixtures")


class TestAggregationOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_brain_atlas = MagicMock()
        cls.mock_brain_atlas.get_labels.return_value = 0
        cls.mock_regions = MagicMock()
        cls.mock_regions.remap.return_value = 0
        cls.mock_brain_atlas.regions = cls.mock_regions
        # Extract the zip file to a temp directory
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.extract_dir = Path(cls.temp_dir.name)
        zip_path = FIXTURE_PATH.joinpath("output.zip")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cls.temp_dir.name)

        cls.extract_dir = cls.extract_dir / "output"

        # Get all probe-level directories (immediate subdirs of 'output')
        cls.probe_dirs = [p for p in cls.extract_dir.glob("*") if p.is_dir()]
        # Aggregate all probes into a single DataFrame
        cls.snippets_df = aggregate_all_probes(cls.probe_dirs, cls.extract_dir)
        # Get all unique pids
        cls.pids = cls.snippets_df["pid"].unique()
        # DataFrame for only one pid
        cls.snippets_df_one_pid = cls.snippets_df[
            cls.snippets_df["pid"] == cls.pids[0]
        ].reset_index(drop=True)
        # DataFrame for two pids, but only one snippet (t0) per pid
        cls.snippets_df_two_pids_one_snippet = cls.snippets_df.iloc[[0, 2]]

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def _run_and_check_outputs(self, snippets_df, input_dir, output_dir, expected_pids):
        # Run the function
        df_channels, df_raw_ephys, df_features_denoise = produce_output_dataframes(
            snippets_df, input_dir, output_dir
        )

        # Check that the output files exist
        output_dir = Path(output_dir)
        self.assertTrue((output_dir / "snippets_df.pqt").exists())
        # Check that the DataFrames are not empty
        self.assertFalse(df_channels.empty)
        self.assertFalse(df_raw_ephys.empty)
        self.assertFalse(df_features_denoise.empty)
        # Check that 'pid' is in the index names
        self.assertIn("pid", df_raw_ephys.index.names)
        self.assertIn("pid", df_features_denoise.index.names)
        self.assertIn("pid", df_channels.index.names)

        # Check that the pids in the output match expected
        self.assertTrue(
            set(df_raw_ephys.index.get_level_values("pid").unique()).issubset(
                set(expected_pids)
            )
        )
        self.assertTrue(
            set(df_features_denoise.index.get_level_values("pid").unique()).issubset(
                set(expected_pids)
            )
        )
        # self.assertTrue(set(df_channels.index.get_level_values('pid').unique()).issubset(set(expected_pids)))
        _ = ephysatlas.data.read_features_from_disk(
            output_dir, brain_atlas=self.mock_brain_atlas, strict=True, mappings=[]
        )

    def test_produce_output_dataframes_all_pids(self):
        with tempfile.TemporaryDirectory() as outdir:
            self._run_and_check_outputs(
                self.snippets_df, self.extract_dir, outdir, self.pids
            )

    def test_produce_output_dataframes_one_pid(self):
        with tempfile.TemporaryDirectory() as outdir:
            self._run_and_check_outputs(
                self.snippets_df_one_pid, self.extract_dir, outdir, [self.pids[0]]
            )

    def test_produce_output_dataframes_two_pids_one_snippet(self):
        with tempfile.TemporaryDirectory() as outdir:
            self._run_and_check_outputs(
                self.snippets_df_two_pids_one_snippet,
                self.extract_dir,
                outdir,
                self.pids,
            )


if __name__ == "__main__":
    unittest.main()
