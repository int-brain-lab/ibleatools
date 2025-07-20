import unittest
import tempfile
import shutil
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import os

import ephysatlas.aggregation as aggregation

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ZIP_PATH = FIXTURES_DIR / "output.zip"

class TestAggregationWithZip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Extract the zip file to a temp directory for all tests
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.extract_path = Path(cls.temp_dir.name)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(cls.extract_path)
        
        cls.extract_path = cls.extract_path / "output"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_aggregate_all_probes(self):
        # Find all probe root directories (UUIDs)
        probe_dirs = [d for d in self.extract_path.iterdir() if d.is_dir()]
        result = aggregation.aggregate_all_probes(probe_dirs, self.extract_path)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_concatenate_channels_data(self):
        # For each probe, test aggregation of channels.pqt
        channels_files = []
        for probe_dir in self.extract_path.iterdir():
            if not probe_dir.is_dir():
                continue
            channels_file = probe_dir / "channels.pqt"
            if not channels_file.exists():
                continue
            else:
                channels_files.append(channels_file)
            
        # Test with single file
        df = aggregation.concatenate_channels_data([channels_file])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("pid", df.index.names)
        self.assertIn("channel", df.index.names)
        self.assertEqual(df.index.names[0], "pid")
        self.assertEqual(df.index.names[1], "channel")

        # Test with multiple files
        df = aggregation.concatenate_channels_data(channels_files)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("pid", df.index.names)
        self.assertIn("channel", df.index.names)
        self.assertEqual(df.index.names[0], "pid")
        self.assertEqual(df.index.names[1], "channel")

    def test_get_features_from_snippets(self):
        # For each probe, for each snippet dir, test feature extraction
        for probe_dir in self.extract_path.iterdir():
            if not probe_dir.is_dir():
                continue
            for snippet_dir in probe_dir.iterdir():
                if not snippet_dir.is_dir():
                    continue
                # Only test if channels.pqt exists in parent
                if not (probe_dir / "channels.pqt").exists():
                    continue
                df = aggregation.get_features_from_snippets(snippet_dir)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertIn("channel", df.columns)

    def test_concat_raw_features(self):
        # Build input_df for concat_raw_features
        rows = []
        for probe_dir in self.extract_path.iterdir():
            if not probe_dir.is_dir():
                continue
            pid = probe_dir.name
            for snippet_dir in probe_dir.iterdir():
                if not snippet_dir.is_dir():
                    continue
                rows.append({"pid": pid, "base_level_dir": self.extract_path, "snippet_level_dir": snippet_dir.relative_to(self.extract_path)})
        input_df = pd.DataFrame(rows)
        df = aggregation.concat_raw_features(input_df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_aggregate_raw_features(self):
        # Build input_df for concat_raw_features
        rows = []
        for probe_dir in self.extract_path.iterdir():
            if not probe_dir.is_dir():
                continue
            pid = probe_dir.name
            for snippet_dir in probe_dir.iterdir():
                if not snippet_dir.is_dir():
                    continue
                rows.append({"pid": pid, "base_level_dir": self.extract_path,  "snippet_level_dir": snippet_dir.relative_to(self.extract_path)})
        input_df = pd.DataFrame(rows)
        df = aggregation.get_aggregated_raw_features(input_df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("channel", df.index.names)
        self.assertIn("pid", df.index.names)

if __name__ == "__main__":
    unittest.main() 