"""
Unit tests for ephysatlas.utils module.
"""

import unittest
import tempfile
import hashlib
from pathlib import Path
import pandas as pd

from ephysatlas.utils import (
    setup_output_directory,
    get_aggregated_snippets_df,
    add_metadata_to_parquet_files,
    _update_parquet_metadata,
)


class TestSetupOutputDirectory(unittest.TestCase):
    """Test the setup_output_directory function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_setup_output_directory_with_pid(self):
        """Test directory setup with PID parameter."""
        params = {
            "pid": "test_pid",
            "t_start": 300.0,
            "duration": 5.0,
            "output_dir": str(self.base_path),
        }

        probe_level_dir, snippet_level_dir = setup_output_directory(params)

        # Check that directories were created
        self.assertTrue(probe_level_dir.exists())
        self.assertTrue(snippet_level_dir.exists())

        # Check directory names
        self.assertEqual(probe_level_dir.name, "test_pid")
        expected_snippet_name = "probe_test_pid_000300.0_05.0"
        self.assertEqual(snippet_level_dir.name, expected_snippet_name)

        # Check directory structure
        self.assertEqual(snippet_level_dir.parent, probe_level_dir)
        self.assertEqual(probe_level_dir.parent, self.base_path)

    def test_setup_output_directory_with_filename(self):
        """Test directory setup with AP file parameter."""
        params = {
            "pid": None,
            "filename": "/path/to/test_ap.cbin",
            "t_start": 100.5,
            "duration": 25.0,
            "output_dir": str(self.base_path),
        }

        probe_level_dir, snippet_level_dir = setup_output_directory(params)

        # Check that directories were created
        self.assertTrue(probe_level_dir.exists())
        self.assertTrue(snippet_level_dir.exists())

        # Check that probe level directory uses hash of AP filename
        ap_file_hash = hashlib.md5("test_ap.cbin".encode()).hexdigest()[:12]
        self.assertEqual(probe_level_dir.name, ap_file_hash)

        # Check snippet level directory name (should use None for pid)
        expected_snippet_name = "probe_None_000100.5_25.0"
        self.assertEqual(snippet_level_dir.name, expected_snippet_name)

    def test_setup_output_directory_padding(self):
        """Test that t_start and duration are properly padded."""
        params = {
            "pid": "test_pid",
            "t_start": 123.456,
            "duration": 7.89,
            "output_dir": str(self.base_path),
        }

        probe_level_dir, snippet_level_dir = setup_output_directory(params)

        # Check padding format
        expected_snippet_name = "probe_test_pid_000123.5_07.9"
        self.assertEqual(snippet_level_dir.name, expected_snippet_name)

    def test_setup_output_directory_existing_directories(self):
        """Test that function works with existing directories."""
        params = {
            "pid": "test_pid",
            "t_start": 0.0,
            "duration": 100.0,
            "output_dir": str(self.base_path),
        }

        # Create directories manually first
        probe_level_dir = self.base_path / "test_pid"
        snippet_level_dir = probe_level_dir / "probe_test_pid_000000.0_100.0"
        probe_level_dir.mkdir(parents=True, exist_ok=True)
        snippet_level_dir.mkdir(parents=True, exist_ok=True)

        # Call function again - should not raise error
        result_probe, result_snippet = setup_output_directory(params)

        # Should return the same paths
        self.assertEqual(result_probe, probe_level_dir)
        self.assertEqual(result_snippet, snippet_level_dir)

    def test_setup_output_directory_nonexistent_parent(self):
        """Test directory setup with nonexistent parent directory."""
        params = {
            "pid": "test_pid",
            "t_start": 0.0,
            "duration": 100.0,
            "output_dir": str(self.base_path / "nonexistent" / "subdir"),
        }

        # Should create parent directories
        probe_level_dir, snippet_level_dir = setup_output_directory(params)

        self.assertTrue(probe_level_dir.exists())
        self.assertTrue(snippet_level_dir.exists())
        self.assertTrue(probe_level_dir.parent.exists())

    def test_setup_output_directory_edge_cases(self):
        """Test edge cases for directory setup."""
        # Test with zero values
        params = {
            "pid": "test_pid",
            "t_start": 0.0,
            "duration": 0.0,
            "output_dir": str(self.base_path),
        }

        probe_level_dir, snippet_level_dir = setup_output_directory(params)
        expected_snippet_name = "probe_test_pid_000000.0_00.0"
        self.assertEqual(snippet_level_dir.name, expected_snippet_name)

        # Test with large values
        params = {
            "pid": "test_pid",
            "t_start": 999999.9,
            "duration": 9999.9,
            "output_dir": str(self.base_path),
        }

        probe_level_dir, snippet_level_dir = setup_output_directory(params)
        expected_snippet_name = "probe_test_pid_999999.9_9999.9"
        self.assertEqual(snippet_level_dir.name, expected_snippet_name)

    def test_setup_output_directory_hash_consistency(self):
        """Test that AP file hashing is consistent."""
        params1 = {
            "pid": None,
            "filename": "/path/to/test_ap.cbin",
            "t_start": 0.0,
            "duration": 100.0,
            "output_dir": str(self.base_path),
        }

        params2 = {
            "pid": None,
            "filename": "/different/path/to/test_ap.cbin",  # Same filename, different path
            "t_start": 0.0,
            "duration": 100.0,
            "output_dir": str(self.base_path),
        }

        probe_level_dir1, _ = setup_output_directory(params1)
        probe_level_dir2, _ = setup_output_directory(params2)

        # Should have same hash (same filename)
        self.assertEqual(probe_level_dir1.name, probe_level_dir2.name)

        # Test with different filename
        params3 = {
            "pid": None,
            "filename": "/path/to/different_ap.cbin",
            "t_start": 0.0,
            "duration": 100.0,
            "output_dir": str(self.base_path),
        }

        probe_level_dir3, _ = setup_output_directory(params3)

        # Should have different hash
        self.assertNotEqual(probe_level_dir1.name, probe_level_dir3.name)


class TestGetAggregatedSnippetsDf(unittest.TestCase):
    """Test the get_aggregated_snippets_df function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_aggregated_snippets_df_empty_directory(self):
        """Test with empty probe level directory."""
        df = get_aggregated_snippets_df(self.base_path)
        self.assertTrue(df.empty)
        self.assertEqual(len(df), 0)

    def test_get_aggregated_snippets_df_with_parquet_files(self):
        """Test with snippet directories containing parquet files."""
        # Create snippet directories
        snippet_dir1 = self.base_path / "snippet1"
        snippet_dir2 = self.base_path / "snippet2"
        snippet_dir1.mkdir()
        snippet_dir2.mkdir()

        # Create test dataframes with metadata
        df1 = pd.DataFrame({"data": [1, 2, 3]})
        df1.attrs["pid"] = "test_pid_1"
        df1.attrs["t_start"] = 100.0

        df2 = pd.DataFrame({"data": [4, 5, 6]})
        df2.attrs["pid"] = "test_pid_2"
        df2.attrs["t_start"] = 200.0

        # Save parquet files
        df1.to_parquet(snippet_dir1 / "data.parquet")
        df2.to_parquet(snippet_dir2 / "data.pqt")

        # Test function
        result_df = get_aggregated_snippets_df(self.base_path)

        # Check results
        self.assertEqual(len(result_df), 2)
        self.assertIn("pid", result_df.columns)
        self.assertIn("t_start", result_df.columns)
        self.assertIn("test_pid_1", result_df["pid"].values)
        self.assertIn("test_pid_2", result_df["pid"].values)

    def test_get_aggregated_snippets_df_mixed_content(self):
        """Test with directories containing both parquet files and other content."""
        # Create snippet directory
        snippet_dir = self.base_path / "snippet1"
        snippet_dir.mkdir()

        # Create a text file (should be ignored)
        (snippet_dir / "data.txt").write_text("test")

        # Create parquet file with metadata
        df = pd.DataFrame({"data": [1, 2, 3]})
        df.attrs["pid"] = "test_pid"
        df.to_parquet(snippet_dir / "data.pqt")

        # Test function
        result_df = get_aggregated_snippets_df(self.base_path)

        # Should only process parquet files
        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]["pid"], "test_pid")


class TestAddMetadataToParquetFiles(unittest.TestCase):
    """Test the add_metadata_to_parquet_files function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_metadata_to_parquet_files(self):
        """Test adding metadata to parquet files."""
        # Create test dataframe
        df = pd.DataFrame({"data": [1, 2, 3, 4, 5]})
        df.to_parquet(self.base_path / "test.pqt")

        # Test adding metadata
        snippet_attrs = {
            "base_level_dir": str(self.base_path),
            "snippet_level_dir": ".",
            "pid": "test_pid",
            "t_start": 100.0,
            "duration": 5.0,
        }

        add_metadata_to_parquet_files(**snippet_attrs)

        # Verify metadata was added
        result_df = pd.read_parquet(self.base_path / "test.pqt")
        self.assertEqual(result_df.attrs["pid"], "test_pid")
        self.assertEqual(result_df.attrs["t_start"], 100.0)
        self.assertEqual(result_df.attrs["duration"], 5.0)

    def test_add_metadata_to_parquet_files_multiple_formats(self):
        """Test adding metadata to both .parquet and .pqt files."""
        # Create test dataframes
        df1 = pd.DataFrame({"data": [1, 2, 3]})
        df2 = pd.DataFrame({"data": [4, 5, 6]})

        df1.to_parquet(self.base_path / "test1.parquet")
        df2.to_parquet(self.base_path / "test2.pqt")

        # Test adding metadata
        snippet_attrs = {
            "base_level_dir": str(self.base_path),
            "snippet_level_dir": ".",
            "pid": "test_pid",
            "custom_attr": "test_value",
        }

        add_metadata_to_parquet_files(**snippet_attrs)

        # Verify metadata was added to both files
        result_df1 = pd.read_parquet(self.base_path / "test1.parquet")
        result_df2 = pd.read_parquet(self.base_path / "test2.pqt")

        self.assertEqual(result_df1.attrs["pid"], "test_pid")
        self.assertEqual(result_df1.attrs["custom_attr"], "test_value")
        self.assertEqual(result_df2.attrs["pid"], "test_pid")
        self.assertEqual(result_df2.attrs["custom_attr"], "test_value")

    def test_add_metadata_to_parquet_files_nonexistent_directory(self):
        """Test behavior with nonexistent directory."""
        snippet_attrs = {
            "base_level_dir": str(self.base_path),
            "snippet_level_dir": "nonexistent",
            "pid": "test_pid",
        }

        # Should not raise error, just log warning
        add_metadata_to_parquet_files(**snippet_attrs)


class TestUpdateParquetMetadata(unittest.TestCase):
    """Test the _update_parquet_metadata function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_parquet_metadata_success(self):
        """Test successful metadata update."""
        # Create test dataframe
        df = pd.DataFrame({"data": [1, 2, 3]})
        file_path = self.base_path / "test.parquet"
        df.to_parquet(file_path)

        # Test updating metadata
        snippet_attrs = {
            "pid": "test_pid",
            "t_start": 100.0,
            "duration": 5.0,
        }

        _update_parquet_metadata(file_path, **snippet_attrs)

        # Verify metadata was updated
        result_df = pd.read_parquet(file_path)
        self.assertEqual(result_df.attrs["pid"], "test_pid")
        self.assertEqual(result_df.attrs["t_start"], 100.0)
        self.assertEqual(result_df.attrs["duration"], 5.0)

    def test_update_parquet_metadata_preserves_existing_attrs(self):
        """Test that existing metadata attributes are preserved."""
        # Create test dataframe with existing metadata
        df = pd.DataFrame({"data": [1, 2, 3]})
        df.attrs["existing_attr"] = "existing_value"
        file_path = self.base_path / "test.parquet"
        df.to_parquet(file_path)

        # Test updating metadata
        snippet_attrs = {
            "pid": "test_pid",
            "new_attr": "new_value",
        }

        _update_parquet_metadata(file_path, **snippet_attrs)

        # Verify both old and new metadata are present
        result_df = pd.read_parquet(file_path)
        self.assertEqual(result_df.attrs["existing_attr"], "existing_value")
        self.assertEqual(result_df.attrs["pid"], "test_pid")
        self.assertEqual(result_df.attrs["new_attr"], "new_value")

    def test_update_parquet_metadata_nonexistent_file(self):
        """Test behavior with nonexistent file."""
        file_path = self.base_path / "nonexistent.parquet"
        snippet_attrs = {"pid": "test_pid"}

        # Should not raise error, just log warning
        _update_parquet_metadata(file_path, **snippet_attrs)


if __name__ == "__main__":
    unittest.main()
