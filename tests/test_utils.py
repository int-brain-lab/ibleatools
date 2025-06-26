"""
Unit tests for ephysatlas.utils module.
"""

import unittest
import tempfile
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

from ephysatlas.utils import setup_output_directory


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
        self.assertEqual(probe_level_dir.name, "test_pid_123")
        expected_snippet_name = "probe_test_pid_00000300.0_0050.0"
        self.assertEqual(snippet_level_dir.name, expected_snippet_name)

        # Check directory structure
        self.assertEqual(snippet_level_dir.parent, probe_level_dir)
        self.assertEqual(probe_level_dir.parent, self.base_path)

    def test_setup_output_directory_with_ap_file(self):
        """Test directory setup with AP file parameter."""
        params = {
            "pid": None,
            "ap_file": "/path/to/test_ap.cbin",
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
        expected_snippet_name = "probe_None_00000100.5_0025.0"
        self.assertEqual(snippet_level_dir.name, expected_snippet_name)

    def test_setup_output_directory_default_output_dir(self):
        """Test directory setup with default output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                # Change to temp directory
                os.chdir(temp_dir)

                params = {
                    "pid": "test_pid",
                    "t_start": 0.0,
                    "duration": 100.0,
                    # No output_dir specified
                }

                probe_level_dir, snippet_level_dir = setup_output_directory(params)

                # Check that directories were created in current directory
                self.assertTrue(probe_level_dir.exists())
                self.assertTrue(snippet_level_dir.exists())
                self.assertEqual(probe_level_dir.parent, Path.cwd())

            finally:
                # Restore original working directory
                os.chdir(original_cwd)

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
        expected_snippet_name = "probe_test_pid_00000123.5_0007.9"
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
        snippet_level_dir = probe_level_dir / "probe_test_pid_00000000.0_0100.0"
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
        expected_snippet_name = "probe_test_pid_00000000.0_0000.0"
        self.assertEqual(snippet_level_dir.name, expected_snippet_name)

        # Test with large values
        params = {
            "pid": "test_pid",
            "t_start": 999999.9,
            "duration": 9999.9,
            "output_dir": str(self.base_path),
        }

        probe_level_dir, snippet_level_dir = setup_output_directory(params)
        expected_snippet_name = "probe_test_pid_09999999.9_9999.9"
        self.assertEqual(snippet_level_dir.name, expected_snippet_name)

    def test_setup_output_directory_hash_consistency(self):
        """Test that AP file hashing is consistent."""
        params1 = {
            "pid": None,
            "ap_file": "/path/to/test_ap.cbin",
            "t_start": 0.0,
            "duration": 100.0,
            "output_dir": str(self.base_path),
        }

        params2 = {
            "pid": None,
            "ap_file": "/different/path/to/test_ap.cbin",  # Same filename, different path
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
            "ap_file": "/path/to/different_ap.cbin",
            "t_start": 0.0,
            "duration": 100.0,
            "output_dir": str(self.base_path),
        }

        probe_level_dir3, _ = setup_output_directory(params3)

        # Should have different hash
        self.assertNotEqual(probe_level_dir1.name, probe_level_dir3.name)


if __name__ == "__main__":
    unittest.main()
