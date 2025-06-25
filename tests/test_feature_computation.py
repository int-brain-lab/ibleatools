import unittest
import numpy as np
from pathlib import Path
import tempfile

from ephysatlas.feature_computation import load_data_from_files, add_target_coordinates


class TestFeatureComputation(unittest.TestCase):
    
    def test_load_data_from_files_nonexistent_files(self):
        """Test load_data_from_files with non-existent files"""
        with self.assertRaises(RuntimeError):
            load_data_from_files("nonexistent_ap.cbin", "nonexistent_lf.cbin", Path("."))
    
    def test_load_data_from_files_invalid_file_types(self):
        """Test load_data_from_files with invalid file types"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp_ap, \
             tempfile.NamedTemporaryFile(suffix='.txt') as tmp_lf:
            with self.assertRaises(RuntimeError):
                load_data_from_files(tmp_ap.name, tmp_lf.name, Path("."))
    
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
        channels = {
            "rawInd": np.arange(10),
            "axial_um": np.linspace(0, 1000, 10) 
        }
        
        traj_dict = {
            "x": 1000,
            "y": 2000,
            "z": 3000,
            "depth": 4000,
            "theta": 0,
            "phi": 0
        }
        
        result = add_target_coordinates(channels=channels, traj_dict=traj_dict)
        
        # Check output format
        self.assertIsInstance(result, dict)
        self.assertIn('x_target', result)
        self.assertIn('y_target', result)
        self.assertIn('z_target', result)
        self.assertEqual(len(result['x_target']), len(channels['rawInd']))
        self.assertEqual(len(result['y_target']), len(channels['rawInd']))
        self.assertEqual(len(result['z_target']), len(channels['rawInd']))
    
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
            "phi": 0
        }
        
        result = add_target_coordinates(channels=channels, traj_dict=traj_dict)
        
        # Check that rawInd was added
        self.assertIn('rawInd', result)
        self.assertEqual(len(result['rawInd']), 384)
        self.assertTrue(np.array_equal(result['rawInd'], np.arange(384)))


if __name__ == '__main__':
    unittest.main() 