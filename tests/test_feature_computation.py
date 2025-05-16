import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from ephysatlas.feature_computation import load_data_from_files, get_target_coordinates

def test_load_data_from_files():
    # Test with non-existent files
    with pytest.raises(RuntimeError):
        load_data_from_files("nonexistent_ap.cbin", "nonexistent_lf.cbin")
    
    # Test with invalid file types
    with tempfile.NamedTemporaryFile(suffix='.txt') as tmp_ap, \
         tempfile.NamedTemporaryFile(suffix='.txt') as tmp_lf:
        with pytest.raises(RuntimeError):
            load_data_from_files(tmp_ap.name, tmp_lf.name)
    
    #Todo add smaller .cbin files and see if they are loaded correctly

def test_get_target_coordinates():
    # Test with invalid input combinations
    with pytest.raises(ValueError):
        get_target_coordinates()  # No arguments provided
    
    with pytest.raises(ValueError):
        get_target_coordinates(pid="test_pid")  # Only pid provided
    
    with pytest.raises(ValueError):
        get_target_coordinates(one="test_one")  # Only one provided
    
    # Test with valid trajectory dictionary
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
    
    df = get_target_coordinates(channels=channels, traj_dict=traj_dict)
    
    # Check output format
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['x_target', 'y_target', 'z_target'])
    assert len(df) == len(channels['rawInd'])
    assert all(df.index == channels['rawInd']) 