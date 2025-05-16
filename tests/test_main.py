import pytest
from pathlib import Path
import yaml
import tempfile
import os

from main import load_config, parse_arguments, get_parameters


def test_imports():
    """Test that all required modules can be imported"""
    import numpy as np
    import pandas as pd
    import yaml
    from one.api import ONE
    from iblutil.util import setup_logger
    from ephysatlas.feature_computation import compute_features
    from ephysatlas.region_inference import infer_regions
    from ephysatlas.plots import plot_results
    from ephysatlas import decoding
    
    # If we get here, all imports worked
    assert True 


def test_load_config():
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        config_data = {
            'pid': 'test_pid',
            't_start': 0.0,
            'duration': 100.0,
            'mode': 'both'
        }
        yaml.dump(config_data, tmp)
        tmp_path = tmp.name

    try:
        # Test config loading
        loaded_config = load_config(tmp_path)
        assert loaded_config == config_data
    finally:
        # Delete the temp file.
        os.unlink(tmp_path)

def test_parse_arguments():
    # Test with valid arguments
    args = ['--config', 'test_config.yaml']
    parsed_args = parse_arguments(args)
    assert parsed_args.config == 'test_config.yaml'

    # Test with missing required argument
    with pytest.raises(SystemExit):
        parse_arguments([])

def test_get_parameters():
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        config_data = {
            'pid': 'test_pid',
            't_start': 0.0,
            'duration': 100.0,
            'mode': 'both'
        }
        yaml.dump(config_data, tmp)
        tmp_path = tmp.name

    try:
        # Create args namespace
        class Args:
            def __init__(self, config):
                self.config = config
        
        args = Args(tmp_path)
        
        # Test getting parameters
        params = get_parameters(args)
        assert params['pid'] == 'test_pid'
        assert params['t_start'] == 0.0
        assert params['duration'] == 100.0
        assert params['mode'] == 'both'
        
        # Test with missing required parameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_invalid:
            invalid_config = {'mode': 'both'}  # Missing required parameters
            yaml.dump(invalid_config, tmp_invalid)
            tmp_invalid_path = tmp_invalid.name
        
        try:
            args_invalid = Args(tmp_invalid_path)
            with pytest.raises(ValueError):
                get_parameters(args_invalid)
        finally:
            os.unlink(tmp_invalid_path)
            
    finally:
        os.unlink(tmp_path)