import os
import argparse
from typing import List, Optional, Dict, Any
from src.feature_computation import compute_features
from src.region_inference import infer_regions
from one.api import ONE
import numpy as np
import yaml
from pathlib import Path
import pandas as pd
from src.logger_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_arguments(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    logger.debug("Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Electrophysiology feature computation and region inference")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    return parser.parse_args(args)


def get_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    """Get parameters from config file."""
    logger.info("Loading configuration from YAML file")
    config = load_config(args.config)
    
    # Validate required parameters
    required_params = ['pid', 't_start', 'duration']
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise ValueError(f"Missing required parameters in config file: {', '.join(missing_params)}")
    
    return {
        'pid': config['pid'],
        't_start': config['t_start'],
        'duration': config['duration'],
        'mode': config.get('mode', 'both'),
        'features_path': config.get('features_path'),
        'model_path': config.get('model_path')
    }


def main(args: Optional[List[str]] = None) -> int:
    """Main function that can be called with arguments or use command line arguments."""
    logger.info("Starting main function")
    if args is None:
        import sys
        args = sys.argv[1:]
    
    # Parse arguments
    parsed_args = parse_arguments(args)
    
    # Get parameters from config file
    params = get_parameters(parsed_args)
    logger.info(f"Processing probe ID: {params['pid']}")
    
    # Initialize ONE
    one = ONE()
    logger.info("ONE client initialized")
    
    df_features = None
    # Determine features file path
    features_path = params.get('features_path')
    if features_path is None:
        features_path = Path(f"features_{params['pid']}.parquet")
    else:
        features_path = Path(features_path)
        # Ensure the file has .parquet extension
        if features_path.suffix != '.parquet':
            features_path = features_path.with_suffix('.parquet')
    
    # Compute features if mode is 'features' or 'both'
    if params['mode'] in ['features', 'both']:
        logger.info("Starting feature computation")
        df_features = compute_features(
            params['pid'], 
            params['t_start'], 
            params['duration'], 
            one
        )
        logger.info(f"Feature computation completed. Shape: {df_features.shape}")
        
        # Save features to parquet file
        logger.info(f"Saving features to {features_path}")
        df_features.to_parquet(features_path, index=False)
    
    # Infer regions if mode is 'inference' or 'both'
    if params['mode'] in ['inference', 'both']:
        logger.info("Starting region inference")
        # Get model path from parameters or use default
        model_path = params.get('model_path')
        if model_path is None:
            model_path = Path("/Users/pranavrai/Downloads/models/2024_W50_Cosmos_voter-snap-pudding/")
        else:
            model_path = Path(model_path)
        
        # If df_features is None, load from file.
        if df_features is None:
            # This should only in inference mode.
            assert params['mode'] == 'inference'
            if not features_path.exists():
                raise ValueError(f"Features file not found at {features_path}. Please compute features first.")
            logger.info(f"Loading features from {features_path}")
            df_features = pd.read_parquet(features_path)
        
        predicted_regions, predicted_probas = infer_regions(df_features, params['pid'], one, model_path)
        logger.info(f"Predicted regions shape: {predicted_regions.shape}")
        logger.info(f"Prediction probabilities shape: {predicted_probas.shape}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    import sys
    sys.exit(exit_code)