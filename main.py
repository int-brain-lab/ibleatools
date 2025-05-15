import argparse
from typing import List, Optional, Dict, Any
from ephysatlas.feature_computation import compute_features
from ephysatlas.region_inference import infer_regions
from one.api import ONE
import numpy as np
import yaml
from pathlib import Path
import pandas as pd
from ephysatlas.logger_config import setup_logger
from ephysatlas.plots import plot_results
from ephysatlas import decoding
import random
import string

# from iblutil.util import setup_logger
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
    if 'pid' in config:
        # PID-based configuration
        required_params = ['pid', 't_start', 'duration']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"Missing required parameters in config file: {', '.join(missing_params)}")
    else:
        # File-based configuration
        required_params = ['ap_file', 'lf_file', 't_start', 'duration']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"Missing required parameters in config file: {', '.join(missing_params)}")
    
    return {
        'pid': config.get('pid'),
        'ap_file': config.get('ap_file'),
        'lf_file': config.get('lf_file'),
        't_start': config['t_start'],
        'duration': config['duration'],
        'mode': config.get('mode', 'both'),
        'features_path': config.get('features_path'),
        'model_path': config.get('model_path'),
        'traj_dict': config.get('traj_dict')
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
    
    # Initialize ONE if using PID
    one = ONE()
    if params['pid'] is not None:
        logger.info("ONE client initialized")
        logger.info(f"Processing probe ID: {params['pid']}")
    else:
        logger.info(f"Processing files: AP={params['ap_file']}, LF={params['lf_file']}")
    
    df_features = None
    # Determine features file path
    features_path = params.get('features_path')
    if features_path is None:
        if params['pid'] is not None:
            features_path = Path(f"features_{params['pid']}.parquet")
        else:
            # Generate 8 character alphanumeric filename
            filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            features_path = Path(f"features_{filename}.parquet")
            logger.info(f"Generated features filename: {features_path}")
    else:
        features_path = Path(features_path)
        # Ensure the file has .parquet extension
        if features_path.suffix != '.parquet':
            features_path = features_path.with_suffix('.parquet')
    
    # Compute features if mode is 'features' or 'both'
    if params['mode'] in ['features', 'both']:
        logger.info("Starting feature computation")
        df_features = compute_features(
            pid=params.get('pid'),
            t_start=params['t_start'],
            duration=params['duration'],
            one=one,
            ap_file=params.get('ap_file'),
            lf_file=params.get('lf_file'),
            traj_dict=params.get('traj_dict')
        )
        logger.info(f"Feature computation completed. Shape: {df_features.shape}")
        
        # Save features to parquet file
        logger.info(f"Saving features to {features_path}")
        df_features.to_parquet(features_path, index=True)
    
    # Infer regions if mode is 'inference' or 'both'
    if params['mode'] in ['inference', 'both']:
        logger.info("Starting region inference")
        # Get model path from parameters or use default
        model_path = params.get('model_path')
        if model_path is None:
            model_path = Path("/Users/pranavrai/Downloads/models/2024_W50_Cosmos_voter-snap-pudding/")
        else:
            model_path = Path(model_path)
        
        # If df_features is None, load from file
        if df_features is None:
            # This should only happen in inference mode
            assert params['mode'] == 'inference'
            if not features_path.exists():
                raise ValueError(f"Features file not found at {features_path}. Please compute features first.")
            logger.info(f"Loading features from {features_path}")
            df_features = pd.read_parquet(features_path)
        
        
        predicted_probas, predicted_regions = infer_regions(df_features, model_path)
        logger.info(f"Predicted regions shape: {predicted_regions.shape}")
        logger.info(f"Prediction probabilities shape: {predicted_probas.shape}")

        # Save numpy arrays
        output_dir = features_path.parent
        np_probas_path = output_dir / f"probas_{params['pid']}.npy"
        np_regions_path = output_dir / f"regions_{params['pid']}.npy"
        
        logger.info(f"Saving prediction probabilities as numpy array to {np_probas_path}")
        np.save(np_probas_path, predicted_probas)
        
        logger.info(f"Saving predicted regions as numpy array to {np_regions_path}")
        np.save(np_regions_path, predicted_regions)

        #Plot the results
        #Todo need to have better interface than calling dict_model here just for plotting.
        dict_model = decoding.load_model(model_path.joinpath(f'FOLD04'))
        fig, ax = plot_results(df_features, predicted_probas, dict_model)
        import matplotlib.pyplot as plt
        plt.savefig(output_dir / f"results_{params['pid']}.png")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    import sys
    sys.exit(exit_code)