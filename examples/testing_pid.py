from one.api import ONE
from ephysatlas.feature_computation import compute_features
from deploy.iblsdsc import OneSdsc
# from ephysatlas.regionclassifier.region_inference import infer_regions
from pathlib import Path
import pandas as pd
import logging

# Set the logging level at the beginning of your script
logging.basicConfig(level=logging.INFO)

# local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Decoder_June')
# model_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Decoder_June/model/2024_W50_Cosmos_voter-snap-pudding')
# Note: the model needs to be manually shared for now, ask OW or PR
local_data_path = Path(".")
one = ONE()
# one =OneSdsc(mode='local' , cache_rest=None)
force_feature_compute = False

pid = "4cb60c5c-d15b-4abd-8cfd-776bc5a81dbe"
features_to_compute = []

## Compute features
features_path = local_data_path.joinpath(f'{pid}__features.pqt')
if not features_path.exists() or force_feature_compute:
    # Compute features
    df = compute_features(
        pid=pid,
        t_start=300.0,
        duration=1.0,
        one=one,
        features_to_compute=features_to_compute,
        recompute_channels=False,
        )
    # Save dataframe
    df.to_parquet(features_path)
else:
    # Load features
    df = pd.read_parquet(features_path)

## Run inferences
# predicted_probas, predicted_regions = infer_regions(df, model_path)
