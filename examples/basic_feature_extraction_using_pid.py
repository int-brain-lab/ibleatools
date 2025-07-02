from one.api import ONE
from ephysatlas.feature_computation import compute_features
from pathlib import Path
import pandas as pd
import tempfile

#Change the local data path to your desired location
local_data_path = Path(tempfile.gettempdir())
print(f"Data will be saved at {local_data_path}")

one = ONE()

pid = "4cb60c5c-d15b-4abd-8cfd-776bc5a81dbe"

## Compute features
features_path = local_data_path.joinpath(f'{pid}__features.pqt')
df = compute_features(
    pid=pid,
    t_start=300.0,
    duration=1.0,
    one=one,
    features_to_compute=["lf", "csd", "ap"])
# Save dataframe
df.to_parquet(features_path)
