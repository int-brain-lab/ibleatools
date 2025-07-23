# %% Step 1: Create the reveal deck of images
from pathlib import Path
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing


from one.api import ONE
import ephysatlas.fixtures
import ephysatlas.reveal

pid = str(np.random.choice(ephysatlas.fixtures.benchmark_pids))
path_features = Path("/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding/features/2025_W28")
df_features = ephysatlas.data.read_features_from_disk(path_features, strict=False)

one = ONE(base_url="https://alyx.internationalbrainlab.org")
path_model = Path(
    "/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding/models/2025_W28_Cosmos_living-olivedrab-cassowary")
df_predictions = pd.read_parquet(path_model / "predictions.pqt")
path_figures = Path(f"/datadisk/Data/paper-ephys-atlas/reveal")
pids = df_predictions.index.get_level_values(level=0).unique()
path_reveal = Path.home().joinpath('Documents/JS/reveal.internationalbrainlab.org')
# learn the scaling (ideally this should be in the model data itself)
scaler = sklearn.preprocessing.RobustScaler()
x_list = ephysatlas.features.voltage_features_set()
scaler.fit(df_features.loc[:, x_list])


# %%
pid = np.random.choice(ephysatlas.fixtures.benchmark_pids)
pid = 'dab512bd-a02d-4c1f-8dbc-9155a163efc0'
df_pid = df_features.loc[pid]
ar = ephysatlas.reveal.AtlasReveal(one, pid=pid, df_pid=df_pid)

# You can also save all figures at once
# todo: figure 01 normalize using the full list of features
ar.figure_01_features_with_histology_columns(scaler=scaler)
# ar.figure_02_classifier_results(df_predictions=df_predictions.loc[pid], path_model=path_model)
ar.figure_03_histology_slices()
# ar.figure_04_ap_voltage()
# ar.figure_05_lfp_voltage()
# ar.figure_06_bad_channels()

# %%


