"""
Figure 01: ce - Features with checkerboard pattern, make sure to add the Atlas ID, Cosmos and the unique atlas ID next to it
Figure 02: bcg - Prediction of vanilla model +  HMM + confidence + post localization
Figure 04: a(c) - AP band snippet (raw / destriped)
Figure 05: a(c) - LF band snippet (raw / destriped)
Figure 06: ad - Bad channel AP (NB: also plot the actual outcome from the dataframe, ie. the one in ALF)
Figure 07: h(ci) - Raster + behaviour start/stop times + snippets (computed and the one displayed) + spike sorting version

Data types:
a- raw data
b- target coordinates
c- ground truth: ephys aligned coordinates
d- bad channels
e- features (denoised)
f- encoding model - outlier predictions
g- decoding model - region predictions
h- spike sorting data
i- behaviour events

"""

import addcopyfighandler  # noqa

# %%
from pathlib import Path

from brainbox.io.one import SpikeSortingLoader
from one.api import ONE

import pandas as pd

import ephysatlas.features
import ephysatlas.data
import ephysatlas.anatomy
import ephysatlas.regionclassifier
import ephysatlas.plots

STREAM = False
one = ONE(base_url="https://alyx.internationalbrainlab.org")

ba = ephysatlas.anatomy.ClassifierAtlas()
path_features = Path("/mnt/s0/ephys-atlas-decoding/features/2024_W50")  # parede
path_model = Path(
    "/mnt/s0/ephys-atlas-decoding/models/2024_W50_Cosmos_medical-rosewood-kestrel/"
)
pid = "749cb2b7-e57e-4453-a794-f6230e4d0226"  # mrsicflogellab/Subjects/SWC_038/2020-07-30/001/alf/probe01

ssl = SpikeSortingLoader(pid=pid, one=one)
raw_ap = ssl.raw_electrophysiology(band="ap", stream=STREAM)
raw_lf = ssl.raw_electrophysiology(band="lf", stream=STREAM)

df_features = ephysatlas.data.read_features_from_disk(path_features)
df_pid = df_features.loc[pid]
df_predictions = pd.read_parquet(path_model.joinpath("cross_validation.pqt"))
# TODO tilt

# %% Figure 01
# Plot overall displays
x_list = ephysatlas.features.voltage_features_set()
xy = df_pid[["lateral_um", "axial_um"]].to_numpy()

fig, axs = ephysatlas.plots.figure_features_channel_space(
    df_pid, x_list, xy, pid=pid, mapping="Allen", cmap="Spectral", br=ba.regions
)


# %% Figure 02
ephysatlas.regionclassifier.load_model(path_model.joinpath("FOLD00"))
df_predictions

# TODO clims / hlims using quantiles
# clim = np.array([np.nanquantile(feature, 0.1), np.nanquantile(feature, 0.9)])
# hlim = np.array([np.nanquantile(feature, 0.01), np.nanquantile(feature, 0.99)])
