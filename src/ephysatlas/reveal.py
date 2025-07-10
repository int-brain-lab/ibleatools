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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics

from brainbox.io.one import SpikeSortingLoader
import brainbox.ephys_plots
from one.api import ONE

import ephysatlas.features
import ephysatlas.data
import ephysatlas.anatomy
import ephysatlas.regionclassifier
import ephysatlas.plots

STREAM = False
one = ONE(base_url="https://alyx.internationalbrainlab.org")

ba = ephysatlas.anatomy.ClassifierAtlas()
path_features = Path("/mnt/s0/ephys-atlas-decoding/features/2025_W27")  # parede
path_model = Path(
    "/mnt/s0/ephys-atlas-decoding/models/2025_W27_Cosmos_magical-emerald-starling"
)
pid = "749cb2b7-e57e-4453-a794-f6230e4d0226"  # mrsicflogellab/Subjects/SWC_038/2020-07-30/001/alf/probe01

ssl = SpikeSortingLoader(pid=pid, one=one)
raw_ap = ssl.raw_electrophysiology(band="ap", stream=STREAM)
raw_lf = ssl.raw_electrophysiology(band="lf", stream=STREAM)

df_features = ephysatlas.data.read_features_from_disk(path_features, strict=False)
x_list = ephysatlas.features.voltage_features_set()
df_pid = df_features.loc[pid]
df_predictions = pd.read_parquet(path_model.joinpath("predictions.pqt"))

# %% Figure 01
# Plot overall displays
xy = df_pid[["lateral_um", "axial_um"]].to_numpy()

fig, axs = ephysatlas.plots.figure_features_channel_space(
    df_pid, x_list, xy, pid=pid, mapping="Allen", cmap="Spectral", br=ba.regions
)


# %% Figure 02: results of the channel regions classifer
classifier, model_info = ephysatlas.regionclassifier.load_model(path_model)
df_predictions = pd.read_parquet(path_model.joinpath("predictions.pqt"))

rids = model_info["CLASSES"]
xy = df_pid[["lateral_um", "axial_um"]].to_numpy()
df_pid_merged = df_pid.merge(df_predictions.loc[pid], left_index=True, right_index=True)
accuracy = sklearn.metrics.accuracy_score(
    df_pid_merged["prediction"].values, df_pid_merged["Cosmos_id"].values
)
df_pid_merged["confidence"] = np.max(
    df_pid_merged.loc[:, [str(c) for c in rids]], axis=1
)
df_depths = df_pid_merged.drop("acronym", axis=1).groupby("axial_um").mean()


fig, axs = plt.subplots(
    1, 8, figsize=(10, 6), gridspec_kw={"width_ratios": [0.3, 1, 0.3, 1, 1, 1, 0.2, 5]}
)
# brain regions column with ALlen leaf labels
ax = axs[0]
brainbox.ephys_plots.plot_brain_regions(
    df_pid["atlas_id"].values,
    channel_depths=xy[:, 1],
    brain_regions=ba.regions,
    display=True,
    ax=axs[0],
)
ax = axs[1]
ephysatlas.plots.plot_probe_rect2(
    xy, color=ephysatlas.plots.get_color_br(df_pid, ba.regions, mapping="Allen"), ax=ax
)
ax.set_title("Allen True labels")


# brain regions column with Cosmos leaf labels
ax = axs[2]
brainbox.ephys_plots.plot_brain_regions(
    df_pid["Cosmos_id"].values,
    channel_depths=xy[:, 1],
    brain_regions=ba.regions,
    display=True,
    ax=ax,
)
# ax.set_yticklabels(ax.get_yticklabels(), rotation=90)

ax = axs[3]
ephysatlas.plots.plot_probe_rect2(
    xy, color=ephysatlas.plots.get_color_br(df_pid, ba.regions, mapping="Cosmos"), ax=ax
)
ax.set_title("Cosmos True labels")

# show predictions
ax = axs[4]
ephysatlas.plots.plot_probe_rect2(
    xy, color=ba.regions.get(df_pid_merged["prediction"].values).rgb / 255, ax=ax
)
ax.set_title("Cosmos prediction")

# show confidence
ax = axs[5]
img = ephysatlas.plots.plot_probe_rect2(
    xy,
    color=ephysatlas.plots.get_color_feat(
        df_pid_merged["confidence"], cmap_name="magma", min_val=0, max_val=1
    ),
    ax=ax,
    colorbar=True,
)
ax.set_title("Confidence")

axs[6].set_axis_off()

# show cumulative probabilities
ax = axs[7]
ephysatlas.plots.plot_cumulative_probas(
    df_depths.loc[:, [str(c) for c in rids]].values,
    df_depths.index,
    np.array(rids),
    regions=ba.regions,
    ax=ax,
)
ax.set_title("Classifier predicted probabilities")

# Move y-axis to the right side
ax.yaxis.set_label_position("right")
ax.tick_params(
    axis="y", which="both", left=False, right=True, labelleft=False, labelright=True
)

# Optional: Add a y-axis label
ax.set_ylabel("Depth (μm)", rotation=270, labelpad=15)  # Adjust labelpad as needed


fig.suptitle(
    f"PID {pid} \n accuracy {accuracy:0.2} \n confidence {np.mean(df_pid_merged['confidence']): 0.2}",
    y=0.08,
    fontweight="bold",
)


# %% Also plot all of the features
