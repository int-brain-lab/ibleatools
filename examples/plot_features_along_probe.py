"""Plot the features of one probe along its depth, beside the histology.

The data are the feature tables committed under ``tests/fixtures/features/2025_W28`` (13
probes, 384 channels each), so the script runs without downloading anything. To plot other
features, point ``path`` at a downloaded vintage (``ephysatlas.data.download_tables``) or at
the output of ``produce_output_dataframes``.

Run it from the repository root, whole or cell by cell (``# %%`` cells). The figures are saved
to ``figures/``.
"""

# %% Imports and data
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from iblatlas.regions import BrainRegions

import ephysatlas.plots
from ephysatlas.features import EphysTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
path = REPO / "tests" / "fixtures" / "features" / "2025_W28"
outdir = Path("figures")
outdir.mkdir(exist_ok=True)

df_features = pd.read_parquet(path / "raw_ephys_features_denoised.pqt")
df_channels = pd.read_parquet(path / "channels.pqt")
pid = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"  # crosses Isocortex, HPF and TH

# The features of the heatmaps, grouped by band. Leave feature_groups out to draw every
# feature of the table, grouped by family
feature_groups = {
    "LF": [
        "rms_lf", "psd_delta", "psd_theta", "psd_alpha", "psd_beta", "psd_gamma",
        "psd_lfp",
    ],
    "CSD": [
        "rms_lf_csd", "psd_delta_csd", "psd_theta_csd", "psd_alpha_csd",
        "psd_beta_csd", "psd_gamma_csd", "psd_lfp_csd",
    ],
    "AP": ["rms_ap", "cor_ratio"],
    "Spikes": [
        "spike_count", "alpha_mean", "alpha_std", "peak_val", "trough_val", "tip_val",
        "depolarisation_slope", "repolarisation_slope", "recovery_slope", "polarity",
    ],
}  # fmt: skip

# %% 1. Z-scored features down one probe, beside the histology
# Each feature is z-scored over every channel of the 13 probes, then drawn for one probe
fig, axs = ephysatlas.plots.figure_features_heatmap(
    df_features,
    pid,
    channels=df_channels,
    feature_groups=feature_groups,
    title=f"Denoised features along probe {pid[:8]}",
)
fig.savefig(outdir / "features_zscored_along_probe.png", dpi=110)
logger.info("wrote %s", outdir / "features_zscored_along_probe.png")

# %% 2. The same probe before and after denoising, on one colour scale
features = [f for group in feature_groups.values() for f in group]
df_raw = pd.read_parquet(path / "raw_ephys_features.pqt")
# Put the aggregated table in the denoised units first (rms_* to dB, spike_count to log2),
# so the only remaining difference between the two panels is the denoising filter
transformer = EphysTransformer()
transformer.fit(df_raw[features])
df_raw_t = transformer.transform(df_raw[features])

# Both panels use the z-score of the first table
fig, axs = ephysatlas.plots.figure_features_heatmap(
    {"aggregated, before denoising": df_raw_t, "after denoising": df_features},
    pid,
    channels=df_channels,
    feature_groups=feature_groups,
    show_group_labels=False,
    title=f"Denoising, probe {pid[:8]}",
)
fig.savefig(outdir / "features_raw_vs_denoised.png", dpi=110)
logger.info("wrote %s", outdir / "features_raw_vs_denoised.png")

# %% 3. Each feature as a strip on the probe geometry
# figure_features_channel_space wants features and channel geometry in one table, with a
# region id column in the map it colours by
duplicate_cols = set(df_features.columns) & set(df_channels.columns)
df = df_features.merge(
    df_channels.drop(columns=duplicate_cols), left_index=True, right_index=True
)
br = BrainRegions()
df["Cosmos_id"] = br.remap(
    df["atlas_id"].to_numpy(), source_map="Allen", target_map="Cosmos"
)
pid_df = df.loc[[pid]].copy()
fig, axs = ephysatlas.plots.figure_features_channel_space(
    pid_df,
    features=["rms_lf", "psd_delta", "psd_theta", "psd_gamma", "rms_ap", "spike_count"],
    xy=pid_df[["lateral_um", "axial_um"]].to_numpy(),  # lateral first, then axial
    pid=pid,
    br=br,
    mapping="Cosmos",
)
fig.savefig(outdir / "features_along_probe.png", dpi=150, bbox_inches="tight")
logger.info("wrote %s", outdir / "features_along_probe.png")

# %% 4. The distribution of 24 features over every channel of the dataset
fig, axs = ephysatlas.plots.plot_features_distributions(
    df,
    x_list=features[:24],
    title="Feature distributions (2025_W28 fixture, 13 probes)",
)
fig.savefig(outdir / "features_distributions.png", dpi=150, bbox_inches="tight")
logger.info("wrote %s", outdir / "features_distributions.png")
plt.close("all")
