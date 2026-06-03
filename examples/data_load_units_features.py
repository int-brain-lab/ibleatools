# %%
"""
Load and visualise per-unit features produced by the SDSC cells map/reduce pipeline.


Files under CELLS_AGG_PATH
---------------------------
waveforms.voltage.npy         (n_traces_total, 128)  float16  – all neighbourhood traces (large ~8 GB)
clusters.waveforms_peak.npy   (n_clusters, 128)       float16  – peak-channel trace per cluster
clusters.acgs_log.npy         (n_clusters, 128)       float16  – log-binned ACGs normalised by spike_count (sp/sp)
acgs_log.times.npy            (128,)                  float64  – bin centres in seconds
clusters_good.stpc.npy        (n_good, 1000)          float16  – spike-triggered power coupling
clusters_good.stlfp.npy       (n_good, 250)           float16  – spike-triggered LFP

clusters.table.pqt            (n_clusters, ~59 cols)           – all clusters: QC, anatomy, burstiness/memory, waveform features
clusters_good.table.pqt       (n_good, ~61 cols)               – good clusters (bitwise_fail==0): same + coupling_delay/coupling_strength
waveforms.table.pqt           (n_traces_total, 3 cols)         – pid / cluster_id / abs_channel index into waveforms.voltage.npy
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from iblatlas.regions import BrainRegions
from iblutil.numerical import ismember
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rastermap import Rastermap

from viewephys.gui import viewephys
import addcopyfighandler  # noqa: F401
import ephysatlas.data

sns.set_theme(context="notebook")

PROJECT = "ibl_neuropixel_brainwide_01"
ROOT_PATH = Path("/Users/olivier/Documents/datadisk/paper-ephys-atlas")
PATH_PROJECT = ROOT_PATH.joinpath(PROJECT)
CELLS_AGG_PATH = PATH_PROJECT.joinpath("cells_aggregates")
SAMPLING_RATE = 30_000  # Hz
FIGURES_PATH = Path.home().joinpath("Documents", "figures")

# %% Download data if not yet available
if not CELLS_AGG_PATH.joinpath("clusters.table.pqt").exists():
    from one.api import ONE

    one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="remote")
    ephysatlas.data.download_cells_features(ROOT_PATH, project=PROJECT, one=one)

# %% Load tables and small arrays
data = ephysatlas.data.read_cells_features(PATH_PROJECT)
df_clusters = data["df_clusters"]
df_clusters_good = data["df_clusters_good"]
acgs_log = data["acgs_log"]
acgs_log_times = data["acgs_log_times"]
waveforms_peak = data["waveforms_peak"]

good_pos = np.where(df_clusters["bitwise_fail"] == 0)[0]
waveforms_peak_good = waveforms_peak[good_pos]
acgs_log_good = acgs_log[good_pos]

print(f"n_clusters              {df_clusters.shape[0]:,}")
print(f"n_good_clusters         {df_clusters_good.shape[0]:,}")
print(f"waveforms_peak          {waveforms_peak.shape}")
print(f"acgs_log                {acgs_log.shape}")
print(f"clusters.table columns:\n  {list(df_clusters.columns)}")

# %% Visualise waveforms and ACGs for a handful of good units
rng = np.random.default_rng()
sample_idx = rng.choice(len(df_clusters_good), size=6, replace=False)

fig, axs = plt.subplots(2, 6, figsize=(14, 7))
ns = waveforms_peak_good.shape[1]
t_wf = np.arange(ns) / SAMPLING_RATE * 1e3  # ms

for col, idx in enumerate(sample_idx):
    cid = df_clusters_good.index[idx]

    ax_wf = axs[0, col]
    ax_wf.plot(t_wf, waveforms_peak_good[idx] * 1e6, color="steelblue", lw=1)
    ax_wf.set(
        title=f"cid {cid}",
        xlabel="ms" if col == 0 else "",
        ylabel="µV" if col == 0 else "",
    )
    ax_wf.axhline(0, color="k", lw=0.5, ls="--")

    ax_acg = axs[1, col]
    ax_acg.semilogx(acgs_log_times * 1e3, acgs_log_good[idx], color="coral", lw=1)
    ax_acg.set(xlabel="lag (ms)" if col == 0 else "", ylabel="sp/s" if col == 0 else "")

axs[0, 0].set_ylabel("waveform (µV)")
axs[1, 0].set_ylabel("ACG (sp/s)")
fig.suptitle("Example units — peak-channel waveform (top) and log-ACG (bottom)")
fig.tight_layout()
fig.savefig(FIGURES_PATH.joinpath("2026-05-29_units_waveforms_acgs.png"), dpi=150)


# %%
br = BrainRegions()
_, rids = ismember(df_clusters_good["atlas_id"].values, br.id)
sort_idx = np.argsort(br.order[rids])
eqcs = {}
eqcs["waveforms"] = viewephys(
    waveforms_peak_good[sort_idx],
    fs=SAMPLING_RATE,
    title="waveforms by region",
    br=br,
    channels=df_clusters_good.iloc[sort_idx],
)


# %% Rastermap sort
model = Rastermap(
    n_PCs=64,
    n_clusters=100,
    grid_upsample=0,
    locality=0.75,
    time_lag_window=0,
    bin_size=1,
    symmetric=False,
).fit(waveforms_peak_good)

# %% Matplotlib: region sort vs region-primary + rastermap-secondary
rastermap_rank = np.argsort(model.isort)
cosmos_ids = br.remap(
    df_clusters_good["atlas_id"].values, source_map="Allen", target_map="Cosmos"
)
_, cosmos_rids = ismember(cosmos_ids, br.id)
cosmos_order = br.order[cosmos_rids]
sort_idx_cosmos_rm = np.lexsort((rastermap_rank, cosmos_order))

image_region = br.rgb[rids].astype(np.uint8)
t_ms = np.arange(waveforms_peak_good.shape[1]) / (SAMPLING_RATE / 1000)
vmax = np.nanpercentile(np.abs(waveforms_peak_good), 99)

fig, axs = plt.subplots(
    4, 1, figsize=(22, 10), gridspec_kw={"height_ratios": [0.2, 3, 0.2, 3]}
)
titles = ["Allen region order", "Rastermap within Cosmos regions"]
for col, (sidx, title) in enumerate(zip([sort_idx, sort_idx_cosmos_rm], titles)):
    axr, axw = axs[col * 2], axs[col * 2 + 1]
    axr.imshow(image_region[np.newaxis, sidx], aspect="auto")
    axr.axis("off")
    axr.set_title(title)
    im = axw.imshow(
        waveforms_peak_good[sidx].T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[0, sidx.size, t_ms[-1], t_ms[0]],
    )
    axw.set(xlabel="Unit #" if col == 1 else "", ylabel="Time (ms)")
    for ax, visible in ((axr, False), (axw, col == 1)):
        cax = make_axes_locatable(ax).append_axes("right", size="1%", pad=0.05)
        if visible:
            fig.colorbar(im, cax=cax, label="Amplitude (V)")
        else:
            cax.axis("off")

fig.tight_layout()
fig.savefig(
    FIGURES_PATH.joinpath("2026-06-03_waveforms_region_vs_rastermap.png"), dpi=150
)
