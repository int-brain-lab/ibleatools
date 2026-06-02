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
import pandas as pd
import seaborn as sns

from viewephys.gui import viewephys
import addcopyfighandler  # noqa: F401

sns.set_theme(context='notebook')

CELLS_AGG_PATH = Path('/datadisk/ephys-atlas/cells_aggregates')
CELLS_AGG_PATH = Path('/Users/olivier/Documents/datadisk/paper-ephys-atlas/cells_aggregates')
SAMPLING_RATE = 30_000  # Hz
FIGURES_PATH = Path.home().joinpath('Documents', 'figures')

# %% Load tables and small arrays
df_clusters = pd.read_parquet(CELLS_AGG_PATH.joinpath('clusters.table.pqt'))
df_clusters_good = pd.read_parquet(CELLS_AGG_PATH.joinpath('clusters_good.table.pqt'))
acgs_log = np.load(CELLS_AGG_PATH.joinpath('clusters.acgs_log.npy')).astype(np.float32)
acgs_log_times = np.load(CELLS_AGG_PATH.joinpath('acgs_log.times.npy'))
waveforms_peak = np.load(CELLS_AGG_PATH.joinpath('clusters.waveforms_peak.npy')).astype(np.float32)

print(f'n_clusters              {df_clusters.shape[0]:,}')
print(f'n_good_clusters         {df_clusters_good.shape[0]:,}')
print(f'waveforms_peak          {waveforms_peak.shape}')
print(f'acgs_log                {acgs_log.shape}')
print(f'clusters.table columns:\n  {list(df_clusters.columns)}')

# %% Visualise waveforms and ACGs for a handful of good units
rng = np.random.default_rng()
good_mask = df_clusters['bitwise_fail'] == 0
good_idx = np.where(good_mask)[0]
sample_idx = rng.choice(good_idx, size=6, replace=False)

fig, axs = plt.subplots(2, 6, figsize=(14, 7))
ns = waveforms_peak.shape[1]
t_wf = np.arange(ns) / SAMPLING_RATE * 1e3  # ms

for col, idx in enumerate(sample_idx):
    cid = df_clusters.index[idx]

    ax_wf = axs[0, col]
    ax_wf.plot(t_wf, waveforms_peak[idx] * 1e6, color='steelblue', lw=1)
    ax_wf.set(title=f'cid {cid}', xlabel='ms' if col == 0 else '', ylabel='µV' if col == 0 else '')
    ax_wf.axhline(0, color='k', lw=0.5, ls='--')

    ax_acg = axs[1, col]
    ax_acg.semilogx(acgs_log_times * 1e3, acgs_log[idx], color='coral', lw=1)
    ax_acg.set(xlabel='lag (ms)' if col == 0 else '', ylabel='sp/s' if col == 0 else '')

axs[0, 0].set_ylabel('waveform (µV)')
axs[1, 0].set_ylabel('ACG (sp/s)')
fig.suptitle('Example units — peak-channel waveform (top) and log-ACG (bottom)')
fig.tight_layout()
fig.savefig(FIGURES_PATH.joinpath('2026-05-29_units_waveforms_acgs.png'), dpi=150)

# %% Feature distributions across good units — waveform features and burstiness/memory are all in clusters.table
features_to_plot = ['peak_to_trough_ratio', 'half_width', 'repolarisation_slope',
                    'recovery_slope', 'tip_ratio', 'burstiness', 'memory']
features_to_plot = [f for f in features_to_plot if f in df_clusters_good.columns]

n = len(features_to_plot)
ncols = (n + 1) // 2
fig, axs = plt.subplots(2, ncols, figsize=(3 * ncols, 6))
for i, (ax, feat) in enumerate(zip(axs.flatten(), features_to_plot)):
    vals = df_clusters_good[feat].dropna()
    ax.hist(vals, bins=100, color='steelblue', edgecolor='none')
    ax.set(xlabel=feat, ylabel='count' if i % ncols == 0 else '')
for ax in axs.flatten()[n:]:
    ax.set_visible(False)

fig.suptitle('Waveform feature + burstiness/memory distributions (good units)')
fig.tight_layout()
fig.savefig(FIGURES_PATH.joinpath('2026-05-29_units_feature_distributions.png'), dpi=150)

# %%
eqc = viewephys(waveforms_peak, fs=SAMPLING_RATE)

# %% Burstiness vs memory cross-plot per Cosmos region (Allen atlas colours, good units)
from iblatlas.regions import BrainRegions

br = BrainRegions()

df_plot = df_clusters_good[['burstiness', 'memory', 'atlas_id']].dropna().copy()
atlas_ids = df_plot['atlas_id'].values

# Map each cell to its Cosmos parent and keep its Allen atlas RGB
# colour
cosmos_ids = br.id2id(atlas_ids, mapping='Cosmos')
rgb = br.get(atlas_ids).rgb / 255.0  # (N, 3) float, Allen colours per cell

# Unique Cosmos regions, dropping void / root
void_root = set(br.acronym2id(['void', 'root']).tolist())
cosmos_unique = np.array([c for c in np.unique(cosmos_ids) if c not in void_root])
cosmos_info = br.get(cosmos_unique)

n = len(cosmos_unique)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))

fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True)
for ax in axs.flatten():
    ax.set_visible(False)

for i, (cid, acronym, color) in enumerate(zip(cosmos_unique, cosmos_info.acronym, cosmos_info.rgb)):
    ax = axs.flatten()[i]
    ax.set_visible(True)
    mask = cosmos_ids == cid
    ax.scatter(
        df_plot['burstiness'].values[mask],
        df_plot['memory'].values[mask],
        c=rgb[mask],
        s=1, alpha=0.5, linewidths=0, rasterized=True,
    )
    ax.set_title(acronym, color=color / 255.0, fontweight='bold', fontsize=9)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='k', lw=0.3, ls='--')
    ax.axvline(0, color='k', lw=0.3, ls='--')

fig.supxlabel('burstiness')
fig.supylabel('memory')
fig.suptitle('Burstiness vs memory by Cosmos region — Allen atlas colours (good units, Cosmos mapping)')
fig.tight_layout()
fig.savefig(FIGURES_PATH.joinpath('2026-06-01_burstiness_memory_cosmos.png'), dpi=150)
