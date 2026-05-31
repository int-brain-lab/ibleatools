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
SAMPLING_RATE = 30_000  # Hz
FIGURES_PATH = Path.home().joinpath('Documents', 'figures')

# %% Load tables and small arrays
df_clusters = pd.read_parquet(CELLS_AGG_PATH.joinpath('clusters.table.pqt'))
df_clusters_good = pd.read_parquet(CELLS_AGG_PATH.joinpath('clusters_good.table.pqt'))
acgs_log = np.load(CELLS_AGG_PATH.joinpath('clusters.acgs_log.npy')).astype(np.float32)
acgs_log_old = np.load(Path('/datadisk/ephys-atlas/cells_aggregates-old/acgs_log_bins.npy'))
acgs_log_times = np.load(CELLS_AGG_PATH.joinpath('acgs_log.times.npy'))
waveforms_peak = np.load(CELLS_AGG_PATH.joinpath('clusters.waveforms_peak.npy')).astype(np.float32)

print(f'n_clusters              {df_clusters.shape[0]:,}')
print(f'n_good_clusters         {df_clusters_good.shape[0]:,}')
print(f'waveforms_peak          {waveforms_peak.shape}')
print(f'acgs_log                {acgs_log.shape}')
print(f'clusters.table columns:\n  {list(df_clusters.columns)}')

# %% Visualise waveforms and ACGs for a handful of good units
rng = np.random.default_rng(54)
good_mask = df_clusters['bitwise_fail'] == 0
good_idx = np.where(good_mask)[0]
sample_idx = rng.choice(good_idx, size=6, replace=False)

fig, axs = plt.subplots(2, 6, figsize=(18, 6))
ns = waveforms_peak.shape[1]
t_wf = np.arange(ns) / SAMPLING_RATE * 1e3  # ms

for col, idx in enumerate(sample_idx):
    cid = df_clusters.index[idx]

    ax_wf = axs[0, col]
    ax_wf.plot(t_wf, waveforms_peak[idx] * 1e6, color='steelblue', lw=1)
    ax_wf.set(title=f'cid {cid}', xlabel='ms' if col == 0 else '', ylabel='µV' if col == 0 else '')
    ax_wf.axhline(0, color='k', lw=0.5, ls='--')

    ax_acg = axs[1, col]
    ax_acg.semilogx(acgs_log_times * 1e3, acgs_log[idx], color='coral', lw=2)
    ax_acg.semilogx(acgs_log_times * 1e3, acgs_log_old[idx] / df_clusters['spike_count'].iloc[idx], color='green', lw=1)
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

fig, axs = plt.subplots(1, len(features_to_plot), figsize=(4 * len(features_to_plot), 4))
for ax, feat in zip(axs, features_to_plot):
    vals = df_clusters_good[feat].dropna()
    ax.hist(vals, bins=100, color='steelblue', edgecolor='none')
    ax.set(xlabel=feat, ylabel='count')

fig.suptitle('Waveform feature + burstiness/memory distributions (good units)')
fig.tight_layout()
fig.savefig(FIGURES_PATH.joinpath('2026-05-29_units_feature_distributions.png'), dpi=150)

# %%
eqc = viewephys(waveforms_peak, fs=SAMPLING_RATE)

# %%
SKILL_SOURCES=(
  "$HOME/Documents/ibldevtools/00_on_call/skills"
  "$HOME/Documents/ibl-ai-agent/skills"
  "$HOME/PycharmProjects/EphysAtlas/paper-ephys-atlas/.claude/skills"
)