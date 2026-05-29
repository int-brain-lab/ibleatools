# %%
"""
Load and visualise per-unit features produced by the SDSC cells map/reduce pipeline.

Files under CELLS_AGG_PATH
---------------------------
avg_waveforms.npy            (n_traces_total, ns)  float32  – all neighbourhood traces (large ~8 GB)
avg_waveform_peak_channel.npy (n_clusters, ns)     float32  – peak-channel trace per cluster
acgs_log_bins.npy            (n_clusters, n_log_bins) float32 – log-binned ACGs
acgs_log_times.npy           (n_log_bins,)          float64  – bin centres in seconds
df_clusters.pqt              (n_clusters, ~35 cols)          – cluster table after ssl merge
avg_waveforms_index.pqt      (n_traces_total, 3 cols)        – pid / cluster_id / abs_channel
avg_waveform_features.pqt    (n_clusters, ~23 cols)          – waveform shape features
df_clusters_extended.pqt     (n_clusters, 2 cols)            – burstiness / memory
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
df_clusters = pd.read_parquet(CELLS_AGG_PATH.joinpath('df_clusters.pqt'))
df_wf_features = pd.read_parquet(CELLS_AGG_PATH.joinpath('avg_waveform_features.pqt'))
df_clusters_extended = pd.read_parquet(CELLS_AGG_PATH.joinpath('df_clusters_extended.pqt'))
acgs_log_bins = np.load(CELLS_AGG_PATH.joinpath('acgs_log_bins.npy'))
acgs_log_times = np.load(CELLS_AGG_PATH.joinpath('acgs_log_times.npy'))
avg_waveform_peak_channel = np.load(CELLS_AGG_PATH.joinpath('avg_waveform_peak_channel.npy'))

print(f'n_clusters              {df_clusters.shape[0]:,}')
print(f'avg_waveform_peak_ch    {avg_waveform_peak_channel.shape}')
print(f'acgs_log_bins           {acgs_log_bins.shape}')
print(f'df_clusters columns:\n  {list(df_clusters.columns)}')
print(f'df_wf_features columns:\n  {list(df_wf_features.columns)}')

# %% Visualise waveforms and ACGs for a handful of good units
rng = np.random.default_rng(0)
good_mask = df_clusters['bitwise_fail'] == 0
good_idx = np.where(good_mask)[0]
sample_idx = rng.choice(good_idx, size=6, replace=False)

fig, axs = plt.subplots(2, 6, figsize=(18, 6))
ns = avg_waveform_peak_channel.shape[1]
t_wf = np.arange(ns) / 30_000 * 1e3  # ms, assuming 30 kHz

for col, idx in enumerate(sample_idx):
    pid = df_clusters['pid'].iloc[idx] if 'pid' in df_clusters.columns else ''
    cid = df_clusters.index[idx]

    ax_wf = axs[0, col]
    ax_wf.plot(t_wf, avg_waveform_peak_channel[idx] * 1e6, color='steelblue', lw=1)
    ax_wf.set(title=f'cid {cid}', xlabel='ms' if col == 0 else '', ylabel='µV' if col == 0 else '')
    ax_wf.axhline(0, color='k', lw=0.5, ls='--')

    ax_acg = axs[1, col]
    acg_norm = acgs_log_bins[idx] / df_clusters['spike_count'].iloc[idx]
    ax_acg.semilogx(acgs_log_times * 1e3, acg_norm, color='coral', lw=1)
    ax_acg.set(xlabel='lag (ms)' if col == 0 else '', ylabel='sp/s' if col == 0 else '')

axs[0, 0].set_ylabel('waveform (µV)')
axs[1, 0].set_ylabel('ACG (sp/s)')
fig.suptitle('Example units — peak-channel waveform (top) and log-ACG (bottom)')
fig.tight_layout()
fig.savefig(FIGURES_PATH.joinpath('2026-05-29_units_waveforms_acgs.png'), dpi=150)

# %% Feature distributions across good units
df_good = df_wf_features.loc[good_mask]
df_ext_good = df_clusters_extended.loc[good_mask]

features_to_plot = ['peak_to_trough_ratio', 'half_width', 'repolarisation_slope',
                    'recovery_slope', 'tip_ratio']
features_to_plot = [f for f in features_to_plot if f in df_good.columns]

fig, axs = plt.subplots(1, len(features_to_plot) + 2, figsize=(4 * (len(features_to_plot) + 2), 4))
for ax, feat in zip(axs, features_to_plot):
    vals = df_good[feat].dropna()
    ax.hist(vals, bins=100, color='steelblue', edgecolor='none')
    ax.set(xlabel=feat, ylabel='count')

for ax, feat in zip(axs[len(features_to_plot):], ['burstiness', 'memory']):
    vals = df_ext_good[feat].dropna()
    ax.hist(vals, bins=100, color='coral', edgecolor='none')
    ax.set(xlabel=feat, ylabel='count')

fig.suptitle('Waveform feature + burstiness/memory distributions (good units)')
fig.tight_layout()
fig.savefig(FIGURES_PATH.joinpath('2026-05-29_units_feature_distributions.png'), dpi=150)

# %%
eqc = viewephys(avg_waveform_peak_channel, fs=30_000)

# %%
SKILL_SOURCES=(
  "$HOME/Documents/ibldevtools/00_on_call/skills"
  "$HOME/Documents/ibl-ai-agent/skills"
  "$HOME/PycharmProjects/EphysAtlas/paper-ephys-atlas/.claude/skills"
)