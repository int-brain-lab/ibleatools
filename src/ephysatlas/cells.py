"""
This module serves as a collection of feature extraction methods at the cell level.
"""

from pathlib import Path

import numpy as np
import scipy.signal
import tqdm
import matplotlib.pyplot as plt

import ibldsp.voltage
import ibldsp.utils
import spikeglx

import iblatlas.atlas
from iblutil.numerical import bincount2D
from ibldsp.utils import WindowGenerator
from iblutil.numerical import ismember
import brainbox.ephys_plots
import iblatlas.regions


BINSIZE = .001
LAG = .5

def get_neighbours_members(df_clusters, radius_um):
    """
    Get neighbouring clusters but exclude self
    :param df_clusters:
    :param radius_um:
    :return:
    """
    xy = df_clusters['lateral_um'] + 1j * df_clusters['axial_um']
    xy, xyt = np.meshgrid(xy, xy)
    neighbours = np.abs(xy - xyt) < radius_um
    neighbours[np.diag_indices(df_clusters.shape[0], 2)] = False

    return neighbours


def spike_triggered_population_coupling(spikes, df_clusters, file_stpc=None):
    """
    :param spikes:
    :param df_clusters:
    :return:
    """
    lag = LAG  # one sided correlation lag, in seconds
    binsize = BINSIZE
    wl = 10  # window length in seconds
    radius_um = 500
    drop_bad_units = False  # while we compute only for good units, the population rate is actually the sum of all sorted units, regardless of their QC
    tbounds = np.array((spikes['times'][0], spikes['times'][-1]))
    nbins = int(np.diff(tbounds) / binsize)  # number of bins
    # precompute firing rates, and a table of neighbouring clusters

    i_good = np.where(df_clusters['bitwise_fail'] == 0)[0]
    if drop_bad_units:
        df_clusters = df_clusters[i_good]
        st = spikes['times'][np.isin(spikes['clusters'], df_clusters.index)]
        sc = spikes['clusters'][np.isin(spikes['clusters'], df_clusters.index)]
        _, sc = ismember(sc, df_clusters.index)
    else:
        st, sc = spikes['times'], spikes['clusters']
    nc_all = df_clusters.shape[0]
    nc_good = i_good.size

    firing_rates = np.bincount(sc, minlength=nc_all) / np.diff(tbounds)

    sos = scipy.signal.butter(3, 20, 'lp', fs=1 / binsize, output='sos')

    sb = ((st - tbounds[0]) / binsize).astype(int)
    neighbours = get_neighbours_members(df_clusters, radius_um)
    nsw = int(wl / binsize)
    icenter = np.searchsorted((np.arange(nsw) - nsw // 2) * binsize, (-lag, lag))

    if file_stpc is not None and file_stpc.exists():
        stpc = np.load(file_stpc)
    else:
        stpc = np.zeros((nc_good, int(np.diff(icenter))))
        wg = WindowGenerator(nbins, nsw, nsw // 2)
        # indices of the window around 0 lag to keep
        for first, last in tqdm.tqdm(wg.firstlast, total=wg.nwin):
            ispikes = np.searchsorted(sb, (first, last))
            binned_spikes, time_bins, clusters_bins = bincount2D(
                st[slice(*ispikes)] - tbounds[0], sc[slice(*ispikes)], binsize, 1,
                ylim=[0, nc_all - 1], xlim=[first * binsize, (first + wg.nswin) * binsize]
            )
            binned_spikes = binned_spikes.astype(np.float32)
            for ic_out, ic in enumerate(i_good):
                popestimate = binned_spikes[neighbours[ic], :]
                # popestimate = popestimate - - np.mean(binned_spikes[neighbours[ic], :], axis=0)
                popestimate = np.sum(popestimate, axis=0) / np.sum(firing_rates[neighbours[ic]])
                cc = scipy.signal.correlate(binned_spikes[ic], popestimate, mode='same')
                stpc[ic_out] += cc[slice(*icenter)]
        stpc = scipy.signal.sosfiltfilt(sos, stpc)
        stpc = stpc - np.mean(stpc, axis=1)[:, np.newaxis]
        stpc = stpc / firing_rates[i_good, np.newaxis]
        if file_stpc is not None:
            file_stpc.parent.mkdir(parents=True, exist_ok=True)
            np.save(file_stpc, stpc)

    # % Get coupling strength and coupling delay
    tscale = np.arange(stpc.shape[1]) * binsize - lag
    taper = ibldsp.utils.fcn_cosine([-lag, -lag / 2])(tscale) - ibldsp.utils.fcn_cosine([lag / 2, lag])(tscale)
    i0 = np.searchsorted(tscale, 0)
    coupling_strength = stpc[:, i0]
    coupling_delay = np.sum(tscale * np.abs(stpc), axis=1) / np.sum(np.abs(stpc), axis=1)
    df_clusters['coupling_delay'] = np.nan
    df_clusters['coupling_strength'] = np.nan
    df_clusters.loc[i_good, 'coupling_strength'] = coupling_strength
    df_clusters.loc[i_good, 'coupling_delay'] = coupling_delay

    return df_clusters, stpc, tscale, coupling_strength, taper, coupling_delay, firing_rates


def display_stpc(df_clusters, stpc, tscale, coupling_strength, coupling_delay, firing_rates, br=None, label=None, save_file=None):
    """
    Display spike-triggered population coupling results.

    This function visualizes STPC traces for the units marked as "good"
    (i.e. `bitwise_fail == 0`), sorted by probe depth, and shows a compact
    summary alongside the matrix plot.

    Parameters
    ----------
    df_clusters : pandas.DataFrame
        Cluster-level metadata. Must include at least:
        - `bitwise_fail`: QC flag, where 0 indicates a good unit
        - `depths`: unit depth used for sorting
        - `atlas_id`: region id used for brain-region plotting
    stpc : numpy.ndarray
        STPC traces with shape `(n_units, n_time_bins)`.
    tscale : numpy.ndarray
        Time axis for `stpc`, in seconds.
    coupling_strength : numpy.ndarray
        Coupling strength for each unit, typically the STPC value at zero lag.
    coupling_delay : numpy.ndarray
        Coupling delay for each unit, computed from the absolute STPC profile.
    firing_rates : numpy.ndarray
        Firing rate for each unit, in Hz.
    save_file : Path | none, optional
        If it is a file Path, save the figure to disk instead of keeping it open.
    br : iblatlas.regions.BrainRegions, optional
        Brain regions object. If not provided, a default instance is created.

    Returns
    -------
    None
        The function creates a plot and optionally saves it.
    """
    br = iblatlas.regions.BrainRegions()
    i_good = np.where(df_clusters['bitwise_fail'] == 0)[0]
    idepth_sorted = np.argsort(df_clusters.loc[i_good]['depths'])
    # firing_rates = np.bincount(sc, minlength=nc_all) / np.diff(tbounds)

    ch = df_clusters.loc[i_good[idepth_sorted]].reset_index()
    fig, ax = plt.subplots(1, 5, gridspec_kw={"width_ratios": [4, 1, 1, 1, 0.4]}, sharey=True,
                           figsize=(13, 8))

    cdepths = ch['depths'].values
    cdepths = np.arange(len(cdepths))

    ax[0].matshow(
        stpc[idepth_sorted],
        # stpc[idepth_sorted] / np.mean(np.abs(stpc[idepth_sorted]), axis=1)[:, np.newaxis],
        aspect='auto', cmap='magma', origin='lower', extent=(-LAG, LAG, 0, len(cdepths)))
    ax[1].plot(coupling_strength[idepth_sorted], cdepths, color='k', linewidth=1)
    ax[2].plot(coupling_delay[idepth_sorted], cdepths, color='k', linewidth=1)
    ax[3].plot(firing_rates[i_good[idepth_sorted]], cdepths, color='k', linewidth=1)
    ax[1].set(xlabel='coupling strength')

    ax[2].set(xlabel='coupling delay')
    ax[3].set(xlabel='firing rate (Hz)')
    brainbox.ephys_plots.plot_brain_regions(
        ch["atlas_id"].values,
        channel_depths=cdepths,
        brain_regions=br,
        ax=ax[-1],
    )
    ax[0].set(xlim=(-0.25, 0.25), xlabel='Spike-triggered population average (s)')
    if label is not None:
        fig.suptitle(label)
    if save_file is not None:
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file)
        plt.close(fig)


def spike_triggered_lfp(file_rsamp_lfp, spikes, df_clusters, event_window=(-0.5, 0.5), fs_ap=30_000, fs=2500 // 10, file_stlfp=None):
    if not file_rsamp_lfp.exists():
        raise FileNotFoundError()
    # read npy file
    sr = spikeglx.Reader(file_rsamp_lfp, fs=fs)
    w = sr[:, :]  # this is intentionally casting the memmap in memory !
    w = w - np.median(w, axis=1)[:, np.newaxis]  # common average referencing
    i_good = np.where(df_clusters['bitwise_fail'] == 0)[0]
    event_window_samples = (np.array(event_window) * fs).astype(int)
    if file_stlfp.exists():
        lfp_spike_triggered = np.load(file_stlfp)
    else:
        lfp_spike_triggered = np.zeros((i_good.size, np.sum(np.abs(event_window_samples))))
        for i, iclu in tqdm.tqdm(enumerate(i_good)):
            # the spike samples on the AP band
            st = spikes['samples'][spikes['clusters'] == iclu]
            # the spike samples on the resampled LF band
            st = (st / (fs_ap / fs))
            event_samples = st[slice(*np.searchsorted(st, [event_window[0] * fs, sr.ns - event_window[1] * fs]))]
            residual_dt = event_samples - event_samples.astype(int)  # positive dt means the t0 will have to be delayed
            # compute a vector of indices corresponding to the perievent window at the given sampling rate
            sample_window = np.round(np.arange(event_window_samples[0], event_window_samples[1])).astype(int)
            # we inflate this vector to a 2d array where each column corresponds to an event
            idx_psth = np.tile(sample_window[:, np.newaxis], (1, event_samples.size))
            # we add the index of each event too their respective column
            idx_psth += event_samples.astype(int)
            ich = df_clusters.loc[iclu, 'channels']
            p = w[idx_psth, ich].astype(np.float32)  # psth is a 2d array (ntimes, nevents)b
            # p = p - np.mean(p, axis=1)[:, np.newaxis]
            p = ibldsp.fourier.fshift(p, residual_dt, axis=0)
            lfp_spike_triggered[i, :] = np.median(p, axis=-1)
        np.save(file_stlfp, lfp_spike_triggered)
