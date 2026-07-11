"""
This module serves as a collection of feature extraction methods at the cell level.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandera.pandas as pa
import scipy.signal
import tqdm
import matplotlib.pyplot as plt

import phylib.stats

import ibldsp.voltage
import ibldsp.utils
import spikeglx

import iblatlas.atlas
from iblutil.numerical import bincount2D
from ibldsp.utils import WindowGenerator
from iblutil.numerical import ismember
import brainbox.ephys_plots
import iblatlas.regions


BINSIZE = 0.001
LAG = 0.5

# Standard bitwise_fail threshold for the sliding-RP bit (brainbox.metrics.single_units
# .compute_labels, METRICS_PARAMS['RPmax_confidence']), percent (0-100) scale.
BITWISE_FAIL_RP_CONFIDENCE_THRESHOLD = 90.0
# bitwise_fail bit layout (brainbox.metrics.single_units.compute_labels): bit 0 (value 1)
# = sliding-RP confidence, bit 1 (value 2) = noise_cutoff, bit 2 (value 4) = amp_median.
_BITWISE_FAIL_NOISE_AMP_MASK = 0b110


def select_good_units_relaxed_rp(df_clusters, rp_confidence_threshold=70.0):
    """Good-unit mask with a relaxed sliding-refractory-period inclusion criterion.

    The standard good-unit definition (``bitwise_fail == 0``, computed upstream by
    ``brainbox.metrics.single_units.compute_labels``) requires the legacy sliding-RP
    metric to reach :data:`BITWISE_FAIL_RP_CONFIDENCE_THRESHOLD` (90%) confidence
    (bit 0 of ``bitwise_fail``). This keeps the noise-cutoff and amplitude vetoes
    (bits 1-2) unchanged, but swaps that RP bit for a relaxed threshold on the v2
    sliding-RP metric (``slidingRP2_max_confidence``, computed by
    ``compute_sliding_rp_v2`` in the SDSC cells pipeline), letting through units
    whose refractory-period confidence is lower than 90% but still above
    `rp_confidence_threshold`.

    Parameters
    ----------
    df_clusters : pandas.DataFrame
        Cluster-level metadata; must include ``bitwise_fail`` and
        ``slidingRP2_max_confidence`` columns.
    rp_confidence_threshold : float, optional
        Minimum ``slidingRP2_max_confidence`` (0-100 scale) to pass. Defaults to 70.0,
        vs the standard 90.0 baked into ``bitwise_fail``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, one entry per row of `df_clusters`. Clusters with a NaN
        `slidingRP2_max_confidence` (too few spikes to compute) are treated as fail.
    """
    required_columns = {"bitwise_fail", "slidingRP2_max_confidence"}
    missing_columns = required_columns - set(df_clusters.columns)
    assert not missing_columns, f"df_clusters is missing columns: {missing_columns}"

    noise_amp_pass = (
        df_clusters["bitwise_fail"].to_numpy() & _BITWISE_FAIL_NOISE_AMP_MASK
    ) == 0
    rp_pass = (
        df_clusters["slidingRP2_max_confidence"].to_numpy() >= rp_confidence_threshold
    )
    return noise_amp_pass & rp_pass


def get_neighbours_members(df_clusters, radius_um):
    """
    Get neighbouring clusters but exclude self
    :param df_clusters:
    :param radius_um:
    :return:
    """
    xy = df_clusters["lateral_um"] + 1j * df_clusters["axial_um"]
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
    tbounds = np.array((spikes["times"][0], spikes["times"][-1]))
    nbins = int(np.diff(tbounds) / binsize)  # number of bins
    # precompute firing rates, and a table of neighbouring clusters

    i_good = np.where(df_clusters["bitwise_fail"] == 0)[0]
    if drop_bad_units:
        df_clusters = df_clusters[i_good]
        st = spikes["times"][np.isin(spikes["clusters"], df_clusters.index)]
        sc = spikes["clusters"][np.isin(spikes["clusters"], df_clusters.index)]
        _, sc = ismember(sc, df_clusters.index)
    else:
        st, sc = spikes["times"], spikes["clusters"]
    nc_all = df_clusters.shape[0]
    nc_good = i_good.size

    firing_rates = np.bincount(sc, minlength=nc_all) / np.diff(tbounds)

    sos = scipy.signal.butter(3, 20, "lp", fs=1 / binsize, output="sos")

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
                st[slice(*ispikes)] - tbounds[0],
                sc[slice(*ispikes)],
                binsize,
                1,
                ylim=[0, nc_all - 1],
                xlim=[first * binsize, (first + wg.nswin) * binsize],
            )
            binned_spikes = binned_spikes.astype(np.float32)
            for ic_out, ic in enumerate(i_good):
                popestimate = binned_spikes[neighbours[ic], :]
                # popestimate = popestimate - - np.mean(binned_spikes[neighbours[ic], :], axis=0)
                popestimate = np.sum(popestimate, axis=0) / np.sum(
                    firing_rates[neighbours[ic]]
                )
                cc = scipy.signal.correlate(binned_spikes[ic], popestimate, mode="same")
                stpc[ic_out] += cc[slice(*icenter)]
        stpc = scipy.signal.sosfiltfilt(sos, stpc)
        stpc = stpc - np.mean(stpc, axis=1)[:, np.newaxis]
        stpc = stpc / firing_rates[i_good, np.newaxis]
        if file_stpc is not None:
            file_stpc.parent.mkdir(parents=True, exist_ok=True)
            np.save(file_stpc, stpc)

    # % Get coupling strength and coupling delay
    tscale = np.arange(stpc.shape[1]) * binsize - lag
    taper = ibldsp.utils.fcn_cosine([-lag, -lag / 2])(tscale) - ibldsp.utils.fcn_cosine(
        [lag / 2, lag]
    )(tscale)
    i0 = np.searchsorted(tscale, 0)
    coupling_strength = stpc[:, i0]
    coupling_delay = np.sum(tscale * np.abs(stpc), axis=1) / np.sum(
        np.abs(stpc), axis=1
    )
    df_clusters["coupling_delay"] = np.nan
    df_clusters["coupling_strength"] = np.nan
    df_clusters.loc[i_good, "coupling_strength"] = coupling_strength
    df_clusters.loc[i_good, "coupling_delay"] = coupling_delay

    return (
        df_clusters,
        stpc,
        tscale,
        coupling_strength,
        taper,
        coupling_delay,
        firing_rates,
    )


def display_stpc(
    df_clusters,
    stpc,
    tscale,
    coupling_strength,
    coupling_delay,
    firing_rates,
    br=None,
    label=None,
    save_file=None,
):
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
    i_good = np.where(df_clusters["bitwise_fail"] == 0)[0]
    idepth_sorted = np.argsort(df_clusters.loc[i_good]["depths"])
    # firing_rates = np.bincount(sc, minlength=nc_all) / np.diff(tbounds)

    ch = df_clusters.loc[i_good[idepth_sorted]].reset_index()
    fig, ax = plt.subplots(
        1,
        5,
        gridspec_kw={"width_ratios": [4, 1, 1, 1, 0.4]},
        sharey=True,
        figsize=(13, 8),
    )

    cdepths = ch["depths"].values
    cdepths = np.arange(len(cdepths))

    ax[0].matshow(
        stpc[idepth_sorted],
        # stpc[idepth_sorted] / np.mean(np.abs(stpc[idepth_sorted]), axis=1)[:, np.newaxis],
        aspect="auto",
        cmap="magma",
        origin="lower",
        extent=(-LAG, LAG, 0, len(cdepths)),
    )
    ax[1].plot(coupling_strength[idepth_sorted], cdepths, color="k", linewidth=1)
    ax[2].plot(coupling_delay[idepth_sorted], cdepths, color="k", linewidth=1)
    ax[3].plot(firing_rates[i_good[idepth_sorted]], cdepths, color="k", linewidth=1)
    ax[1].set(xlabel="coupling strength")

    ax[2].set(xlabel="coupling delay")
    ax[3].set(xlabel="firing rate (Hz)")
    brainbox.ephys_plots.plot_brain_regions(
        ch["atlas_id"].values,
        channel_depths=cdepths,
        brain_regions=br,
        ax=ax[-1],
    )
    ax[0].set(xlim=(-0.25, 0.25), xlabel="Spike-triggered population average (s)")
    if label is not None:
        fig.suptitle(label)
    if save_file is not None:
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file)
        plt.close(fig)


def spike_triggered_lfp(
    file_rsamp_lfp,
    spikes,
    df_clusters,
    event_window=(-0.5, 0.5),
    fs_ap=30_000,
    fs=2500 // 10,
    file_stlfp=None,
):
    if not file_rsamp_lfp.exists():
        raise FileNotFoundError()
    # read npy file
    sr = spikeglx.Reader(file_rsamp_lfp, fs=fs)
    w = sr[:, :]  # this is intentionally casting the memmap in memory !
    w = w - np.median(w, axis=1)[:, np.newaxis]  # common average referencing
    i_good = np.where(df_clusters["bitwise_fail"] == 0)[0]
    event_window_samples = (np.array(event_window) * fs).astype(int)
    if file_stlfp.exists():
        lfp_spike_triggered = np.load(file_stlfp)
    else:
        lfp_spike_triggered = np.zeros(
            (i_good.size, np.sum(np.abs(event_window_samples)))
        )
        for i, iclu in tqdm.tqdm(enumerate(i_good)):
            # the spike samples on the AP band
            st = spikes["samples"][spikes["clusters"] == iclu]
            # the spike samples on the resampled LF band
            st = st / (fs_ap / fs)
            event_samples = st[
                slice(
                    *np.searchsorted(
                        st, [event_window[0] * fs, sr.ns - event_window[1] * fs]
                    )
                )
            ]
            residual_dt = event_samples - event_samples.astype(
                int
            )  # positive dt means the t0 will have to be delayed
            # compute a vector of indices corresponding to the perievent window at the given sampling rate
            sample_window = np.round(
                np.arange(event_window_samples[0], event_window_samples[1])
            ).astype(int)
            # we inflate this vector to a 2d array where each column corresponds to an event
            idx_psth = np.tile(sample_window[:, np.newaxis], (1, event_samples.size))
            # we add the index of each event too their respective column
            idx_psth += event_samples.astype(int)
            ich = df_clusters.loc[iclu, "channels"]
            p = w[idx_psth, ich].astype(
                np.float32
            )  # psth is a 2d array (ntimes, nevents)b
            # p = p - np.mean(p, axis=1)[:, np.newaxis]
            p = ibldsp.fourier.fshift(p, residual_dt, axis=0)
            lfp_spike_triggered[i, :] = np.median(p, axis=-1)
        np.save(file_stlfp, lfp_spike_triggered)


class ModelClusters(pa.DataFrameModel):
    """Schema for cluster-level features (good_clusters.pqt / all_clusters.pqt). pid is a plain column."""

    pid: str = pa.Field(description="Probe insertion UUID")
    uuids: str = pa.Field(description="Cluster UUID")
    cluster_id: int = pa.Field(coerce=True)
    channels: int = pa.Field(coerce=True)
    depths: float = pa.Field(coerce=True)
    firing_rate: float = pa.Field(coerce=True, nullable=True)
    amp_median: float = pa.Field(coerce=True, nullable=True)
    amp_max: float = pa.Field(coerce=True, nullable=True)
    amp_min: float = pa.Field(coerce=True, nullable=True)
    amp_std_dB: float = pa.Field(coerce=True, nullable=True)
    contamination: float = pa.Field(coerce=True, nullable=True)
    contamination_alt: float = pa.Field(coerce=True, nullable=True)
    drift: float = pa.Field(coerce=True, nullable=True)
    missed_spikes_est: float = pa.Field(coerce=True, nullable=True)
    noise_cutoff: float = pa.Field(coerce=True, nullable=True)
    presence_ratio: float = pa.Field(coerce=True, nullable=True)
    presence_ratio_std: float = pa.Field(coerce=True, nullable=True)
    slidingRP_viol: float = pa.Field(coerce=True, nullable=True)
    slidingRP_viol_forced: Optional[float] = pa.Field(coerce=True, nullable=True)
    spike_count: float = pa.Field(coerce=True, nullable=True)
    label: float = pa.Field(coerce=True, nullable=True)
    bitwise_fail: int = pa.Field(coerce=True)
    ks2_label: Optional[str] = pa.Field(nullable=True)
    x: float = pa.Field(coerce=True, nullable=True, description="MNI x coordinate (m)")
    y: float = pa.Field(coerce=True, nullable=True, description="MNI y coordinate (m)")
    z: float = pa.Field(coerce=True, nullable=True, description="MNI z coordinate (m)")
    acronym: Optional[str] = pa.Field(nullable=True)
    atlas_id: int = pa.Field(coerce=True, nullable=True)
    axial_um: float = pa.Field(coerce=True, nullable=True)
    lateral_um: float = pa.Field(coerce=True, nullable=True)
    coupling_delay: Optional[float] = pa.Field(coerce=True, nullable=True)
    coupling_strength: Optional[float] = pa.Field(coerce=True, nullable=True)


def compute_burstiness_and_memory(spike_train):
    """
    Computes the burstiness (B) and memory (M) metrics from a neuronal spike train.

    Metrics Summary:
    - Burstiness (B): Reflects the variability of a unit's inter-spike intervals (ISIs)[cite: 195].
      It solely describes the distribution of ISI durations, normalized to a range between
      -1 (completely regular) and 1 (maximally bursty)[cite: 195, 996].
    - Memory (M): Reflects the temporal ordering of ISIs[cite: 195]. It is defined as the
      Pearson correlation coefficient between subsequent ISIs[cite: 195, 1002].

    Args:
        spike_train (array-like): A sequence of timestamps representing action potentials (spikes).

    Returns:
        tuple: (burstiness, memory). Returns (np.nan, np.nan) if there are fewer
               than 6 spikes, as reliable estimation requires sufficient data[cite: 1005].
    """
    # Convert input to a numpy array
    spikes = np.asarray(spike_train)

    # Metrics are only computed in epochs with at least six spikes available[cite: 1005].
    if len(spikes) < 6:
        return np.nan, np.nan

    # Calculate Inter-Spike Intervals (ISIs)
    isis = np.diff(spikes)

    # ----------------------------------------------------
    # 1. Compute Burstiness (B)
    # ----------------------------------------------------
    mean_isi = np.mean(isis)
    std_isi = np.std(isis, ddof=1)  # Sample standard deviation

    # Prevent division by zero
    if (std_isi + mean_isi) == 0:
        burstiness = np.nan
    else:
        burstiness = (std_isi - mean_isi) / (std_isi + mean_isi)

    # ----------------------------------------------------
    # 2. Compute Memory (M)
    # ----------------------------------------------------
    isis_current = isis[:-1]
    isis_next = isis[1:]

    # Prevent Pearson correlation errors if the ISIs are entirely constant
    if np.std(isis_current, ddof=1) == 0 or np.std(isis_next, ddof=1) == 0:
        memory = np.nan
    else:
        # np.corrcoef returns a 2x2 correlation matrix; we want the off-diagonal value
        memory = np.corrcoef(isis_current, isis_next)[0, 1]

    return burstiness, memory


def compute_log_acg(
    spike_times,
    fs,
    spike_clusters=None,
    bin_size=0.2e-3,
    win_size=2.0,
    n_log_bins=512,
    log_trim=1e-3,
):
    """
    Compute a long autocorrelogram with log-spaced bins.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times in seconds.
    fs : float
        Sampling rate of the recording in Hz.
    spike_clusters : numpy.ndarray, optional
        Cluster label for each spike. If provided, the function computes one ACG per
        unique cluster and returns a 2-D array whose rows are ordered by
        ``np.unique(spike_clusters)``. If None (default), ``spike_times`` is treated as a
        single cluster and a 1-D array is returned.
    bin_size : float, optional
        Base bin resolution in seconds. Default 0.2e-3 s.
    win_size : float, optional
        One-sided window length in seconds. Default 2.0 s.
    n_log_bins : int, optional
        Number of log-spaced output bins between ``log_trim`` and ``win_size``. Default 512.
        Bins narrower than ``bin_size`` (near ``log_trim``) will be zero due to the
        refractory period, which is physically correct.
    log_trim : float, optional
        Start of the output lag axis in seconds; bins below this lag are excluded.
        Default 1e-3 s (refractory period).

    Returns
    -------
    acg_log : numpy.ndarray
        Log-binned ACG in **spike pairs · s⁻¹** (raw coincident pair counts divided
        by log-bin width; not normalised by recording duration or firing rate).
        The asymptotic value at long lags (τ ≫ any temporal correlation) is
        λ² × T = λ × n_spikes, where λ is the firing rate and T the recording duration.
        To obtain sp/s units with an asymptote equal to the firing rate, divide by
        ``n_spikes``; to obtain a dimensionless ACG, divide by ``n_spikes × λ``.
        Shape ``(n_log_bins,)`` when ``spike_clusters`` is None, or
        ``(n_unique_clusters, n_log_bins)`` otherwise.
    t_log : numpy.ndarray
        Geometric centre of each log bin in seconds, shape ``(n_log_bins,)``.
    """
    n_bins = int(win_size / bin_size) + 1
    t_bins = np.arange(n_bins) * bin_size

    # log bin structure in time-space — computed once, n_log_bins guaranteed
    t_edges = np.geomspace(log_trim, win_size, n_log_bins + 1)
    t_log = np.sqrt(t_edges[:-1] * t_edges[1:])  # geometric bin centres
    bin_widths = np.diff(t_edges)
    log_bin_idx = np.searchsorted(t_edges, t_bins, side="right") - 1
    valid = (log_bin_idx >= 0) & (log_bin_idx < n_log_bins)
    idx_v = log_bin_idx[valid]

    def _single_acg(st):
        if st.size < 2:
            return np.zeros(n_log_bins)
        autocorr = phylib.stats.correlograms(
            st,
            np.zeros(st.size, dtype=int),
            np.array([0], dtype=int),
            sample_rate=fs,
            bin_size=bin_size,
            window_size=2 * win_size,
            symmetrize=False,
        ).squeeze()
        return (
            np.bincount(
                idx_v, weights=autocorr[valid].astype(float), minlength=n_log_bins
            )
            / bin_widths
        )

    if spike_clusters is None:
        return _single_acg(spike_times), t_log

    cluster_ids = np.unique(spike_clusters)
    acg_log = np.array(
        [_single_acg(spike_times[spike_clusters == cid]) for cid in cluster_ids]
    )
    return acg_log, t_log


# ── 3D ACG (firing-rate decile x log-time-lag) ───────────────────────────────
# Matches Han Yu's NEMO/ICLR pipeline (compute_3dACG_IBL.py): linear ACG at
# cbin=1 ms / cwin=2000 ms via spikeinterface (same algorithm as
# npyx.c4.fast_acg3d), then log-time resampled -> (n_clusters, 10, 201). The
# log-resampling step is vendored in ephysatlas.iblnpyx (see that module's
# docstring for why) rather than depended on directly; expect to swap it back
# to a plain npyx dependency once that's resolved upstream. Recompute on a new
# dataset with these same constants to reproduce the released format exactly.
ACG3D_WINDOW_MS = 2000.0
ACG3D_BIN_MS = 1.0
ACG3D_NUM_FIRING_RATE_QUANTILES = 10
ACG3D_SMOOTHING_MS = 250.0
ACG3D_N_LOG_BINS = 100  # -> 2 * 100 + 1 = 201 bins after mirroring


def compute_3d_acgs(spike_times, spike_clusters, cluster_ids, fs):
    """
    Firing-rate-decile x log-time-lag 3D autocorrelogram, one per cluster.

    Computes the linear-time 3D ACG (spikeinterface's ``compute_acgs_3d``, the
    same algorithm as ``npyx.c4.fast_acg3d``), then resamples each cluster onto
    Han Yu's NEMO/ICLR log-time axis via
    :func:`ephysatlas.iblnpyx.convert_acg_log`. Uses this module's ``ACG3D_*``
    constants; call with unmodified constants to reproduce a reference dataset
    exactly.

    Requires the optional ``spikeinterface`` dependency (``pip install
    ibleatools[full]``); imported lazily here so the rest of this module stays
    usable without it.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times for the whole recording, in seconds.
    spike_clusters : numpy.ndarray
        Cluster id of each spike, same length as `spike_times`.
    cluster_ids : numpy.ndarray
        Clusters to compute the ACG for; defines the output row order.
    fs : float
        Sampling frequency of `spike_times`, in Hz.

    Returns
    -------
    acgs_3d : numpy.ndarray
        (len(cluster_ids), ACG3D_NUM_FIRING_RATE_QUANTILES, 201) float32 array.
    t_log : numpy.ndarray
        (201,) log-time bin centres, in ms (negative, zero, then positive lags).
    """
    from spikeinterface.core import NumpySorting
    from spikeinterface.postprocessing import compute_acgs_3d

    from ephysatlas.iblnpyx import convert_acg_log

    sorting = NumpySorting.from_samples_and_labels(
        samples_list=np.round(spike_times * fs).astype(np.int64),
        labels_list=spike_clusters,
        sampling_frequency=fs,
        unit_ids=cluster_ids,
    )
    acgs_3d_lin, _, _ = compute_acgs_3d(
        sorting,
        window_ms=ACG3D_WINDOW_MS,
        bin_ms=ACG3D_BIN_MS,
        num_firing_rate_quantiles=ACG3D_NUM_FIRING_RATE_QUANTILES,
        smoothing_factor=ACG3D_SMOOTHING_MS,
        n_jobs=1,
    )
    n_bins = 2 * ACG3D_N_LOG_BINS + 1
    acgs_3d = np.empty(
        (acgs_3d_lin.shape[0], ACG3D_NUM_FIRING_RATE_QUANTILES, n_bins),
        dtype=np.float32,
    )
    t_log = None
    for i, acg_lin in enumerate(acgs_3d_lin):
        acgs_3d[i], t_log = convert_acg_log(
            acg_lin,
            cbin=ACG3D_BIN_MS,
            cwin=ACG3D_WINDOW_MS,
            n_log_bins=ACG3D_N_LOG_BINS,
        )
    return acgs_3d, t_log
