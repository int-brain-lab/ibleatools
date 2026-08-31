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
from iblutil.util import Bunch
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
    ``brainbox.metrics.single_units.compute_labels``) requires the sliding-RP metric
    (``max_confidence``) to reach :data:`BITWISE_FAIL_RP_CONFIDENCE_THRESHOLD` (90%)
    confidence (bit 0 of ``bitwise_fail``). This keeps the noise-cutoff and amplitude
    vetoes (bits 1-2) unchanged, but relaxes that RP bit to `rp_confidence_threshold`
    applied directly to ``max_confidence``, letting through units whose
    refractory-period confidence is lower than 90% but still above the threshold.

    Parameters
    ----------
    df_clusters : pandas.DataFrame
        Cluster-level metadata; must include ``bitwise_fail`` and ``max_confidence``
        columns.
    rp_confidence_threshold : float, optional
        Minimum ``max_confidence`` (0-100 scale) to pass. Defaults to 70.0, vs the
        standard 90.0 baked into ``bitwise_fail``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, one entry per row of `df_clusters`. Clusters with a NaN
        `max_confidence` (too few spikes to compute) are treated as fail.
    """
    required_columns = {"bitwise_fail", "max_confidence"}
    missing_columns = required_columns - set(df_clusters.columns)
    assert not missing_columns, f"df_clusters is missing columns: {missing_columns}"

    noise_amp_pass = (
        df_clusters["bitwise_fail"].to_numpy() & _BITWISE_FAIL_NOISE_AMP_MASK
    ) == 0
    rp_pass = df_clusters["max_confidence"].to_numpy() >= rp_confidence_threshold
    return noise_amp_pass & rp_pass


def _coupling_lag_axis(lags, binsize):
    """(n_lags_neg, n_lags_pos, tscale) for a `lags` window/`binsize`, see
    :func:`spike_triggered_population_coupling`."""
    lag_min, lag_max = (-abs(lags), abs(lags)) if np.isscalar(lags) else lags
    n_lags_neg = int(round(-lag_min / binsize))
    n_lags_pos = int(round(lag_max / binsize))
    tscale = np.arange(-n_lags_neg, n_lags_pos + 1) * binsize
    return n_lags_neg, n_lags_pos, tscale


def _coupling_strength_and_delay(stpc, tscale):
    """Zero-lag value and (signed) centre-of-mass delay of `stpc`, see
    :func:`spike_triggered_population_coupling`."""
    coupling_strength = stpc[:, np.searchsorted(tscale, 0)]
    with np.errstate(invalid="ignore", divide="ignore"):
        coupling_delay = np.sum(tscale * stpc, axis=1) / np.sum(stpc, axis=1)
    return coupling_strength, coupling_delay


def get_neighbours_members(lateral_um, axial_um, radius_um):
    """
    Boolean adjacency matrix of clusters located within a given radius of one another on
    the probe (2-D, lateral x axial), excluding self.

    Parameters
    ----------
    lateral_um, axial_um : array_like
        1-D coordinate arrays, one entry per cluster, in micrometres (the
        ``lateral_um`` / ``axial_um`` columns of :class:`ModelClusters`).
    radius_um : float
        Inclusion radius, in micrometres.

    Returns
    -------
    numpy.ndarray of bool, shape (n_clusters, n_clusters)
        ``neighbours[i, j]`` is True if cluster `j` lies within `radius_um` of cluster
        `i`. The diagonal is always False (a cluster is not its own neighbour).
    """
    xy = np.asarray(lateral_um) + 1j * np.asarray(axial_um)
    distance = np.abs(xy[:, np.newaxis] - xy[np.newaxis, :])
    neighbours = distance <= radius_um
    np.fill_diagonal(neighbours, False)
    return neighbours


def spike_triggered_population_coupling(
    spike_times,
    spike_clusters,
    cluster_ids,
    lateral_um,
    axial_um,
    radius_um,
    lags=0.08,
    binsize=BINSIZE,
    lowpass_hz=20.0,
):
    """
    Spike-triggered population coupling of each cluster to its local, spatially-restricted
    population, as defined in Bimbard, Harris & Carandini, "Invariant activity sequences
    across the mouse brain" (bioRxiv 2025.12.20.695676), Methods section "Population
    coupling", itself an extension of Okun et al. 2015 (Nature 521, 511-515).

    For cluster `i`, let :math:`g_i(t)` be the mean, across every *other* cluster within
    `radius_um` of it on the probe, of the mean-centred binned firing:

    .. math::

        g_i(t) = \\frac{1}{N_i - 1}\\sum_{j \\in \\mathrm{neighbours}(i)} (f_j(t) - \\mu_j)

    The coupling at lag :math:`\\tau` is then the spike-triggered average of :math:`g_i`,
    i.e. the (normalised) cross-correlation of cluster `i`'s spike train with :math:`g_i`:

    .. math::

        c_{i,\\tau} = 100 \\times \\frac{1}{\\lVert f_i \\rVert}
            \\int f_i(t - \\tau)\\, g_i(t)\\, dt

    where :math:`\\lVert f_i \\rVert` is the number of spikes fired by cluster `i`. The
    result is expressed as a percentage and, by construction, is uncorrelated with a
    neuron's own firing rate.

    Parameters
    ----------
    spike_times : array_like
        Spike times (s), one entry per spike.
    spike_clusters : array_like
        Cluster id of each spike (same length as `spike_times`); values are looked up in
        `cluster_ids`, so ids need not be contiguous or start at 0.
    cluster_ids : array_like
        Cluster ids to compute the coupling for, and from which each cluster's local
        population is built; defines the row order of the outputs and must align with
        `lateral_um`, `axial_um`.
    lateral_um, axial_um : array_like
        Cluster coordinates on the probe, one entry per entry of `cluster_ids`, in
        micrometres (the ``lateral_um`` / ``axial_um`` columns of :class:`ModelClusters`).
    radius_um : float
        Inclusion radius for a cluster's local population, in micrometres. A cluster with
        no other cluster within `radius_um` gets an all-nan coupling row.
    lags : float or (float, float), optional
        Symmetric half-window (single float) or explicit ``(lag_min, lag_max)`` window of
        lags, in seconds, over which the coupling is returned. Default 0.08, i.e.
        [-80, 80] ms, as in Bimbard et al. 2025 ("we computed coupling within a
        [-80, 80] ms window").
    binsize : float, optional
        Bin size (s) used to bin the spike trains before cross-correlating. Default 1 ms,
        as in Bimbard et al. 2025.
    lowpass_hz : float or None, optional
        Cutoff (Hz) of the zero-phase, 3rd-order Butterworth low-pass filter applied to
        each coupling curve ("low-passed filtered the coupling with a 20 Hz cutoff").
        None disables filtering. Default 20.0.

    Returns
    -------
    iblutil.util.Bunch
        stpc : numpy.ndarray (n_clusters, n_lags)
            Population coupling, in %, one row per entry of `cluster_ids`.
        tscale : numpy.ndarray (n_lags,)
            Lag axis (s) matching axis 1 of `stpc`.
        coupling_strength : numpy.ndarray (n_clusters,)
            Coupling at zero lag, ``stpc[:, tscale == 0]``.
        coupling_delay : numpy.ndarray (n_clusters,)
            Centre of mass of the (filtered) coupling curve, in seconds, per the literal
            Methods definition (no absolute value). For a cluster whose coupling is weak
            and noisy enough that the curve's signed sum is close to zero, this ratio can
            be ill-conditioned and land outside the `lags` window; such rows are more
            reliably screened out by `coupling_strength` than trusted for their delay.
        firing_rate : numpy.ndarray (n_clusters,)
            Mean firing rate (Hz) of each cluster over
            ``[spike_times.min(), spike_times.max()]``.
        n_neighbours : numpy.ndarray (n_clusters,)
            Number of other clusters within `radius_um`, i.e. :math:`N_i - 1`.

    Notes
    -----
    Memory and compute scale with ``n_clusters ** 2 * n_bins`` (the local-population sum)
    and ``n_clusters * n_bins`` (the batched FFT cross-correlation), where
    ``n_bins = duration / binsize``. For very long recordings, pre-restrict `spike_times`
    / `spike_clusters` to the epoch of interest (e.g. spontaneous-activity periods, as in
    Bimbard et al. 2025) before calling this function.
    """
    spike_times = np.asarray(spike_times)
    spike_clusters = np.asarray(spike_clusters)
    cluster_ids = np.asarray(cluster_ids)
    n_clusters = cluster_ids.size

    in_clusters, sc = ismember(spike_clusters, cluster_ids)
    st = spike_times[in_clusters]

    n_lags_neg, n_lags_pos, tscale = _coupling_lag_axis(lags, binsize)

    counts, _, _ = bincount2D(
        st,
        sc,
        xbin=binsize,
        ybin=1,
        xlim=[st.min(), st.max()],
        ylim=[0, n_clusters - 1],
    )
    n_bins = counts.shape[1]
    n_spikes = np.bincount(sc, minlength=n_clusters).astype(float)
    firing_rate = n_spikes / (st.max() - st.min())

    neighbours = get_neighbours_members(lateral_um, axial_um, radius_um)
    n_neighbours = neighbours.sum(axis=1)

    centred = counts - counts.mean(axis=1)[:, np.newaxis]
    with np.errstate(all="ignore"):
        # `all="ignore"` also silences a spurious "divide by zero"/"overflow" warning that
        # some Accelerate/OpenBLAS builds raise on large matmuls (numpy gh-21196); it does
        # not affect the (correct) result, and the real 0/0 from `n_neighbours == 0` rows
        # is handled explicitly below.
        population = (neighbours @ centred) / n_neighbours[:, np.newaxis]

    # Spike-triggered average of `population` for every cluster and every lag at once is a
    # per-row cross-correlation of `counts` with `population`, computed here in a single
    # batched FFT rather than per-cluster / per-window loops (order of operands matters:
    # irfft(rfft(population) * conj(rfft(counts)))[k] == sum_t counts[t] * population[t+k]).
    # The circular wrap-around this introduces is negligible since n_lags << n_bins.
    xcorr = np.fft.irfft(
        np.fft.rfft(population, axis=1) * np.conj(np.fft.rfft(counts, axis=1)),
        n=n_bins,
        axis=1,
    )
    lag_index = np.arange(-n_lags_neg, n_lags_pos + 1) % n_bins
    with np.errstate(invalid="ignore", divide="ignore"):
        stpc = 100 * xcorr[:, lag_index] / n_spikes[:, np.newaxis]
    stpc[(n_neighbours == 0) | (n_spikes == 0), :] = np.nan

    if lowpass_hz is not None:
        sos = scipy.signal.butter(3, lowpass_hz, "lp", fs=1 / binsize, output="sos")
        ok = np.all(np.isfinite(stpc), axis=1)
        stpc[ok] = scipy.signal.sosfiltfilt(sos, stpc[ok], axis=1)

    coupling_strength, coupling_delay = _coupling_strength_and_delay(stpc, tscale)

    return Bunch(
        stpc=stpc,
        tscale=tscale,
        coupling_strength=coupling_strength,
        coupling_delay=coupling_delay,
        firing_rate=firing_rate,
        n_neighbours=n_neighbours,
    )


def spike_triggered_population_coupling_df(
    spikes,
    df_clusters,
    radius_um=500,
    lags=0.08,
    binsize=BINSIZE,
    lowpass_hz=20.0,
    good_units_only=True,
    file_stpc=None,
):
    """
    ``df_clusters`` / ``spikes``-Bunch convenience wrapper around
    :func:`spike_triggered_population_coupling`, for interoperability with the rest of the
    ephys-atlas cell-features pipeline.

    The population that couplings are computed against always includes every cluster in
    `df_clusters` regardless of QC -- matching Bimbard et al. 2025's use of "the summed
    activity of all neurons" as the reference population -- irrespective of
    `good_units_only`, which only restricts which clusters are *reported*.

    Parameters
    ----------
    spikes : dict-like
        Must expose ``times`` (s) and ``clusters`` (cluster id, matching `df_clusters`'s
        index).
    df_clusters : pandas.DataFrame
        Cluster metadata, indexed by cluster id. Must contain ``lateral_um``,
        ``axial_um`` and ``bitwise_fail`` (QC flag, 0 == good).
    radius_um, lags, binsize, lowpass_hz : see :func:`spike_triggered_population_coupling`.
    good_units_only : bool, optional
        If True (default, matching this module's existing cell-features pipeline), only
        report rows for clusters with ``bitwise_fail == 0``. Set to False to report every
        cluster. `file_stpc`, if given, only ever holds the reported rows, so a cache
        written with one setting cannot be reused with the other.
    file_stpc : Path, optional
        If given and it exists, load `stpc` (already restricted to the reported
        clusters) from it instead of recomputing; otherwise compute and save it there.

    Returns
    -------
    iblutil.util.Bunch
        Same as :func:`spike_triggered_population_coupling`, restricted to the reported
        clusters, plus:
        df_clusters : pandas.DataFrame
            `df_clusters` with ``coupling_strength`` / ``coupling_delay`` columns
            added/overwritten (nan for clusters not reported or with no neighbours).
        i_reported : numpy.ndarray
            Positional index into `df_clusters` of the rows reported in `stpc` etc.
    """
    cluster_ids = df_clusters.index.to_numpy()
    lateral_um = df_clusters["lateral_um"].to_numpy()
    axial_um = df_clusters["axial_um"].to_numpy()

    if good_units_only:
        i_reported = np.where(df_clusters["bitwise_fail"].to_numpy() == 0)[0]
    else:
        i_reported = np.arange(df_clusters.shape[0])

    if file_stpc is not None and file_stpc.exists():
        stpc = np.load(file_stpc)
        _, _, tscale = _coupling_lag_axis(lags, binsize)
        coupling_strength, coupling_delay = _coupling_strength_and_delay(stpc, tscale)
        in_clusters, sc = ismember(np.asarray(spikes["clusters"]), cluster_ids)
        st = np.asarray(spikes["times"])[in_clusters]
        firing_rate_all = np.bincount(sc, minlength=cluster_ids.size) / (
            st.max() - st.min()
        )
        n_neighbours_all = get_neighbours_members(lateral_um, axial_um, radius_um).sum(
            axis=1
        )
        firing_rate = firing_rate_all[i_reported]
        n_neighbours = n_neighbours_all[i_reported]
    else:
        full = spike_triggered_population_coupling(
            spikes["times"],
            spikes["clusters"],
            cluster_ids,
            lateral_um,
            axial_um,
            radius_um,
            lags=lags,
            binsize=binsize,
            lowpass_hz=lowpass_hz,
        )
        tscale = full["tscale"]
        stpc = full["stpc"][i_reported]
        coupling_strength = full["coupling_strength"][i_reported]
        coupling_delay = full["coupling_delay"][i_reported]
        firing_rate = full["firing_rate"][i_reported]
        n_neighbours = full["n_neighbours"][i_reported]
        if file_stpc is not None:
            file_stpc.parent.mkdir(parents=True, exist_ok=True)
            np.save(file_stpc, stpc)

    df_clusters = df_clusters.copy()
    df_clusters["coupling_strength"] = np.nan
    df_clusters["coupling_delay"] = np.nan
    df_clusters.loc[df_clusters.index[i_reported], "coupling_strength"] = (
        coupling_strength
    )
    df_clusters.loc[df_clusters.index[i_reported], "coupling_delay"] = coupling_delay

    return Bunch(
        df_clusters=df_clusters,
        stpc=stpc,
        tscale=tscale,
        coupling_strength=coupling_strength,
        coupling_delay=coupling_delay,
        firing_rate=firing_rate,
        n_neighbours=n_neighbours,
        i_reported=i_reported,
    )


def get_neighbours_members_legacy(df_clusters, radius_um):
    """
    Original (pre-fix) implementation, kept only so results can be compared side by side
    against :func:`spike_triggered_population_coupling_df` on real data. Do not use for
    new work -- see :func:`spike_triggered_population_coupling_legacy` for what it gets
    wrong.
    """
    xy = df_clusters["lateral_um"] + 1j * df_clusters["axial_um"]
    xy, xyt = np.meshgrid(xy, xy)
    neighbours = np.abs(xy - xyt) < radius_um
    neighbours[np.diag_indices(df_clusters.shape[0], 2)] = False

    return neighbours


def spike_triggered_population_coupling_legacy(spikes, df_clusters, file_stpc=None):
    """
    Original (pre-fix) implementation of spike-triggered population coupling, kept only
    for side-by-side comparison against :func:`spike_triggered_population_coupling_df` on
    real data. Do not use for new work; relative to the Methods of Bimbard et al. 2025 it:

    - sums neighbours' raw (non-mean-centred) counts and divides by their *summed firing
      rate* rather than averaging their mean-centred counts by neighbour count;
    - double-counts most bins via overlapping (50%-hop) correlation windows instead of a
      single pass over the whole recording;
    - normalises the result by firing rate (Hz) instead of spike count, and uses
      ``abs(stpc)`` as the centre-of-mass weight instead of the signed curve;
    - restricts neighbours to `lateral_um`/`axial_um` (2-D, probe-local) coordinates and a
      hardcoded ``radius_um = 500`` and ``LAG = 0.5`` s (±500 ms) window, both no longer
      configurable.
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
    neighbours = get_neighbours_members_legacy(df_clusters, radius_um)
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
    firing_rate,
    i_reported,
    br=None,
    label=None,
    save_file=None,
    **_unused,
):
    """
    Display spike-triggered population coupling results.

    This function visualizes the STPC traces of the reported units, sorted by probe
    depth, and shows a compact summary alongside the matrix plot. Intended to be called
    as ``display_stpc(**result)`` on the :class:`iblutil.util.Bunch` returned by
    :func:`spike_triggered_population_coupling_df`.

    Parameters
    ----------
    df_clusters : pandas.DataFrame
        Cluster-level metadata (all clusters, not just reported ones). Must include at
        least `depths` (unit depth used for sorting) and `atlas_id` (region id used for
        brain-region plotting).
    stpc, coupling_strength, coupling_delay, firing_rate : numpy.ndarray
        As returned by :func:`spike_triggered_population_coupling_df`: one row/entry per
        reported cluster (see `i_reported`).
    tscale : numpy.ndarray
        Time axis for `stpc`, in seconds.
    i_reported : numpy.ndarray
        Positional index into `df_clusters` of the clusters reported in `stpc` etc., as
        returned by :func:`spike_triggered_population_coupling_df`.
    save_file : Path | None, optional
        If it is a file Path, save the figure to disk instead of keeping it open.
    br : iblatlas.regions.BrainRegions, optional
        Brain regions object. If not provided, a default instance is created.
    **_unused
        Swallows extra keys (e.g. ``n_neighbours``) when called as
        ``display_stpc(**result)``.

    Returns
    -------
    None
        The function creates a plot and optionally saves it.
    """
    br = iblatlas.regions.BrainRegions() if br is None else br
    df_reported = df_clusters.iloc[i_reported]
    idepth_sorted = np.argsort(df_reported["depths"].to_numpy())

    ch = df_reported.iloc[idepth_sorted].reset_index()
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
        aspect="auto",
        cmap="magma",
        origin="lower",
        extent=(tscale[0], tscale[-1], 0, len(cdepths)),
    )
    ax[1].plot(coupling_strength[idepth_sorted], cdepths, color="k", linewidth=1)
    ax[2].plot(coupling_delay[idepth_sorted], cdepths, color="k", linewidth=1)
    ax[3].plot(firing_rate[idepth_sorted], cdepths, color="k", linewidth=1)
    ax[1].set(xlabel="coupling strength")

    ax[2].set(xlabel="coupling delay")
    ax[3].set(xlabel="firing rate (Hz)")
    brainbox.ephys_plots.plot_brain_regions(
        ch["atlas_id"].values,
        channel_depths=cdepths,
        brain_regions=br,
        ax=ax[-1],
    )
    ax[0].set(
        xlim=(tscale[0], tscale[-1]), xlabel="Spike-triggered population average (s)"
    )
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
