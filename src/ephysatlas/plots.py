import logging

import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from iblatlas.atlas import BrainRegions
from iblutil.numerical import ismember
from ibl_style.style import figure_style
from ibl_style.utils import MM_TO_INCH
import brainbox.ephys_plots
from matplotlib import (
    cm,
)  # This is deprecated, but cannot import matplotlib.colormaps as cm

import ephysatlas.features


_logger = logging.getLogger(__name__)

figure_style()

QUANTILES = [0.01, 0.1, 0.9, 0.99]
BINS = 50


def plot_histogram(
    series, ax=None, quantiles=None, bins=None, xlabel=None, title=None, normalise=False
):
    quantiles = quantiles if quantiles is not None else QUANTILES
    quantile_values = np.quantile(series, quantiles)

    bins = bins if bins is not None else BINS

    hist_values, bin_edges = np.histogram(series, bins=bins)
    if normalise:
        hist_values = hist_values / len(series)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    color_indices = np.digitize(bin_centers, quantile_values, right=True)
    colors = cm.viridis(color_indices / color_indices.max())

    if ax is None:
        fig, ax = plt.subplots()

    ax.bar(
        bin_edges[:-1],
        hist_values,
        width=np.diff(bin_edges),
        color=colors,
        align="edge",
    )

    ax.set_xlabel(xlabel)
    if normalise:
        ax.set_ylabel("Normalised Count")
    else:
        ax.set_ylabel("Count")
    ax.set_title(title)
    ax.text(
        0.95,
        0.95,
        f"{len(series):,} samples",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out", length=6)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()


def plot_cumulative_probas(probas, depths, aids, regions=None, ax=None, legend=False):
    """
    Plot cumulative probabilities of brain regions along probe depths.

    Creates a stacked area plot showing the probability distribution of different brain regions
    at each depth along a probe trajectory. Each region is colored according to its standard
    atlas color.

    Parameters
    ----------
    probas : numpy.ndarray
        Array of shape (ndepths, nregions) containing probabilities for each region at each depth.
        Values should sum to 1 across regions for each depth.
    depths : numpy.ndarray
        Vector of length ndepths containing the depth values along the probe trajectory.
    aids : numpy.ndarray
        Vector of length nregions containing the atlas IDs for each region.
    regions : iblatlas.BrainRegions, optional
        BrainRegions object containing region information. If None, a new instance is created.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If None, the current axes will be used.
    legend : bool, default False
        Whether to display a legend with region names.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    """
    regions = regions or BrainRegions()
    _, rids = ismember(aids, regions.id)
    cprobas = probas.cumsum(axis=1)
    for i, ir in enumerate(rids):
        ax.fill_betweenx(
            depths,
            cprobas[:, i],
            label=regions.acronym[ir],
            zorder=-i,
            color=regions.rgb[ir] / 255,
        )
    ax.margins(y=0)
    ax.set_xlim(0, 1)
    ax.set_axisbelow(False)
    if legend:
        ax.legend()
    return ax


# How to add ground truth(histology data) to the plot?
def plot_results(df, predicted_probas, dict_model, regions=None):
    features = dict_model["meta"]["FEATURES"][:-4]
    aids = np.array(dict_model["meta"]["CLASSES"])
    n_folds, n_channels, n_classes = predicted_probas.shape
    if predicted_probas.ndim == 2:
        predicted_probas = predicted_probas[np.newaxis, ...]

    df_depths = df.groupby("axial_um").mean()
    entropies = np.mean(-predicted_probas * np.log2(predicted_probas), axis=2).T

    fig, ax = plt.subplots(
        1,
        1 + n_folds + 1,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [1] + [0.4] * n_folds + [0.2]},
        sharey=True,
    )

    ax[0].imshow(
        scipy.stats.zscore(df_depths.loc[:, features].to_numpy().astype(float)),
        extent=[0, len(features) + 1, df["axial_um"].min(), df["axial_um"].max()],
        vmin=-2,
        vmax=2,
        cmap="Spectral",
        aspect="auto",
    )
    ax[0].set_xticks(np.arange(len(features)) + 0.5)
    ax[0].set_xticklabels(features, rotation=90)
    for i in range(n_folds):
        plot_cumulative_probas(
            predicted_probas[i],
            df["axial_um"].values,
            aids=aids,
            regions=regions,
            ax=ax[i + 1],
        )
        ax[-1].plot(entropies[:, i], df["axial_um"], label=f"Fold {i}", alpha=0.2)
        ax[i + 1].set_title("Fold {i}")
    ax[-1].plot(entropies.mean(axis=1), df["axial_um"], label="Mean", color="k")
    # ax[-1].legend()
    ax[-1].set_title("Entropies")
    return fig, ax


def select_series(df, features=None, acronym=None, id=None, mapping="Allen"):
    if features is None:  # Take the whole set
        features = ephysatlas.features.voltage_features_set()
    if acronym is not None:
        series = df.loc[df[f"{mapping}_acronym"] == acronym, features]
    elif id is not None:
        series = df.loc[df[f"{mapping}_id"] == id, features]
    return series


def get_color_feat(x, cmap_name="viridis", min_val=None, max_val=None):
    min_val = np.min(x) if min_val is None else min_val
    max_val = np.max(x) if max_val is None else max_val
    # Normalise between 0-1
    cmap = matplotlib.colormaps[cmap_name]
    x_norm = (x - min_val) / (max_val - min_val)
    # x_norm = scipy.stats.zscore(x)
    color = cmap(x_norm)
    return color


def get_color_br(pid_ch_df, br, mapping="Allen"):
    region_info = br.get(pid_ch_df[mapping + "_id"])
    color = region_info.rgb / 255
    return color


def plot_probe_rect(xy, color, ax, width=16, height=40):
    """
    This function uses rectangles painted around the yx coordinates
    :param xy:
    :param color:
    :param ax:
    :param width:
    :param height:
    :return:
    """
    # Add rectangles
    for i in range(0, len(color)):
        a_x = xy[i, 0]
        a_y = xy[i, 1]
        a_color = color[i]
        ax.add_patch(
            matplotlib.patches.Rectangle(
                xy=(a_x - width / 2, a_y - height / 2),
                width=width,
                height=height,
                linewidth=1,
                color=a_color,
                fill=True,
            )
        )
    ax.set_xlim([min(xy[:, 0]) - width / 2, max(xy[:, 0]) + width / 2])
    ax.set_ylim([min(xy[:, 1]) - height / 2, max(xy[:, 1]) + height / 2])
    # plt.show()


def plot_probe_rect2(xy, color, ax, width=16, height=40, colorbar=False):
    """
    This function uses imshow to draw rectangles painted around the yx coordinates
    :param xy:
    :param color:
    :param ax:
    :param width:
    :param height:
    :param colorbar: if True, add a colorbar to the plot
    :return:
    """

    # HACK: stretch the probe in the X direction to improve readability of the plots with very
    # long thin probes
    xy = xy.copy()
    k = 3
    xy[:, 1] /= k

    xmin, ymin = xy.min(axis=0)
    ymin = 0
    xmax, ymax = xy.max(axis=0)
    hw, hh = width / 2, height / 2
    # extent = [xmin - hw, xmax + hw, ymin - hh, ymax + hh]
    extent = [xmin - hw, xmax + hw, ymin, ymax]
    X = round(extent[1] - extent[0]) + 1
    Y = round(extent[3] - extent[2]) + 1

    im = np.zeros((Y, X, 4), dtype=np.float32)
    im[..., 3] = 1

    for a_x, a_y, a_color in zip(xy[:, 0], xy[:, 1], color):
        i0 = max(0, round(a_y - hh))
        i1 = min(Y, round(a_y + hh) + 1)
        j0 = max(0, round(a_x - hw))
        j1 = min(X, round(a_x + hw) + 1)
        im[i0:i1, j0:j1, :3] = a_color.ravel()[:3]

    img = ax.imshow(im, extent=extent, origin="lower", aspect="auto")

    ax.set_xlim(*extent[:2])
    ax.set_xticks([])
    ax.set_ylim(ymin, ymax + 1)
    yticks = np.arange(0, ymax, 500)
    ax.set_yticks(yticks, labels=map(int, yticks * k))

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(img, cax=cax)


def figure_features_channel_space(
    pid_df: pd.DataFrame,
    features: list[str],
    xy: np.ndarray,
    pid: str,
    fig: plt.Figure = None,
    axs: np.ndarray = None,
    br: BrainRegions = None,
    mapping: str = "Cosmos",
    plot_rect: callable = plot_probe_rect2,
    cmap: str = "viridis",
    scaler: object = None,
    vmin: float = None,
    vmax: float = None,
):
    """
    Create a figure displaying electrophysiological features and brain regions along a probe.

    This function visualizes multiple features along a probe's channels in physical space,
    as well as brain region information. It creates a multi-panel figure where each panel
    shows a different feature or brain region mapping.

    Parameters
    ----------
    pid_df : pandas.DataFrame
        Dataframe containing channels and voltage information for a given probe ID (PID).
        Must contain columns for the specified features and brain region mapping.
        Example on how to prepare it:
        # Merge the voltage and channels dataframe
        df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
        # Select a PID and create the single probe dataframe
        pid = '0228bcfd-632e-49bd-acd4-c334cf9213e9'
        pid_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()

    features : list of str
        List of feature names to display, e.g. ['rms_lf', 'psd_delta', 'rms_ap'].
        These must be column keys in pid_df.

    xy : numpy.ndarray
        Matrix of spatial channel positions (in μm), with shape [N_channels x 2].
        First column is lateral_um (x) and second column is axial_um (y).

    pid : str
        Probe ID to be displayed in the figure title.

    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, a new figure is created.

    axs : array of matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new axes are created.

    br : iblatlas.atlas.BrainRegions, optional
        BrainRegions object for region color mapping. If None, a new one is created.

    mapping : str, default "Cosmos"
        Brain region mapping to use. The function will look for columns named
        "{mapping}_id" in pid_df.

    plot_rect : callable, default plot_probe_rect2
        Function to use for plotting rectangles. Should accept xy, color, and ax parameters.

    cmap : str, default "viridis"
        Colormap name to use for feature visualization.

    scaler : object, optional
        Scaling to be applied to feature values before displaying. Should have a transform method
        (like sklearn.preprocessing.StandardScaler).

    vmin : float, optional
        Minimum value for color normalization. If None, the minimum value in the data is used.

    vmax : float, optional
        Maximum value for color normalization. If None, the maximum value in the data is used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.

    axs : array of matplotlib.axes.Axes
        The axes objects for each subplot.
    """
    if br is None:
        br = BrainRegions()
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, len(features) + 4, sharey=False, figsize=(9, 5))
    if scaler is not None:
        pid_df.loc[:, features] = scaler.transform(pid_df.loc[:, features])

    brainbox.ephys_plots.plot_brain_regions(
        pid_df[mapping+"_id"].values,
        channel_depths=xy[:, 1],
        brain_regions=br,
        display=True,
        ax=axs[0],
    )
    axs[0].set_title("mapping", rotation=90)

    for i_feat, feature in enumerate(features):
        ax = axs[i_feat + 4]
        feat_arr = pid_df[[feature]].to_numpy()
        # Plot feature
        # todo OW use the min/max values from the pandera schemes instead
        color = get_color_feat(feat_arr, cmap_name=cmap, min_val=vmin, max_val=vmax)
        plot_rect(xy, color, ax=ax)
        ax.set_title(feature, rotation=90)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Plot brain region in space in unique colors
    ax = axs[2]
    d_uni = np.unique(pid_df[mapping + "_id"].to_numpy(), return_inverse=True)[1]
    d_uni = d_uni.astype(np.float32)
    color = get_color_feat(d_uni, cmap_name="Blues")
    plot_probe_rect2(xy, color, ax=axs[2])
    ax.set_title("unique region", rotation=90)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Plot brain region along probe depth with color code
    ax = axs[1]
    color = get_color_br(pid_df, br, mapping=mapping)
    plot_probe_rect2(xy, color, ax=ax)
    ax.set_title(mapping, rotation=90)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    axs[3].axis("off")
    # Add pid as suptitle
    # pid = pid_df.index[0][0]
    fig.suptitle(f"PID {pid}", y=0.08, fontweight="bold")

    # now adjust the figure
    adjust = 7.5
    # Depending on the location of axis labels leave a bit more space
    extra_left = 7.5
    extra_top = 20
    extra_bottom = 4
    width, height = fig.get_size_inches() / MM_TO_INCH
    fig.subplots_adjust(
        top=1 - (extra_top + adjust) / height,
        bottom=(adjust + extra_bottom) / height,
        left=(adjust + extra_left) / width,
        right=1 - adjust / width,
        wspace=0,
    )
    return fig, axs


def plot_features_distributions(df_features, x_list=None, title=""):
    """
    Create a grid of histograms displaying the distribution of electrophysiological features.

    This function generates a multi-panel figure with histograms for each feature in x_list.
    Each histogram is color-coded according to feature values and accompanied by a colorbar.
    The function uses quantile-based limits to handle outliers in the data visualization.

    Parameters
    ----------
    None : The function uses global variables:
        x_list : list
            List of feature names to plot
        df_features : pandas.DataFrame
            DataFrame containing the feature values with feature names as columns
        title : str
            Title for the figure

    Returns
    -------
    tuple
        A tuple containing:
        - fig : matplotlib.figure.Figure
            The figure object containing all histograms
        - axs : numpy.ndarray
            Array of matplotlib.axes.Axes objects for each subplot
    """
    if x_list is None:
        x_list = ephysatlas.features.voltage_features_set()
    fig, axs = plt.subplots(
        4, 12, figsize=(16, 9), gridspec_kw={"width_ratios": [4, 0.2] * 6}
    )
    axs = axs.flatten()
    i = 0
    for feature_name in x_list:
        ax = axs[i]
        if feature_name not in df_features.columns:
            _logger.warning(
                f"'{feature_name}' not found in the DataFrame. Skipping this feature."
            )
            continue
        feature = df_features.loc[:, feature_name].values

        clim = np.array([np.nanquantile(feature, 0.1), np.nanquantile(feature, 0.9)])
        hlim = np.array(
            [np.nanquantile(feature, 0.005), np.nanquantile(feature, 0.995)]
        )

        # Main histogram plot with box and grid
        c, x = np.histogram(feature, bins=np.linspace(hlim[0], hlim[1], 64))
        bars = ax.bar(x[:-1], c / np.sum(c), width=np.diff(x)[0])
        cmap = plt.get_cmap("PuOr")
        norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
        for bar, bin_center in zip(bars, x[:-1]):
            bar.set_color(cmap(norm(bin_center)))

        # Set box style and grid
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_title(f"Feature: {feature_name}")

        # Add colorbar in second axis
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axs[i + 1])
        cb.set_label("Feature value")
        i += 2
    for ax in axs[i:]:
        ax.axis("off")
    fig.suptitle(title)
    return fig, axs
