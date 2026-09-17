"""
Electrophysiological plotting and visualization module.

This module provides comprehensive plotting and visualization tools for
electrophysiological data analysis, including feature distributions, probe
visualizations, brain region mappings, and statistical plots.

The module includes:
- Histogram plotting with quantile-based coloring
- Cumulative probability plots for brain regions
- Probe visualization in physical space
- Feature distribution analysis
- Brain region mapping and visualization
- Statistical plotting utilities

Functions
---------
plot_histogram
    Create histograms with quantile-based color coding
plot_cumulative_probas
    Plot cumulative probabilities of brain regions along probe depths
plot_results
    Visualize model prediction results and feature distributions
select_series
    Select data series based on features and brain region criteria
get_color_feat
    Generate colors for feature values using colormaps
get_color_br
    Generate colors for brain regions
plot_probe_rect
    Plot probe channels as rectangles with specified colors
plot_probe_rect2
    Plot probe channels using imshow for better visualization
figure_features_channel_space
    Create comprehensive probe visualization with features and brain regions
plot_features_distributions
    Create grid of histograms for feature distributions
plot_region_bar
    Draw brain regions as a coloured bar along the probe depth
plot_features_heatmap
    Draw the features of one probe as an image, channels sorted by depth
figure_features_heatmap
    Z-scored features of one probe beside its histology, for one or several tables
figure_predictions_along_probe
    Histology, predicted region and region probabilities along one probe
figure_voltage_snippets
    Voltage images of one or several snippets, side by side
figure_encoder_coronal
    Features predicted by the spatial encoder on a coronal plane

Constants
---------
QUANTILES : list
    Default quantile values for histogram coloring
BINS : int
    Default number of bins for histograms

Examples
--------
>>> from ephysatlas.plots import plot_histogram, plot_probe_rect2
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>>
>>> # Create sample data
>>> data = np.random.randn(1000)
>>>
>>> # Plot histogram
>>> plot_histogram(data, xlabel="Value", title="Sample Distribution")
>>>
>>> # Plot probe visualization
>>> xy = np.column_stack([np.arange(64), np.zeros(64)])
>>> colors = np.random.rand(64, 3)
>>> fig, ax = plt.subplots()
>>> plot_probe_rect2(xy, colors, ax)

Notes
-----
This module integrates with the IBL (International Brain Laboratory) ecosystem
and uses their styling conventions and brain region atlases. It provides
both simple plotting functions and complex multi-panel visualizations for
electrophysiological data analysis.

See Also
--------
ephysatlas.features : Feature extraction and processing
iblatlas.atlas : Brain region atlas functionality
brainbox.ephys_plots : Additional electrophysiology plotting tools
"""

import logging

import numpy as np
import pandas as pd
import scipy.stats
import sklearn

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from iblatlas.atlas import AllenAtlas, BrainRegions
from iblutil.numerical import ismember
from ibl_style.style import figure_style
from ibl_style.utils import MM_TO_INCH
import brainbox.ephys_plots
import ibldsp.plots
from matplotlib import (
    cm,
)  # This is deprecated, but cannot import matplotlib.colormaps as cm

import ephysatlas.features
from ephysatlas.anatomy import NEW_VOID, ClassifierRegions


_logger = logging.getLogger(__name__)

figure_style()

QUANTILES = [0.01, 0.1, 0.9, 0.99]
BINS = 50


def plot_histogram(
    series, ax=None, quantiles=None, bins=None, xlabel=None, title=None, normalise=False
):
    """Create histograms with quantile-based color coding.

    This function creates histograms with color coding based on quantile values,
    providing visual distinction between different ranges of the data distribution.

    Args:
        series (pd.Series or np.ndarray): Data series to plot as histogram.
        ax (matplotlib.axes.Axes, optional): Axes on which to plot. If None,
            a new figure and axes will be created.
        quantiles (list, optional): Quantile values for color coding.
            Defaults to QUANTILES constant [0.01, 0.1, 0.9, 0.99].
        bins (int, optional): Number of histogram bins. Defaults to BINS constant (50).
        xlabel (str, optional): Label for the x-axis.
        title (str, optional): Title for the plot.
        normalise (bool, optional): Whether to normalize the histogram counts.
            Defaults to False.

    Returns:
        None: The function modifies the provided axes or creates a new plot.

    Note:
        The function uses the viridis colormap for quantile-based coloring.
        Sample count is displayed in the top-right corner of the plot.
    """
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
    """Plot cumulative probabilities of brain regions along probe depths.

    Creates a stacked area plot showing the probability distribution of different brain regions
    at each depth along a probe trajectory. Each region is colored according to its standard
    atlas color.

    Args:
        probas (np.ndarray): Array of shape (ndepths, nregions) containing probabilities
            for each region at each depth. Values should sum to 1 across regions for each depth.
        depths (np.ndarray): Vector of length ndepths containing the depth values along
            the probe trajectory.
        aids (np.ndarray): Vector of length nregions containing the atlas IDs for each region.
        regions (iblatlas.BrainRegions, optional): BrainRegions object containing region
            information. If None, a new instance is created.
        ax (matplotlib.axes.Axes, optional): Axes on which to plot. If None, the current
            axes will be used.
        legend (bool, optional): Whether to display a legend with region names.
            Defaults to False.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.

    Note:
        The function creates a stacked area plot where each brain region is represented
        by a different color from the atlas. The y-axis represents depth along the probe.
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
    """Visualize model prediction results and feature distributions.

    This function creates a comprehensive visualization of model prediction results,
    including feature heatmaps, cumulative probability plots for different folds,
    and entropy analysis across channels.

    Args:
        df (pd.DataFrame): DataFrame containing channel data and features.
        predicted_probas (np.ndarray): Array of predicted probabilities with shape
            (n_folds, n_channels, n_classes) or (n_channels, n_classes).
        dict_model (dict): Model dictionary containing metadata including features
            and class information.
        regions (iblatlas.BrainRegions, optional): BrainRegions object for region
            visualization. If None, a new instance is created.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object containing all plots.
            - axs (np.ndarray): Array of matplotlib axes objects.

    Note:
        The function creates a multi-panel figure with feature heatmaps, probability
        plots for each fold, and entropy analysis. It automatically handles both
        single-fold and multi-fold prediction arrays.
    """
    features = dict_model["FEATURES"][:-4]
    aids = np.array(dict_model["CLASSES"])
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
    """Select data series based on features and brain region criteria.

    This function filters a DataFrame to select specific features based on
    brain region criteria (acronym or ID) and returns the selected data series.

    Args:
        df (pd.DataFrame): DataFrame containing the data to filter.
        features (list, optional): List of feature names to select. If None,
            uses all available voltage features. Defaults to None.
        acronym (str, optional): Brain region acronym to filter by.
            Mutually exclusive with id parameter.
        id (int, optional): Brain region ID to filter by.
            Mutually exclusive with acronym parameter.
        mapping (str, optional): Brain region mapping system to use.
            Defaults to "Allen".

    Returns:
        pd.DataFrame: Filtered DataFrame containing only the selected features
            for the specified brain region.

    Note:
        Either acronym or id should be provided, but not both. If neither is
        provided, the function will return None.
    """
    if features is None:  # Take the whole set
        features = ephysatlas.features.voltage_features_set()
    if acronym is not None:
        series = df.loc[df[f"{mapping}_acronym"] == acronym, features]
    elif id is not None:
        series = df.loc[df[f"{mapping}_id"] == id, features]
    return series


def get_color_feat(x, cmap_name="viridis", min_val=None, max_val=None):
    """Generate colors for feature values using colormaps.

    This function normalizes feature values to the range [0, 1] and maps them
    to colors using a specified colormap. Useful for creating color-coded
    visualizations of feature values.

    Args:
        x (np.ndarray): Array of feature values to colorize.
        cmap_name (str, optional): Name of the matplotlib colormap to use.
            Defaults to "viridis".
        min_val (float, optional): Minimum value for normalization. If None,
            uses the minimum value in x. Defaults to None.
        max_val (float, optional): Maximum value for normalization. If None,
            uses the maximum value in x. Defaults to None.

    Returns:
        np.ndarray: Array of RGBA colors with the same shape as x.

    Note:
        The function performs min-max normalization and maps the normalized
        values to colors using the specified colormap. Values are clipped to
        the [0, 1] range during normalization.
    """
    min_val = np.nanmin(x) if min_val is None else min_val
    max_val = np.nanmax(x) if max_val is None else max_val
    # Normalise between 0-1
    cmap = matplotlib.colormaps[cmap_name]
    x_norm = (x - min_val) / (max_val - min_val)
    # x_norm = scipy.stats.zscore(x)
    color = cmap(x_norm)
    return color


def get_color_br(pid_ch_df, br, mapping="Allen"):
    """Generate colors for brain regions.

    This function extracts brain region IDs from a DataFrame and maps them
    to their corresponding RGB colors from the brain regions atlas.

    Args:
        pid_ch_df (pd.DataFrame): DataFrame containing brain region mapping
            columns (e.g., "Allen_id").
        br (iblatlas.atlas.BrainRegions): BrainRegions object containing
            region information and colors.
        mapping (str, optional): Brain region mapping system to use.
            Defaults to "Allen".

    Returns:
        np.ndarray: Array of RGB colors normalized to [0, 1] range.

    Note:
        The function looks for a column named "{mapping}_id" in the DataFrame
        and uses the brain regions atlas to map these IDs to RGB colors.
    """
    region_info = br.get(pid_ch_df[mapping + "_id"])
    color = region_info.rgb / 255
    return color


def plot_probe_rect(xy, color, ax, width=16, height=40):
    """Plot probe channels as rectangles with specified colors.

    This function uses matplotlib rectangles to visualize probe channels
    at their spatial coordinates with specified colors and dimensions.

    Args:
        xy (np.ndarray): Array of shape (n_channels, 2) containing x,y coordinates
            for each channel in micrometers.
        color (np.ndarray): Array of shape (n_channels, 3) or (n_channels, 4)
            containing RGB or RGBA colors for each channel.
        ax (matplotlib.axes.Axes): Axes on which to plot the rectangles.
        width (float, optional): Width of each rectangle in micrometers.
            Defaults to 16.
        height (float, optional): Height of each rectangle in micrometers.
            Defaults to 40.

    Returns:
        None: The function modifies the provided axes.

    Note:
        The function automatically adjusts the plot limits to accommodate all
        rectangles. Each channel is represented by a filled rectangle centered
        at its spatial coordinates.
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
    """Plot probe channels using imshow for better visualization.

    This function uses matplotlib's imshow to visualize probe channels as
    colored rectangles, providing better performance and visualization quality
    compared to individual rectangle patches.

    Args:
        xy (np.ndarray): Array of shape (n_channels, 2) containing x,y coordinates
            for each channel in micrometers.
        color (np.ndarray): Array of shape (n_channels, 3) or (n_channels, 4)
            containing RGB or RGBA colors for each channel.
        ax (matplotlib.axes.Axes): Axes on which to plot the visualization.
        width (float, optional): Width of each channel representation in micrometers.
            Defaults to 16.
        height (float, optional): Height of each channel representation in micrometers.
            Defaults to 40.
        colorbar (bool, optional): Whether to add a colorbar to the plot.
            Defaults to False.

    Returns:
        None: The function modifies the provided axes.

    Note:
        The function stretches the probe in the X direction (factor of 3) to improve
        readability for very long thin probes. It creates a rasterized representation
        using numpy arrays and imshow for efficient rendering.
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
    """Create a figure displaying electrophysiological features and brain regions along a probe.

    This function visualizes multiple features along a probe's channels in physical space,
    as well as brain region information. It creates a multi-panel figure where each panel
    shows a different feature or brain region mapping.

    Args:
        pid_df (pd.DataFrame): Dataframe containing channels and voltage information for
            a given probe ID (PID). Must contain columns for the specified features and
            brain region mapping.
        features (list[str]): List of feature names to display, e.g. ['rms_lf', 'psd_delta', 'rms_ap'].
            These must be column keys in pid_df.
        xy (np.ndarray): Matrix of spatial channel positions (in μm), with shape [N_channels x 2].
            First column is lateral_um (x) and second column is axial_um (y).
        pid (str): Probe ID to be displayed in the figure title.
        fig (matplotlib.figure.Figure, optional): Existing figure to plot on. If None,
            a new figure is created.
        axs (np.ndarray, optional): Existing axes to plot on. If None, new axes are created.
        br (iblatlas.atlas.BrainRegions, optional): BrainRegions object for region color mapping.
            If None, a new one is created.
        mapping (str, optional): Brain region mapping to use. The function will look for
            columns named "{mapping}_id" in pid_df. Defaults to "Cosmos".
        plot_rect (callable, optional): Function to use for plotting rectangles. Should accept
            xy, color, and ax parameters. Defaults to plot_probe_rect2.
        cmap (str, optional): Colormap name to use for feature visualization.
            Defaults to "viridis".
        scaler (object, optional): Scaling to be applied to feature values before displaying.
            Should have a transform method (like sklearn.preprocessing.StandardScaler).
        vmin (float, optional): Minimum value for color normalization. If None, the minimum
            value in the data is used.
        vmax (float, optional): Maximum value for color normalization. If None, the maximum
            value in the data is used.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object containing the plots.
            - axs (np.ndarray): The axes objects for each subplot.

    Note:
        The function creates a multi-panel figure with brain region visualization,
        feature plots, and probe layout. It automatically handles figure sizing and
        subplot arrangement for optimal visualization.

    Example:
        # Merge the voltage and channels dataframe
        df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
        # Select a PID and create the single probe dataframe
        pid = '0228bcfd-632e-49bd-acd4-c334cf9213e9'
        pid_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()
    """
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, len(features) + 4, sharey=False, figsize=(9, 5))
    if scaler is not None:
        with sklearn.config_context(assume_finite=True):
            pid_df.loc[:, features] = scaler.transform(pid_df.loc[:, features])

    # plot all the features
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

    # plot brain regions if available
    if br is None:
        axs[0].axis("off")
        axs[1].axis("off")
        axs[2].axis("off")
    else:
        brainbox.ephys_plots.plot_brain_regions(
            pid_df[mapping + "_id"].values,
            channel_depths=xy[:, 1],
            brain_regions=br,
            display=True,
            ax=axs[0],
        )
        axs[0].set_title(mapping, rotation=90)

        # Plot brain region along probe depth with color code
        ax = axs[1]
        color = get_color_br(pid_df, br, mapping=mapping)
        plot_probe_rect2(xy, color, ax=ax)
        ax.set_title(mapping, rotation=90)
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
    """Create a grid of histograms displaying the distribution of electrophysiological features.

    This function generates a multi-panel figure with histograms for each feature in x_list.
    Each histogram is color-coded according to feature values and accompanied by a colorbar.
    The function uses quantile-based limits to handle outliers in the data visualization.

    Args:
        df_features (pd.DataFrame): DataFrame containing the feature values with feature
            names as columns.
        x_list (list, optional): List of feature names to plot. If None, uses all
            available voltage features. Defaults to None.
        title (str, optional): Title for the figure. Defaults to "".

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object containing all histograms.
            - axs (np.ndarray): Array of matplotlib.axes.Axes objects for each subplot.

    Note:
        The function creates a 4x12 grid layout with histograms and colorbars.
        Each histogram uses quantile-based limits (0.1-0.9 for color range, 0.005-0.995
        for histogram range) to handle outliers gracefully. The PuOr colormap is used
        for feature value coloring.
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


# The feature families of the heatmaps, from the feature schemas. The spike family lists the
# raw waveform columns and the spike-shape columns that replace them in denoised tables
FEATURE_FAMILIES = {
    "LF": [ephysatlas.features.ModelLfFeatures],
    "CSD": [ephysatlas.features.ModelCsdFeatures],
    "AP": [ephysatlas.features.ModelApFeatures],
    "Spikes": [
        ephysatlas.features.ModelSpikeFeatures,
        ephysatlas.features.ModelSpikeShapeFeatures,
    ],
}


def _classifier_regions():
    """Brain regions that also know the region decoders' extra ``void_fluid`` class (id 2000).

    Returns:
        ephysatlas.anatomy.ClassifierRegions: The iblatlas regions plus ``void_fluid``.
    """
    regions = ClassifierRegions()
    regions.add_new_region(NEW_VOID)
    return regions


def _panel_axes(panels, width_ratios, figsize, fig=None, axs=None):
    """Create one axes per panel in a row, or check and name the axes the caller gives.

    Args:
        panels (list[str]): Names of the panels, left to right.
        width_ratios (list[float]): Relative width of each panel, for new axes.
        figsize (tuple): Figure size in inches, for new axes.
        fig (matplotlib.figure.Figure, optional): Figure of the given axes.
        axs (Sequence[matplotlib.axes.Axes], optional): One axes per panel, left to right.

    Returns:
        tuple: The figure and a dict of axes keyed by panel name.

    Raises:
        ValueError: If no panel is shown, or the given axes do not match the panels.
    """
    if len(panels) == 0:
        raise ValueError("every panel is hidden: nothing to draw")
    if axs is None:
        fig, axs = plt.subplots(
            1,
            len(panels),
            figsize=figsize,
            width_ratios=width_ratios,
            constrained_layout=True,
            squeeze=False,
        )
        axs = axs[0]
    axs = list(np.atleast_1d(axs))
    if len(axs) != len(panels):
        raise ValueError(f"{len(axs)} axes given for {len(panels)} panels: {panels}")
    fig = fig if fig is not None else axs[0].figure
    return fig, dict(zip(panels, axs))


def plot_region_bar(
    ax,
    depths,
    atlas_ids,
    regions=None,
    show_labels=True,
    label_min_fraction=0.06,
    fontsize=8,
    title=None,
):
    """Draw brain regions as a coloured bar along the probe depth.

    Each channel fills the depth range half-way to its neighbours, in its region's colour. The
    acronym of a run of channels in one region is written inside the run, when the run is
    taller than ``label_min_fraction`` of the bar.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on. Its x-axis is hidden.
        depths (np.ndarray): Depth of each channel along the probe (µm), in any order.
        atlas_ids (np.ndarray): Atlas id of each channel's region, in the same order, in the
            region map to draw (remap Allen ids first, e.g. ``regions.remap``).
        regions (iblatlas.regions.BrainRegions, optional): Region table for the colours and
            acronyms. Defaults to the iblatlas regions plus the decoders' ``void_fluid``.
        show_labels (bool, optional): Write the region acronyms. Defaults to True.
        label_min_fraction (float, optional): Smallest run, as a fraction of the bar height,
            that gets an acronym. Defaults to 0.06.
        fontsize (float, optional): Font size of the acronyms. Defaults to 8.
        title (str, optional): Axes title.

    Returns:
        matplotlib.axes.Axes: The axes drawn on.

    Raises:
        ValueError: If an atlas id is not in ``regions``.
    """
    regions = regions if regions is not None else _classifier_regions()
    depths = np.asarray(depths, dtype=float)
    atlas_ids = np.asarray(atlas_ids)
    order = np.argsort(depths, kind="stable")
    depths, atlas_ids = depths[order], atlas_ids[order]
    found, ir = ismember(atlas_ids, regions.id)  # region index of each channel
    if not found.all():
        raise ValueError(
            f"atlas ids not in the regions: {np.unique(atlas_ids[~found])}"
        )
    # Each channel fills the half-way points to its neighbours
    edges = np.r_[depths[0], (depths[1:] + depths[:-1]) / 2, depths[-1]]
    for i in range(depths.size):
        ax.axhspan(edges[i], edges[i + 1], color=regions.rgb[ir[i]] / 255, lw=0)
    if show_labels:
        # One label per run of channels in one region, only where the run can hold it
        starts = np.r_[0, np.flatnonzero(np.diff(ir)) + 1]
        stops = np.r_[starts[1:], ir.size]
        for start, stop in zip(starts, stops):
            if edges[stop] - edges[start] > label_min_fraction * (
                depths[-1] - depths[0]
            ):
                ax.text(
                    0.5,
                    (edges[start] + edges[stop]) / 2,
                    regions.acronym[ir[start]],
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    transform=ax.get_yaxis_transform(),
                )
    ax.set(xticks=[], ylim=(depths[0], depths[-1]))
    if title is not None:
        ax.set_title(title)
    return ax


def plot_features_heatmap(
    values,
    depths,
    features,
    ax,
    feature_groups=None,
    vmin=-2,
    vmax=2,
    cmap="RdBu_r",
    show_feature_names=True,
    show_group_labels=True,
    show_group_separators=True,
    fontsize=8,
):
    """Draw the features of one probe as an image: channels sorted by depth, one column each.

    Args:
        values (np.ndarray): Feature values, shape (n_channels, n_features), typically
            z-scored. Rows in any order.
        depths (np.ndarray): Depth of each channel along the probe (µm), in the row order.
        features (list[str]): Feature names, in the column order of ``values``.
        ax (matplotlib.axes.Axes): Axes to draw on.
        feature_groups (dict[str, list[str]], optional): Named groups of features, e.g.
            ``{"LF": [...], "AP": [...]}``. Concatenated, they must equal ``features``.
        vmin (float, optional): Lower end of the colour range. Defaults to -2.
        vmax (float, optional): Upper end of the colour range. Defaults to 2.
        cmap (str, optional): Matplotlib colormap name. Defaults to "RdBu_r".
        show_feature_names (bool, optional): Name each column. Defaults to True.
        show_group_labels (bool, optional): Name each group above the image. Defaults to True.
        show_group_separators (bool, optional): Draw a white line between groups. Defaults
            to True.
        fontsize (float, optional): Font size of the feature names. Defaults to 8.

    Returns:
        matplotlib.image.AxesImage: The image, e.g. for a colorbar.

    Raises:
        ValueError: If ``feature_groups`` does not list ``features`` in order.
    """
    features = list(features)
    if feature_groups is not None:
        grouped = [f for group in feature_groups.values() for f in group]
        if grouped != features:
            raise ValueError(
                "feature_groups must list the features, in the column order of values"
            )
    values = np.asarray(values, dtype=float)
    depths = np.asarray(depths, dtype=float)
    order = np.argsort(depths, kind="stable")
    depths = depths[order]
    im = ax.imshow(
        values[order],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=[0, len(features), depths[0], depths[-1]],
        interpolation="none",
    )
    if show_feature_names:
        ax.set_xticks(
            np.arange(len(features)) + 0.5, features, rotation=90, fontsize=fontsize
        )
    else:
        ax.set_xticks([])
    if feature_groups is not None:
        x = 0
        for k, (name, group) in enumerate(feature_groups.items()):
            if show_group_labels:
                ax.text(
                    x + len(group) / 2,
                    1.01,
                    name,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    transform=ax.get_xaxis_transform(),
                )
            x += len(group)
            if show_group_separators and k < len(feature_groups) - 1:
                ax.axvline(x, color="w", lw=2)
    return im


def figure_features_heatmap(
    tables,
    pid,
    channels=None,
    feature_groups=None,
    mean=None,
    std=None,
    mapping="Cosmos",
    regions=None,
    vmin=-2,
    vmax=2,
    cmap="RdBu_r",
    show_histology=True,
    show_colorbar=True,
    show_group_labels=True,
    show_feature_names=True,
    fontsize=8,
    title=None,
    fig=None,
    axs=None,
):
    """Z-scored features of one probe against depth, beside its histology.

    Each feature is z-scored over every channel of every probe of the first table, so the
    probe is compared with the whole dataset. Several tables (the aggregated and the denoised
    one, for example) are drawn side by side on that same scale.

    Args:
        tables (pd.DataFrame or dict[str, pd.DataFrame]): Feature table(s) indexed by
            ``(pid, channel)``. A dict draws one panel per table, titled by its key.
        pid (str): The probe to draw.
        channels (pd.DataFrame, optional): Channel table indexed by ``(pid, channel)`` with
            ``axial_um``, and ``{mapping}_id`` or an Allen ``atlas_id`` for the histology.
            Defaults to the first table, which then needs those columns.
        feature_groups (dict[str, list[str]] or list[str], optional): Features to draw,
            grouped by name, or a plain list. Defaults to the LF, CSD, AP and Spikes families
            of the feature schemas, restricted to the columns of the first table. Features
            missing from the first table are skipped with a warning.
        mean (pd.Series, optional): Per-feature mean of the z-score. Defaults to the mean
            over the first table.
        std (pd.Series, optional): Per-feature standard deviation of the z-score. Defaults to
            the standard deviation over the first table.
        mapping (str, optional): Region map of the histology. Defaults to "Cosmos".
        regions (iblatlas.regions.BrainRegions, optional): Region table. Defaults to the
            iblatlas regions plus the decoders' ``void_fluid``.
        vmin (float, optional): Lower end of the colour range. Defaults to -2.
        vmax (float, optional): Upper end of the colour range. Defaults to 2.
        cmap (str, optional): Matplotlib colormap name. Defaults to "RdBu_r".
        show_histology (bool, optional): Draw the histology bar. Defaults to True.
        show_colorbar (bool, optional): Draw the colorbar. Defaults to True.
        show_group_labels (bool, optional): Name the feature groups. Defaults to True.
        show_feature_names (bool, optional): Name each feature. Defaults to True.
        fontsize (float, optional): Font size of the feature names. Defaults to 8.
        title (str, optional): Figure title.
        fig (matplotlib.figure.Figure, optional): Figure of ``axs``.
        axs (Sequence[matplotlib.axes.Axes], optional): One axes per panel, left to right:
            the histology (if shown), then one per table.

    Returns:
        tuple: The figure, and a dict of axes keyed by panel: ``"histology"``, the table
        names (``"features"`` for a single table) and ``"colorbar"``.
    """
    tables = tables if isinstance(tables, dict) else {"features": tables}
    first = next(iter(tables.values()))
    # The features to draw, grouped, keeping those the first table has
    if feature_groups is None:
        feature_groups = {}
        for name, schemas in FEATURE_FAMILIES.items():
            columns = dict.fromkeys(c for s in schemas for c in s.to_schema().columns)
            present = [
                c for c in columns if c in first.columns and c != "channel_labels"
            ]
            if present:
                feature_groups[name] = present
    elif not isinstance(feature_groups, dict):
        feature_groups = {"": list(feature_groups)}
    missing = [f for g in feature_groups.values() for f in g if f not in first.columns]
    if missing:
        _logger.warning(f"features not in the table, skipped: {missing}")
    feature_groups = {
        name: [f for f in group if f in first.columns]
        for name, group in feature_groups.items()
    }
    feature_groups = {name: group for name, group in feature_groups.items() if group}
    features = [f for group in feature_groups.values() for f in group]
    # One z-score scale for every panel: the statistics of the first table, all probes
    mean = first[features].mean() if mean is None else mean
    std = first[features].std() if std is None else std
    ch = (channels if channels is not None else first).loc[pid]
    depths = ch["axial_um"].to_numpy()

    panels = (["histology"] if show_histology else []) + list(tables)
    width_ratios = [1 if p == "histology" else 12 for p in panels]
    fig, axs = _panel_axes(
        panels, width_ratios, figsize=(1 + 9 * len(tables), 6), fig=fig, axs=axs
    )
    regions = regions if regions is not None else _classifier_regions()
    if show_histology:
        if f"{mapping}_id" in ch.columns:
            atlas_ids = ch[f"{mapping}_id"].to_numpy()
        else:
            atlas_ids = regions.remap(
                ch["atlas_id"].to_numpy(), source_map="Allen", target_map=mapping
            )
        plot_region_bar(
            axs["histology"], depths, atlas_ids, regions=regions, title="Histology"
        )
    for name, table in tables.items():
        # The probe's channels in the order of the channel table, so they match the depths
        z = ((table.loc[pid, features] - mean[features]) / std[features]).reindex(
            ch.index
        )
        im = plot_features_heatmap(
            z.to_numpy(),
            depths,
            features,
            axs[name],
            # a plain list of features has no group names to show
            feature_groups=feature_groups if "" not in feature_groups else None,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show_feature_names=show_feature_names,
            show_group_labels=show_group_labels,
            fontsize=fontsize,
        )
        if len(tables) > 1:
            # Above the group names, which sit just over the image
            axs[name].set_title(name, pad=16 if show_group_labels else 6)
    axs[panels[0]].set_ylabel("depth from probe tip (µm)")
    if show_colorbar:
        heatmaps = [axs[name] for name in tables]
        cbar = fig.colorbar(im, ax=heatmaps, label="z-score", shrink=0.6)
        axs["colorbar"] = cbar.ax
    if title is not None:
        fig.suptitle(title)
    return fig, axs


def figure_predictions_along_probe(
    out,
    depths,
    classes,
    histology=None,
    regions=None,
    mapping="Cosmos",
    show_histology=True,
    show_prediction=True,
    show_probabilities=True,
    show_legend=False,
    show_labels=True,
    ylim=None,
    fig=None,
    axs=None,
):
    """Region predictions along one probe: histology, predicted region and probabilities.

    Args:
        out (pd.DataFrame): The output of a region decoder's ``predict`` for one probe, with
            ``predicted_atlas_id`` and one ``p_<acronym>`` column per class, in class order.
        depths (np.ndarray): Depth along the probe (µm) of each row of ``out``.
        classes (np.ndarray): Atlas id of each class, in the order of the ``p_`` columns:
            the model's ``config["classes"]``.
        histology (pd.DataFrame or dict, optional): The channels' histology, with
            ``axial_um`` and an Allen ``atlas_id``, e.g. from
            ``SpikeSortingLoader.load_channels()``. Without it, no histology panel is drawn.
        regions (iblatlas.regions.BrainRegions, optional): Region table. Defaults to the
            iblatlas regions plus the decoders' ``void_fluid``, so no class is dropped.
        mapping (str, optional): Region map the histology is drawn in, the model's
            ``config["region_map"]``. Defaults to "Cosmos".
        show_histology (bool, optional): Draw the histology bar. Defaults to True.
        show_prediction (bool, optional): Draw the predicted-region bar. Defaults to True.
        show_probabilities (bool, optional): Draw the probabilities as stacked bands.
            Defaults to True.
        show_legend (bool, optional): Add a legend of the classes, right of the
            probabilities. Defaults to False.
        show_labels (bool, optional): Write the acronyms on the region bars. Defaults to True.
        ylim (tuple, optional): Depth range of every panel. Defaults to that of ``depths``.
        fig (matplotlib.figure.Figure, optional): Figure of ``axs``.
        axs (Sequence[matplotlib.axes.Axes], optional): One axes per shown panel, left to
            right. Use it to draw several probes on one figure.

    Returns:
        tuple: The figure, and a dict of axes keyed by panel: ``"histology"``,
        ``"prediction"`` and ``"probabilities"``.

    Raises:
        ValueError: If the number of ``p_`` columns differs from the number of classes.
    """
    prob_cols = [c for c in out.columns if c.startswith("p_")]
    classes = np.asarray(classes)
    if len(prob_cols) != classes.size:
        raise ValueError(
            f"{len(prob_cols)} probability columns but {classes.size} classes: pass the "
            f"model's config['classes']"
        )
    regions = regions if regions is not None else _classifier_regions()
    depths = np.asarray(depths, dtype=float)
    order = np.argsort(depths, kind="stable")
    panels = [
        name
        for name, show in [
            ("histology", show_histology and histology is not None),
            ("prediction", show_prediction),
            ("probabilities", show_probabilities),
        ]
        if show
    ]
    width_ratios = [2 if p == "probabilities" else 1 for p in panels]
    fig, axs = _panel_axes(
        panels, width_ratios, figsize=(1.6 * sum(width_ratios), 7), fig=fig, axs=axs
    )
    if "histology" in axs:
        atlas_ids = regions.remap(
            np.asarray(histology["atlas_id"]), source_map="Allen", target_map=mapping
        )
        plot_region_bar(
            axs["histology"],
            np.asarray(histology["axial_um"], dtype=float),
            atlas_ids,
            regions=regions,
            show_labels=show_labels,
            title="Histology",
        )
    if "prediction" in axs:
        plot_region_bar(
            axs["prediction"],
            depths,
            out["predicted_atlas_id"].to_numpy(),
            regions=regions,
            show_labels=show_labels,
            title="Predicted",
        )
    if "probabilities" in axs:
        plot_cumulative_probas(
            probas=out[prob_cols].to_numpy()[order],
            depths=depths[order],
            aids=classes,
            regions=regions,
            ax=axs["probabilities"],
        )
        axs["probabilities"].set(xlabel="cumulative probability", title="Probabilities")
        if show_legend:
            # Right of the panel, so the legend covers none of the bands
            axs["probabilities"].legend(
                loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False
            )
    ylim = ylim if ylim is not None else (depths.min(), depths.max())
    for name, ax in axs.items():
        ax.set_ylim(ylim)
        if name != panels[0]:
            ax.set_yticks([])
    axs[panels[0]].set_ylabel("depth from probe tip (µm)")
    return fig, axs


def figure_voltage_snippets(
    snippets,
    vrange=None,
    cmap="Greys_r",
    show_colorbar=True,
    show_titles=True,
    fig=None,
    axs=None,
):
    """Voltage images (channels × time) of one or several snippets, side by side.

    Each panel is drawn by ``ibldsp.plots.voltageshow``. Crop the time window before calling,
    e.g. ``data[:, :int(0.05 * fs)]`` for 50 ms.

    Args:
        snippets (dict[str, tuple[np.ndarray, float]]): ``{title: (data, fs)}``, with
            ``data`` of shape (n_channels, n_samples) in volts and ``fs`` in Hz. For example
            the destriped AP and LF bands, or the raw and destriped AP.
        vrange (float, optional): Colour range, ± µV, for every panel. Defaults to 4 times
            the median channel standard deviation of each panel, so the channels outside the
            brain, which are much larger, do not grey out the others.
        cmap (str, optional): Matplotlib colormap name. Defaults to "Greys_r".
        show_colorbar (bool, optional): Draw a colorbar beside each panel. Defaults to True.
        show_titles (bool, optional): Title each panel with its key. Defaults to True.
        fig (matplotlib.figure.Figure, optional): Figure of ``axs``.
        axs (Sequence[matplotlib.axes.Axes], optional): One axes per snippet.

    Returns:
        tuple: The figure, and a dict of axes keyed by snippet title.
    """
    fig, axs = _panel_axes(
        list(snippets),
        [1] * len(snippets),
        figsize=(5.5 * len(snippets), 5.5),
        fig=fig,
        axs=axs,
    )
    for title, (data, fs) in snippets.items():
        # float32 before scaling to µV: float16 snippets would overflow
        data = np.asarray(data, dtype=np.float32)
        panel_vrange = (
            vrange if vrange is not None else 4 * np.median(np.std(data * 1e6, axis=1))
        )
        im = ibldsp.plots.voltageshow(
            data,
            fs,
            cmap=cmap,
            ax=axs[title],
            vrange=panel_vrange,
            ylabel="channel (0 = probe tip)",
        )
        if show_colorbar:
            fig.colorbar(im, ax=axs[title], label="µV", shrink=0.8)
        if show_titles:
            axs[title].set_title(title)
    return fig, axs


def figure_encoder_coronal(
    encoder,
    features,
    ap_m,
    brain_atlas=None,
    step=4,
    mapping="Cosmos",
    percentiles=(2, 98),
    scales=None,
    units=None,
    cmap="magma",
    show_boundaries=True,
    show_colorbar=True,
    show_titles=True,
    fig=None,
    axs=None,
):
    """Features predicted by the spatial encoder on a coronal plane, one panel per feature.

    Predicts every ``step``-th voxel of the plane that lies inside the brain, in one call to
    ``encoder.predict_xyz``, then draws each feature with the region outlines on top.

    Args:
        encoder (ephysatlas.models.encoder_inpainting.SpatialEncoder): The spatial encoder,
            from ``load_pretrained("int-brain-lab/ea-encoder-channel", ...)``.
        features (list[str]): Features to draw, from ``encoder.features``.
        ap_m (float): Antero-posterior coordinate of the plane, in metres from bregma.
        brain_atlas (iblatlas.atlas.BrainAtlas, optional): Atlas for the brain mask, the
            voxel coordinates and the outlines. Defaults to ``AllenAtlas(res_um=25)``, which
            downloads the Allen volumes on first use.
        step (int, optional): Predict every ``step``-th voxel along ML and DV. Defaults to 4
            (100 µm with the 25 µm atlas).
        mapping (str, optional): Region map of the outlines. Defaults to "Cosmos".
        percentiles (tuple, optional): Percentiles of each panel's values that set its
            colour range. Defaults to (2, 98).
        scales (dict[str, float], optional): Factor applied to a feature before drawing, e.g.
            ``{"spike_amplitude": 1e6}`` for µV. Defaults to none.
        units (dict[str, str], optional): Colorbar label of a feature. Defaults to none.
        cmap (str, optional): Matplotlib colormap name. Defaults to "magma".
        show_boundaries (bool, optional): Draw the region outlines. Defaults to True.
        show_colorbar (bool, optional): Draw a colorbar beside each panel. Defaults to True.
        show_titles (bool, optional): Title each panel with its feature. Defaults to True.
        fig (matplotlib.figure.Figure, optional): Figure of ``axs``.
        axs (Sequence[matplotlib.axes.Axes], optional): One axes per feature.

    Returns:
        tuple: The figure, and a dict of axes keyed by feature.

    Raises:
        ValueError: If a feature is not predicted by the encoder.
    """
    unknown = [f for f in features if f not in encoder.features]
    if unknown:
        raise ValueError(f"the encoder does not predict {unknown}")
    ba = brain_atlas if brain_atlas is not None else AllenAtlas(res_um=25)
    scales, units = scales or {}, units or {}
    # Every step-th voxel of the plane that is inside the brain, as (x, y, z) in metres
    label = ba.label[ba.bc.y2i(ap_m), ::step, ::step]  # ML × DV
    ix, iz = np.nonzero(label)
    xyz_m = np.c_[
        ba.bc.xscale[::step][ix], np.full(ix.size, ap_m), ba.bc.zscale[::step][iz]
    ]
    # (n_voxels, n_features), in the order of encoder.features
    values = encoder.predict_xyz(xyz_m)
    fig, axs = _panel_axes(
        list(features),
        [1] * len(features),
        figsize=(4.3 * len(features), 3.8),
        fig=fig,
        axs=axs,
    )
    for feature in features:
        ax = axs[feature]
        image = np.full(label.shape, np.nan)
        image[ix, iz] = values[:, encoder.features.index(feature)] * scales.get(
            feature, 1
        )
        vmin, vmax = np.nanpercentile(image, percentiles)
        im = ax.imshow(
            image.T, extent=ba.extent(axis=1), cmap=cmap, vmin=vmin, vmax=vmax
        )
        if show_boundaries:
            ba.plot_cslice(ap_m, volume="boundary", mapping=mapping, ax=ax)
        if show_titles:
            ax.set_title(feature)
        ax.set_axis_off()
        if show_colorbar:
            fig.colorbar(im, ax=ax, shrink=0.7, label=units.get(feature, ""))
    return fig, axs
