import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches

from iblatlas.atlas import BrainRegions
from iblutil.numerical import ismember
from ibl_style.style import figure_style

import ephysatlas.features

figure_style()


def plot_cumulative_probas(probas, depths, aids, regions=None, ax=None, legend=False):
    """
    :param probas: (ndepths x nregions) array of probabilities for each region that sum to 1 for each depth
    :param depths: (ndepths) vector of depths
    :param aids: (nregions) vector of atlas_ids
    :param regions: optional: iblatlas.BrainRegion object
    :param ax:
    :param legend:
    :return:
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


def get_color_feat(x, cmap_name="viridis"):
    # Normalise between 0-1
    cmap = matplotlib.colormaps[cmap_name]
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
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


def plot_probe_rect2(xy, color, ax, width=16, height=40):
    """
    This function uses imshow to draw rectangles painted around the yx coordinates
    :param xy:
    :param color:
    :param ax:
    :param width:
    :param height:
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

    ax.imshow(im, extent=extent, origin="lower", aspect="auto")

    ax.set_xlim(*extent[:2])
    ax.set_xticks([])
    ax.set_ylim(ymin, ymax + 1)
    yticks = np.arange(0, ymax, 500)
    ax.set_yticks(yticks, labels=map(int, yticks * k))


def figure_features_chspace(
    pid_df,
    features,
    xy,
    pid,
    fig=None,
    axs=None,
    br=None,
    mapping="Cosmos",
    plot_rect=plot_probe_rect2,
    cmap="viridis",
):
    """

    :param pid_df: Dataframe containing channels and voltage information for a given PID
    Example on how to prepare it:
    # Merge the voltage and channels dataframe
    df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
    # Select a PID and create the single probe dataframe
    pid = '0228bcfd-632e-49bd-acd4-c334cf9213e9'
    pid_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()

    :param features: list of feature names to display, e.g. ['rms_lf', 'psd_delta', 'rms_ap']
    These have to bey columns keys of the pid_df
    :param xy: Matrix of spatial channel position (in um), lateral_um (x) and axial_um (y) [N channel x2]
    :return:
    """
    if br is None:
        br = BrainRegions()
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, len(features) + 2, sharey=True)  # +2

    for i_feat, feature in enumerate(features):
        feat_arr = pid_df[[feature]].to_numpy()
        # Plot feature
        color = get_color_feat(feat_arr, cmap_name=cmap)
        plot_rect(xy, color, ax=axs[i_feat])
        axs[i_feat].set_title(feature, rotation=90)

    # Plot brain region in space in unique colors
    # TODO Change to use rectangle function as plot_probe does not work anymore
    d_uni = np.unique(pid_df[mapping + "_id"].to_numpy(), return_inverse=True)[1]
    d_uni = d_uni.astype(np.float32)
    color = get_color_feat(d_uni)
    plot_rect(xy, color, ax=axs[len(features)])
    axs[len(features)].set_title("unique region", rotation=90)

    # Plot brain region along probe depth with color code
    color = get_color_br(pid_df, br, mapping=mapping)
    plot_rect(xy, color, ax=axs[len(features) + 1])  # + 1
    axs[len(features) + 1].set_title(mapping, rotation=90)

    # Add pid as suptitle
    # pid = pid_df.index[0][0]
    fig.suptitle(f"PID {pid}", y=0.08)

    return fig, axs
