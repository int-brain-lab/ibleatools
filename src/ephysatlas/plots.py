import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from iblatlas.atlas import BrainRegions
from iblutil.numerical import ismember
from ibl_style.style import figure_style

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
def plot_results(df, predicted_probas, dict_model, regions = None):
    features = dict_model['meta']['FEATURES'][:-4]
    aids = np.array(dict_model['meta']['CLASSES'])
    n_folds, n_channels, n_classes = predicted_probas.shape
    if predicted_probas.ndim == 2:
        predicted_probas = predicted_probas[np.newaxis, ...]

    df_depths = df.groupby('axial_um').mean()
    entropies = np.mean(-predicted_probas * np.log2(predicted_probas), axis=2).T

    fig, ax = plt.subplots(1, 1 + n_folds + 1, figsize=(16, 8), gridspec_kw={'width_ratios': [1] + [0.4] * n_folds + [0.2]}, sharey=True)

    ax[0].imshow(
        scipy.stats.zscore(df_depths.loc[:, features].to_numpy().astype(float)),
        extent=[0, len(features) + 1, df['axial_um'].min(), df['axial_um'].max()],
        vmin=-2,
        vmax=2,
        cmap="Spectral",
        aspect='auto',
    )
    ax[0].set_xticks(np.arange(len(features)) + 0.5)
    ax[0].set_xticklabels(features, rotation=90)
    for i in range(n_folds):
        plot_cumulative_probas(predicted_probas[i], df['axial_um'].values, aids=aids, regions=regions, ax=ax[i + 1])
        ax[-1].plot(entropies[:, i], df['axial_um'], label=f'Fold {i}', alpha=0.2)
        ax[i + 1].set_title('Fold {i}')
    ax[-1].plot(entropies.mean(axis=1), df['axial_um'], label='Mean', color='k')
    # ax[-1].legend()
    ax[-1].set_title('Entropies')
    return fig, ax

# plot_results(df_inference_denoised, predicted_probas)

