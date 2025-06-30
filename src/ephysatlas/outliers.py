import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import iqr
import tqdm

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from ephysatlas.plots import select_series
from ephysatlas.anatomy import ClassifierRegions, NEW_VOID

# from ephys_atlas.plots import BINS
BINS = 50


def compute_histogram(series, bins=None):
    bins = bins if bins is not None else BINS

    hist_values, bin_edges = np.histogram(series, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist_values, bin_centers


def detect_outlier_kstest(train_data: np.ndarray, test_data: np.ndarray):
    '''
    For a single feature, compute channel by channel the KS test against the distribution

    Parameters:
    - train_data: (N,) numpy array, training dataset (assumed to represent the true distribution).
    - test_data: (M,) numpy array, test dataset (points to evaluate for outlier probability).

    Returns:
    - outlier_statistic: (M,) numpy array, KS statistic of each test sample being an outlier.
    '''
    out = np.zeros(test_data.shape)
    for count, sample in enumerate(test_data):  # Test on each channel value independently
        ks_stat = stats.kstest(sample, train_data)
        out[count] = ks_stat.statistic
    return out


##

def kde_proba_distribution(train_data, test_data, n_samples=50, bandwidth_factor=16, interp_kind='linear'):
    '''
    For a single feature, compute channel by channel the outlier score using KDE for
    the test set against the training distribution.

    There are 3 main steps:
    # Step 0 : Filter the train data to remove large outliers.
    # Step 1: Compute the probability density "histogram" using KDE, on linearly spaced vector x_train
    # Step 2: Get score of the test samples by interpolating values on the made histogram

    Parameters:
    - train_data: (N,) numpy array, training dataset (assumed to represent the true distribution).
    - test_data: (M,) numpy array, test dataset (points to evaluate for outlier probability).
    - n_samples: int, default 50, number of samples used to estimate density from KDE fit
    - bandwidth_factor: int, default 16, dividing factor for the bandwidth of the KDE fit
    - interp_kind: string, default 'linear', type of interpolation for the KDE fit

    Returns:
    - outp: (M,) numpy array, outlier probability statistic of each test sample being an outlier.
    - x_train: (E, ) numpy array, x values of the histogram formed using training dataset
    - hist_train: (E, ) numpy array, y values of the histogram formed using training dataset
    '''
    # Step 0 : Filter the train data to remove large outliers
    train_data = train_data[np.abs(train_data - np.median(train_data)) < 5 * iqr(train_data)]

    # Step 1: Compute the histogram using KDE, on linearly spaced vector x_train
    # Note that for n_samples > 50 the performance of kde.score_samples drops and run takes longer
    x_train = np.linspace(np.min(train_data), np.max(train_data), n_samples)
    bin_width = x_train[1] - x_train[0]
    # Fit KDE
    # Note : STD/16 is fragile to use when there is bimodal or outliers
    robust_std = iqr(train_data) / 1.349  # approximate standard deviation assuming normality
    bandwidth = robust_std / bandwidth_factor
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(train_data.reshape(-1, 1))  # Reshape for usage in kde
    # Get probability score
    # The KDE returns the log_pdf. A density can exceed 1 and is not necessarily between 0 and 1.
    density = np.exp(kde.score_samples(x_train.reshape(-1, 1)))  # Reshape for usage in kde
    # Normalize to get probability-like values between 0-1
    hist_train = density / np.sum(density)
    n_original = hist_train.shape[0]

    # Step 2: Get score of the test samples.
    # Note: the kde.score_samples is too slow for N>50 samples so we by-pass it by interpolating values
    # on the made histogram
    # Pad above and below with bins of N=0 samples if test data has larger x-values; add 3 bins to be safe
    # This is necessary otherwise the interpolation may not have a wide-enough range
    n_padbin_add = 3
    n_above = 0
    n_below = 0
    if np.max(test_data) > np.max(x_train):
        # Pad above
        add_x = np.arange(np.max(x_train), np.max(test_data) + n_padbin_add * bin_width, bin_width)
        add_y = np.zeros(np.shape(add_x))
        n_above = add_x.shape[0]
        x_train = np.concatenate((x_train, add_x))
        hist_train = np.concatenate((hist_train, add_y))

    if np.min(test_data) < np.min(x_train):
        # Pad below
        add_x = np.arange(np.min(test_data) - n_padbin_add * bin_width, np.min(x_train), bin_width)
        add_y = np.zeros(np.shape(add_x))
        n_below = add_x.shape[0]
        x_train = np.concatenate((add_x, x_train))
        hist_train = np.concatenate((add_y, hist_train))

    assert n_original + n_above + n_below == hist_train.shape[0]

    # Create the interpolation function
    y_train = hist_train
    interp_func = interp1d(x_train, y_train, kind=interp_kind)  # 'linear' or 'cubic', 'quadratic', etc.

    # Generate new y-values from test data
    y_test = interp_func(test_data)

    # The outlier probability is the inverse
    outp = 1 - y_test

    return outp, x_train, hist_train


def kde_proba_1pid(df_base, df_new, features, mapping, p_thresh=0.999999, min_ch=15):
    # Regions
    regions = np.unique(df_new[mapping + '_id']).astype(int)
    # Store the features that are outlier per brain region in a dict
    dictout = dict((el, list()) for el in regions.tolist())
    dictout['mapping'] = mapping
    dictout['features'] = features

    for count, region in tqdm.tqdm(enumerate(regions), total=len(regions)):

        # Get channel indices that are in region, but keeping all info besides features
        idx_reg = np.where(df_new[mapping + '_id'] == region)
        df_new_compute = df_new.iloc[idx_reg].copy()
        df_new_compute['has_outliers'] = False

        listout = list()
        for feature in features:
            # Load data for that regions
            df_train = select_series(df_base, features=[feature],
                                     acronym=None, id=region, mapping=mapping)

            # Get channel indices that are in region, keeping only feature values
            df_test = select_series(df_new, features=[feature],
                                    acronym=None, id=region, mapping=mapping)

            # For all channels at once, test if outside the distribution for the given features
            train_data = df_train.to_numpy()
            test_data = df_test.to_numpy()
            score_out, _, _ = kde_proba_distribution(train_data, test_data)
            # score_out, _, _ = detect_outlier_histV2(train_data, test_data)
            # Save into new column
            df_new_compute[feature + '_q'] = score_out
            df_new_compute[feature + '_extremes'] = 0
            df_new_compute.loc[df_new_compute[feature + '_q'] > p_thresh, feature + '_extremes'] = 1
            # A region is assigned as having outliers if more than half its channels are outliers
            # Condition on N minimum channel.
            has_outliers = sum(df_new_compute[feature + '_extremes']) > np.floor(len(test_data) / 2)
            if len(test_data) >= min_ch and has_outliers:
                listout.append(feature)
                if sum(df_new_compute['has_outliers'] == 0):  # Reassign only if entirely False
                    df_new_compute['has_outliers'] = True
        # Save appended list of feature in dict
        dictout[region] = listout

        # Concatenate dataframes
        if count == 0:
            df_save = df_new_compute.copy()
        else:
            df_save = pd.concat([df_save, df_new_compute])

    if df_save['has_outliers'].sum() > 0:
        has_outlier = True
    else:
        has_outlier = False

    return df_save, dictout, has_outlier



def save_score_kde(df_save, dictout, has_outlier, local_save_data, filenamebase):
    # Compute outlier score using:
    # df_save, dictout, has_outlier = kde_proba_1pid(df_base, df_new, features, mapping)

    if has_outlier:
        # Save only if outlier are present
        df_save.to_parquet(local_save_data.joinpath(f'{filenamebase}_df_save.pqt'))
        np.save(local_save_data.joinpath(f'{filenamebase}_dictout.npy'), dictout)


def compute_misaligned_proba(aids, aids_ch, predicted_probas):
    # aids : numpy array (M,) :  predicted atlas ID of the model
    #        typically loaded as: np.array(dict_model["meta"]["CLASSES"])
    # aids_ch : numpy array (N,) :original atlas ID of the N channels using mapping of the model
    #         typically loaded as: df[mapping_model + "_id"].to_numpy()
    # predicted_probas: output probas of the infer_region model
    #          typically loaded as: predicted_probas, _ = infer_regions(df, model_path)

    # Average
    predicted_probas_avg = np.mean(predicted_probas, axis=0)

    list_val = list()
    for i_ch in range(0, len(aids_ch)):
        idx = np.where(aids == aids_ch[i_ch])[0][0]
        # Find column in predicted proba avg
        probval = predicted_probas_avg[i_ch, idx]

        list_val.append(probval)
    return np.array(list_val)


def remap_df__original_to_model(df_new, mapping_original, mapping_model, regions=None):
    # Remap original labels to prediction mapping
    if regions is None:
        regions = ClassifierRegions()
        regions.add_new_region(NEW_VOID)

    # remap Beryl onto Cosmos as the model only runs on Cosmos
    if mapping_original != mapping_model:
        df_new[mapping_model + "_id"] = regions.remap(
            df_new[mapping_original + "_id"], source_map=mapping_original, target_map=mapping_model
        )
        df_new[mapping_model + "_acronym"] = regions.id2acronym(
            df_new[mapping_model + "_id"]
        )
    return df_new


def df_add_channel_label(df, pid, one=None):
    # Get the channel labels information
    if one is None:
        one = ONE()
    ssl = SpikeSortingLoader(pid=pid, one=one)
    channels = ssl.load_channels()
    assert channels.get('labels') is not None

    df_label = pd.DataFrame(channels)[['labels', 'rawInd']].rename(columns={'rawInd': 'channel'})
    df_label['pid'] = pid
    df_label = df_label.set_index(['pid', 'channel'])

    # Merge to get the labels column
    df = df.merge(df_label, left_index=True, right_index=True)
    return df