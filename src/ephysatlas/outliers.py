import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import iqr
import tqdm

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from ephysatlas.plots import select_series
from ephysatlas.anatomy import ClassifierRegions, NEW_VOID

from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity

# from ephys_atlas.plots import BINS
BINS = 50


def compute_histogram(series, bins=None):
    bins = bins if bins is not None else BINS

    hist_values, bin_edges = np.histogram(series, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist_values, bin_centers


def detect_outlier_kstest(train_data: np.ndarray, test_data: np.ndarray):
    """
    For a single feature, compute channel by channel the KS test against the distribution

    Parameters:
    - train_data: (N,) numpy array, training dataset (assumed to represent the true distribution).
    - test_data: (M,) numpy array, test dataset (points to evaluate for outlier probability).

    Returns:
    - outlier_statistic: (M,) numpy array, KS statistic of each test sample being an outlier.
    """
    out = np.zeros(test_data.shape)
    for count, sample in enumerate(
        test_data
    ):  # Test on each channel value independently
        ks_stat = stats.kstest(sample, train_data)
        out[count] = ks_stat.statistic
    return out


##


def kde_proba_distribution(
    train_data,
    test_data,
    n_samples=50,
    bandwidth_factor=16,
    interp_kind="linear",
    n_min_sample_train=300,
):
    """
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
    - n_min_sample_train: int, default 300, number of samples minimal to remove outliers from based distribution

    Returns:
    - outp: (M,) numpy array, outlier probability statistic of each test sample being an outlier.
    - x_train: (E, ) numpy array, x values of the histogram formed using training dataset
    - hist_train: (E, ) numpy array, y values of the histogram formed using training dataset
    """
    if len(train_data) > n_min_sample_train:
        # Step 0 : Filter the train data to remove large outliers
        train_data = train_data[
            np.abs(train_data - np.median(train_data)) <= 5 * iqr(train_data)
        ]

    # Step 1: Compute the histogram using KDE, on linearly spaced vector x_train
    # Note that for n_samples > 50 the performance of kde.score_samples drops and run takes longer
    x_train = np.linspace(np.min(train_data), np.max(train_data), n_samples)
    bin_width = x_train[1] - x_train[0]
    # Fit KDE
    # Note : STD/16 is fragile to use when there is bimodal or outliers
    robust_std = (
        iqr(train_data) / 1.349
    )  # approximate standard deviation assuming normality
    if robust_std == 0.0:  # Fall back in case IQR is 0
        robust_std = np.std(train_data) / 16
    bandwidth = robust_std / bandwidth_factor
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(train_data.reshape(-1, 1))  # Reshape for usage in kde
    # Get probability score
    # The KDE returns the log_pdf. A density can exceed 1 and is not necessarily between 0 and 1.
    density = np.exp(
        kde.score_samples(x_train.reshape(-1, 1))
    )  # Reshape for usage in kde
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

    # For extremely high value (e.g. >10 IQR), remove them, put outlier score to 1
    val_replace = np.mean(train_data)  # Use the mean of the train data so it's well within boundaries
    idx_replace = np.where( (test_data > np.mean(train_data) + 10 * iqr(train_data)) |
                            (test_data < np.mean(train_data) - 10 * iqr(train_data))
                            )[0]
    test_data[idx_replace] = val_replace

    if np.max(test_data) > np.max(x_train):
        # Pad above
        add_x = np.arange(
            np.max(x_train), np.max(test_data) + n_padbin_add * bin_width, bin_width
        )
        add_y = np.zeros(np.shape(add_x))
        n_above = add_x.shape[0]
        x_train = np.concatenate((x_train, add_x))
        hist_train = np.concatenate((hist_train, add_y))

    if np.min(test_data) < np.min(x_train):
        # Pad below
        add_x = np.arange(
            np.min(test_data) - n_padbin_add * bin_width, np.min(x_train), bin_width
        )
        add_y = np.zeros(np.shape(add_x))
        n_below = add_x.shape[0]
        x_train = np.concatenate((add_x, x_train))
        hist_train = np.concatenate((add_y, hist_train))

    assert n_original + n_above + n_below == hist_train.shape[0]

    # Create the interpolation function
    y_train = hist_train
    interp_func = interp1d(
        x_train, y_train, kind=interp_kind
    )  # 'linear' or 'cubic', 'quadratic', etc.

    # Generate new y-values from test data
    y_test = interp_func(test_data)

    # The outlier probability is the inverse
    outp = 1 - y_test
    outp[idx_replace] = 1  # Replace extreme outliers that were set to default value to outlier score 1

    return outp, x_train, hist_train


def kde_proba_1pid(df_base, df_new, features, mapping, p_thresh=0.999999,
                   min_ch=15, n_pid=3, min_ch_compute=100):
    # Regions
    regions = np.unique(df_new[mapping + "_id"]).astype(int)
    # Store the features that are outlier per brain region in a dict
    dictout = dict((el, list()) for el in regions.tolist())
    dictout["mapping"] = mapping
    dictout["features"] = features

    for count, region in tqdm.tqdm(enumerate(regions), total=len(regions)):
        # Get channel indices that are in region, but keeping all info besides features
        idx_reg = np.where(df_new[mapping + "_id"] == region)
        df_new_compute = df_new.iloc[idx_reg].copy()
        df_new_compute["has_outliers"] = False

        listout = list()
        for feature in features:
            # print(f"{feature} [{region}]")
            # Load data for that regions
            df_train = select_series(
                df_base, features=[feature], acronym=None, id=region, mapping=mapping
            )
            # Get channel indices that are in region, keeping only feature values
            df_test = select_series(
                df_new, features=[feature], acronym=None, id=region, mapping=mapping
            )

            df_pid = select_series(
                df_base, features=['pid'], acronym=None, id=region, mapping=mapping
            )

            # For all channels at once, test if outside the distribution for the given features
            train_data = df_train.to_numpy()
            test_data = df_test.to_numpy()
            # score_out = 0 if N pid or N channel too small in training set
            if bool((df_pid.nunique().values[0] >= n_pid) & (df_pid.shape[0] >= min_ch_compute)):
                score_out, _, _ = kde_proba_distribution(train_data, test_data)
            else:
                score_out = np.zeros(test_data.shape)
            # Save into new column
            df_new_compute[feature + "_q"] = score_out
            df_new_compute[feature + "_extremes"] = 0
            df_new_compute.loc[
                df_new_compute[feature + "_q"] > p_thresh, feature + "_extremes"
            ] = 1
            # A region is assigned as having outliers if more than half its channels are outliers
            # Condition on N minimum channel.
            has_outliers = sum(df_new_compute[feature + "_extremes"]) > np.floor(
                len(test_data) / 2
            )
            if len(test_data) >= min_ch and has_outliers:
                listout.append(feature)
                if sum(
                    df_new_compute["has_outliers"] == 0
                ):  # Reassign only if entirely False
                    df_new_compute["has_outliers"] = True
        # Save appended list of feature in dict
        dictout[region] = listout

        # Concatenate dataframes
        if count == 0:
            df_save = df_new_compute.copy()
        else:
            df_save = pd.concat([df_save, df_new_compute])

    if df_save["has_outliers"].sum() > 0:
        has_outlier = True
    else:
        has_outlier = False

    # Resort by channel
    df_save = df_save.sort_values(by=["channel"])
    return df_save, dictout, has_outlier


def save_score_kde(df_save, dictout, has_outlier, local_save_data, filenamebase):
    # Compute outlier score using:
    # df_save, dictout, has_outlier = kde_proba_1pid(df_base, df_new, features, mapping)

    if has_outlier:
        # Save only if outlier are present
        df_save.to_parquet(local_save_data.joinpath(f"{filenamebase}_df_save.pqt"))
        np.save(local_save_data.joinpath(f"{filenamebase}_dictout.npy"), dictout)


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
            df_new[mapping_original + "_id"],
            source_map=mapping_original,
            target_map=mapping_model,
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
    assert channels.get("labels") is not None

    df_label = pd.DataFrame(channels)[["labels", "rawInd"]].rename(
        columns={"rawInd": "channel"}
    )
    df_label["pid"] = pid
    df_label = df_label.set_index(["pid", "channel"])

    # Merge to get the labels column
    df = df.merge(df_label, left_index=True, right_index=True)
    return df

# ===========================================================
# ====== SVM ================================================

def detect_modality_auto_bandwidth(data, peak_height_frac=0.1, resolution=500):
    """
    Detect unimodal or multimodal distribution using KDE with Silverman's bandwidth.

    Parameters
    ----------
    data : array-like
        Raw data points.
    peak_height_frac : float, optional
        Fraction of maximum density for minimum peak height.
    resolution : int, optional
        Number of points in the KDE evaluation grid.

    Returns
    -------
    modality : str
        "unimodal" or "multimodal"
    peaks : array
        Indices of detected peaks in x_grid.
    x_grid : array
        Grid of x values for plotting/debugging.
    density : array
        KDE density values on x_grid.
    bandwidth : float
        Bandwidth used for KDE (Silverman's rule).
    """

    data = np.asarray(data)
    n = len(data)
    sigma = np.std(data)

    # Silverman's rule of thumb
    bandwidth = 1.06 * sigma * n ** (-1 / 5)

    # KDE fit
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data.reshape(-1, 1))

    # Evaluate density
    x_grid = np.linspace(data.min(), data.max(), resolution)
    log_dens = kde.score_samples(x_grid[:, None])
    density = np.exp(log_dens)

    # Peak detection
    peaks, _ = find_peaks(density, height=max(density) * peak_height_frac)

    modality = "unimodal" if len(peaks) <= 1 else "multimodal"
    return modality, peaks, x_grid, density, bandwidth


def get_gamma_svm(X_train, f=0.4):
    '''
    f : how many times larger you want gamma
    '''
    modality, peaks, x_grid, density, bandwidth = detect_modality_auto_bandwidth(X_train)
    num_peaks = len(peaks)

    if num_peaks <= 1 :
        # Unimodal
        sigma = ( iqr(X_train) / 1.349 )  # approximate standard deviation assuming normality
        if sigma == 0.0:  # Fall back in case IQR is 0
            sigma = np.std(X_train) / 16

        sigma = sigma / np.sqrt(f)

    else:
        # Multimodal
        # Suppose you detected k peaks
        k = num_peaks  # from your detection method

        # Cluster the data into k groups
        data_reshaped = X_train.reshape(-1, 1)
        labels = KMeans(n_clusters=k, n_init=10).fit_predict(data_reshaped)

        # Compute per-cluster STD
        cluster_stds = [np.std(X_train[labels == i]) for i in range(k)]

        # Use average STD for gamma
        sigma = np.mean(cluster_stds)

    sigma = sigma / np.sqrt(f)
    gamma = 1 / (2 * sigma ** 2)
    return gamma


def outlier_score_svm(X_train, X_test, nu=0.2, f=0.4, kernel='rbf'):
    # Step 0 : Filter the train data to remove large outliers
    X_train = X_train[
        np.abs(X_train - np.median(X_train)) <= 5 * iqr(X_train)
        ]

    gamma = get_gamma_svm(X_train, f=f)
    model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    model.fit(X_train.reshape(-1, 1))  # This is slow for high N ; e.g. 29 seconds for 69k samples
    pred_svm = model.predict(X_test.reshape(-1, 1))
    return pred_svm


def generate_testset(X_train):
    val_iqr = 3 * iqr(X_train)
    X_test = np.linspace(min(X_train) - val_iqr, max(X_train) + val_iqr, 100).reshape(-1, 1)
    return X_test


def score_svm_1pid(df_base, df_new, features, mapping,
                   min_ch_outlier=20, n_pid=3, min_ch_compute=100, max_ch_compute = 3000, p_thresh_kde=0.98):
    # Regions
    regions = np.unique(df_new[mapping + "_id"]).astype(int)
    # Store the features that are outlier per brain region in a dict
    dictout = dict((el, list()) for el in regions.tolist())
    dictout["mapping"] = mapping
    dictout["features"] = features

    for count, region in tqdm.tqdm(enumerate(regions), total=len(regions)):
        # Get channel indices that are in region, but keeping all info besides features
        idx_reg = np.where(df_new[mapping + "_id"] == region)
        df_new_compute = df_new.iloc[idx_reg].copy()
        df_new_compute["has_outliers"] = False

        listout = list()
        for feature in features:
            # print(f"{feature} [{region}]")
            # Load data for that regions
            df_train = select_series(
                df_base, features=[feature], acronym=None, id=region, mapping=mapping
            )
            # Get channel indices that are in region, keeping only feature values
            df_test = select_series(
                df_new, features=[feature], acronym=None, id=region, mapping=mapping
            )

            df_pid = select_series(
                df_base, features=['pid'], acronym=None, id=region, mapping=mapping
            )

            # For all channels at once, test if outside the distribution for the given features
            train_data = df_train.to_numpy()
            test_data = df_test.to_numpy()

            print(f'Train N: {len(train_data)}, Test N: {len(test_data)}')

            # Outlier → 1
            # Inlier → 0

            # score_out = 0 if N pid or N channel too small in training set
            if bool((df_pid.nunique().values[0] >= n_pid) & (df_pid.shape[0] >= min_ch_compute)):
                if len(train_data) > max_ch_compute:  # Apply KDE method as SVM model.fit is too slow for high N
                    scoreq_out, _, _ = kde_proba_distribution(train_data,
                                                              test_data)  # This is a score to be thresholded
                    scoreq_out = scoreq_out.squeeze()  # TODO this should not be necessary, place in kde_proba_distribution
                    score_out = scoreq_out > p_thresh_kde
                    score_out = score_out.astype(int)

                else:
                    score_svm = outlier_score_svm(train_data, test_data)
                    # map the OneClassSVM output from {1, -1} into {0, 1}
                    score_out = np.where(score_svm == -1, 1, 0)
            else:
                score_out = np.zeros(test_data.shape)
            # Save into new column
            df_new_compute[feature + "_extremes"] = score_out
            # A region is assigned as having outliers if more than N minimum channels are outliers
            has_outliers = sum(df_new_compute[feature + "_extremes"]) > np.floor(min_ch_outlier)
            if has_outliers:
                listout.append(feature)
                if sum(
                    df_new_compute["has_outliers"] == 0
                ):  # Reassign only if entirely False
                    df_new_compute["has_outliers"] = True
        # Save appended list of feature in dict
        dictout[region] = listout

        # Concatenate dataframes
        if count == 0:
            df_save = df_new_compute.copy()
        else:
            df_save = pd.concat([df_save, df_new_compute])

    if df_save["has_outliers"].sum() > 0:
        has_outlier = True
    else:
        has_outlier = False

    # Resort by channel
    df_save = df_save.sort_values(by=["channel"])
    return df_save, dictout, has_outlier
