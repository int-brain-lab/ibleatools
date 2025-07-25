from abc import ABC
import logging
from pathlib import Path
import random
import shutil
import string
import tempfile
from typing_extensions import Annotated, List

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pydantic
import scipy.signal
import skimage.restoration
from pandera.typing import Series
import sklearn.base

import ibldsp.waveforms
import ibldsp.cadzow
import ibldsp.utils
import ibldsp.voltage

# Set up logger
logger = logging.getLogger(__name__)

__features_version__ = (
    "2025.07.01"  # this is the version of this feature extractor code
)


# TODO - Scratch_dir path is not working as expected. Even if I pass the scratch_dir argument in the main compute_features function, here I am gettig log from Path("/scratch/dartsort/")
def _setup_scratch_directory(scratch_dir=None):
    """
    Set up scratch directory with fallback logic.

    Args:
        scratch_dir (Path or str, optional): Preferred scratch directory path.
            If None, will try system defaults.

    Returns:
        Path: Path to the created scratch directory
    """
    if scratch_dir is not None:
        scratch_path = Path(scratch_dir)
    else:
        # Try SDSC scratch directory first
        scratch_path = Path("/scratch/dartsort/")

    try:
        scratch_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using scratch directory: {scratch_path}")
        return scratch_path
    except Exception as e:
        # Fallback to system temp directory if preferred directory fails
        logger.warning(f"Error creating scratch directory {scratch_path}: {e}")
        fallback_path = Path(tempfile.gettempdir()) / "dartsort"
        fallback_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using fallback scratch directory: {fallback_path}")
        return fallback_path


floats = Annotated[pa.Float, pa.Float32]
BANDS = {
    "delta": [0, 4],
    "theta": [4, 10],
    "alpha": [8, 12],
    "beta": [15, 30],
    "gamma": [30, 90],
    "lfp": [0, 90],
}

FEATURES_LIST = ["raw_ap", "raw_lf", "localisation", "waveforms"]


def get_feature_cmin(feature_name):
    # todo
    pass


class DartParameters(pydantic.BaseModel):
    localization_radius: pydantic.PositiveFloat = 150
    chunk_length_samples: pydantic.PositiveInt = 2**15
    trough_offset: pydantic.PositiveInt = (42,)
    scratch_dir: Path | str | None = pydantic.Field(
        default=None,
        description="Scratch directory for temporary files. If None, will use system defaults.",
    )


class ChannelDataFrameSchema(pa.DataFrameModel):
    pid: Series[str] = pa.Field()
    channel: Series[int] = pa.Field()
    x: Series[float] = pa.Field()
    y: Series[float] = pa.Field()
    z: Series[float] = pa.Field()
    axial_um: Series[float] = pa.Field(coerce=True)
    lateral_um: Series[float] = pa.Field(coerce=True)
    acronym: Series[str] = pa.Field()
    atlas_id: Series[int] = pa.Field()


class BaseChannelFeatures(pa.DataFrameModel):
    pass  # channel: Index[int] = pa.Field(check_name=True)


class ModelLfFeatures(BaseChannelFeatures):
    rms_lf: Series[float] = pa.Field(
        coerce=True, metadata={"transform": lambda x: 20 * np.log10(x)}
    )
    psd_delta: Series[float] = pa.Field(coerce=True)
    psd_theta: Series[float] = pa.Field(coerce=True)
    psd_alpha: Series[float] = pa.Field(coerce=True)
    psd_beta: Series[float] = pa.Field(coerce=True)
    psd_gamma: Series[float] = pa.Field(coerce=True)
    psd_lfp: Series[float] = pa.Field(coerce=True)


class ModelCsdFeatures(BaseChannelFeatures):
    rms_lf_csd: Series[float] = pa.Field(
        coerce=True, metadata={"transform": lambda x: 20 * np.log10(x)}
    )
    psd_delta_csd: Series[float] = pa.Field(coerce=True)
    psd_theta_csd: Series[float] = pa.Field(coerce=True)
    psd_alpha_csd: Series[float] = pa.Field(coerce=True)
    psd_beta_csd: Series[float] = pa.Field(coerce=True)
    psd_gamma_csd: Series[float] = pa.Field(coerce=True)
    psd_lfp_csd: Series[float] = pa.Field(coerce=True)


class ModelApFeatures(BaseChannelFeatures):
    rms_ap: Series[float] = pa.Field(
        coerce=True, metadata={"transform": lambda x: 20 * np.log10(x)}
    )
    cor_ratio: Series[float] = pa.Field(coerce=True)
    channel_labels: Series[int] = pa.Field(coerce=True)


class ModelSpikeFeatures(BaseChannelFeatures):
    alpha_mean: Series[float] = pa.Field(coerce=True)
    alpha_std: Series[float] = pa.Field(coerce=True)
    depolarisation_slope: Series[float] = pa.Field(coerce=True)
    peak_time_secs: Series[float] = pa.Field(coerce=True)
    peak_val: Series[float] = pa.Field(coerce=True)
    polarity: Series[float] = pa.Field(coerce=True)
    recovery_slope: Series[float] = pa.Field(coerce=True)
    recovery_time_secs: Series[float] = pa.Field(coerce=True)
    repolarisation_slope: Series[float] = pa.Field(coerce=True)
    spike_count: float = pa.Field(
        coerce=True,
        metadata={
            "transform": lambda x: np.where(x == 0, np.nan, np.log2(x.astype(float)))
        },
    )
    tip_time_secs: Series[float] = pa.Field(coerce=True)
    tip_val: Series[float] = pa.Field(coerce=True)
    trough_time_secs: Series[float] = pa.Field(coerce=True)
    trough_val: Series[float] = pa.Field(coerce=True)


class ModelChannelLayout(BaseChannelFeatures):
    axial_um: Series[float] = pa.Field(coerce=True)
    lateral_um: Series[float] = pa.Field(coerce=True)


class ModelHistologyPlanned(BaseChannelFeatures):
    x_target: Series[float] = pa.Field(coerce=True)
    y_target: Series[float] = pa.Field(coerce=True)
    z_target: Series[float] = pa.Field(coerce=True)


class ModelHistologyResolved(BaseChannelFeatures):
    x: Series[float] = pa.Field(coerce=True)
    y: Series[float] = pa.Field(coerce=True)
    z: Series[float] = pa.Field(coerce=True)
    atlas_id: Series[int] = pa.Field(coerce=True)
    acronym: Series[str] = pa.Field(coerce=True)


class ModelRawFeatures(
    ModelSpikeFeatures,
    ModelCsdFeatures,
    ModelApFeatures,
    ModelLfFeatures,
    ModelChannelLayout,
):
    pass


def voltage_features_set(features_list=FEATURES_LIST):
    """
    THis function returns the list of features columns names depending on their provenance.
    This is useful to select the columns for training
    :param features_list: optional, defaults to ['raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms', 'micro-manipulator'], or 'all'
    :return:
    """
    if features_list == "all":
        features_list = [
            "raw_ap",
            "raw_lf",
            "raw_lf_csd",
            "waveforms",
            "micro-manipulator",
        ]
    # the looping preserves the order of the features groups in the list
    x_list = []
    for feature_group in features_list:
        match feature_group:
            case "raw_ap":
                x_list += sorted(
                    list(
                        set(ModelApFeatures.to_schema().columns.keys())
                        - set(["channel"])
                    )
                )
            case "raw_lf":
                x_list += sorted(
                    list(
                        set(ModelLfFeatures.to_schema().columns.keys())
                        - set(["channel"])
                    )
                )
            case "raw_lf_csd":
                x_list += sorted(
                    list(
                        set(ModelCsdFeatures.to_schema().columns.keys())
                        - set(["channel"])
                    )
                )
            case "waveforms":
                x_list += sorted(
                    list(
                        set(ModelSpikeFeatures.to_schema().columns.keys())
                        - set(["channel"])
                    )
                )
            case "micro-manipulator":
                x_list += sorted(
                    list(
                        set(ModelHistologyPlanned.to_schema().columns.keys())
                        - set(["channel"])
                    )
                )
    return x_list


def _get_power_in_band(fscale, period, band):
    band = np.array(band)
    # weight the frequencies
    fweights = ibldsp.utils.fcn_cosine([-np.diff(band), 0])(
        -abs(fscale - np.mean(band))
    )
    p = 10 * np.log10(
        np.sum(period * fweights / np.sum(fweights), axis=-1)
    )  # dB relative to v/sqrt(Hz)
    return p


def lf(data, fs, bands=None):
    """
    Computes the LF features from a numpy array
    :param data: numpy array with the data (channels, samples)
    :param fs: sampling interval (Hz)
    :param bands: dictionary with the bands to compute (default: BANDS constant)
    :return: pandas dataframe with the columns ['channel', 'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta',
       'psd_gamma', 'psd_lfp']
    """
    bands = BANDS if bands is None else bands
    nc = data.shape[0]  # number of channels
    fscale, period = scipy.signal.periodogram(data, fs)
    df_lf = pd.DataFrame()
    df_lf["channel"] = np.arange(nc)
    df_lf["rms_lf"] = ibldsp.utils.rms(data, axis=-1)
    for b in BANDS:
        df_lf[f"psd_{b}"] = _get_power_in_band(fscale, period, bands[b])
    ModelLfFeatures.validate(df_lf)
    return df_lf


def csd(data, fs, geometry, bands=None, decimate=10):
    """
    Computes the CSD features from a numpy array
    :param data: numpy array with the data (channels, samples)
    :param fs: sampling interval (Hz)
    :param geometry: dictionary with the geometry (x, y) of the channels
    :param bands: dictionary with the bands to compute (default: BANDS constant)
    :params decimate: decimation factor for the CSD calculation (default: 10)
    :return: pandas dataframe with the columns ['channel', 'rms_lf_csd', 'psd_delta_csd', 'psd_theta_csd', 'psd_alpha_csd',
       'psd_beta_csd', 'psd_gamma_csd', 'psd_lfp_csd']
    """
    data_rs = scipy.signal.decimate(data, decimate, axis=1, ftype="fir")
    data_rs = ibldsp.cadzow.cadzow_np1(data_rs, rank=2, fs=fs, niter=1, fmax=90)
    data_rs = ibldsp.voltage.current_source_density(data_rs, h=geometry)
    df_csd = lf(data_rs, fs, bands=bands)
    df_csd = df_csd.rename(
        columns={c: f"{c}_csd" for c in df_csd.columns if c not in ["channel"]}
    )
    ModelCsdFeatures.validate(df_csd)
    return df_csd


def ap(data, geometry=None, channel_labels=None):
    """
    Computes the LF features from a numpy array
    :param data: numpy array with the AP band data (channels, samples)
    :return: pandas dataframe with the columns ['channel', 'rms_ap']
    """
    assert geometry is not None, "Geometry is required for AP band computation"
    assert channel_labels is not None, "Channel labels are required"
    df_ap = pd.DataFrame()
    nc = data.shape[0]  # number of channels
    df_ap["channel"] = np.arange(nc)
    df_ap["rms_ap"] = ibldsp.utils.rms(data, axis=-1)
    df_ap["cor_ratio"] = xcor_acor_ratio(data, geometry=geometry)
    df_ap["channel_labels"] = channel_labels
    ModelApFeatures.validate(df_ap)
    return df_ap


def dart_subtraction_numpy(data, fs, geometry, **params):
    """
    :param data: [nc, ns] numpy array of voltage traces, z-scored or not
    :return:
    """

    params = DartParameters() if params is None else DartParameters(**params)
    # pip install ephys-atlas[gpu]
    import dartsort  # 04a23714d77f28c1bbf3351ed9e21601395d1bca is a working commit
    import spikeinterface.core as sc
    import h5py

    dart_xy = np.c_[geometry["x"], geometry["y"]]

    zdata = data / ibldsp.utils.rms(data, axis=-1)[:, np.newaxis]
    rec_np = sc.NumpyRecording(zdata.T, sampling_frequency=fs)
    rec_np.set_dummy_probe_from_locations(dart_xy)

    # I'm making configuration objects here that don't require fitting any
    # models. For instance, if you have do_tpca_denoise=True, dartsort will try
    # to load up many waveforms from the recording to fit a PCA, but the recording
    # is too short for that and it takes time.
    denoising_cfg = dartsort.FeaturizationConfig(
        denoise_only=True,
        do_tpca_denoise=False,
        localization_radius=params.localization_radius,
    )
    subtraction_cfg = dartsort.SubtractionConfig(
        subtraction_denoising_config=denoising_cfg,
        extract_radius=params.localization_radius,
        chunk_length_samples=params.chunk_length_samples,
    )
    # this determines what features you get out at the end
    # the nn localizer is another model which needs to be fitted, so turning
    # that off is good
    featurization_cfg = dartsort.FeaturizationConfig(
        nn_localization=False,
        save_output_waveforms=True,  # save final nn denoised waveforms
        save_input_waveforms=True,  # save collision-cleaned, but not NN-denoised, waveforms
        localization_radius=params.localization_radius,
    )

    # we make sure that each runs get a different temp folder
    temp_suffix = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for _ in range(8)]
    )

    # Ensure scratch directory exists
    scratch_dir = _setup_scratch_directory(params.scratch_dir)

    detected_spikes, h5_filename = dartsort.subtract(
        rec_np,
        temp_folder := scratch_dir.joinpath(f"dart_{temp_suffix}"),
        featurization_config=featurization_cfg,
        subtraction_config=subtraction_cfg,
        n_jobs=1,
        # if you set n_jobs=1, this will initialize CUDA in a separate process, so GPU memory will be freed. with n_jobs=0, the cuda runtime will be initialized in the main process
        show_progress=True,
    )

    df_spikes = pd.DataFrame(
        {
            "sample": detected_spikes.times_samples,
            "channel": detected_spikes.channels,
            "ptp": detected_spikes.denoised_ptp_amplitudes,
            "xloc": detected_spikes.point_source_localizations[:, 0],  # xyza
            "yloc": detected_spikes.point_source_localizations[:, 1],  # xyza
            "zloc": detected_spikes.point_source_localizations[:, 2],  # xyza
            "alpha": detected_spikes.point_source_localizations[:, 3],  # xyza
        }
    )

    h5file = h5py.File(h5_filename)
    d_waveforms = {  # n_spikes, nsw, ncw
        "raw": np.array(h5file["collisioncleaned_waveforms"]),
        "denoised": np.array(h5file["denoised_waveforms"]),
        "channel_index": np.array(h5file["channel_index"]),
    }
    shutil.rmtree(temp_folder)
    return df_spikes, d_waveforms


def spikes(
    data, fs: int, geometry: dict, return_waveforms=True, scratch_dir=None, **params
):
    """
    :param data:
    :param fs:
    :param geometry:
    :param params:
    :return:
    """
    params = DartParameters() if params is None else DartParameters(**params)
    logger.info("Starting spike detection")
    # Update params with scratch_dir if provided
    if scratch_dir is not None:
        params.scratch_dir = scratch_dir
    df_spikes_, d_waveforms = dart_subtraction_numpy(data, fs, geometry, params=params)
    logger.info("Spike detection completed")
    df_waveforms = ibldsp.waveforms.compute_spike_features(d_waveforms["denoised"])
    df_spikes = df_spikes_.merge(df_waveforms, left_index=True, right_index=True)
    # we cast the float32 values as float64
    df_spikes[df_spikes.select_dtypes(np.float32).columns] = df_spikes.select_dtypes(
        np.float32
    ).astype(np.float64)
    fcn_mean_time = lambda x: np.mean((x - params.trough_offset)) / fs  # NOQA
    # aggregation by channel of the spikes / waveforms features
    df_spiking = (
        df_spikes.groupby("channel")
        .agg(
            alpha_mean=pd.NamedAgg(column="alpha", aggfunc="mean"),
            alpha_std=pd.NamedAgg(column="alpha", aggfunc=lambda x: np.std(x, ddof=0)),
            spike_count=pd.NamedAgg(column="alpha", aggfunc="count"),
            peak_time_secs=pd.NamedAgg(column="peak_time_idx", aggfunc=fcn_mean_time),
            peak_val=pd.NamedAgg(column="peak_val", aggfunc="mean"),
            trough_time_secs=pd.NamedAgg(
                column="trough_time_idx", aggfunc=fcn_mean_time
            ),
            trough_val=pd.NamedAgg(column="trough_val", aggfunc="mean"),
            tip_time_secs=pd.NamedAgg(column="tip_time_idx", aggfunc=fcn_mean_time),
            tip_val=pd.NamedAgg(column="tip_val", aggfunc="mean"),
            recovery_time_secs=pd.NamedAgg(
                column="recovery_time_idx", aggfunc=fcn_mean_time
            ),
            depolarisation_slope=pd.NamedAgg(
                column="depolarisation_slope", aggfunc="mean"
            ),
            repolarisation_slope=pd.NamedAgg(
                column="repolarisation_slope", aggfunc="mean"
            ),
            recovery_slope=pd.NamedAgg(column="recovery_slope", aggfunc="mean"),
            polarity=pd.NamedAgg(
                column="invert_sign_peak", aggfunc=lambda x: -x.mean()
            ),
        )
        .reset_index()
    )
    ModelSpikeFeatures.validate(df_spiking)
    if return_waveforms:
        return df_spiking, d_waveforms | {"df_spikes": df_spikes}
    else:
        return df_spiking


def xcor_acor_ratio(v: np.ndarray, geometry: dict, n_neighbor: int = 3) -> np.ndarray:
    """
    Cross corr over auto-correlation ratio
    :param v: voltage array for AP band (nc, ns)
    :param geometry: geometry dict with 'x' and 'y' arrays for the electrode positions (nc, )
    :param n_diags: number of n
    :return: np.ndarray of size (nc, )
    """
    # %% on calcule la matrice de covariance
    n_mirror = 12
    n_diags = 8
    nc = v.shape[0]
    i_mirror = np.r_[
        np.arange(n_mirror, 0, -1),
        np.arange(nc),
        np.arange(nc - 2, nc - n_mirror - 2, -1),
    ]
    ncm = i_mirror.size
    i0, i1 = np.meshgrid(i_mirror, i_mirror)
    dxy = (
        geometry["x"][i0]
        - geometry["x"][i1]
        + (geometry["y"][i0] - geometry["y"][i1]) * 1j
    )
    cov = v[i_mirror] @ v[i_mirror].T

    # Here for each channel we extract the covariances of neighbouring channels
    diags = np.zeros((n_diags * 2 + 1, ncm))
    diags_xy = np.zeros_like(diags, dtype=np.complex64)
    for i, di in enumerate(np.arange(-n_diags, n_diags + 1)):
        if di == 0:
            diags[i, :] = np.diag(cov)
            continue
        if di < 0:
            ic = np.s_[-di:]
        elif di > 0:
            ic = np.s_[:-di]
        d = np.diag(cov, di).copy()
        d[np.diag(i0, di) == np.diag(i1, di)] = np.nan
        diags[i, ic] = d
        diags_xy[i, ic] = np.diag(dxy, di)

    cor_ratio = np.nanmean(diags, axis=0) / diags[n_diags]
    # # the metric is the ratio of cross-correlation of the neighouring channels over to the auto-correlation
    # fig, ax = plt.subplots(2, 1, sharex=True)
    # ax[0].matshow(diags / diags[n_diags], aspect='auto', extent=[cscale[0], cscale[-1], -n_diags, n_diags])
    # ax[1].plot(cscale, cor_ratio)
    return cor_ratio[n_mirror:-n_mirror]


def denoise_shank(
    feature: np.ndarray, xy: np.ndarray, labels: np.ndarray | None = None, fac: int = 1
) -> np.ndarray:
    """
    Denoise the AP feature using a maximum variation filter. Interpolates the feature in a square grid,
    performs the filtering, and then interpolates back to the original grid.

    :param feature: AP feature to denoise (nc)
    :param xy: Coordinates of the AP feature (nc, 2)
    :param labels: Channels quality annotation (nc), if different than 0, channel is discarded and interpolated. Set to None for no annotation.
    :param fac: Factor for the TV denoising in median deviation units(default 1)
    :return: Denoised AP (nc)
    """
    isvalid = ~np.isnan(feature)
    xyu = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    x, y = np.meshgrid(*xyu)
    xyi = np.c_[x.flatten(), y.flatten()]
    feature_image = scipy.interpolate.griddata(
        xy[isvalid, :], feature[isvalid], xyi
    ).reshape(x.shape)
    feature_image_nearest = scipy.interpolate.griddata(
        xy[isvalid, :], feature[isvalid], xyi, method="nearest"
    ).reshape(x.shape)
    feature_image[np.isnan(feature_image)] = feature_image_nearest[
        np.isnan(feature_image)
    ]
    feature_image_dn = skimage.restoration.denoise_tv_chambolle(
        feature_image, weight=np.median(np.abs(feature_image)) * fac
    )
    denoised_feature = scipy.interpolate.RegularGridInterpolator(
        xyu, feature_image_dn.T, bounds_error=False
    )(xy)
    return denoised_feature


class _EphysTransformerInterface(
    ABC,
    sklearn.base.OneToOneFeatureMixin,
    sklearn.base.TransformerMixin,
    sklearn.base.BaseEstimator,
):
    def __init__(self):
        super().__init__()
        self.set_output(transform="pandas")

    def _get_feature_names(self, X: pd.DataFrame = None) -> List[str]:
        # the features to work with are the intersection of the dataframe columns and the defined schemas
        return list(
            set(voltage_features_set(["raw_ap", "raw_lf", "raw_lf_csd", "waveforms"]))
            & set(X.columns)
        )

    def validate_X(self, X: pd.DataFrame) -> None:
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"

    def fit_transform(self, X: pd.DataFrame = None, y=None):
        self.fit(X)
        return self.transform(X)


class EphysTransformer(_EphysTransformerInterface):
    def __init__(self):
        super().__init__()
        self.set_output(transform="pandas")

    def fit(self, X: pd.DataFrame = None, y=None):
        self.validate_X(X)
        raw_features_schema = ModelRawFeatures.to_schema()
        self.fcn_transform_ = {}
        for feature_name in self._get_feature_names(X):
            if (
                metadata := raw_features_schema.columns[feature_name].metadata
            ) is not None:
                self.fcn_transform_[feature_name] = metadata["transform"]

    def transform(self, X: pd.DataFrame, y=None):
        self.validate_X(X)
        xt = X.copy()
        for column_name in X.columns:
            if column_name in self.fcn_transform_:
                xt.loc[:, column_name] = self.fcn_transform_[column_name](
                    X[column_name].to_numpy()
                )
        return xt


class EphysDenoiser(_EphysTransformerInterface):
    def __init__(self, fac=1, channel_labels=None):
        super().__init__()
        self.fac = fac
        self.channel_labels = channel_labels

    def _get_channel_labels(self, X: pd.DataFrame = None) -> np.ndarray:
        if self.channel_labels is None:
            if "channel_labels" in X.columns:
                self.channel_labels = X["channel_labels"].to_numpy()
            else:
                self.channel_labels = np.zeros(X.shape[0], dtype=int)
        return self.channel_labels

    def transform(self, X: pd.DataFrame, y=None):
        self.validate_X(X)
        channel_labels = self._get_channel_labels(X)
        ns = X.shape[0]
        for feature_name in self._get_feature_names(X):
            if (
                feature_name == "channel_labels"
            ):  # we do not want to apply any denoising to this feature
                continue
            fval = np.copy(X[feature_name].to_numpy())
            fval[channel_labels != 0] = np.nan
            logger.info(f"Calculation for feature_name = {feature_name}")
            denoised_values = denoise_shank(
                feature=fval,
                xy=X[["lateral_um", "axial_um"]].values,
                fac=self.fac,
            )
            # Check that the denoised values have the expected length
            if len(denoised_values) != ns:
                raise ValueError(
                    f"Length mismatch for feature '{feature_name}': "
                    f"denoised values length ({len(denoised_values)}) != "
                    f"DataFrame length ({ns})"
                )
            # Let pandas determine the appropriate dtype for the new values
            X.loc[:, feature_name] = denoised_values
        return X

    def fit(self, X: pd.DataFrame = None, y=None):
        return


def denoise_dataframe(df_pid, fac=1, channel_labels=None):
    """
    Applies total variation filter denoising to the features of a single probe insertion dataframe.

    This function processes electrophysiological features by applying a total variation filter
    to denoise them. If a transformation is defined in the metadata schema for a feature,
    it will be applied before denoising. Channels marked with non-zero labels are treated
    as invalid and their values are interpolated from neighboring channels.

    Parameters
    ----------
    df_pid : pandas.DataFrame
        DataFrame containing probe insertion data with features to denoise.
        Must contain 'lateral_um', 'axial_um', and 'labels' columns.
    fac : float, default=1
        Factor for the TV denoising in median deviation units. Higher values
        result in stronger denoising.

    Returns
    -------
    pandas.DataFrame
        A new dataframe with the same structure as the input, but with denoised feature values.
        Non-feature columns are copied without modification.
    """
    df_transformed = EphysTransformer().fit_transform(df_pid)
    df_denoised = EphysDenoiser(fac=fac, channel_labels=channel_labels).fit_transform(
        df_transformed
    )
    return df_denoised
