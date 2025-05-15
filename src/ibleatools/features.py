import dataclasses
from pathlib import Path
import random
import shutil
import string

import numpy as np
import pandas as pd
import pandera
import pydantic
import scipy.signal
import skimage.restoration

import ibldsp.waveforms
import ibldsp.cadzow
import ibldsp.utils
import ibldsp.voltage

from pandera.typing import Series

from typing_extensions import Annotated
from typing import Dict, Any, Tuple
from .logger_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

floats = Annotated[pandera.Float, pandera.Float32]
BANDS = {
    "delta": [0, 4],
    "theta": [4, 10],
    "alpha": [8, 12],
    "beta": [15, 30],
    "gamma": [30, 90],
    "lfp": [0, 90],
}

FEATURES_LIST = ["raw_ap", "raw_lf", "localisation", "waveforms"]


class DartParameters(pydantic.BaseModel):
    localization_radius: pydantic.PositiveFloat = 150
    chunk_length_samples: pydantic.PositiveInt = 2**15
    trough_offset: pydantic.PositiveInt = (42,)


class BaseChannelFeatures(pandera.DataFrameModel):
    channel: int


class ModelLfFeatures(BaseChannelFeatures):
    rms_lf: Series[float] = pandera.Field(coerce=True)
    psd_delta: Series[float] = pandera.Field(coerce=True)
    psd_theta: Series[float] = pandera.Field(coerce=True)
    psd_alpha: Series[float] = pandera.Field(coerce=True)
    psd_beta: Series[float] = pandera.Field(coerce=True)
    psd_gamma: Series[float] = pandera.Field(coerce=True)
    psd_lfp: Series[float] = pandera.Field(coerce=True)


class ModelCsdFeatures(BaseChannelFeatures):
    rms_lf_csd: Series[float] = pandera.Field(coerce=True)
    psd_delta_csd: Series[float] = pandera.Field(coerce=True)
    psd_theta_csd: Series[float] = pandera.Field(coerce=True)
    psd_alpha_csd: Series[float] = pandera.Field(coerce=True)
    psd_beta_csd: Series[float] = pandera.Field(coerce=True)
    psd_gamma_csd: Series[float] = pandera.Field(coerce=True)
    psd_lfp_csd: Series[float] = pandera.Field(coerce=True)


class ModelApFeatures(BaseChannelFeatures):
    rms_ap: Series[float] = pandera.Field(coerce=True)
    cor_ratio: Series[float] = pandera.Field(coerce=True)


class ModelSpikeFeatures(BaseChannelFeatures):
    alpha_mean: Series[float] = pandera.Field(coerce=True)
    alpha_std: Series[float] = pandera.Field(coerce=True)
    depolarisation_slope: Series[float] = pandera.Field(coerce=True)
    peak_time_secs: Series[float] = pandera.Field(coerce=True)
    peak_val: Series[float] = pandera.Field(coerce=True)
    polarity: Series[float] = pandera.Field(coerce=True)
    recovery_slope: Series[float] = pandera.Field(coerce=True)
    recovery_time_secs: Series[float] = pandera.Field(coerce=True)
    repolarisation_slope: Series[float] = pandera.Field(coerce=True)
    spike_count: int = pandera.Field(coerce=True, metadata={'transform': lambda x: x.astype(float)})
    tip_time_secs: Series[float] = pandera.Field(coerce=True)
    tip_val: Series[float] = pandera.Field(coerce=True)
    trough_time_secs: Series[float] = pandera.Field(coerce=True)
    trough_val: Series[float] = pandera.Field(coerce=True)


class ModelChannelLayout(BaseChannelFeatures):
    axial_um: Series[float] = pandera.Field(coerce=True)
    lateral_um: Series[float] = pandera.Field(coerce=True)
    labels: Series[int] = pandera.Field(coerce=True, nullable=True)


class ModelHistologyPlanned(BaseChannelFeatures):
    x_target: Series[float] = pandera.Field(coerce=True)
    y_target: Series[float] = pandera.Field(coerce=True)
    z_target: Series[float] = pandera.Field(coerce=True)


class ModelHistologyResolved(BaseChannelFeatures):
    x: Series[float] = pandera.Field(coerce=True)
    y: Series[float] = pandera.Field(coerce=True)
    z: Series[float] = pandera.Field(coerce=True)
    atlas_id: Series[int] = pandera.Field(coerce=True)
    acronym: Series[str] = pandera.Field(coerce=True)


class ModelRawFeatures(
    ModelSpikeFeatures, ModelCsdFeatures, ModelApFeatures, ModelLfFeatures, ModelChannelLayout
):
    pass


def voltage_features_set(features_list=FEATURES_LIST):
    """
    THis function returns the list of features columns names depending on their provenance.
    This is useful to select the columns for training
    :param features_list: optional, defaults to ['raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms', 'micro-manipulator'], or 'all'
    :return:
    """
    if features_list == 'all':
        features_list = ['raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms', 'micro-manipulator']
    # the looping preserves the order of the features groups in the list
    x_list = []
    for feature_group in features_list:
        match feature_group:
            case 'raw_ap':
                x_list += list(set(ModelApFeatures.to_schema().columns.keys()) - set(['channel']))
            case "raw_lf":
                x_list += list(set(ModelLfFeatures.to_schema().columns.keys()) - set(['channel']))
            case "raw_lf_csd":
                x_list += list(set(ModelCsdFeatures.to_schema().columns.keys()) - set(['channel']))
            case "waveforms":
                x_list += list(set(ModelSpikeFeatures.to_schema().columns.keys()) - set(['channel']))
            case "micro-manipulator":
                x_list += list(set(ModelHistologyPlanned.to_schema().columns.keys()) - set(['channel']))
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


def ap(data, geometry=None):
    """
    Computes the LF features from a numpy array
    :param data: numpy array with the AP band data (channels, samples)
    :return: pandas dataframe with the columns ['channel', 'rms_ap']
    """
    assert geometry is not None, "Geometry is required for AP band computation"
    df_ap = pd.DataFrame()
    nc = data.shape[0]  # number of channels
    df_ap["channel"] = np.arange(nc)
    df_ap["rms_ap"] = ibldsp.utils.rms(data, axis=-1)
    df_ap["cor_ratio"] = xcor_acor_ratio(data, geometry=geometry)
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
    scratch_dir = Path.home().joinpath("scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
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


def spikes(data, fs: int, geometry: dict, return_waveforms=True, **params):
    """
    :param data:
    :param fs:
    :param geometry:
    :param params:
    :return:
    """
    params = DartParameters() if params is None else DartParameters(**params)
    logger.info(f"Starting spike detection")
    df_spikes_, d_waveforms = dart_subtraction_numpy(data, fs, geometry, params=params)
    logger.info(f"Spike detection completed")
    df_waveforms = ibldsp.waveforms.compute_spike_features(d_waveforms["denoised"])
    df_spikes = df_spikes_.merge(df_waveforms, left_index=True, right_index=True)
    # we cast the float32 values as float64
    df_spikes[df_spikes.select_dtypes(np.float32).columns] = df_spikes.select_dtypes(
        np.float32
    ).astype(np.float64)
    fcn_mean_time = lambda x: np.mean((x - params.trough_offset)) / fs
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


def denoise_shank(feature: np.ndarray, xy: np.ndarray, labels: np.ndarray | None = None, fac: int = 1) -> np.ndarray:
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
    feature_image = scipy.interpolate.griddata(xy[isvalid, :], feature[isvalid], xyi).reshape(x.shape)
    feature_image_nearest = scipy.interpolate.griddata(xy[isvalid, :], feature[isvalid], xyi, method='nearest').reshape(x.shape)
    feature_image[np.isnan(feature_image)] = feature_image_nearest[np.isnan(feature_image)]
    feature_image_dn = skimage.restoration.denoise_tv_chambolle(feature_image, weight=np.median(np.abs(feature_image)) * fac)
    denoised_feature = scipy.interpolate.RegularGridInterpolator(xyu, feature_image_dn.T, bounds_error=False)(xy)
    return denoised_feature


def denoise_dataframe(df_pid, feature_names=None, fac=1):
    """
    Applies total variation filter denoising to the features of a single pid datframe.
    :param df_pid:
    :param feature_names:
    :param fac:
    :return:
    """
    if feature_names is None:
        feature_names = list(set(voltage_features_set(['raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms'])) & set(df_pid.columns))
    df_pid_denoise = df_pid.loc[:, list(set(df_pid.columns) - set(feature_names))].copy()
    raw_features_schema = ModelRawFeatures.to_schema()
    for feature_name in feature_names:
        if (metadata := raw_features_schema.columns[feature_name].metadata) is not None:
            fval = metadata['transform'](np.copy(df_pid[feature_name].to_numpy()))
        else:
            fval = np.copy(df_pid[feature_name].to_numpy())
        fval[df_pid['labels'] != 0] = np.nan
        df_pid_denoise.loc[:, feature_name] = denoise_shank(
            feature=fval,
            xy=df_pid[['lateral_um', 'axial_um']].values,
            fac=fac,
        )
    return df_pid_denoise
