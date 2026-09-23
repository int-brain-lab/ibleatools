"""
Electrophysiological feature extraction and processing module.

This module provides comprehensive tools for extracting, processing, and analyzing
electrophysiological features from neural recordings. It supports multiple backends
for spike detection and feature computation, including Dartsort and SpikeInterface.

The module includes:
- Feature extraction from AP (action potential) and LF (local field potential) bands
- Current source density (CSD) computation
- Spike detection and waveform analysis
- Feature denoising using total variation filters
- Data validation using Pandera schemas
- Transformer classes for scikit-learn compatibility

Classes
-------
DartParameters
    Configuration parameters for Dartsort backend
ChannelDataFrameSchema
    Pandera schema for channel data validation
ModelLfFeatures
    Schema for local field potential features
ModelCsdFeatures
    Schema for current source density features
ModelApFeatures
    Schema for action potential features
ModelSpikeFeatures
    Schema for spike waveform features
ModelSpikeShapeFeatures
    Sparse, neuroscientist-facing remapping of ModelSpikeFeatures
ModelChannelLayout
    Schema for channel layout information
ModelHistologyPlanned
    Schema for planned histology coordinates
ModelHistologyResolved
    Schema for resolved histology coordinates
ModelRawFeatures
    Combined schema for all raw features
EphysTransformer
    Transformer for applying feature transformations
EphysDenoiser
    Transformer for denoising electrophysiological features

Functions
---------
_setup_scratch_directory
    Set up scratch directory with fallback logic
voltage_features_set
    Get list of feature column names by provenance
lf
    Compute LF features from numpy array
csd
    Compute CSD features from numpy array
ap
    Compute AP features from numpy array
dart_subtraction_numpy
    Perform spike detection using Dartsort
spikes
    Spike detection and feature extraction with multiple backend support
remap_waveform_shape_features
    Remap ModelSpikeFeatures' 18 raw waveform columns onto a sparser set
remap_waveform_shape_features_volume
    Same remapping, for a brainwide encoding volume instead of a dataframe
xcor_acor_ratio
    Compute cross-correlation over auto-correlation ratio
denoise_shank
    Denoise AP features using total variation filter
denoise_dataframe
    Apply total variation filter denoising to features

Constants
---------
__features_version__ : str
    Version of the feature extractor code
BANDS : dict
    Frequency bands for spectral analysis
FEATURES_LIST : list
    List of available feature types

Examples
--------
>>> from ephysatlas.features import lf, csd, ap
>>> import numpy as np
>>>
>>> # Generate sample data
>>> data = np.random.randn(64, 30000)  # 64 channels, 30k samples
>>> fs = 30000  # 30 kHz sampling rate
>>>
>>> # Compute LF features
>>> lf_features = lf(data, fs)
>>>
>>> # Compute CSD features
>>> geometry = {'x': np.arange(64), 'y': np.zeros(64)}
>>> csd_features = csd(data, fs, geometry)
>>>
>>> # Compute AP features
>>> ap_data = np.random.randn(64, 10000)
>>> ap_features = ap(ap_data, geometry, np.zeros(64))

Notes
-----
This module requires several dependencies including numpy, pandas, scipy,
scikit-image, and optionally dartsort for advanced spike detection.
GPU acceleration is supported through the dartsort backend.

See Also
--------
ibldsp.waveforms : Waveform processing utilities
ibldsp.cadzow : Cadzow denoising algorithms
ibldsp.voltage : Voltage processing utilities
"""

from abc import ABC
import logging
from pathlib import Path
import random
import shutil
import string
import tempfile
from typing_extensions import Annotated, List
from typing import Optional

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pydantic
import scipy.signal
import skimage.restoration
import sklearn.base

import ibldsp.waveforms
import ibldsp.cadzow
import ibldsp.utils
import ibldsp.voltage

# Set up logger
logger = logging.getLogger(__name__)

__features_version__ = (
    "2026.09.20"  # this is the version of this feature extractor code
)


# TODO - Scratch_dir path is not working as expected. Even if I pass the scratch_dir argument in the main compute_features function, here I am gettig log from Path("/scratch/dartsort/")
def _setup_scratch_directory(scratch_dir=None):
    """Set up scratch directory with fallback logic.

    Args:
        scratch_dir (Path or str, optional): Preferred scratch directory path.
            If None, will try system defaults.

    Returns:
        Path: Path to the created scratch directory.

    Note:
        This function first tries to use the SDSC scratch directory (/scratch/dartsort/),
        and falls back to the system temp directory if that fails.
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
    """Get the minimum value for a given feature.

    Args:
        feature_name (str): Name of the feature.

    Returns:
        float: Minimum value for the feature.

    Note:
        This function is currently a placeholder and needs implementation.
    """
    # TODO: Implement feature minimum value retrieval
    pass


class DartParameters(pydantic.BaseModel):
    """Configuration parameters for Dartsort backend.

    This class defines the parameters used for spike detection and feature
    extraction using the Dartsort algorithm.

    Attributes:
        localization_radius (float): Radius in micrometers for spike localization.
            Defaults to 150.
        chunk_length_samples (int): Length of data chunks in samples for processing.
            Defaults to 2^15 (32768).
        trough_offset (int): Offset in samples from spike peak to trough.
            Defaults to 42.
        scratch_dir (Path or str, optional): Scratch directory for temporary files.
            If None, will use system defaults.
        n_jobs (int): DARTsort worker count used to build its
            ``ComputationConfig``. ``0`` runs in the main process; positive values
            select that many workers. Defaults to 0.
        detection_threshold (float): Peak detection threshold, in units of channel
            RMS. Defaults to 4.0.
        spatial_dedup_radius_um (float): Radius in micrometres within which only the
            largest simultaneous peak is kept. Defaults to 150.0.
        positive_temporal_dedup_radius_samples (int): Samples around a trough in
            which positive peaks are suppressed. Defaults to 7.
        residnorm_decrease_threshold (float): Minimum residual-norm decrease for a
            detected spike to be subtracted. Defaults to 3.162 (sqrt(10)).
    """

    localization_radius: pydantic.PositiveFloat = 150
    chunk_length_samples: pydantic.PositiveInt = 2**15
    trough_offset: pydantic.PositiveInt = 42
    detection_threshold: pydantic.PositiveFloat = 4.0
    spatial_dedup_radius_um: pydantic.PositiveFloat = 150.0
    positive_temporal_dedup_radius_samples: pydantic.PositiveInt = 7
    residnorm_decrease_threshold: pydantic.PositiveFloat = 3.162
    scratch_dir: Path | str | None = pydantic.Field(
        default=None,
        description="Scratch directory for temporary files. If None, will use system defaults.",
    )
    n_jobs: pydantic.NonNegativeInt = 0


class ChannelDataFrameSchema(pa.DataFrameModel):
    """Pandera schema for channel data validation.

    This schema defines the structure and validation rules for channel
    information including spatial coordinates and anatomical labels.

    Note:
        ``x``/``y``/``z`` are in **metres**, while ``axial_um``/``lateral_um`` are in
        micrometres. The two length units coexist in one table: the coordinates follow the
        IBL/iblatlas convention (metres from Bregma), whereas the probe
        geometry is expressed in the micrometres.

    Attributes:
        pid (Series[str]): Probe insertion ID.
        channel (Series[int]): Channel index.
        x (Series[float]): X-coordinate in metres from Bregma (IBL coordinates space).
        y (Series[float]): Y-coordinate in metres from Bregma (IBL coordinates space).
        z (Series[float]): Z-coordinate in metres from Bregma (IBL coordinates space).
        axial_um (Series[float]): Axial distance in micrometers.
        lateral_um (Series[float]): Lateral distance in micrometers.
        acronym (Series[str]): Brain region acronym.
        atlas_id (Series[int]): Atlas region identifier.
    """

    pid: str = pa.Field(
        coerce=True, description="Probe insertion ID", metadata={"raw_unit": "N/A"}
    )
    channel: int = pa.Field(
        coerce=True, description="Channel index", metadata={"raw_unit": "index"}
    )
    x: float = pa.Field(
        coerce=True,
        description="x-position in metres from Bregma (in IBL coordinates space)",
        metadata={"raw_unit": "m"},
    )
    y: float = pa.Field(
        coerce=True,
        description="y-position in metres from Bregma (in IBL coordinates space)",
        metadata={"raw_unit": "m"},
    )
    z: float = pa.Field(
        coerce=True,
        description="z-position in metres from Bregma (in IBL coordinates space)",
        metadata={"raw_unit": "m"},
    )
    axial_um: float = pa.Field(
        coerce=True,
        description="Distance along the probe length (depth)",
        metadata={"raw_unit": "um"},
    )
    lateral_um: float = pa.Field(
        coerce=True,
        description="Distance along the probe width",
        metadata={"raw_unit": "um"},
    )
    acronym: str = pa.Field(
        description="Brain region acronym in the Allen mapping",
        metadata={"raw_unit": "N/A"},
    )
    atlas_id: int = pa.Field(
        description="Atlas region identifier in Allen mapping",
        metadata={"raw_unit": "N/A"},
    )
    distance_to_tip_um: Optional[float] = pa.Field(
        coerce=True,
        nullable=True,
        description="Distance from the electrode to the tip of the probe in micrometers",
        metadata={"raw_unit": "um"},
    )


class BaseChannelFeatures(pa.DataFrameModel):
    """Base class for channel-based feature schemas.

    This is an abstract base class that provides the foundation for
    all channel-based feature validation schemas.

    Note:
        The channel field is expected to be an index in derived classes.
    """

    pass  # channel: Index[int] = pa.Field(check_name=True)


class ModelLfFeatures(BaseChannelFeatures):
    """Schema for local field potential features.

    This schema defines the structure and validation rules for local field
    potential (LFP) features including RMS values and power spectral density
    across different frequency bands.

    Attributes:
        rms_lf (Series[float]): Root mean square of LFP signal in dB.
        psd_delta (Series[float]): Power spectral density in delta band (0-4 Hz).
        psd_theta (Series[float]): Power spectral density in theta band (4-10 Hz).
        psd_alpha (Series[float]): Power spectral density in alpha band (8-12 Hz).
        psd_beta (Series[float]): Power spectral density in beta band (15-30 Hz).
        psd_gamma (Series[float]): Power spectral density in gamma band (30-90 Hz).
        psd_lfp (Series[float]): Power spectral density in full LFP band (0-90 Hz).
    """

    rms_lf: float = pa.Field(
        coerce=True,
        description="Root mean square of LFP signal in V. The value is transformed to dB using 20 * np.log10(x)",
        metadata={
            "raw_unit": "V",
            "transformed_unit": "dB rel. V",
            "transform": lambda x: 20 * np.log10(x),
        },
    )
    psd_lfp: float = pa.Field(
        coerce=True,
        description="Power in the band 0 - 90 Hz in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_delta: float = pa.Field(
        coerce=True,
        description="Power in the band 0 - 4 Hz in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_theta: float = pa.Field(
        coerce=True,
        description="Power in the band 4 - 10 Hz in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_alpha: float = pa.Field(
        coerce=True,
        description="Power in the band 8 - 12 Hz in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_beta: float = pa.Field(
        coerce=True,
        description="Power in the band 15 - 30 Hz in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_gamma: float = pa.Field(
        coerce=True,
        description="Power in the band 30 - 90 Hz in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_residual_lfp: Optional[float] = pa.Field(
        description="Power in the band 0 - 90 Hz in decibels relative to V ** 2 / Hz after removing the linear fit of the psd decay in log-log space",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
        nullable=True,
    )
    psd_residual_delta: Optional[float] = pa.Field(
        description="Power in the band 0 - 4 Hz in decibels relative to V ** 2 / Hz after removing the linear fit of the psd decay in log-log space",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
        nullable=True,
    )
    psd_residual_theta: Optional[float] = pa.Field(
        description="Power in the band 4 - 10 Hz in decibels relative to V ** 2 / Hz after removing the linear fit of the psd decay in log-log space",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
        nullable=True,
    )
    psd_residual_alpha: Optional[float] = pa.Field(
        description="Power in the band 8 - 12 Hz in decibels relative to V ** 2 / Hz after removing the linear fit of the psd decay in log-log space",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
        nullable=True,
    )
    psd_residual_beta: Optional[float] = pa.Field(
        description="Power in the band 15 - 30 Hz in decibels relative to V ** 2 / Hz after removing the linear fit of the psd decay in log-log space",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
        nullable=True,
    )
    psd_residual_gamma: Optional[float] = pa.Field(
        description="Power in the band 30 - 90 Hz in decibels relative to V ** 2 / Hz after removing the linear fit of the psd decay in log-log space",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
        nullable=True,
    )
    aperiodic_offset: Optional[float] = pa.Field(
        nullable=True,
        description="Y-intercept for the fit of the psd decay in log-log space",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    aperiodic_exponent: Optional[float] = pa.Field(
        nullable=True,
        description="Slope for the fit of the psd decay in log-log space",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    decay_fit_error: Optional[float] = pa.Field(
        nullable=True,
        description="RMS error of the fit of the psd decay in log-log space",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    decay_fit_r_squared: Optional[float] = pa.Field(
        nullable=True,
        description="R-squared value of the fit of the psd decay in log-log space",
        metadata={"raw_unit": "dimensionless"},
    )
    decay_n_peaks: Optional[float] = pa.Field(
        coerce=True,
        nullable=True,
        description="Number of peaks detected in the residual plot after removing the linear fit of the psd decay in log-log space",
        metadata={"raw_unit": "count"},
    )
    rms_lf_no_car: Optional[float] = pa.Field(
        coerce=True,
        nullable=True,
        description="Root mean square of LFP signal in V without common-average reference (no CAR). The value is transformed to dB using 20 * np.log10(x)",
        metadata={
            "raw_unit": "V",
            "transformed_unit": "dB rel. V",
            "transform": lambda x: 20 * np.log10(x),
        },
    )


class ModelCsdFeatures(BaseChannelFeatures):
    """Schema for current source density features.

    This schema defines the structure and validation rules for current source
    density (CSD) features including RMS values and power spectral density
    across different frequency bands.

    Attributes:
        rms_lf_csd (Series[float]): Root mean square of CSD signal in dB.
        psd_delta_csd (Series[float]): CSD power spectral density in delta band (0-4 Hz).
        psd_theta_csd (Series[float]): CSD power spectral density in theta band (4-10 Hz).
        psd_alpha_csd (Series[float]): CSD power spectral density in alpha band (8-12 Hz).
        psd_beta_csd (Series[float]): CSD power spectral density in beta band (15-30 Hz).
        psd_gamma_csd (Series[float]): CSD power spectral density in gamma band (30-90 Hz).
        psd_lfp_csd (Series[float]): CSD power spectral density in full LFP band (0-90 Hz).
    """

    rms_lf_csd: float = pa.Field(
        coerce=True,
        description="Root mean square of CSD signal in V. The value is transformed to dB using 20 * np.log10(x)",
        metadata={
            "raw_unit": "V",
            "transformed_unit": "dB rel. V",
            "transform": lambda x: 20 * np.log10(x),
        },
    )
    psd_delta_csd: float = pa.Field(
        coerce=True,
        description="Power in the band 0 - 4 Hz after current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_theta_csd: float = pa.Field(
        coerce=True,
        description="Power in the band 4 - 10 Hz after current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_alpha_csd: float = pa.Field(
        coerce=True,
        description="Power in the band 8 - 12 Hz after current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_beta_csd: float = pa.Field(
        coerce=True,
        description="Power in the band 15 - 30 Hz after current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_gamma_csd: float = pa.Field(
        coerce=True,
        description="Power in the band 30 - 90 Hz after current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )
    psd_lfp_csd: float = pa.Field(
        coerce=True,
        description="Power in the band 0 - 90 Hz after current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )

    rms_lf_csd_diff1: Optional[float] = pa.Field(
        coerce=True,
        description="Root mean square of Diff1 CSD signal in V. The value is transformed to dB using 20 * np.log10(x)",
        metadata={
            "raw_unit": "V",
            "transformed_unit": "dB rel. V",
            "transform": lambda x: 20 * np.log10(x),
        },
    )

    psd_delta_csd_diff1: Optional[float] = pa.Field(
        coerce=True,
        description="Power in the band 0 - 4 Hz after Diff1 current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )

    psd_theta_csd_diff1: Optional[float] = pa.Field(
        coerce=True,
        description="Power in the band 4 - 10 Hz after Diff1 current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )

    psd_alpha_csd_diff1: Optional[float] = pa.Field(
        coerce=True,
        description="Power in the band 8 - 12 Hz after Diff1 current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )

    psd_beta_csd_diff1: Optional[float] = pa.Field(
        coerce=True,
        description="Power in the band 15 - 30 Hz after Diff1 current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )

    psd_gamma_csd_diff1: Optional[float] = pa.Field(
        coerce=True,
        description="Power in the band 30 - 90 Hz after Diff1 current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )

    psd_lfp_csd_diff1: Optional[float] = pa.Field(
        coerce=True,
        description="Power in the band 0 - 90 Hz after Diff1 current source density estimation in decibels relative to V ** 2 / Hz",
        metadata={"raw_unit": "dB rel. V**2/Hz"},
    )


class ModelApFeatures(BaseChannelFeatures):
    """Schema for action potential features.

    This schema defines the structure and validation rules for action potential
    (AP) features including RMS values and correlation ratios.

    Attributes:
        rms_ap (Series[float]): Root mean square of AP signal in dB.
        cor_ratio (Series[float]): Cross-correlation over auto-correlation ratio.
        channel_labels (Series[int]): Quality labels for channels.
    """

    rms_ap: float = pa.Field(
        coerce=True,
        description="Root mean square of AP signal in V. The value is transformed to dB using 20 * np.log10(x)",
        metadata={
            "raw_unit": "V",
            "transformed_unit": "dB rel. V",
            "transform": lambda x: 20 * np.log10(x),
        },
    )
    cor_ratio: float = pa.Field(
        coerce=True,
        description="Ratio of the median of zero-lag cross-correlations with neighbouring channels over zero-lag autocorrelation",
        metadata={"raw_unit": "dimensionless"},
    )
    channel_labels: int = pa.Field(
        coerce=True,
        description=(
            "Quality labels for channels. 0 means a good channel, values higher than 0 means a bad channel. "
            "1: dead low coherence / amplitude, "
            "2: noisy, "
            "3: outside of the brain"
        ),
        metadata={"raw_unit": ""},
    )


class ModelSpikeSharedFeatures(BaseChannelFeatures):
    """Spike-shape fields shared verbatim by `ModelSpikeFeatures` and
    `ModelSpikeShapeFeatures` (the sparse remap keeps these unchanged).

    Attributes:
        alpha_mean (Series[float]): Mean alpha parameter for spike localization (N/A).
        alpha_std (Series[float]): Standard deviation of alpha parameter (N/A).
        depolarisation_slope (Series[float]): Slope during depolarization phase (V/s).
        polarity (Series[float]): Spike polarity, dimensionless.
        recovery_slope (Series[float]): Slope during recovery phase (V/s).
        repolarisation_slope (Series[float]): Slope during repolarization phase (V/s).
        slowness_s_per_m (Series[float]): Signed axial slowness of the multi-channel
            waveform (s/m).
        slowness_s_per_m_std (Series[float]): Across-spike standard deviation of
            `slowness_s_per_m` on the channel (s/m).
        spatial_spread_um (Series[float]): Amplitude-weighted spatial spread of the
            multi-channel waveform (um).
        spatial_spread_um_std (Series[float]): Across-spike standard deviation of
            `spatial_spread_um` on the channel (um).
        spike_count (Series[float]): Spike rate, spikes/s (log2 transformed).
        tip_val (Series[float]): Tip amplitude value (V).

    Note:
        `tip_val` and the 3 slope columns are computed from waveforms that
        `dart_subtraction_numpy` has already rescaled back to Volts (DARTsort
        z-scores internally by channel RMS for detection only).
        `alpha_mean`/`alpha_std` are DARTsort's localisation amplitude, fit
        jointly across a multichannel snippet whose channels each carry a
        different RMS; there is no single scale factor to recover Volts from
        them, so they stay in DARTsort's native, non-physical units.
        `slowness_s_per_m`/`spatial_spread_um` are also weighted by these
        Volts-scale amplitudes (see `ibldsp.waveforms.compute_slowness`/
        `compute_spatial_spread`), but the values themselves -- and those of
        their `_std` counterparts -- are geometric (seconds per metre /
        micrometres), not amplitudes.
    """

    alpha_mean: float = pa.Field(
        coerce=True,
        description="Average brightness of the spike (output of the spike localisation code)",
        metadata={"raw_unit": "N/A"},
    )
    alpha_std: float = pa.Field(
        coerce=True,
        description="Standard deviation of the brightness of the spike (output of the spike localisation code)",
        metadata={"raw_unit": "N/A"},
    )
    depolarisation_slope: float = pa.Field(
        coerce=True,
        description="Slope of the line crossing the tip and peak points.",
        metadata={"raw_unit": "V/s"},
    )
    polarity: float = pa.Field(
        coerce=True,
        description="Sum of each spike polarity divided by the total number of spikes",
        metadata={"raw_unit": "dimensionless"},
    )
    recovery_slope: float = pa.Field(
        coerce=True,
        description="Slope of the line crossing the trough and recovery points.",
        metadata={"raw_unit": "V/s"},
    )
    repolarisation_slope: float = pa.Field(
        coerce=True,
        description="Slope of the line crossing the peak and trough points.",
        metadata={"raw_unit": "V/s"},
    )
    slowness_s_per_m: Optional[float] = pa.Field(
        coerce=True,
        nullable=True,
        description="Signed slowness (inverse apparent velocity) of the waveform along "
        "the probe axis, from a weighted linear fit of per-channel cross-correlation "
        "pick time vs axial distance from the peak channel. Positive means later pick "
        "times at larger distance from the probe tip (wave moving 'up'); negative means "
        "'down'. Reported as slowness (proportional to the timing difference) rather "
        "than velocity (its inverse) because velocity blows up whenever the fit's "
        "slope is near zero (near-simultaneous arrival across the neighbourhood).",
        metadata={"raw_unit": "s/m"},
    )
    slowness_s_per_m_std: Optional[float] = pa.Field(
        coerce=True,
        nullable=True,
        description="Standard deviation of `slowness_s_per_m` across the spikes detected "
        "on the channel (sample standard deviation, ddof=1). NaN where fewer than two "
        "spikes on the channel have a usable slowness.",
        metadata={"raw_unit": "s/m"},
    )
    spatial_spread_um: Optional[float] = pa.Field(
        coerce=True,
        nullable=True,
        description="Amplitude-weighted mean distance of the multi-channel waveform's "
        "neighbour channels from the peak channel.",
        metadata={"raw_unit": "um"},
    )
    spatial_spread_um_std: Optional[float] = pa.Field(
        coerce=True,
        nullable=True,
        description="Standard deviation of `spatial_spread_um` across the spikes detected "
        "on the channel (sample standard deviation, ddof=1): how consistently spread out "
        "the channel's waveforms are, as opposed to how spread out they are on average. "
        "NaN where fewer than two spikes on the channel have a usable spread.",
        metadata={"raw_unit": "um"},
    )
    spike_count: float = pa.Field(
        coerce=True,
        description="log2 transformed value of mean spike rate (spikes/s, independent of "
        "the snippet duration; the mean is calculated across the snippets by "
        "replacing the null values with 0)",
        metadata={
            "raw_unit": "count/s",
            "transformed_unit": "log2 (count/s)",
            "transform": lambda x: np.where(x == 0, np.nan, np.log2(x.astype(float))),
        },
    )
    tip_val: float = pa.Field(
        coerce=True,
        description="Waveform tip amplitude.",
        metadata={"raw_unit": "V"},
    )


class ModelSpikeFeatures(ModelSpikeSharedFeatures):
    """Schema for spike waveform features.

    This schema defines the structure and validation rules for spike waveform
    features including timing, amplitude, and slope characteristics. Adds
    the raw per-spike-event columns to `ModelSpikeSharedFeatures`.

    Attributes:
        peak_time_secs (Series[float]): Time to peak in seconds.
        peak_val (Series[float]): Peak amplitude value (V).
        recovery_time_secs (Series[float]): Recovery time in seconds.
        tip_time_secs (Series[float]): Time to tip in seconds.
        trough_time_secs (Series[float]): Time to trough in seconds.
        trough_val (Series[float]): Trough amplitude value (V).

    Note:
        `peak_val`/`trough_val` are computed from waveforms that
        `dart_subtraction_numpy` has already rescaled back to Volts; see
        `ModelSpikeSharedFeatures` for `tip_val`/the slopes/`alpha`.
    """

    peak_time_secs: float = pa.Field(coerce=True)
    peak_val: float = pa.Field(coerce=True, metadata={"raw_unit": "V"})
    recovery_time_secs: float = pa.Field(coerce=True)
    tip_time_secs: float = pa.Field(coerce=True)
    trough_time_secs: float = pa.Field(coerce=True)
    trough_val: float = pa.Field(coerce=True, metadata={"raw_unit": "V"})


class ModelSpikeShapeFeatures(ModelSpikeSharedFeatures):
    """Schema for the sparse, neuroscientist-facing spike-shape feature set.

    Output of `remap_waveform_shape_features`. Adds the reparametrised
    columns to `ModelSpikeSharedFeatures`'s unchanged pass-through fields.
    Note this schema's 'peak' is the negative deflection and 'trough' the
    positive rebound after it, opposite common neurophysiology usage, so
    `spike_width_secs` is the classic trough-to-peak width despite the name
    order.

    The 3 inherited slope columns correlate strongly with the
    amplitude/duration columns below (r=0.96-0.99 with algebraic
    reconstructions from them) and add no real extra information, but are
    kept precomputed anyway since they're a commonly-used, handy quantity on
    their own.

    Attributes:
        spike_width_secs (Series[float]): Trough-to-peak spike duration (s).
        predepolarisation_width_secs (Series[float]): Tip-to-peak duration (s).
        spike_amplitude (Series[float]): Trough-to-peak amplitude (V).
        peak_to_trough_ratio_log (Series[float]): Log amplitude ratio (dimensionless).
    """

    spike_width_secs: float = pa.Field(
        coerce=True,
        description="Trough-to-peak spike duration: trough_time_secs minus peak_time_secs.",
        metadata={"raw_unit": "s"},
    )
    predepolarisation_width_secs: float = pa.Field(
        coerce=True,
        description="Tip-to-peak duration: peak_time_secs minus tip_time_secs.",
        metadata={"raw_unit": "s"},
    )
    spike_amplitude: float = pa.Field(
        coerce=True,
        description="Overall spike amplitude: trough_val minus peak_val.",
        metadata={"raw_unit": "V"},
    )
    peak_to_trough_ratio_log: float = pa.Field(
        coerce=True,
        description="log(|peak_val / trough_val|): waveform shape asymmetry, independent of "
        "overall amplitude scale.",
        metadata={"raw_unit": "dimensionless"},
    )


# Shared ModelSpikeSharedFeatures columns that are nullable and were added after some
# on-disk feature datasets were computed (#123 for the means, #127 for their standard
# deviations): remap_waveform_shape_features/_volume treat these as all-NaN when
# altogether absent from the input, rather than raising.
_NULLABLE_SHARED_SPIKE_FEATURES = frozenset(
    {
        "slowness_s_per_m",
        "slowness_s_per_m_std",
        "spatial_spread_um",
        "spatial_spread_um_std",
    }
)


def _remap_waveform_shape_arrays(get):
    """Core of the waveform-shape remapping, agnostic to the container.

    Parameters
    ----------
    get : callable
        ``get(name)`` returns the raw `ModelSpikeFeatures` column/slice for
        `name` (a `pandas.Series` or an `(nx, ny, nz)` array both work: only
        elementwise `+`, `-`, `/`, `np.log`, `np.abs` are used below).

    Returns
    -------
    dict
        Maps each `ModelSpikeShapeFeatures` column name to its array.
    """
    # Keyed in ModelSpikeShapeFeatures.to_schema() column order (inherited
    # ModelSpikeSharedFeatures fields first, then this schema's own).
    return {
        "alpha_mean": get("alpha_mean"),
        "alpha_std": get("alpha_std"),
        "depolarisation_slope": get("depolarisation_slope"),
        "polarity": get("polarity"),
        "recovery_slope": get("recovery_slope"),
        "repolarisation_slope": get("repolarisation_slope"),
        "slowness_s_per_m": get("slowness_s_per_m"),
        "slowness_s_per_m_std": get("slowness_s_per_m_std"),
        "spatial_spread_um": get("spatial_spread_um"),
        "spatial_spread_um_std": get("spatial_spread_um_std"),
        "spike_count": get("spike_count"),
        "tip_val": get("tip_val"),
        "spike_width_secs": get("trough_time_secs") - get("peak_time_secs"),
        "predepolarisation_width_secs": get("peak_time_secs") - get("tip_time_secs"),
        "spike_amplitude": get("trough_val") - get("peak_val"),
        "peak_to_trough_ratio_log": np.log(np.abs(get("peak_val") / get("trough_val"))),
    }


def remap_waveform_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remap `ModelSpikeFeatures`'s 18 raw waveform columns onto the sparser `ModelSpikeShapeFeatures` set.

    The 14 columns the PCA was run over are redundant: it needs only ~7-8 components for 95%
    of their variance. Only ``recovery_time_secs`` (exact constant offset of
    ``trough_time_secs``) and ``peak_val``/``trough_val``/``peak_time_secs``/
    ``trough_time_secs``/``tip_time_secs`` (reparametrised into
    ``spike_amplitude``/``peak_to_trough_ratio_log``/``spike_width_secs``/
    ``predepolarisation_width_secs`` without losing relative information) are
    actually dropped. ``tip_val`` and the 3 slopes
    (``depolarisation_slope``/``repolarisation_slope``/``recovery_slope``)
    are kept unchanged despite correlating with the amplitude/duration
    columns (r=0.79-0.99) -- handy precomputed quantities in their own right,
    even if not strictly independent information. ``slowness_s_per_m``/
    ``spatial_spread_um`` (#123) and their ``_std`` counterparts (#127) are
    geometric, postdate that PCA redundancy analysis and were not part of it,
    and are also kept unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns of `ModelSpikeFeatures` (e.g. the output of
        `spike`, raw or denoised). `slowness_s_per_m`/`spatial_spread_um` (#123)
        and their `_std` counterparts (#127), all nullable, are treated as
        all-NaN if altogether absent, so data saved before they existed can
        still be remapped.

    Returns
    -------
    pandas.DataFrame
        Columns of `ModelSpikeShapeFeatures`, same index as `df`.
    """

    def get(name):
        if name in df.columns:
            return df[name]
        if name in _NULLABLE_SHARED_SPIKE_FEATURES:
            return pd.Series(np.nan, index=df.index, name=name)
        raise KeyError(name)

    out = pd.DataFrame(_remap_waveform_shape_arrays(get), index=df.index)
    return ModelSpikeShapeFeatures.validate(out)


def remap_waveform_shape_features_volume(ephys_atlas_vol, feature_names):
    """Same transform as `remap_waveform_shape_features`, for a `download_encoding_volume` array.

    Values there are unnormalised already, so no de-z-scoring is needed.
    Does not derive `mean_per_feature`/`std_per_feature` for the new
    columns -- those come from the original per-channel dataset, not the
    volume itself.

    Parameters
    ----------
    ephys_atlas_vol : numpy.ndarray
        Shape ``(nx, ny, nz, n_features)``.
    feature_names : numpy.ndarray or sequence of str
        Length ``n_features``, naming `ephys_atlas_vol`'s last axis; must
        include the 18 raw columns of `ModelSpikeFeatures`.

    Returns
    -------
    remapped_vol : numpy.ndarray
        Shape ``(nx, ny, nz, 16)``, dtype float32.
    remapped_feature_names : numpy.ndarray
        Length 16, naming `remapped_vol`'s last axis (`ModelSpikeShapeFeatures`
        column order).

    Raises
    ------
    KeyError
        If a required raw feature name is absent from `feature_names` (except
        `slowness_s_per_m`/`spatial_spread_um`, #123, and their `_std`
        counterparts, #127, all nullable: treated as all-NaN if altogether
        absent, so volumes computed before they existed can still be remapped).
    """
    index_of = {name: i for i, name in enumerate(np.asarray(feature_names))}
    assert len(index_of) == len(feature_names), "duplicate names in feature_names"

    def get(name):
        if name in index_of:
            return ephys_atlas_vol[..., index_of[name]].astype(np.float32)
        if name in _NULLABLE_SHARED_SPIKE_FEATURES:
            return np.full(ephys_atlas_vol.shape[:-1], np.nan, dtype=np.float32)
        raise KeyError(name)

    arrays = _remap_waveform_shape_arrays(get)
    remapped_vol = np.stack(list(arrays.values()), axis=-1)
    remapped_feature_names = np.array(list(arrays.keys()))
    return remapped_vol, remapped_feature_names


class ModelChannelLayout(BaseChannelFeatures):
    """Schema for channel layout information.

    This schema defines the structure and validation rules for channel
    layout features including spatial positioning.

    Attributes:
        axial_um (Series[float]): Axial distance in micrometers.
        lateral_um (Series[float]): Lateral distance in micrometers.
    """

    axial_um: float = pa.Field(
        coerce=True,
        description="Distance along the probe length (depth)",
        metadata={"raw_unit": "um"},
    )
    lateral_um: float = pa.Field(
        coerce=True,
        description="Distance along the probe width",
        metadata={"raw_unit": "um"},
    )


class ModelHistologyPlanned(BaseChannelFeatures):
    """Schema for planned histology coordinates.

    This schema defines the structure and validation rules for planned
    histology coordinates before actual histological analysis. The coordinates come from
    the best trajectory Alyx holds for the insertion -- micro-manipulator, else planned,
    else histology track -- see feature_computation.PROVENANCE_PREFERENCE.

    Note:
        Same coordinate space as :class:`ModelHistologyResolved` (IBL, metres from Bregma);
        Populated by ``feature_computation.add_target_coordinates``, which queries Alyx
        trajectories with ``provenance__lte,50`` and takes the best provenance available
        per ``PROVENANCE_PREFERENCE``: ``Micro-manipulator`` (30), else ``Planned`` (10),
        else ``Histology track`` (50); an insertion with none of the three raises
        ``ValueError``. A micro-manipulator or planned trajectory is converted from the
        needles (in-vivo) atlas into Allen space before being interpolated along the
        track; a histology track is already in Allen space and is interpolated as is.

    Attributes:
        x_target (Series[float]): Target X-coordinate in metres from Bregma (IBL coordinates space).
        y_target (Series[float]): Target Y-coordinate in metres from Bregma (IBL coordinates space).
        z_target (Series[float]): Target Z-coordinate in metres from Bregma (IBL coordinates space).
    """

    x_target: float = pa.Field(
        coerce=True,
        description="Target x-position in metres from Bregma (in IBL coordinates space), from the best available Alyx trajectory (micro-manipulator, else planned, else histology track)",
        metadata={"raw_unit": "m"},
    )
    y_target: float = pa.Field(
        coerce=True,
        description="Target y-position in metres from Bregma (in IBL coordinates space), from the best available Alyx trajectory (micro-manipulator, else planned, else histology track)",
        metadata={"raw_unit": "m"},
    )
    z_target: float = pa.Field(
        coerce=True,
        description="Target z-position in metres from Bregma (in IBL coordinates space), from the best available Alyx trajectory (micro-manipulator, else planned, else histology track)",
        metadata={"raw_unit": "m"},
    )


class ModelHistologyResolved(BaseChannelFeatures):
    """Schema for resolved histology coordinates.

    This schema defines the structure and validation rules for resolved
    histology coordinates after actual histological analysis.

    Note:
        Same coordinate space as :class:`ModelHistologyPlanned`. These come from
        ``SpikeSortingLoader.load_channels()`` -- the highest-provenance alignment Alyx holds
        for the insertion, above the micro-manipulator estimate the ``*_target`` columns use,
        hence "at the most recent histology step".

    Attributes:
        x (Series[float]): Resolved X-coordinate in metres from Bregma (IBL coordinates space).
        y (Series[float]): Resolved Y-coordinate in metres from Bregma (IBL coordinates space).
        z (Series[float]): Resolved Z-coordinate in metres from Bregma (IBL coordinates space).
        atlas_id (Series[int]): Atlas region identifier.
        acronym (Series[str]): Brain region acronym.
    """

    # Metres from Bregma, the IBL coordinate convention iblatlas expects.
    x: float = pa.Field(
        coerce=True,
        description="x-position in metres from Bregma (in IBL coordinates space). Coordinates at the most recent histology step",
        metadata={"raw_unit": "m"},
    )
    y: float = pa.Field(
        coerce=True,
        description="y-position in metres from Bregma (in IBL coordinates space). Coordinates at the most recent histology step",
        metadata={"raw_unit": "m"},
    )
    z: float = pa.Field(
        coerce=True,
        description="z-position in metres from Bregma (in IBL coordinates space). Coordinates at the most recent histology step",
        metadata={"raw_unit": "m"},
    )
    atlas_id: int = pa.Field(
        coerce=True,
        description="Atlas region identifier in Allen mapping",
        metadata={"raw_unit": "N/A"},
    )
    acronym: str = pa.Field(
        coerce=True,
        description="Brain region acronym in the Allen mapping",
        metadata={"raw_unit": "N/A"},
    )


class ModelRawFeatures(
    ModelSpikeFeatures,
    ModelCsdFeatures,
    ModelApFeatures,
    ModelLfFeatures,
    ModelChannelLayout,
):
    """Combined schema for all raw features.

    This schema combines all individual feature schemas into a single
    comprehensive schema for raw electrophysiological data validation.

    Note:
        This class inherits from multiple feature schemas to provide
        a unified interface for all feature types.
    """

    pass


class ModelDenoisedFeatures(
    ModelSpikeShapeFeatures,
    ModelCsdFeatures,
    ModelApFeatures,
    ModelLfFeatures,
    ModelChannelLayout,
):
    """Combined schema for denoised features, after the waveform-shape remap.

    Identical to `ModelRawFeatures` apart from the spike block: `denoise_raw_features_data`
    applies `remap_waveform_shape_features`, which drops `ModelSpikeFeatures`' 6 raw
    timing/amplitude columns in favour of `ModelSpikeShapeFeatures`' 4 reparametrised ones.

    Note:
        Denoised tables written before that remap still match `ModelRawFeatures`, so
        `ephysatlas.data.read_features_from_disk` chooses between the two schemas on the
        columns actually present rather than on which file it loaded.
    """

    pass


class ModelProbeDetails(pa.DataFrameModel):
    """Schema for probe insertion metadata (df_probe_details.pqt). One row per probe insertion."""

    pid: str = pa.Field(description="Probe insertion UUID")
    eid: str = pa.Field(description="Session UUID")
    probe_name: Optional[str] = pa.Field(nullable=True)
    probe_serial: Optional[str] = pa.Field(nullable=True)
    neuropixel_version: Optional[str] = pa.Field(nullable=True)
    probe_model: Optional[str] = pa.Field(
        nullable=True,
        description="Probe part number from SpikeGLX meta-data (imDatPrb_pn), "
        "e.g. 'NP2013'. See spikeglx.get_probe_model.",
    )
    referencing_scheme: Optional[str] = pa.Field(
        nullable=True,
        description="Referencing scheme from SpikeGLX meta-data (imroTbl): "
        "'external', 'tip', 'ground' or 'on_shank'. See spikeglx.get_referencing_scheme.",
    )
    lab: Optional[str] = pa.Field(nullable=True)
    pname: Optional[str] = pa.Field(nullable=True)
    spike_sorting: Optional[str] = pa.Field(nullable=True)
    spike_sorting_version: Optional[str] = pa.Field(nullable=True)
    histology: Optional[str] = pa.Field(nullable=True)
    record_length: Optional[float] = pa.Field(coerce=True, nullable=True)
    channel_count: Optional[float] = pa.Field(coerce=True, nullable=True)
    bwm: bool = pa.Field(description="Part of Brain-Wide Map freeze")


def feature_group_of_column_map() -> dict:
    """Map each denoisable feature column name to its provenance group.

    Returns
    -------
    dict
        Maps feature column name to one of 'raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms'.
    """
    schema_by_group = {
        "raw_ap": ModelApFeatures,
        "raw_lf": ModelLfFeatures,
        "raw_lf_csd": ModelCsdFeatures,
        "waveforms": ModelSpikeFeatures,
    }
    return {
        column_name: group
        for group, model in schema_by_group.items()
        for column_name in model.to_schema().columns.keys()
    }


# Default per-group TV-denoise fac, selected by bracketing fac per group
# against Cosmos_id classification accuracy (5-fold CV): see
# https://oliche.github.io/oliche-quarto/analyses/2026-09-tv-denoise-fac-bracket/
DEFAULT_FAC = {
    "raw_ap": 0.1,
    "raw_lf": 0.1,
    "raw_lf_csd": 0.1,
    "waveforms": 3,
}


def voltage_features_set(features_list=FEATURES_LIST):
    """Get list of feature column names by provenance.

    This function returns the list of features columns names depending on their provenance.
    This is useful to select the columns for training.

    Args:
        features_list (list, optional): List of feature groups to include.
            Defaults to ['raw_ap', 'raw_lf', 'localisation', 'waveforms'].
            Use 'all' to include all available feature groups.

    Returns:
        list: Sorted list of feature column names excluding the 'channel' column.

    Note:
        The looping preserves the order of the features groups in the list.
        Available feature groups: 'raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms', 'micro-manipulator'.
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
                x_list += list(ModelApFeatures.to_schema().columns.keys())
            case "raw_lf":
                x_list += list(ModelLfFeatures.to_schema().columns.keys())
            case "raw_lf_csd":
                x_list += list(ModelCsdFeatures.to_schema().columns.keys())
            case "waveforms":
                x_list += list(ModelSpikeFeatures.to_schema().columns.keys())
            case "micro-manipulator":
                x_list += list(ModelHistologyPlanned.to_schema().columns.keys())
    return x_list


def _get_power_in_band(fscale, period, band):
    """Calculate power in a specific frequency band.

    Args:
        fscale (np.ndarray): Frequency scale array.
        period (np.ndarray): Periodogram values.
        band (list): Frequency band [low, high] in Hz.

    Returns:
        np.ndarray: Power in the specified band in dB relative to v/sqrt(Hz).

    Note:
        This function weights the frequencies using a cosine window and
        computes the weighted average power in the specified band.
    """
    band = np.array(band)
    # weight the frequencies
    fweights = ibldsp.utils.fcn_cosine([-np.diff(band), 0])(
        -abs(fscale - np.mean(band))
    )
    p = 10 * np.log10(
        np.sum(period * fweights / np.sum(fweights), axis=-1)
    )  # dB relative to v/sqrt(Hz)
    return p


def get_psd_decay_features(
    data, fs, fscale, period, bands, nperseg=2048, PSD_range=BANDS["lfp"]
):
    """
    Extract power spectral density decay features from electrophysiological data.

    This function computes spectral parameterization features that characterize the
    aperiodic (1/f) component of the power spectral density (PSD) using the specparam
    library. It fits a model to separate periodic peaks from the aperiodic background
    in the frequency domain, providing insights into the underlying neural dynamics.

    Additionally, it computes residual power features by removing the fitted aperiodic
    component from the observed PSD, highlighting periodic components across different
    frequency bands.

    The aperiodic component of neural signals is thought to reflect the balance of
    excitation and inhibition in neural circuits, making these features particularly
    useful for characterizing brain states and pathological conditions.

    Parameters
    ----------
    data : np.ndarray
        Input electrophysiological data with shape (n_channels, n_samples).
        Each row represents a different recording channel.
    fs : float
        Sampling frequency of the data in Hz.
    fscale : np.ndarray
        Frequency scale array corresponding to the period data.
    period : np.ndarray
        2D array of power spectral density values with shape (n_channels, n_frequencies).
        Each row represents the PSD for a different recording channel.
    bands : dict
        Dictionary mapping frequency band names to [min_freq, max_freq] ranges in Hz.
        Used for computing residual power in specific frequency bands.
    nperseg : int, optional
        Length of each segment for Welch's method PSD estimation, by default 2048.
        Larger values provide better frequency resolution but less temporal averaging.
    PSD_range : list of float, optional
        Frequency range [min_freq, max_freq] in Hz for spectral parameterization,
        by default BANDS["lfp"] which is [0, 90] Hz.

    Returns
    -------
    pd.DataFrame
        One row per channel. Aperiodic-component columns: ``aperiodic_offset``
        (log10 power at 1 Hz), ``aperiodic_exponent`` (1/f slope in log-log
        space), ``decay_fit_error`` (RMS fit error), ``decay_fit_r_squared``
        (goodness of fit) and ``decay_n_peaks`` (number of periodic peaks).
        Residual-power columns (periodic power per band after aperiodic removal):
        ``psd_residual_delta``, ``psd_residual_theta``, ``psd_residual_alpha``,
        ``psd_residual_beta``, ``psd_residual_gamma`` and ``psd_residual_lfp``.

    Notes
    -----
    The function uses the specparam library (formerly FOOOF) to separate periodic
    and aperiodic components of the PSD. The aperiodic component follows a 1/f^β
    relationship where β is the aperiodic exponent.

    Channels with R-squared < 0.9 are flagged as having poor fits, which may
    indicate artifacts or unusual spectral properties.

    The spectral model is configured with:
    - Peak width limits: [10, 15] Hz
    - Maximum peaks: 4
    - Minimum peak height: 0.1

    The residual curve is computed as:
    residual = 10^(log10(observed_PSD) - log10(fitted_aperiodic_component))

    Examples
    --------
    >>> import numpy as np
    >>> # Generate sample data and frequency arrays
    >>> data = np.random.randn(10, 10000)
    >>> fs = 1000.0  # 1 kHz sampling rate
    >>> fscale = np.linspace(0, 90, 100)
    >>> period = np.random.rand(10, 100)  # Mock PSD data
    >>> bands = {'delta': [0, 4], 'theta': [4, 10], 'alpha': [8, 12]}
    >>> features = get_psd_decay_features(data, fs, fscale, period, bands)
    >>> print(features.columns)
    Index(['aperiodic_offset', 'aperiodic_exponent', 'decay_fit_error',
           'decay_fit_r_squared', 'decay_n_peaks', 'psd_residual_delta',
           'psd_residual_theta', 'psd_residual_alpha', ...], dtype='object')

    References
    ----------
    Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R.,
    et al. (2020). Parameterizing neural power spectra into periodic and aperiodic
    components. Nature Neuroscience, 23(12), 1655-1665.
    """
    assert period.ndim == 2, "Period must be a 2D array"
    from scipy.signal import welch

    # Get a smoothed out version of the PSD
    frequencies, psd_arr = welch(data, fs, nperseg=nperseg)

    from specparam import SpectralModel

    # Initialize a model object for spectral parameterization, with some settings
    fm = SpectralModel(
        peak_width_limits=[10, 15], max_n_peaks=4, min_peak_height=0.1, verbose=False
    )

    result_list = []

    if fscale[0] == 0:
        fscale = fscale[1:]
        period = period[:, 1:]

    # Fit individual PSD over 3-40 Hz range
    for i in range(psd_arr.shape[0]):
        if np.sum(psd_arr[i, :]) == 0:
            result_dict = {
                "aperiodic_offset": np.nan,
                "aperiodic_exponent": np.nan,
                "decay_fit_error": np.nan,
                "decay_fit_r_squared": np.nan,
                "decay_n_peaks": np.nan,
            }
            for b in BANDS:
                result_dict[f"psd_residual_{b}"] = np.nan
            result_list.append(result_dict)
            continue

        fm.fit(frequencies, psd_arr[i, :], PSD_range)

        fit_result = fm.get_results()
        offset, slope = fit_result.aperiodic_params

        result_dict = {
            "aperiodic_offset": fit_result.aperiodic_params[0],
            "aperiodic_exponent": fit_result.aperiodic_params[1],
            "decay_fit_error": fit_result.error,
            "decay_fit_r_squared": fit_result.r_squared,
            "decay_n_peaks": fm.n_peaks_,
        }
        # Get the predicted decay of PSD the curve based on offset and slope
        psd_decay = offset - np.log10(fscale**slope)
        residual_curve = 10 ** (np.log10(period[i, :]) - psd_decay)

        for b in BANDS:
            result_dict[f"psd_residual_{b}"] = _get_power_in_band(
                fscale, residual_curve, bands[b]
            )
        result_list.append(result_dict)

        # if i==0:
        #     import matplotlib.pyplot as plt
        #     fig,ax = plt.subplots(1,2)
        #     fm.plot(ax=ax[0])
        #     ax[0].plot(fscale,psd_decay,'g*', markersize=0.1, alpha=0.5)
        #     ax[1].semilogy(fscale,residual_curve,'r', markersize=0.1, alpha=0.5)
        #     ax[1].set_xlim(0, 90)

    psd_decay_features = pd.DataFrame(result_list)

    low_r_squared_channels = psd_decay_features[
        psd_decay_features["decay_fit_r_squared"] < 0.9
    ].index
    logger.info(
        f"Number of channels with low r squared during psd decay fit: {len(low_r_squared_channels)}"
    )
    if len(low_r_squared_channels) > 0:
        logger.warning(
            f"Channels with low r squared during psd decay fit: {low_r_squared_channels}"
        )

    return psd_decay_features


def lf(data, fs, bands=None, decay_features=True):
    """Compute LF features from a numpy array.

    Computes the local field potential (LF) features from electrophysiological data
    including RMS values and power spectral density across different frequency bands.

    Args:
        data (np.ndarray): Data array with shape (channels, samples).
        fs (float): Sampling frequency in Hz.
        bands (dict, optional): Dictionary with frequency bands to compute.
            Defaults to BANDS constant.

    Returns:
        pd.DataFrame: DataFrame with columns ['channel', 'rms_lf', 'psd_delta',
            'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'psd_lfp'].

    Note:
        The function computes RMS values and power spectral density for each
        frequency band defined in the BANDS constant.
    """
    bands = BANDS if bands is None else bands
    nc = data.shape[0]  # number of channels
    fscale, period = scipy.signal.periodogram(data, fs)
    df_lf = pd.DataFrame()
    df_lf["channel"] = np.arange(nc)
    df_lf["rms_lf"] = ibldsp.utils.rms(data, axis=-1)
    for b in BANDS:
        df_lf[f"psd_{b}"] = _get_power_in_band(fscale, period, bands[b])

    # Caluclate the Aperiodic and Periodic features
    logger.info("Calculating Aperiodic and Periodic features")
    df_decay = get_psd_decay_features(data, fs, fscale, period, bands)
    assert df_decay.shape[0] == df_lf.shape[0]
    assert len(set(df_decay.columns) & set(df_lf.columns)) == 0
    df_lf = pd.concat([df_lf, df_decay], axis=1)

    return ModelLfFeatures.validate(df_lf)


def csd(data, fs, geometry, bands=None, decimate=10, scale=True):
    """Compute CSD features from a numpy array.

    Computes the current source density (CSD) features from electrophysiological data
    including RMS values and power spectral density across different frequency bands.

    Args:
        data (np.ndarray): Data array with shape (channels, samples).
        fs (float): Sampling frequency in Hz.
        geometry (dict): Dictionary with channel geometry containing 'x' and 'y' arrays.
        bands (dict, optional): Dictionary with frequency bands to compute.
            Defaults to BANDS constant.
        decimate (int, optional): Decimation factor for CSD calculation.
            Defaults to 10.
        scale (bool, optional): Forwarded to current_source_density. If True, scale the
            finite difference by the intercontact distance and tissue conductivity;
            if False, return the raw numerical finite difference. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with columns ['channel', 'rms_lf_csd', 'psd_delta_csd',
            'psd_theta_csd', 'psd_alpha_csd', 'psd_beta_csd', 'psd_gamma_csd', 'psd_lfp_csd'].

    Note:
        The function applies Cadzow denoising and current source density computation
        before computing the spectral features.
    """
    data_rs = scipy.signal.decimate(data, decimate, axis=1, ftype="fir")
    data_rs = ibldsp.cadzow.cadzow_denoiser(
        data_rs,
        rank=5,
        fs=fs / decimate,
        niter=1,
        fmax=125,
        nswx=64,
        ovx=32,
        gap_threshold=2.0,
        ppca_k=2.0,
        h=geometry,
    )
    # Calculate the CSD features
    data_rs_diff2 = ibldsp.voltage.current_source_density(
        data_rs, h=geometry, n=2, scale=scale
    )
    df_csd = lf(data_rs_diff2, fs / decimate, bands=bands, decay_features=False)
    df_csd = df_csd.rename(
        columns={c: f"{c}_csd" for c in df_csd.columns if c not in ["channel"]}
    )

    # Calculate the Diff1 CSD features.
    data_rs_diff1 = ibldsp.voltage.current_source_density(
        data_rs, h=geometry, n=1, scale=scale
    )
    df_csd_diff1 = lf(data_rs_diff1, fs / decimate, bands=bands, decay_features=False)
    df_csd_diff1 = df_csd_diff1.rename(
        columns={
            c: f"{c}_csd_diff1" for c in df_csd_diff1.columns if c not in ["channel"]
        }
    )
    assert df_csd_diff1["channel"].equals(df_csd["channel"]), (
        "Channels are not perfectly aligned!"
    )

    df_csd = pd.concat([df_csd, df_csd_diff1.drop(columns=["channel"])], axis=1)
    return ModelCsdFeatures.validate(df_csd)


def ap(data, geometry=None, channel_labels=None):
    """Compute AP features from a numpy array.

    Computes the action potential (AP) features from electrophysiological data
    including RMS values and correlation ratios.

    Args:
        data (np.ndarray): AP band data array with shape (channels, samples).
        geometry (dict): Dictionary with channel geometry containing 'x' and 'y' arrays.
        channel_labels (np.ndarray): Array of channel quality labels.

    Returns:
        pd.DataFrame: DataFrame with columns ['channel', 'rms_ap', 'cor_ratio', 'channel_labels'].

    Raises:
        AssertionError: If geometry or channel_labels are not provided.

    Note:
        This function computes RMS values and cross-correlation ratios for
        action potential band data.
    """
    assert geometry is not None, "Geometry is required for AP band computation"
    assert channel_labels is not None, "Channel labels are required"
    df_ap = pd.DataFrame()
    nc = data.shape[0]  # number of channels
    df_ap["channel"] = np.arange(nc)
    df_ap["rms_ap"] = ibldsp.utils.rms(data, axis=-1)
    df_ap["cor_ratio"] = xcor_acor_ratio(data, geometry=geometry)
    df_ap["channel_labels"] = channel_labels
    return ModelApFeatures.validate(df_ap)


def dart_subtraction_numpy(data, fs, geometry, params=None, scratch_dir=None, **extra):
    """Perform spike detection using Dartsort.

    This function performs spike detection and feature extraction using the
    Dartsort algorithm with configurable parameters.

    Args:
        data (np.ndarray): Voltage traces array with shape [nc, ns] in Volts, where
            nc is number of channels and ns is number of samples.
        fs (float): Sampling frequency in Hz.
        geometry (dict): Dictionary with channel geometry containing 'x' and 'y' arrays.
        **params: Additional parameters for Dartsort configuration.

    Returns:
        tuple: A tuple containing:

            - df_spikes (pd.DataFrame): DataFrame with spike information including
              sample indices, channels, peak-to-peak amplitudes (Volts), and
              localizations.
            - d_waveforms (dict): Dictionary containing raw and denoised waveforms
              (Volts) and channel indices.

    Note:
        This function requires the dartsort package to be installed.
        It creates temporary directories for processing and cleans them up afterward.
        GPU acceleration is supported when available.

        DARTsort detects spikes on ``data`` normalised by its own per-channel RMS
        (a fixed detection threshold in units of channel RMS), but that scaling is
        un-applied before returning: ``ptp`` and the waveform snippets are rescaled
        back to Volts using the same per-channel RMS, so every amplitude derived
        from them downstream (``peak_val``/``trough_val``/``tip_val`` and the
        slopes computed by ``ibldsp.waveforms.compute_spike_features``) comes out
        in real units. The localisation amplitude (``alpha``) is the one exception:
        it is fit jointly across a multichannel snippet whose channels each carry a
        different RMS, so there is no single scale factor to un-apply and it is
        left in DARTsort's native (arbitrary) units.
    """

    # Resolve params from a DartParameters object (as _spikes_dartsort passes it), a
    # dict of fields, or None, then merge an explicit scratch_dir. NOTE: this used to
    # be ``DartParameters(**params)`` with a ``**params`` signature, which silently
    # dropped a passed DartParameters object (pydantic ignores the unknown "params"
    # key), so dartsort parameters were never actually applied.
    if isinstance(params, DartParameters):
        if extra:
            params = params.model_copy(update=extra)
    elif params is None:
        params = DartParameters(**extra)
    else:
        params = DartParameters(**{**dict(params), **extra})
    if scratch_dir is not None:
        params = params.model_copy(update={"scratch_dir": scratch_dir})
    # The spike/waveform stack is an optional dependency: pip install ibleatools[full]
    try:
        import dartsort
        import spikeinterface.core as sc
        import h5py
    except ImportError as e:
        raise ImportError(
            "Spike/waveform feature computation requires the optional spike-detection "
            "stack (DARTsort and its dependencies). Install it with: "
            "pip install ibleatools[full]"
        ) from e

    dart_xy = np.c_[geometry["x"], geometry["y"]]

    # DARTsort's detection threshold is expressed in units of channel RMS, so
    # spikes are detected on RMS-normalised data. This scaling is un-applied
    # below (using this same per-channel RMS, in Volts) before returning.
    rms_channel = ibldsp.utils.rms(data, axis=-1)
    zdata = data / rms_channel[:, np.newaxis]
    rec_np = sc.NumpyRecording(zdata.T, sampling_frequency=fs)
    rec_np.set_dummy_probe_from_locations(dart_xy)

    # Keep DARTsort's default waveform duration while honoring the trough offset
    # used later by ibleatools' waveform features.
    default_waveform_cfg = dartsort.WaveformConfig()
    spike_length_samples = default_waveform_cfg.spike_length_samples(fs)
    if params.trough_offset >= spike_length_samples:
        raise ValueError(
            "trough_offset must be smaller than DARTsort's waveform length "
            f"({spike_length_samples} samples at {fs} Hz)"
        )
    waveform_cfg = dartsort.WaveformConfig.from_samples(
        params.trough_offset,
        spike_length_samples - params.trough_offset,
        # raw calibrated fs errors from_samples' round-trip assert.
        sampling_frequency=round(fs),
    )

    # Preserve the old iblsorter branch's bundled single-channel denoiser.
    # The current DARTsort defaults disable NN feature denoising or select
    # a train-from-scratch Decollider.
    legacy_denoiser = {
        "do_nn_denoise": True,
        "nn_denoiser_class_name": "SingleChannelWaveformDenoiser",
        "nn_denoiser_pretrained_path": dartsort.config.default_pretrained_path,
    }
    denoising_cfg = dartsort.FeaturizationConfig(
        denoise_only=True,
        do_tpca_denoise=False,
        do_enforce_decrease="yes",
        save_input_voltages=False,
        extract_radius=params.localization_radius,
        localization_radius=params.localization_radius,
        tpca_fit_radius=params.localization_radius,
        tpca_centered=True,
        tpca_from_templates=False,
        **legacy_denoiser,
    )
    subtraction_cfg = dartsort.SubtractionConfig(
        subtraction_denoising_cfg=denoising_cfg,
        chunk_length_samples=params.chunk_length_samples,
        # This is behaviour change from older version of dartsort
        # In the older version, there was gradual reduction of threshold.
        # Not it is a fixed threshold
        detection_threshold=params.detection_threshold,  # sigma deviation
        realign_to_denoiser=False,
        relative_peak_radius_um=None,
        spatial_dedup_radius_um=params.spatial_dedup_radius_um,
        positive_temporal_dedup_radius_samples=params.positive_temporal_dedup_radius_samples,
        residnorm_decrease_threshold=params.residnorm_decrease_threshold,
        trough_priority=None,
        whiten=False,
    )
    # this determines what features you get out at the end
    # the nn localizer is another model which needs to be fitted, so turning
    # that off is good
    featurization_cfg = dartsort.FeaturizationConfig(
        nn_localization=False,
        do_enforce_decrease="yes",
        save_input_voltages=False,
        save_output_waveforms=True,  # save final nn denoised waveforms
        save_input_waveforms=True,  # save collision-cleaned, but not NN-denoised, waveforms
        extract_radius=params.localization_radius,
        localization_radius=params.localization_radius,
        tpca_fit_radius=params.localization_radius,
        tpca_centered=True,
        tpca_from_templates=False,
        # A shorter TPCA window changes the output waveform length in DARTsort
        # 0.5, so use the extraction window for both saved waveform datasets.
        input_tpca_waveform_cfg=waveform_cfg,
        **legacy_denoiser,
    )

    # we make sure that each runs get a different temp folder
    temp_suffix = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for _ in range(8)]
    )

    # Ensure scratch directory exists
    scratch_dir = _setup_scratch_directory(params.scratch_dir)

    # New version of DARTsort outputs a single object.
    # And has a different keyword arguments for subtract func.
    temp_folder = scratch_dir.joinpath(f"dart_{temp_suffix}")
    detected_spikes = dartsort.subtract(
        output_dir=temp_folder,
        recording=rec_np,
        waveform_cfg=waveform_cfg,
        featurization_cfg=featurization_cfg,
        subtraction_cfg=subtraction_cfg,
        computation_cfg=dartsort.ComputationConfig.from_n_jobs(params.n_jobs),
        show_progress=True,
    )
    if detected_spikes is None or detected_spikes.parent_h5_path is None:
        raise RuntimeError("DARTsort subtraction did not produce a feature file")

    # denoised_ptp_amplitudes is evaluated at each spike's detection channel, so a
    # single per-spike RMS-normalised amplitude, exactly un-scaled here.
    df_spikes = pd.DataFrame(
        {
            "sample": detected_spikes.times_samples,
            "channel": detected_spikes.channels,
            "ptp": detected_spikes.denoised_ptp_amplitudes
            * rms_channel[detected_spikes.channels],
            "xloc": detected_spikes.point_source_localizations[:, 0],  # xyza
            "yloc": detected_spikes.point_source_localizations[:, 1],  # xyza
            "zloc": detected_spikes.point_source_localizations[:, 2],  # xyza
            "alpha": detected_spikes.point_source_localizations[:, 3],  # xyza
        }
    )

    # Copy arrays before removing DARTsort's temporary output directory.
    with h5py.File(detected_spikes.parent_h5_path) as h5file:
        d_waveforms = {  # n_spikes, nsw, ncw
            "raw": np.array(h5file["collisioncleaned_waveforms"]),
            "denoised": np.array(h5file["denoised_waveforms"]),
            "channel_index": np.array(h5file["channel_index"]),
        }

    # Each waveform snippet column is a different real channel, each with its own
    # RMS: rescale per neighbour channel using DARTsort's channel_index lookup
    # (nc, n_neighbors), padded with a neutral 1.0 scale for its out-of-range
    # sentinel (== nc, used for unused/zero-padded neighbour slots).
    rms_padded = np.append(rms_channel, 1.0)
    spike_channel_scale = rms_padded[
        d_waveforms["channel_index"][detected_spikes.channels]
    ]  # (n_spikes, n_neighbors)
    d_waveforms["raw"] = d_waveforms["raw"] * spike_channel_scale[:, np.newaxis, :]
    d_waveforms["denoised"] = (
        d_waveforms["denoised"] * spike_channel_scale[:, np.newaxis, :]
    )
    d_waveforms["channel_rms"] = rms_channel

    shutil.rmtree(temp_folder)
    return df_spikes, d_waveforms


def _spikes_dartsort(data, fs: int, geometry: dict, scratch_dir=None, **params):
    """Dartsort backend for spike detection.

    This function serves as the Dartsort backend for the main spikes function,
    handling spike detection and feature extraction using Dartsort.

    Args:
        data (np.ndarray): Raw electrophysiology data.
        fs (int): Sampling frequency in Hz.
        geometry (dict): Channel geometry dictionary.
        scratch_dir (str, optional): Directory for temporary files.
        **params: Dartsort parameters.

    Returns:
        tuple: A tuple containing:

            - ``df_spikes_`` (pd.DataFrame): DataFrame with spike information.
            - d_waveforms (dict): Dictionary containing waveform data.
            - params_obj (DartParameters): Dartsort parameters object.
    """
    params_obj = DartParameters() if params is None else DartParameters(**params)
    logger.info("Starting spike detection with Dartsort backend")
    df_spikes_, d_waveforms = dart_subtraction_numpy(
        data, fs, geometry, scratch_dir=scratch_dir, params=params_obj
    )
    logger.info("Spike detection completed with Dartsort backend")
    return df_spikes_, d_waveforms, params_obj


def _spikes_spikeinterface(data, fs: int, geometry: dict, scratch_dir=None, **params):
    """SpikeInterface backend for spike detection.

    This function serves as the SpikeInterface backend for the main spikes function,
    handling spike detection and feature extraction using SpikeInterface.

    Args:
        data (np.ndarray): Raw electrophysiology data.
        fs (int): Sampling frequency in Hz.
        geometry (dict): Channel geometry dictionary.
        scratch_dir (str, optional): Directory for temporary files.
        **params: SpikeInterface parameters.

    Returns:
        tuple: A tuple containing:

            - ``df_spikes_`` (pd.DataFrame): DataFrame with spike information.
            - d_waveforms (dict): Dictionary containing waveform data.
            - params_obj (dict): SpikeInterface parameters object.

    Raises:
        ImportError: If SpikeInterface is not installed.
        NotImplementedError: This function is not yet fully implemented.

    Note:
        This function is currently a placeholder and needs full implementation
        for SpikeInterface backend support.
    """
    try:
        import spikeinterface.core as sc
        from probeinterface.neuropixels_tools import read_spikeglx
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.core.node_pipeline import ExtractDenseWaveforms
        from spikeinterface.sortingcomponents.peak_localization import (
            LocalizeCenterOfMass,
        )
        from spikeinterface.core.node_pipeline import run_node_pipeline, PeakRetriever
    except ImportError as e:
        raise ImportError(
            f"SpikeInterface not installed. Please install the optional spike-sorting "
            f"stack with: pip install ibleatools[full]. Error: {e}"
        )

    logger.info("Starting spike detection with SpikeInterface backend")

    # Create SpikeInterface recording object
    recording = sc.NumpyRecording(data.T, sampling_frequency=fs)

    # Set up probe geometry
    assert params.get("sr_ap_filepath")
    from probeinterface.neuropixels_tools import read_spikeglx

    probe = read_spikeglx(
        params["sr_ap_filepath"]
    )  # TODO: Pending implementation for the case when data is loaded using files, and not pid
    recording = recording.set_probe(probe)

    si_params = {
        "method": params.get("method", "locally_exclusive"),
        "peak_sign": params.get("peak_sign", "neg"),
        "detect_threshold": params.get("detect_threshold", 5.0),
        "exclude_sweep_ms": params.get("exclude_sweep_ms", 0.1),
        "radius_um": params.get("localization_radius", 100),
        "job_kwargs": params.get("job_kwargs", {}),
    }

    peaks = detect_peaks(
        recording,
        method=si_params["method"],
        detect_threshold=si_params["detect_threshold"],
        radius_um=si_params["radius_um"],
        **si_params["job_kwargs"],
    )

    peak_retriever = PeakRetriever(recording, peaks)

    extract_dense_waveforms = ExtractDenseWaveforms(
        recording,
        parents=[peak_retriever],
        ms_before=0.5,
        ms_after=0.5,
        return_output=True,
    )
    pipeline_nodes = [
        peak_retriever,
        extract_dense_waveforms,
        LocalizeCenterOfMass(
            recording,
            parents=[peak_retriever, extract_dense_waveforms],
            radius_um=si_params["radius_um"],
        ),
    ]
    job_name = "localize peaks using center_of_mass"
    waveform_data, peak_locations = run_node_pipeline(
        recording, pipeline_nodes, si_params["job_kwargs"], job_name=job_name
    )
    waveform_data, peak_locations
    # # Create DataFrame similar to Dartsort output
    # df_spikes_ = pd.DataFrame({
    #     'sample': spikes['sample_index'],
    #     'channel': spikes['channel_index'],
    #     'ptp': np.ones(len(spikes)) * 1.0,  # Placeholder - need to compute from waveforms
    #     'xloc': geometry['x'][spikes['channel_index']],  # Approximate localization
    #     'yloc': geometry['y'][spikes['channel_index']],  # Approximate localization
    #     'zloc': np.zeros(len(spikes)),  # Placeholder
    #     'alpha': np.ones(len(spikes)) * 1.0,  # Placeholder - need proper computation
    # })

    # # Create waveforms dictionary in same format as Dartsort
    # d_waveforms = {
    #     "raw": waveforms,  # Raw waveforms
    #     "denoised": waveforms,  # For now, same as raw (could add denoising step)
    #     "channel_index": we.channel_ids,
    # }

    # # Create a simple params object to maintain interface compatibility
    # # For SpikeInterface, we'll use a dict instead of DartParameters
    # params_obj = {
    #     'trough_offset': params.get('trough_offset', 42),  # Default from DartParameters
    #     **params
    # }

    # logger.info("Spike detection completed with SpikeInterface backend")
    # return df_spikes_, d_waveforms, params_obj
    raise NotImplementedError("This function is not implemented yet")


def _neighbour_channel_xy_um(channel_index, spike_channels, x_um, y_um):
    """Per-spike (x, y) coordinates (um) of a waveform array's neighbour channels.

    Args:
        channel_index (np.ndarray): (n_channels_real, n_neighbours) lookup from a
            real detection channel to its neighbour real-channel ids, as returned
            by DARTsort in ``d_waveforms["channel_index"]``. Padded with the
            sentinel ``n_channels_real`` for unused neighbour slots.
        spike_channels (np.ndarray): (n_spikes,) detection channel (real channel
            id) of each spike, e.g. ``df_spikes["channel"]``.
        x_um (np.ndarray): (n_channels_real,) real channel x-coordinate, um.
        y_um (np.ndarray): (n_channels_real,) real channel y-coordinate, um.

    Returns:
        np.ndarray: (n_spikes, n_neighbours, 2) coordinates, NaN for padded/unused
            neighbour slots.
    """
    xy_um = np.c_[x_um, y_um]
    xy_padded = np.vstack([xy_um, [np.nan, np.nan]])
    return xy_padded[channel_index[spike_channels]]


def spikes(
    data,
    fs: int,
    geometry: dict,
    return_waveforms=True,
    backend="dartsort",
    scratch_dir=None,
    **params,
):
    """Spike detection and feature extraction with multiple backend support.

    This function performs spike detection and feature extraction using either
    Dartsort or SpikeInterface backend, with comprehensive feature computation
    including waveform analysis and spike characterization.

    Args:
        data (np.ndarray): Raw electrophysiology data with shape [nc, ns] where
            nc is number of channels and ns is number of samples.
        fs (int): Sampling frequency in Hz.
        geometry (dict): Channel geometry dictionary with 'x' and 'y' arrays.
        return_waveforms (bool, optional): Whether to return waveforms dictionary.
            Defaults to True.
        backend (str, optional): Backend to use ('dartsort' or 'spikeinterface').
            Defaults to 'dartsort'.
        scratch_dir (str, optional): Directory for temporary files.
        **params: Backend-specific parameters.

    Returns:
        pd.DataFrame or tuple: If return_waveforms is False, returns DataFrame with
            aggregated spike features per channel. If True, returns tuple of
            (DataFrame, waveforms_dict).

    Raises:
        ValueError: If an unknown backend is specified.

    Note:
        The function aggregates spike features by channel and computes various
        waveform characteristics including timing, amplitude, and slope features.
        Both backends produce compatible output formats for further processing.
    """
    # Call the appropriate backend function to get raw spike data
    if backend == "dartsort":
        df_spikes_, d_waveforms, params_obj = _spikes_dartsort(
            data, fs, geometry, scratch_dir, **params
        )
    elif backend == "spikeinterface":
        df_spikes_, d_waveforms, params_obj = _spikes_spikeinterface(
            data, fs, geometry, scratch_dir, **params
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: 'dartsort', 'spikeinterface'"
        )

    # Common processing for both backends
    logger.info("Computing waveform features")
    df_waveforms = ibldsp.waveforms.compute_spike_features(d_waveforms["denoised"])
    df_spikes = df_spikes_.merge(df_waveforms, left_index=True, right_index=True)

    # Cast the float32 values as float64
    df_spikes[df_spikes.select_dtypes(np.float32).columns] = df_spikes.select_dtypes(
        np.float32
    ).astype(np.float64)

    # Multi-channel spatial-spread / slowness features (#123): need each spike's
    # neighbour-channel geometry, from d_waveforms["channel_index"] (detection
    # channel -> neighbour real-channel ids, DARTsort-only for now) and the
    # snippet's channel geometry (um).
    if "channel_index" in d_waveforms:
        neighbor_xy_um = _neighbour_channel_xy_um(
            d_waveforms["channel_index"],
            df_spikes["channel"].to_numpy(),
            geometry["x"],
            geometry["y"],
        )
        channel_geometry_3d = np.dstack(
            [
                neighbor_xy_um[:, :, 0],
                neighbor_xy_um[:, :, 1],
                np.zeros_like(neighbor_xy_um[:, :, 0]),
            ]
        )
        df_spikes = ibldsp.waveforms.compute_spatial_spread(
            d_waveforms["denoised"], df_spikes, channel_geometry_3d
        )
        df_spikes = df_spikes.rename(columns={"spatial_spread": "spatial_spread_um"})
        df_spikes = ibldsp.waveforms.compute_slowness(
            d_waveforms["denoised"], df_spikes, channel_geometry_3d
        )
    else:
        df_spikes["spatial_spread_um"] = np.nan
        df_spikes["slowness_s_per_m"] = np.nan

    # Get trough_offset from params_obj (handle both DartParameters object and dict)
    if hasattr(params_obj, "trough_offset"):
        trough_offset = params_obj.trough_offset
    else:
        trough_offset = params_obj.get("trough_offset", 42)

    fcn_mean_time = lambda x: np.mean((x - trough_offset)) / fs  # NOQA
    # A raw count depends on the duration of `data`; divide by it so
    # spike_count is a rate (spikes/s), comparable across snippets of
    # different lengths.
    duration_secs = data.shape[-1] / fs
    fcn_rate = lambda x: x.size / duration_secs  # NOQA

    # Aggregation by channel of the spikes / waveforms features
    df_spiking = (
        df_spikes.groupby("channel")
        .agg(
            alpha_mean=pd.NamedAgg(column="alpha", aggfunc="mean"),
            alpha_std=pd.NamedAgg(column="alpha", aggfunc=lambda x: np.std(x, ddof=0)),
            spike_count=pd.NamedAgg(column="alpha", aggfunc=fcn_rate),
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
            spatial_spread_um=pd.NamedAgg(column="spatial_spread_um", aggfunc="mean"),
            spatial_spread_um_std=pd.NamedAgg(
                column="spatial_spread_um", aggfunc="std"
            ),
            slowness_s_per_m=pd.NamedAgg(column="slowness_s_per_m", aggfunc="mean"),
            slowness_s_per_m_std=pd.NamedAgg(column="slowness_s_per_m", aggfunc="std"),
        )
        .reset_index()
    )

    df_spiking = ModelSpikeFeatures.validate(df_spiking)

    if return_waveforms:
        return df_spiking, d_waveforms | {"df_spikes": df_spikes}
    else:
        return df_spiking


def xcor_acor_ratio(v: np.ndarray, geometry: dict, n_neighbor: int = 3) -> np.ndarray:
    """Compute cross-correlation over auto-correlation ratio.

    This function calculates the ratio of cross-correlation between neighboring
    channels over the auto-correlation for each channel in the AP band data.

    Args:
        v (np.ndarray): Voltage array for AP band with shape (nc, ns) where
            nc is number of channels and ns is number of samples.
        geometry (dict): Geometry dictionary with 'x' and 'y' arrays for
            electrode positions.
        n_neighbor (int, optional): Number of neighboring channels to consider.
            Defaults to 3.

    Returns:
        np.ndarray: Array of size (nc,) containing the correlation ratios.

    Note:
        The function computes covariance matrices and extracts diagonal elements
        to calculate cross-correlation ratios for neighboring channels.
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
    """Denoise AP features using total variation filter.

    Denoise the AP feature using a maximum variation filter. Interpolates the
    feature in a square grid, performs the filtering, and then interpolates
    back to the original grid.

    Args:
        feature (np.ndarray): AP feature to denoise with shape (nc,).
        xy (np.ndarray): Coordinates of the AP feature with shape (nc, 2).
        labels (np.ndarray, optional): Channel quality annotation array with shape (nc,).
            If different than 0, channel is discarded and interpolated.
            Set to None for no annotation. Defaults to None.
        fac (int, optional): Factor for the TV denoising in median deviation units.
            Defaults to 1.

    Returns:
        np.ndarray: Denoised AP features with shape (nc,).

    Note:
        This function uses scikit-image's total variation Chambolle denoising
        algorithm to smooth the feature values while preserving edges.
    """
    isvalid = ~np.isnan(feature)
    if (
        np.count_nonzero(isvalid) < 5
    ):  # Grid data interpolation requires at least 5 valid points
        return feature
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
    """Abstract base class for electrophysiological feature transformers.

    This class provides the interface for transformers that work with
    electrophysiological features, implementing scikit-learn's transformer
    interface and setting pandas as the default output format.

    Note:
        This is an abstract base class that should not be instantiated directly.
    """

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
        """Fit the transformer and transform the data.

        Args:
            X (pd.DataFrame, optional): Input data to fit and transform.
            y: Ignored, present for compatibility with scikit-learn interface.

        Returns:
            pd.DataFrame: Transformed data.
        """
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
            ) is not None and ("transform" in metadata):
                self.fcn_transform_[feature_name] = metadata["transform"]

    def transform(self, X: pd.DataFrame, y=None):
        self.validate_X(X)
        xt = pd.DataFrame(index=X.index)
        for column_name in X.columns:
            if column_name in self.fcn_transform_:
                xt.loc[:, column_name] = self.fcn_transform_[column_name](
                    X[column_name].to_numpy()
                )
            else:
                xt.loc[:, column_name] = X[column_name]
        return xt


class EphysDenoiser(_EphysTransformerInterface):
    def __init__(self, fac=DEFAULT_FAC, channel_labels=None):
        """TV-denoise electrophysiological features, with one weight factor per feature group.

        Parameters
        ----------
        fac : float or dict, default=DEFAULT_FAC
            Factor for the TV denoising in median deviation units. Either a single
            scalar applied to every feature group, or a dict mapping a subset of
            {'raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms'} to their own factor.
            Groups not specified in the dict default to 1.
        channel_labels : np.ndarray, optional
            Channel quality annotation array with shape (nc,).
        """
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

    def _get_fac(self, feature_name: str, feature_group_of_column: dict) -> float:
        if not isinstance(self.fac, dict):
            return self.fac
        return self.fac.get(feature_group_of_column.get(feature_name), 1)

    def transform(self, X: pd.DataFrame, y=None):
        self.validate_X(X)
        channel_labels = self._get_channel_labels(X)
        ns = X.shape[0]
        feature_group_of_column = feature_group_of_column_map()
        for feature_name in self._get_feature_names(X):
            if (
                feature_name == "channel_labels"
            ):  # we do not want to apply any denoising to this feature
                continue
            fval = np.copy(X[feature_name].to_numpy()).astype(float)
            fval[channel_labels != 0] = np.nan
            fac = self._get_fac(feature_name, feature_group_of_column)
            logger.info(f"Calculation for feature_name = {feature_name}, fac = {fac}")
            denoised_values = denoise_shank(
                feature=fval,
                xy=X[["lateral_um", "axial_um"]].values,
                fac=fac,
            )  # .astype(X[feature_name].dtype)
            # Check that the denoised values have the expected length
            if len(denoised_values) != ns:
                raise ValueError(
                    f"Length mismatch for feature '{feature_name}': "
                    f"denoised values length ({len(denoised_values)}) != "
                    f"DataFrame length ({ns})"
                )
            X.loc[:, feature_name] = denoised_values.astype(X[feature_name].dtype)
        return X

    def fit(self, X: pd.DataFrame = None, y=None):
        return


def denoise_dataframe(df_pid, fac=DEFAULT_FAC, channel_labels=None):
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
    fac : float or dict, default=DEFAULT_FAC
        Factor for the TV denoising in median deviation units. Higher values
        result in stronger denoising. Either a single scalar applied to every
        feature group, or a dict mapping a subset of
        {'raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms'} to their own factor
        (see `feature_group_of_column_map`). Groups absent from the dict default to 1.

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
