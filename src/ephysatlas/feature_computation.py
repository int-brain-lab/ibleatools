from functools import reduce
import logging
import scipy.fft
import pandas as pd
import numpy as np
import iblatlas

from brainbox.io.one import SpikeSortingLoader
import ibldsp.voltage
import ibldsp.utils

from iblatlas.atlas import Insertion, NeedlesAtlas, AllenAtlas
from ibllib.pipes.histology import interpolate_along_track

from ephysatlas import features
from ephysatlas import __version__ as ibleatools_version
from ephysatlas.features import __features_version__ as features_version
from pathlib import Path
from ephysatlas.utils import setup_output_directory, add_metadata_to_parquet_files
from filelock import FileLock
import os

# Set up logger
logger = logging.getLogger(__name__)


def add_target_coordinates(pid=None, one=None, channels=None, traj_dict=None):
    """Add micro-manipulator target coordinates to channel information using trajectory data.

    This function calculates the 3D target coordinates (x_target, y_target, z_target) for each channel
    based on the probe insertion trajectory. It supports two modes: retrieving trajectory data from
    the Alyx database using a probe ID, or using a direct trajectory dictionary. The function applies
    pitch correction and coordinate system transformations to convert from in-vivo coordinates to
    the Allen coordinate system.

    Args:
        pid (str, optional): Probe insertion ID. Required if using Alyx database mode (when one is provided).
        one (ONE, optional): ONE client instance. Required if using Alyx database mode (when pid is provided).
        channels (dict): Channel information dictionary containing at least 'axial_um' field. Should also contain
            'rawInd' or 'channel' for channel indexing.
        traj_dict (dict, optional): Dictionary containing trajectory information with keys:
            - x, y, z: coordinates
            - depth, theta, phi: insertion parameters
            Required if not using Alyx database mode.

    Returns:
        dict: Updated channels dictionary with added 'x_target', 'y_target', and 'z_target' fields
            containing the 3D coordinates for each channel.

    Raises:
        ValueError: If neither (pid, one) nor traj_dict is provided.

    Example:
        >>> from one.api import ONE
        >>> one = ONE()
        >>> channels = {'axial_um': np.arange(384) * 20}
        >>> updated_channels = add_target_coordinates(
        ...     pid='probe00', one=one, channels=channels
        ... )

    Note:
        - The function applies a -5 degree pitch correction to account for probe tilt.
        - Coordinates are transformed from in-vivo to Allen coordinate system.
        - If channels don't have 'rawInd' or 'channel' fields, it picks the number of channels from the axial_um field.
        - The function interpolates coordinates along the probe track for each channel.
        - For Alyx database mode, it prioritizes micro-manipulator provenance trajectories.
    """
    # Initialize atlas objects for coordinate transformations
    needles = NeedlesAtlas()
    allen = AllenAtlas()
    # Compute the brain surface for the needles atlas
    needles.compute_surface()

    # Validate input combinations and retrieve trajectory data
    if pid is not None and one is not None:
        # Mode 1: Using Alyx database to retrieve trajectory information
        # Check if one is in local mode or remote mode,
        # TODO - Doing this for SDSC computation but need to do it cleaner.
        if one.mode == "local":
            # For local mode, create a remote ONE client to access Alyx database
            from one.api import ONE

            one_remote = ONE(mode="remote")
            trajs = one_remote.alyx.rest(
                "trajectories", "list", probe_insertion=pid, django="provenance__lte,30"
            )
        else:
            # For remote mode, use the existing ONE client
            trajs = one.alyx.rest(
                "trajectories", "list", probe_insertion=pid, django="provenance__lte,30"
            )
        # Prioritize micro-manipulator trajectories, fallback to first available
        traj = next(
            (t for t in trajs if t["provenance"] == "Micro-manipulator"), trajs[0]
        )
    elif traj_dict is not None:
        # Mode 2: Using direct trajectory dictionary
        traj = traj_dict
    else:
        raise ValueError("Either provide (pid, one) or traj_dict")

    # Apply the pitch correction by using iblatlas.atlas.tilt_spherical()
    # This corrects for the -5 degree tilt of the probe during insertion
    new_theta, new_phi = iblatlas.atlas.tilt_spherical(
        traj["theta"], traj["phi"], tilt_angle=-5
    )
    traj["theta"] = new_theta
    traj["phi"] = new_phi

    # Create an Insertion object from the trajectory data
    ins = Insertion.from_dict(traj, brain_atlas=needles)

    # Get the trajectory coordinates and flip them (deepest point first)
    txyz = np.flipud(ins.xyz)
    # Convert the coordinates from in-vivo to the Allen coordinate system
    # This involves transforming through the needles atlas to Allen atlas
    txyz = allen.bc.i2xyz(needles.bc.xyz2i(txyz / 1e6, round=False, mode="clip")) * 1e6
    # Interpolate coordinates along the probe track for each channel position
    xyz_mm = interpolate_along_track(txyz, channels["axial_um"] / 1e6)

    # Check if the rawInd data exists in the channels dictionary, otherwise use the default 384 channels (Ask OW)
    if ("rawInd" not in channels) and ("channel" not in channels):
        # assert channels["axial_um"].size == 384
        # Use the same number of channels as in the "axial_um" key
        channels["rawInd"] = np.arange(channels["axial_um"].size)

    # we interpolate the channels from the deepest point up. The neuropixel y coordinate is from the bottom of the probe
    # Update the channels dictionary with the target coordinates
    channels["x_target"] = xyz_mm[:, 0]
    channels["y_target"] = xyz_mm[:, 1]
    channels["z_target"] = xyz_mm[:, 2]
    return channels


# TODO - Make this function more modular so that you can get raw_ap, sr_ap and destriped_ap just from specifying the pid
def online_feature_computation(
    sr_lf=None,
    sr_ap=None,
    t0=0.0,
    duration_ap=5.0,
    duration_lf=25.0,
    channels=None,
    features_to_compute=None,
    output_dir=Path("."),
    scratch_dir=None,
    lf_k_filter=False,
    **kwargs,
):
    """Compute electrophysiological features from SpikeGLX readers.

    The function loads a snippet of AP (action potential) and/or LF (local field
    potential) data from SpikeGLX readers, validates the requested time range,
    performs lightweight bad-channel detection, and forwards the raw arrays to
    :func:`compute_features_from_raw`. Either stream can be omitted; the
    underlying feature pipeline will adjust the feature sets and metadata to match
    the available data.

    Args:
        sr_lf (SpikeGLXReader, optional): Reader for the LF stream. Provide ``None``
            to skip LF processing.
        sr_ap (SpikeGLXReader, optional): Reader for the AP stream. Provide ``None``
            to skip AP processing.
        t0 (float): Start time in seconds of the window to process.
        duration (float): Duration in seconds of the processing window.
        channels (dict, optional): Channel metadata used for bad-channel labels.
            When ``channels`` is ``None`` or does not contain ``"labels"``, the
            function will attempt to infer labels from the available data.
        features_to_compute (list, optional): Subset of feature families to pass to
            the downstream computation. ``None`` delegates the decision to
            :func:`compute_features_from_raw`.
        output_dir (Path, optional): Directory where intermediate feature parquet
            files are written. Defaults to the current directory.
        scratch_dir (Path, optional): Location for temporary scratch data (e.g.
            dartsort artifacts).
        lf_k_filter (bool | None, optional): Spatial-filter mode forwarded to
            :func:`ibldsp.voltage.destripe_lfp`. Keep the default ``False`` to
            preserve the current CAR behavior, or pass ``None`` to disable LF
            spatial filtering entirely.
        **kwargs: Additional options forwarded to
            :func:`compute_features_from_raw`.

    Returns:
        pd.DataFrame: Feature table covering the requested time window.

    Raises:
        ValueError: If the requested time window is negative or extends beyond the
            available samples of the provided reader(s).
        IndexError: If SpikeGLX fails to provide the requested samples.

    Example:
        >>> from spikeglx import Reader
        >>> sr_ap = Reader('path/to/ap.bin')
        >>> sr_lf = Reader('path/to/lf.bin')
        >>> df = online_feature_computation(
        ...     sr_lf=sr_lf, sr_ap=sr_ap, t0=100.0, duration=3.0
        ... )

    Notes:
        - FFT sizes are rounded up to the next fast length for efficient spectral
          computations.
        - LF reads include a three-sample latency offset to align with AP timing.
        - Geometry is sourced from whichever reader is available; at least one
          reader must define geometry.
        - Bad-channel detection falls back from full recordings to the extracted
          snippet when necessary.
    """
    # Validate start time is non-negative
    if t0 < 0:
        raise ValueError(f"Start time t0 ({t0}) cannot be negative")
    if sr_ap is not None:
        # Calculate the next fast length for the AP data to optimize FFT operations
        ns_ap = scipy.fft.next_fast_len(int(sr_ap.fs * duration_ap), real=True)
        max_time_ap = sr_ap.ns / sr_ap.fs
        # Validate AP data duration
        if t0 + duration_ap > max_time_ap:
            raise ValueError(
                f"Requested time range ({t0} to {t0 + duration_ap}) exceeds AP data duration ({max_time_ap})"
            )

        # Calculate start indices for data access
        n0_ap = int(sr_ap.fs * t0)

        # Verify channel indices are valid
        n_channels_ap = sr_ap.nc - sr_ap.nsync

        # Load AP data, ignoring sync pulse columns
        try:
            raw_ap = sr_ap[slice(n0_ap, n0_ap + ns_ap), :n_channels_ap].T
        except IndexError as e:
            raise IndexError(
                f"Failed to access AP data: {str(e)}. Check if time range or channel count is valid."
            )

    else:
        raw_ap = None

    if sr_lf is not None:
        # Calculate the next fast length for the LF data to optimize FFT operations
        ns_lf = scipy.fft.next_fast_len(int(sr_lf.fs * duration_lf), real=True)
        max_time_lf = sr_lf.ns / sr_lf.fs
        # Validate AP data duration

        # Validate LF data duration
        if t0 + duration_lf > max_time_lf:
            raise ValueError(
                f"Requested time range ({t0} to {t0 + duration_lf}) exceeds LF data duration ({max_time_lf})"
            )

        # Calculate start indices for data access
        n0_lf = int(sr_lf.fs * t0 + 3)  # Add 3 to account for LF latency

        # Verify channel indices are valid
        n_channels_lf = sr_lf.nc - sr_lf.nsync

        # Load LF data with latency offset
        try:
            raw_lf = sr_lf[slice(n0_lf, n0_lf + ns_lf), :n_channels_lf].T
        except IndexError as e:
            raise IndexError(
                f"Failed to access LF data: {str(e)}. Check if time range or channel count is valid."
            )

    else:
        raw_lf = None

    # Determine channel labels for bad channel detection
    if channels is None or channels.get("labels") is None:
        # If we have access to the whole recording, then we can detect bad channels from the cbin file.
        if sr_ap is not None and sr_ap.file_bin is not None:
            channel_labels = ibldsp.voltage.detect_bad_channels_cbin(sr_ap.file_bin)
        # Else we can detect bad channels from the snippet of data.
        elif raw_ap is not None:
            channel_labels, _ = ibldsp.voltage.detect_bad_channels(raw_ap, fs=sr_ap.fs)
        # There is no need to update the channel labels, since we do it later on during aggregation.
        else:
            channel_labels = None
    else:
        channel_labels = channels["labels"]

    # Delegate feature computation to the raw data processing function
    return compute_features_from_raw(
        raw_ap=raw_ap,
        raw_lf=raw_lf,
        fs_ap=sr_ap.fs if sr_ap is not None else None,
        fs_lf=sr_lf.fs if sr_lf is not None else None,
        geometry=sr_ap.geometry if sr_ap is not None else sr_lf.geometry,
        channel_labels=channel_labels,
        features_to_compute=features_to_compute,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        lf_k_filter=lf_k_filter,
        **kwargs,
    )


def load_data_from_pid(
    pid, one, probe_level_dir, recompute_channels=False, eid=None, probe_name=None
):
    """Load electrophysiological data and channel information using a probe ID from the ONE database.

    This function loads both AP and LF data from the ONE database using a probe ID. It supports both
    standard ONE clients and OneSdsc clients. The function also handles channel information, either
    loading it from a cached file or computing it from the SpikeGLX reader. File locking is used to
    prevent concurrent access to channel files.

    Args:
        pid (str): Probe ID for the data to be loaded.
        one (ONE): ONE client instance for accessing the database.
        probe_level_dir (Path): Directory for probe-level data storage and caching.
        recompute_channels (bool, optional): Whether to recompute channel information even if a cached
            file exists. Defaults to False.
        eid (str, optional): Session ID. Required when using OneSdsc client.
        probe_name (str, optional): Probe name. Required when using OneSdsc client.

    Returns:
        tuple: A tuple containing:
            - sr_ap: SpikeGLX reader for AP data
            - sr_lf: SpikeGLX reader for LF data
            - channels: Dictionary containing channel information

    Raises:
        AssertionError: If required parameters are missing for OneSdsc (eid, probe_name) or if data loading fails.

    Example:
        >>> from one.api import ONE
        >>> one = ONE()
        >>> sr_ap, sr_lf, channels = load_data_from_pid(
        ...     pid='probe00', one=one, probe_level_dir=Path('output')
        ... )

    Note:
        - The function supports both standard ONE and OneSdsc clients with different parameter requirements.
        - Channel information is cached in 'channels.pqt' files to avoid recomputation.
        - File locking prevents concurrent access to channel files during read/write operations.
        - If channel information cannot be loaded, the function falls back to an empty dictionary.
        - The function automatically extracts geometry information from SpikeGLX readers when needed.
    """
    # Log the start of data loading process
    logger.info(f"Loading data using PID: {pid}")

    # Handle different ONE client types (standard vs OneSdsc)
    if one.__class__.__name__ == "OneSdsc":
        logger.info(f"Loading data using OneSdsc: {pid}")
        # Validate required parameters for OneSdsc
        assert pid is not None and eid is not None and probe_name is not None, (
            "pid, eid, and probe_name are required for OneSdsc"
        )

        # Create SpikeSortingLoader for OneSdsc with streaming disabled
        ssl = SpikeSortingLoader(pid=pid, eid=eid, pname=probe_name, one=one)
        stream = False

    else:
        # Standard ONE client handling
        assert pid is not None, "PID must be a string"
        ssl = SpikeSortingLoader(pid=pid, one=one)
        stream = True

    # Load AP and LF electrophysiological data using the SpikeSortingLoader
    sr_ap = ssl.raw_electrophysiology(band="ap", stream=stream)
    sr_lf = ssl.raw_electrophysiology(band="lf", stream=stream)

    # Verify that data loading was successful
    assert sr_ap is not None and sr_lf is not None, "Failed to load data"

    # Set up the channels file path for caching
    if probe_level_dir is not None:
        file_channels = Path(probe_level_dir) / "channels.pqt"

    # Load channel information from cache if available and recomputation is not requested
    if (
        probe_level_dir is not None
        and file_channels.exists()
        and (not recompute_channels)
    ):
        logger.info(f"Loading channels from {file_channels}")
        # Use file locking to prevent concurrent access
        lock_file = str(file_channels) + ".lock"
        lock = FileLock(lock_file, timeout=60)
        logger.info(f"{os.getpid()} : Acquiring lock for reading the channels dataset.")
        with lock:
            logger.info(
                f"{os.getpid()} : Acquired lock for reading the channels dataset."
            )
            # Load channel information from Parquet file
            channels = pd.read_parquet(file_channels)
            logger.info(f"{os.getpid()} : Finished reading the channels dataset.")
        # Convert DataFrame columns to numpy arrays for consistency
        channels = {col: channels[col].to_numpy() for col in channels.columns}
    else:
        # Load channel information directly from the SpikeGLX reader
        logger.info("Getting channels from SpikeGLX reader")
        try:
            channels = ssl.load_channels()
            # Extract geometry information if not already present
            if ("axial_um" not in channels) and ("y" in sr_ap.geometry):
                channels["axial_um"] = sr_ap.geometry["y"]
            if ("lateral_um" not in channels) and ("x" in sr_ap.geometry):
                channels["lateral_um"] = sr_ap.geometry["x"]
        except KeyError as e:
            # Handle missing channel keys gracefully
            logger.info(f"Channels key was not found: {str(e)}")
            channels = {}
        except Exception as e:
            # Handle any other errors during channel loading
            logger.error(f"Failed to load channels: {str(e)}")
            logger.error("Exception details:", exc_info=True)
            channels = {}

    # Log session information for debugging
    logger.info(
        f"Session path: {ssl.session_path}, probe name: {ssl.pname}, eid: {ssl.eid}"
    )
    return sr_ap, sr_lf, channels


# TODO - Handle how the probe level directory and channels data is handled. (Similar to the load_data_from_pid case)
def load_data_from_files(ap_file=None, lf_file=None):
    """Open SpikeGLX `.cbin` recordings and construct channel metadata.

    Args:
        ap_file (str, optional): Path to an AP `.cbin` file. Leave as ``None``
            to load only the LF stream.
        lf_file (str, optional): Path to an LF `.cbin` file. Leave as ``None``
            to load only the AP stream.

    Returns:
        tuple: ``(sr_ap, sr_lf, channels)`` where ``sr_ap`` and ``sr_lf`` are
        SpikeGLX ``Reader`` instances (set to ``None`` when a stream is not
        requested) and ``channels`` is a dictionary containing at least
        ``"rawInd"``, ``"axial_um"``, and ``"lateral_um"`` values derived from
        the available geometry.

    Raises:
        ValueError: If neither AP nor LF file is supplied.
        ImportError: If the ``spikeglx`` package is not installed.
        RuntimeError: If the reader initialization fails for any supplied file.
    """
    logger.info(f"Loading data from files: AP={ap_file}, LF={lf_file}")

    if (ap_file is None) and (lf_file is None):
        raise ValueError("One of the AP and LF .cbin files must be provided")

    try:
        from spikeglx import Reader

        sr_ap = Reader(ap_file) if ap_file is not None else None
        sr_lf = Reader(lf_file) if lf_file is not None else None

        # Todo here I have to add the channel information
        channels = {}
        channels["rawInd"] = (
            np.arange(sr_ap.nc - sr_ap.nsync)
            if sr_ap is not None
            else np.arange(sr_lf.nc - sr_lf.nsync)
        )
        channels["axial_um"] = (
            sr_ap.geometry["y"] if sr_ap is not None else sr_lf.geometry["y"]
        )
        channels["lateral_um"] = (
            sr_ap.geometry["x"] if sr_ap is not None else sr_lf.geometry["x"]
        )

        return sr_ap, sr_lf, channels
    except ImportError:
        raise ImportError("spikeglx package is required to read .cbin files")
    except Exception as e:
        raise RuntimeError(f"Failed to load .cbin files: {str(e)}")


# TODO - Allow pid to be a dict so that it can be used for SDSC computation.
# Also change the name of the variable from pid to something else.
# TODO - In compute features function, first check if channels file exists, if yes, then load from it. There should be an option to foce re-calculate it.
def compute_features(
    pid=None,
    t_start=None,
    duration=None,
    one=None,
    ap_file=None,
    lf_file=None,
    traj_dict=None,
    features_to_compute=None,
    output_dir=Path("."),
    recompute_channels=False,
    **kwargs,
):
    """Compute features from either PID or .cbin files.

    Args:
        pid (str or dict, optional): Probe ID or probe info dict. Required if ap_file and lf_file are not provided.
        t_start (float): Start time in seconds. Defaults to 0.0 if not specified.
        duration (float, optional): Duration in seconds. If None, will use the entire available duration.
        one (ONE, optional): ONE client instance. Required if pid is provided.
        ap_file (str, optional): Path to AP .cbin file. Required if pid is not provided.
        lf_file (str, optional): Path to LF .cbin file. Required if pid is not provided.
        traj_dict (dict, optional): Dictionary containing trajectory information with keys:
            - x, y, z: coordinates
            - depth, theta, phi: insertion parameters
            Required if using .cbin files and want to add xyz target information.
        features_to_compute (list, optional): List of feature sets to compute
        output_dir (Path, optional): Output directory for saving features
        recompute_channels (bool, optional): Whether to recompute channel information. Defaults to False.
        **kwargs: Additional keyword arguments

    Returns:
        pd.DataFrame: DataFrame containing computed features

    Raises:
        ValueError: If ONE client instance is required when using PID, or if both PID and .cbin files are provided,
            or if both AP and LF .cbin files are not provided when not using PID.

    Note:
        This function is deprecated. Please use compute_features_from_pid instead.
    """
    logger.warning(
        "This function is deprecated now, and will be removed soon. Please use compute_features_from_pid instead."
    )
    # Create a dictionary with all the function arguments
    params = {
        "pid": pid,
        "t_start": t_start,
        "duration": duration,
        "ap_file": ap_file,
        "output_dir": output_dir,
    }

    # Setup the output directory
    probe_level_dir, snippet_level_dir = setup_output_directory(params)

    # Validate input combinations
    if pid is not None:
        if one is None:
            raise ValueError("ONE client instance is required when using PID")
        if ap_file is not None or lf_file is not None:
            raise ValueError("Cannot provide both PID and .cbin files")
        sr_ap, sr_lf, channels = load_data_from_pid(
            pid, one, probe_level_dir, recompute_channels
        )
    else:
        if ap_file is None or lf_file is None:
            raise ValueError(
                "Both AP and LF .cbin files must be provided when not using PID"
            )
        sr_ap, sr_lf, channels = load_data_from_files(ap_file, lf_file, probe_level_dir)

    # Convert time parameters to float
    t_start = float(t_start)

    # If duration is None, use the entire available duration
    if duration is None:
        max_time_ap = sr_ap.ns / sr_ap.fs
        max_time_lf = sr_lf.ns / sr_lf.fs
        duration = min(max_time_ap, max_time_lf) - t_start
    else:
        duration = float(duration)

    # Compute features

    df = online_feature_computation(
        sr_ap=sr_ap,
        sr_lf=sr_lf,
        t0=t_start,
        duration=duration,
        channels=channels,
        features_to_compute=features_to_compute,
        output_dir=snippet_level_dir,
        **kwargs,
    )

    # Add xyz target information if available
    if pid is not None and one is not None:
        # Mode 1: Using Alyx database
        # if pid is a dict, then extract eid and probe_name from it
        if isinstance(pid, dict):
            probe_id = pid["pid"]
        else:
            probe_id = pid
        channels = add_target_coordinates(pid=probe_id, one=one, channels=channels)
    elif traj_dict is not None:
        # Mode 2: Using direct trajectory dictionary
        channels = add_target_coordinates(traj_dict=traj_dict, channels=channels)
    else:
        logger.warning(
            "No trajectory information available, skipping xyz target addition"
        )

    # Export the channels file
    file_channels = probe_level_dir / "channels.pqt"
    if not file_channels.exists():
        try:
            df_channels = pd.DataFrame(channels).rename(columns={"rawInd": "channel"})
            df_channels.to_parquet(file_channels)
        except Exception as e:
            logger.info(f"Failed to export channels file: {str(e)}")

    return df


# TODO - Add the time taken to compute the features using a decorator, and it can be used for each feature computation as well.
# TODO  - Make the channel target detection as a optional step so that this function is not IBL specific.
def compute_features_from_pid(
    pid=None,
    eid=None,
    probe_name=None,
    t_start=None,
    duration=None,
    duration_ap=5,
    duration_lf=25,
    one=None,
    features_to_compute=None,
    output_dir=None,
    recompute_channels=False,
    scratch_dir=None,
    feature_params=None,
    **kwargs,
):
    """Compute electrophysiological features from a probe ID using the ONE database.

    This function serves as the main entry point for computing features from a specific probe.
    It handles the complete pipeline: loading data from ONE, setting up output directories,
    processing channel information, computing features, and optionally saving results with metadata.

    Args:
        pid (str, optional): Probe ID. Required for standard ONE usage, also required for OneSdsc.
        eid (str, optional): Session ID. Required when using OneSdsc.
        probe_name (str, optional): Probe name. Required when using OneSdsc.
        t_start (float, optional): Start time in seconds for feature computation. Defaults to 0.0 if not specified.
        duration (float, optional): Duration in seconds for feature computation. If None, uses the entire available duration.
        one (ONE): ONE client instance. Required for data loading.
        features_to_compute (list, optional): List of feature sets to compute. If None, uses default feature sets.
          should be a subset of ['lf', 'ap', 'waveforms', 'csd']
        output_dir (Path, optional): Output directory for saving features and metadata. If None, features are not saved.
        recompute_channels (bool, optional): Whether to recompute channel information even if channels.pqt file is present.
            Defaults to False.
        scratch_dir (Path, optional): Directory for temporary files (e.g., dartsort scratch files).
        feature_params (FeatureParams | dict, optional): Per-feature parameters
            forwarded to the engine, as a ``FeatureParams`` or a nested dict
            (e.g. ``{"csd": {"scale": False}}``). ``None`` uses the defaults.
        **kwargs: Additional keyword arguments passed to the feature computation pipeline.

    Returns:
        pd.DataFrame: DataFrame containing the computed features for the specified time window.

    Raises:
        ValueError: If ONE client instance is not provided.
        AssertionError: If required parameters are missing for OneSdsc (eid, probe_name).

    Example:
        >>> from one.api import ONE
        >>> one = ONE()
        >>> df = compute_features_from_pid(
        ...     pid='probe00',
        ...     t_start=100.0,
        ...     duration=30.0,
        ...     one=one,
        ...     output_dir=Path('output')
        ... )

    Note:
        - The function automatically determines the maximum available duration if duration is None.
        - Cached channel information is used unless recompute_channels is True.
        - Target coordinates are added to channel information if not already present.
        - Features are computed via ephysatlas.feature_calculators.IBLPIDFeatureCalculator.
        - Metadata is added to all output files for provenance tracking.
        - The function uses file locking to prevent concurrent writes to channel files.
    """
    # Lazy import: feature_calculators.base imports compute_features_from_raw from
    # this module, so importing the package at module scope would be circular.
    from ephysatlas.feature_calculators import (
        FeatureComputationOptions,
        IBLPIDFeatureCalculator,
        SnippetWindow,
    )

    # Deprecated single-duration override.
    if duration is not None:
        logger.warning(
            "The 'duration' parameter is deprecated and will be removed in future versions. "
            "Please use 'duration_ap' and 'duration_lf' instead."
        )
        duration_ap = duration_lf = duration

    logger.info(f"ProcessID for the process: {os.getpid()}")

    # Validate input parameters based on ONE client type.
    if one is None:
        raise ValueError("ONE client instance is required when using PID")
    elif one.__class__.__name__ == "OneSdsc":
        assert pid is not None, "PID is required when using SDSC"
        assert eid is not None, "EID is required when using SDSC"
        assert probe_name is not None, "Probe name is required when using SDSC"

    calc = IBLPIDFeatureCalculator(pid=pid, one=one, eid=eid, probe_name=probe_name)

    # Resolve the snippet window: fall back to the maximum available duration
    # when a duration is not specified.
    t_start = float(t_start) if t_start is not None else 0.0
    max_ap, max_lf = calc.available_duration()
    duration_ap = (max_ap - t_start) if duration_ap is None else float(duration_ap)
    duration_lf = (
        (min(max_ap, max_lf) - t_start) if duration_lf is None else float(duration_lf)
    )

    window = SnippetWindow(
        t_start=t_start, duration_ap=duration_ap, duration_lf=duration_lf
    )
    options = FeatureComputationOptions(
        features_to_compute=features_to_compute,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        recompute_channels=recompute_channels,
        include_trajectory=True,
        lf_k_filter=False,
        feature_params=feature_params,
        extra_kwargs=kwargs,
    )
    result = calc.compute_snippet(window, options)

    # Preserve the parquet-metadata side effect the aggregation layer relies on
    # (utils.get_aggregated_snippets_df reads these .attrs to build the manifest).
    if output_dir is not None and result.snippet_level_dir is not None:
        snippet_attrs = {
            "pid": pid,
            "t_start": t_start,
            "duration_ap": duration_ap,
            "duration_lf": duration_lf,
            "base_level_dir": output_dir.as_posix(),
            "snippet_level_dir": result.snippet_level_dir.relative_to(
                output_dir
            ).as_posix(),
        }
        add_metadata_to_parquet_files(**snippet_attrs)

    return result.features


def compute_features_from_file(
    ap_file=None,
    lf_file=None,
    t_start=None,
    duration=None,
    duration_ap=5,
    duration_lf=25,
    traj_dict=None,
    features_to_compute=None,
    output_dir=None,
    scratch_dir=None,
    lf_k_filter=False,
    feature_params=None,
    **kwargs,
):
    """Compute features from .cbin files.

    Features are computed via
    :class:`ephysatlas.feature_calculators.SpikeGLXFileFeatureCalculator`, which
    reads the AP/LF ``.cbin`` files through ``spikeglx.Reader`` and shares the
    computation engine (:func:`compute_features_from_raw`) with
    :func:`compute_features_from_pid`.

    Args:
        ap_file (str, optional): Path to an AP `.cbin` file. Must be supplied if
            ``lf_file`` is ``None``.
        lf_file (str, optional): Path to an LF `.cbin` file. Must be supplied if
            ``ap_file`` is ``None``.
        t_start (float, optional): Start time (seconds) of the snippet. Defaults
            to ``0.0`` when omitted.
        duration (float, optional): Duration in seconds for feature computation. If None, uses the entire available duration.
            Deprecated: use duration_ap and duration_lf instead.
        duration_ap (float, optional): Duration in seconds for AP feature computation. If None, uses the entire available AP duration.
        duration_lf (float, optional): Duration in seconds for LF feature computation. If None, uses the entire available LF duration.
        traj_dict (dict, optional): Trajectory dictionary with keys ``x``,
            ``y``, ``z``, ``depth``, ``theta``, ``phi`` for adding target
            coordinate columns.
        features_to_compute (list, optional): Feature families to compute. ``None``
            lets the feature calculator pick defaults from the available bands.
        output_dir (Path, optional): Root directory for cached outputs. A
            snippet-level directory is created beneath it.
        scratch_dir (Path, optional): Location for temporary scratch files
            generated by downstream feature routines.
        lf_k_filter (bool | None, optional): Spatial-filter mode forwarded to
            :func:`ibldsp.voltage.destripe_lfp`. The default ``False`` preserves
            the current CAR-style LF destriping; use ``None`` to disable LF
            spatial filtering.
        feature_params (FeatureParams | dict, optional): Per-feature parameters
            forwarded to the engine, as a ``FeatureParams`` or a nested dict
            (e.g. ``{"csd": {"scale": False}}``). ``None`` uses the defaults.
        **kwargs: Additional keyword arguments forwarded to
            :func:`compute_features_from_raw`.

    Returns:
        pd.DataFrame: Aggregated feature table for the requested time range.

    Raises:
        ValueError: If neither AP nor LF file is provided.

    Note:
        The on-disk output layout is named after the AP/LF file stem
        (``pid = calculator.name``) via
        :func:`ephysatlas.utils.setup_output_directory`. This replaces the
        earlier md5-hash probe directory and ``probe_unknown_pid_*`` snippet
        directory; the returned DataFrame, the ``channels.pqt`` contents, and the
        per-snippet parquet ``.attrs`` the aggregation layer reads
        (``filename``/``t_start``/``duration_ap``/``duration_lf``) are unchanged.
    """
    # Lazy import: feature_calculators.base imports compute_features_from_raw from
    # this module, so importing the package at module scope would be circular.
    from ephysatlas.feature_calculators import (
        FeatureComputationOptions,
        SnippetWindow,
        SpikeGLXFileFeatureCalculator,
    )

    # Validate input.
    if (ap_file is None) and (lf_file is None):
        raise ValueError("Both AP and LF .cbin files must be provided")

    # Deprecated single-duration override.
    if duration is not None:
        logger.warning(
            "The 'duration' parameter is deprecated and will be removed in future versions. "
            "Please use 'duration_ap' and 'duration_lf' instead."
        )
        duration_ap = duration_lf = duration

    logger.info(f"ProcessID for the process: {os.getpid()}")

    # The trajectory dict is supplied to the calculator (not per snippet call);
    # include_trajectory below only adds target coordinates when it is provided.
    calc = SpikeGLXFileFeatureCalculator(
        ap_file=ap_file, lf_file=lf_file, traj_dict=traj_dict
    )

    # Resolve the snippet window: fall back to the maximum available duration
    # when a duration is not specified.
    t_start = float(t_start) if t_start is not None else 0.0
    max_ap, max_lf = calc.available_duration()
    duration_ap = (max_ap - t_start) if duration_ap is None else float(duration_ap)
    duration_lf = (
        (min(max_ap, max_lf) - t_start) if duration_lf is None else float(duration_lf)
    )

    window = SnippetWindow(
        t_start=t_start, duration_ap=duration_ap, duration_lf=duration_lf
    )
    options = FeatureComputationOptions(
        features_to_compute=features_to_compute,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        # The file path overwrites channels.pqt on every call, so force a rewrite
        # rather than the base class's conditional (keep-if-exists) write.
        recompute_channels=True,
        include_trajectory=traj_dict is not None,
        lf_k_filter=lf_k_filter,
        feature_params=feature_params,
        extra_kwargs=kwargs,
    )
    result = calc.compute_snippet(window, options)

    # Preserve the parquet-metadata side effect the aggregation layer relies on
    # (utils.get_aggregated_snippets_df reads these .attrs to build the manifest).
    # The file path stamps "filename" (compute_features_from_pid stamps "pid").
    if output_dir is not None and result.snippet_level_dir is not None:
        snippet_attrs = {
            "filename": ap_file if ap_file is not None else lf_file,
            "t_start": t_start,
            "duration_ap": duration_ap,
            "duration_lf": duration_lf,
            "base_level_dir": output_dir.as_posix(),
            "snippet_level_dir": result.snippet_level_dir.relative_to(
                output_dir
            ).as_posix(),
        }
        add_metadata_to_parquet_files(**snippet_attrs)

    return result.features


def _validate_arrays_and_labels(arr_ap, arr_lf, geometry, fs_ap, fs_lf, channel_labels):
    """Validate AP/LF arrays against geometry/sampling rates and default labels.

    Shared by :func:`compute_features_from_raw` (raw arrays) and
    :func:`compute_features_from_destriped` (destriped arrays); both require the
    same shapes/geometry alignment. Returns ``channel_labels``, defaulted to
    zeros when ``None``.
    """
    if arr_ap is None and arr_lf is None:
        raise ValueError("One of the AP or LF data must be provided")
    if arr_ap is not None and arr_lf is not None:
        assert arr_ap.shape[0] == arr_lf.shape[0], (
            "Number of channels must match between AP and LF data"
        )
    if arr_ap is not None:
        assert arr_ap.ndim == 2, "Input array must be 2D"
        assert arr_ap.shape[0] == len(geometry["x"]) == len(geometry["y"]), (
            "Number of channels must match geometry"
        )
        assert fs_ap > 0, "Sampling frequencies must be positive"
    if arr_lf is not None:
        assert arr_lf.ndim == 2, "Input array must be 2D"
        assert arr_lf.shape[0] == len(geometry["x"]) == len(geometry["y"]), (
            "Number of channels must match geometry"
        )
        assert fs_lf > 0, "Sampling frequencies must be positive"
    if channel_labels is None:
        channel_labels = (
            np.zeros(arr_ap.shape[0])
            if arr_ap is not None
            else np.zeros(arr_lf.shape[0])
        )
    return channel_labels


def destripe_ap_lf(
    raw_ap,
    raw_lf,
    fs_ap=None,
    fs_lf=None,
    geometry=None,
    channel_labels=None,
    neuropixel_version=1,
    ap_k_filter=False,
    lf_k_filter=False,
    nshank=1,
):
    """Destripe raw AP and LF snippets with ibldsp.

    Single source of truth for destriping, shared by
    :func:`compute_features_from_raw` and
    :meth:`ephysatlas.feature_calculators.base.BaseFeatureCalculator.get_destriped_snippet`,
    so the feature engine and the debug/inspection path cannot diverge.

    Args:
        raw_ap, raw_lf (np.ndarray | None): Raw AP/LF arrays shaped
            ``(n_channels, n_samples)``; ``None`` skips that band.
        fs_ap, fs_lf (float): Sampling frequencies (Hz).
        geometry (dict): Channel geometry passed to ibldsp.
        channel_labels (np.ndarray | None): Bad-channel labels.
        neuropixel_version (int): Neuropixels version.
        ap_k_filter (bool): Spatial-filter mode for AP destriping.
        lf_k_filter (bool | None): Spatial-filter mode for LF destriping.
        nshank (int): Number of probe shanks.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: ``(des_ap, des_lf)``.
    """
    des_ap = None
    if raw_ap is not None:
        des_ap = ibldsp.voltage.destripe(
            raw_ap,
            fs=fs_ap,
            h=geometry,
            neuropixel_version=neuropixel_version,
            channel_labels=channel_labels,
            k_filter=ap_k_filter,
            nshank=nshank,
        )
    des_lf = None
    if raw_lf is not None:
        des_lf = ibldsp.voltage.destripe_lfp(
            raw_lf,
            fs=fs_lf,
            h=geometry,
            neuropixel_version=neuropixel_version,
            channel_labels=channel_labels,
            k_filter=lf_k_filter,
            nshank=nshank,
        )
    return des_ap, des_lf


# TODO - I can make this function more modular so that that specifying just one of the raw_ap or raw_lf can make things more easier.
def compute_features_from_raw(
    raw_ap,
    raw_lf,
    fs_ap=None,
    fs_lf=None,
    geometry=None,
    channel_labels=None,
    neuropixel_version=1,
    features_to_compute=None,
    output_dir=Path("."),
    scratch_dir=None,
    lf_k_filter=False,
    feature_params=None,
    **kwargs,
):
    """Compute electrophysiological features from raw numpy arrays of AP and LF data.

    This function is the core feature computation engine that processes raw electrophysiological data
    and computes various feature sets. It handles data destriping, feature computation for different
    modalities (LF, CSD, AP, waveforms), and optionally saves results to files. The function supports
    both computation and loading of cached features.

    Args:
        raw_ap (np.ndarray, optional): AP voltage array shaped
            ``(n_channels, n_samples)``. Supply ``None`` to disable AP feature
            computation.
        raw_lf (np.ndarray, optional): LF voltage array shaped
            ``(n_channels, n_samples)``. Supply ``None`` to disable LF/CSD
            feature computation.
        fs_ap (float, optional): Sampling frequency (Hz) for the AP stream. Must
            be positive when ``raw_ap`` is provided.
        fs_lf (float, optional): Sampling frequency (Hz) for the LF stream. Must
            be positive when ``raw_lf`` is provided.
        geometry (dict): Mapping containing at least ``"x"`` and ``"y"`` arrays
            describing channel coordinates.
        channel_labels (np.ndarray, optional): Per-channel label array used to
            mask bad channels during destriping. Defaults to zeros matching the
            available data.
        neuropixel_version (int): Neuropixel probe version passed to the
            destriper.
        features_to_compute (list, optional): Feature families to evaluate.
            ``None`` computes all supported families for the given inputs.
            Supported values are ``["lf", "csd", "ap", "waveforms"]``.
        output_dir (Path, optional): Directory where individual feature parquet
            files (and optional waveform artifacts) are written. ``None`` skips
            writing to disk.
        scratch_dir (Path, optional): Location for temporary scratch data used
            by waveform extraction.
        lf_k_filter (bool | None, optional): Spatial-filter mode passed to
            :func:`ibldsp.voltage.destripe_lfp`. The default ``False`` preserves
            the current CAR behavior; use ``None`` to bypass LF spatial
            filtering.
        feature_params (optional): Optional ``FeatureParams``-like object used to
            override the per-feature kwargs. Read duck-typed (``.lf`` / ``.csd``
            attributes) so this module carries no dependency on the
            ``feature_calculators`` package. When ``None`` the defaults reproduce
            today's behavior exactly. A truthy ``feature_params.lf.compute_rms_no_car``
            additionally enables the ``rms_lf_no_car`` LF feature.
        **kwargs: Extra options controlling the workflow, such as
            ``skip_saved_computation`` or ``save_waveforms``.

    Returns:
        pd.DataFrame: Outer-joined feature table keyed by ``channel``.

    Raises:
        ValueError: If neither AP nor LF data is supplied, or if an unsupported
            feature family is requested.
        AssertionError: If provided arrays are not 2D, geometry dimensions do not
            align, or sampling frequencies are invalid for the requested streams.

    Example:
        >>> import numpy as np
        >>> raw_ap = np.random.randn(384, 30000)
        >>> raw_lf = np.random.randn(384, 3000)
        >>> geometry = {'x': np.arange(384), 'y': np.arange(384)}
        >>> df = compute_features_from_raw(
        ...     raw_ap=raw_ap, raw_lf=raw_lf, fs_ap=30000, fs_lf=2500,
        ...     geometry=geometry, features_to_compute=['lf', 'ap']
        ... )

    Notes:
        - AP and LF streams are destriped before feature computation.
        - Existing parquet outputs can be reused when
          ``skip_saved_computation`` is ``True``.
        - Waveform features emit optional waveform files when
          ``save_waveforms`` is set.
        - Each feature DataFrame is annotated with ibleatools and feature module
          versions for provenance tracking.
    """
    channel_labels = _validate_arrays_and_labels(
        raw_ap, raw_lf, geometry, fs_ap, fs_lf, channel_labels
    )

    # Destripe both bands through the shared primitive (AP uses CAR-style
    # k_filter=False, matching the previous inline behavior).
    des_ap, des_lf = destripe_ap_lf(
        raw_ap,
        raw_lf,
        fs_ap=fs_ap,
        fs_lf=fs_lf,
        geometry=geometry,
        channel_labels=channel_labels,
        neuropixel_version=neuropixel_version,
        ap_k_filter=False,
        lf_k_filter=lf_k_filter,
    )

    # rms_lf_no_car needs a second, no-CAR (k_filter=None) LF destripe of the raw
    # signal, which cannot be recovered from des_lf. Compute it here (via the same
    # primitive) and hand it to the feature engine, which emits rms_lf_no_car iff it
    # receives this array.
    compute_rms_no_car = False
    if feature_params is not None:
        lf_p = getattr(feature_params, "lf", None)
        compute_rms_no_car = bool(getattr(lf_p, "compute_rms_no_car", False))
    des_lf_no_car = None
    if compute_rms_no_car and raw_lf is not None:
        _, des_lf_no_car = destripe_ap_lf(
            None,
            raw_lf,
            fs_lf=fs_lf,
            geometry=geometry,
            channel_labels=channel_labels,
            neuropixel_version=neuropixel_version,
            lf_k_filter=None,
        )
    logger.info("Destriped AP and LF data")

    return compute_features_from_destriped(
        des_ap,
        des_lf,
        fs_ap=fs_ap,
        fs_lf=fs_lf,
        geometry=geometry,
        channel_labels=channel_labels,
        des_lf_no_car=des_lf_no_car,
        features_to_compute=features_to_compute,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        feature_params=feature_params,
        **kwargs,
    )


def compute_features_from_destriped(
    des_ap,
    des_lf,
    fs_ap=None,
    fs_lf=None,
    geometry=None,
    channel_labels=None,
    des_lf_no_car=None,
    features_to_compute=None,
    output_dir=Path("."),
    scratch_dir=None,
    feature_params=None,
    **kwargs,
):
    """Compute features from already-destriped AP/LF arrays.

    This is the feature-computation half of :func:`compute_features_from_raw`
    (which destripes and then calls this). Call it directly to compute features on
    pre-destriped or cached data without re-destriping.

    Args:
        des_ap, des_lf (np.ndarray | None): Destriped AP/LF arrays shaped
            ``(n_channels, n_samples)``; ``None`` skips that band's features.
        fs_ap, fs_lf (float): Sampling frequencies (Hz).
        geometry (dict): Channel geometry with ``"x"``/``"y"``.
        channel_labels (np.ndarray, optional): Bad-channel labels; defaults to
            zeros.
        des_lf_no_car (np.ndarray, optional): LF destriped with no common-average
            reference (``k_filter=None``). When provided, ``rms_lf_no_car`` is added
            to the LF features; it cannot be derived from ``des_lf`` alone.
        features_to_compute (list, optional): Feature families to evaluate. ``None``
            computes all families supported by the supplied bands.
        output_dir (Path, optional): Where per-feature parquet files are written.
        scratch_dir (Path, optional): Scratch location for waveform extraction.
        feature_params (optional): Duck-typed ``FeatureParams`` (``.lf``/``.csd``).
        **kwargs: Extra options such as ``skip_saved_computation`` /
            ``save_waveforms``.

    Returns:
        pd.DataFrame: Outer-joined feature table keyed by ``channel``.

    Raises:
        ValueError: If neither band is supplied or an unsupported family is
            requested.
    """
    channel_labels = _validate_arrays_and_labels(
        des_ap, des_lf, geometry, fs_ap, fs_lf, channel_labels
    )

    # Define available feature sets
    available_features = ["lf", "csd", "ap", "waveforms"]

    if des_ap is None:
        if features_to_compute is None:
            features_to_compute = ["lf", "csd"]
        else:
            features_to_compute = [f for f in features_to_compute if f in ["lf", "csd"]]
    elif des_lf is None:
        if features_to_compute is None:
            features_to_compute = ["ap", "waveforms"]
        else:
            features_to_compute = [
                f for f in features_to_compute if f in ["ap", "waveforms"]
            ]

    # Validate requested features or use all available features
    if features_to_compute is None:
        features_to_compute = available_features
    else:
        # Check for invalid feature requests
        invalid_features = [
            f for f in features_to_compute if f not in available_features
        ]
        if invalid_features:
            raise ValueError(
                f"Invalid feature sets requested: {invalid_features}. Available options: {available_features}"
            )

    # Initialize dictionary to store computed features
    df = {}

    # Helper function to save features to Parquet files
    def save_features(feature_name, feature_df):
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"{feature_name}_features.pqt"
            feature_df.to_parquet(file_path)
            logger.info(f"Saved {feature_name} features to {file_path}")

    # Helper function to load features from existing files
    def load_features(feature_name):
        if output_dir is not None:
            file_path = output_dir / f"{feature_name}_features.pqt"
            if file_path.exists():
                logger.info(f"Loading {feature_name} features from {file_path}")
                return pd.read_parquet(file_path)
        return None

    # TODO add a new parameter to the compute_features_from_raw function, which checks if the it was called from PID and then calculate the full list of channels dataset.
    # Compute or load each feature set
    # if 'channels' in features_to_compute:
    #     df["channels"] = pd.DataFrame(
    #         {
    #             "lateral_um": geometry["x"],
    #             "axial_um": geometry["y"],
    #             "labels": channel_labels,
    #             "channel": np.arange(len(channel_labels)),
    #         }
    #     )
    #     save_features('channels', df["channels"])
    # else:
    #     df["channels"] = load_features('channels')
    #     if df["channels"] is None:
    #         raise ValueError("Channels features not found in save directory")

    logger.info(f"Starting {features_to_compute} computation")

    # Derive the per-feature kwargs from an optional FeatureParams-like object.
    # This is duck-typed (reads feature_params.lf/.csd attributes if present) so this
    # module does NOT import the feature_calculators package (avoids a circular import).
    # The defaults below reproduce today's kwargs exactly when feature_params is None.
    lf_kwargs = {}
    csd_kwargs = {"decimate": 10}
    ap_kwargs = {}
    waveforms_kwargs = {"scratch_dir": scratch_dir}
    if feature_params is not None:
        lf_p = getattr(feature_params, "lf", None)
        if lf_p is not None:
            lf_kwargs = {"bands": lf_p.bands, "decay_features": lf_p.decay_features}
        csd_p = getattr(feature_params, "csd", None)
        if csd_p is not None:
            csd_kwargs = {
                "bands": csd_p.bands,
                "decimate": csd_p.decimate,
                "scale": csd_p.scale,
            }
        # ap / waveforms typed params are placeholders for now; keep today's kwargs.

    # Define configuration for each feature type with their computation functions and parameters
    feature_configs = {
        "lf": {
            "func": features.lf,
            "args": {"data": des_lf, "fs": fs_lf},
            "kwargs": lf_kwargs,
        },
        "csd": {
            "func": features.csd,
            "args": {"data": des_lf, "fs": fs_lf, "geometry": geometry},
            "kwargs": csd_kwargs,
        },
        "ap": {
            "func": features.ap,
            "args": {
                "data": des_ap,
                "geometry": geometry,
                "channel_labels": channel_labels,
            },
            "kwargs": ap_kwargs,
        },
        "waveforms": {
            "func": features.spikes,
            "args": {"data": des_ap, "fs": fs_ap, "geometry": geometry},
            "kwargs": waveforms_kwargs,
        },
    }

    # Helper function to compute and save individual features
    def compute_and_save_feature(feature_name, config):
        """Helper function to compute and save a feature"""
        # Check if we should skip computation for existing files
        skip_saved = kwargs.get("skip_saved_computation", False)
        if skip_saved:
            existing_features = load_features(feature_name)
            if existing_features is not None:
                logger.info(
                    f"Skipping {feature_name.upper()} computation - file already exists"
                )
                df[feature_name] = existing_features
                return

        logger.info(f"Starting {feature_name.upper()} computation")

        # Compute the feature with special handling for waveforms
        if feature_name == "waveforms":
            # Special handling for waveforms which returns tuple (features, waveform_data)
            df[feature_name], waveforms = config["func"](
                **config["args"], **config["kwargs"]
            )
            # Convert spike count to integer type for consistency
            df[feature_name]["spike_count"] = df[feature_name]["spike_count"].astype(
                "Int64"
            )

            # Save waveform files if requested from the function call of compute_features_from_raw
            if (output_dir is not None) and kwargs.get("save_waveforms", False):
                waveforms_dir = output_dir / "waveforms"
                waveforms_dir.mkdir(parents=True, exist_ok=True)
                # Save waveform arrays in compressed format
                np.save(waveforms_dir / "raw.npy", waveforms["raw"].astype(np.float16))
                np.save(
                    waveforms_dir / "denoised.npy",
                    waveforms["denoised"].astype(np.float16),
                )
                np.save(
                    waveforms_dir / "waveform_channels.npy", waveforms["channel_index"]
                )
                # Save spike information
                waveforms["df_spikes"].to_parquet(waveforms_dir / "spikes.pqt")
        elif feature_name == "lf":
            # Standard lf computation, then append the no-CAR RMS when the caller
            # supplied the no-CAR-destriped LF. rms_lf (CAR) and rms_lf_no_car
            # (no-CAR) capture distinct information; the latter cannot come from
            # des_lf, so it is passed in as des_lf_no_car.
            df[feature_name] = config["func"](**config["args"], **config["kwargs"])
            if des_lf_no_car is not None:
                df[feature_name]["rms_lf_no_car"] = ibldsp.utils.rms(
                    des_lf_no_car, axis=-1
                )
        else:
            # Standard feature computation
            df[feature_name] = config["func"](**config["args"], **config["kwargs"])

        # Add package version metadata for provenance tracking
        df[feature_name].attrs["ibleatools_version"] = ibleatools_version
        df[feature_name].attrs[f"{feature_name}_version"] = features_version

        # Save the computed feature to file
        save_features(feature_name, df[feature_name])

    # Compute each requested feature using the configuration
    for feature_name in features_to_compute:
        if feature_name in feature_configs:
            compute_and_save_feature(feature_name, feature_configs[feature_name])
        else:
            logger.warning(f"Unknown feature type: {feature_name}")

    # Merge all computed features on the 'channel' column using outer join
    df_voltage = reduce(
        lambda left, right: pd.merge(left, right, on="channel", how="outer"),
        [df[k] for k in df.keys()],
    )

    return df_voltage
