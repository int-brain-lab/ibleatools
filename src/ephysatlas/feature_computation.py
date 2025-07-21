from functools import reduce
import logging
import scipy.fft
import pandas as pd
import numpy as np
import iblatlas

from brainbox.io.one import SpikeSortingLoader
import ibldsp.voltage

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
    """
    Get the micro-manipulator target coordinates either from Alyx database or directly from trajectory dictionary.

    Args:
        pid (str, optional): Probe insertion ID. Required if using Alyx database mode.
        one (ONE, optional): ONE client instance. Required if using Alyx database mode.
        channels (dict): Channel information containing at least 'axial_um' and 'rawInd' fields
        traj_dict (dict, optional): Dictionary containing trajectory information with keys:
            - x, y, z: coordinates
            - depth, theta, phi: insertion parameters
            Required if not using Alyx database mode.

    Returns:
        dict: Updated channels dictionary with target coordinates
    """
    needles = NeedlesAtlas()
    allen = AllenAtlas()
    needles.compute_surface()

    # Validate input combinations
    if pid is not None and one is not None:
        # Mode 1: Using Alyx database
        # Check if one is in local mode or remote mode,
        # TODO - Doing this for SDSC computation but need to do it cleaner.
        if one.mode == "local":
            from one.api import ONE

            one_remote = ONE(mode="remote")
            trajs = one_remote.alyx.rest(
                "trajectories", "list", probe_insertion=pid, django="provenance__lte,30"
            )
        else:
            trajs = one.alyx.rest(
                "trajectories", "list", probe_insertion=pid, django="provenance__lte,30"
            )
        traj = next(
            (t for t in trajs if t["provenance"] == "Micro-manipulator"), trajs[0]
        )
    elif traj_dict is not None:
        # Mode 2: Using direct trajectory dictionary
        traj = traj_dict
    else:
        raise ValueError("Either provide (pid, one) or traj_dict")

    # Apply the pitch correction by using iblatlas.atlas.tilt_spherical()
    new_theta, new_phi = iblatlas.atlas.tilt_spherical(
        traj["theta"], traj["phi"], tilt_angle=-5
    )
    traj["theta"] = new_theta
    traj["phi"] = new_phi

    ins = Insertion.from_dict(traj, brain_atlas=needles)

    txyz = np.flipud(ins.xyz)
    # Convert the coordinates from in-vivo to the Allen coordinate system
    txyz = allen.bc.i2xyz(needles.bc.xyz2i(txyz / 1e6, round=False, mode="clip")) * 1e6
    xyz_mm = interpolate_along_track(txyz, channels["axial_um"] / 1e6)

    # Check if the rawInd data exists in the channels dictionary, otherwise use the default 384 channels (Ask OW)
    if ("rawInd" not in channels) and ("channel" not in channels):
        assert channels["axial_um"].size == 384
        channels["rawInd"] = np.arange(384)

    # we interpolate the channels from the deepest point up. The neuropixel y coordinate is from the bottom of the probe
    # Update the channels dictionary with the target coordinates
    channels["x_target"] = xyz_mm[:, 0]
    channels["y_target"] = xyz_mm[:, 1]
    channels["z_target"] = xyz_mm[:, 2]
    return channels


def online_feature_computation(
    sr_lf,
    sr_ap,
    t0,
    duration,
    channels=None,
    features_to_compute=None,
    output_dir=Path("."),
    scratch_dir=None,
    **kwargs,
):
    """
    Compute features from SpikeGLX reader objects.

    Args:
        sr_lf: SpikeGLX reader for LF data
        sr_ap: SpikeGLX reader for AP data
        t0 (float): Start time in seconds
        duration (float): Duration in seconds
        channels (dict, optional): Dict containing channel information
        features_to_compute (list, optional): List of feature sets to compute
        output_dir (Path, optional): Output directory for saving features
        **kwargs: Additional keyword arguments

    Returns:
        tuple: (channels, df) Updated channels dict and computed features DataFrame
    """
    # Calculate the next fast length for the AP data
    ns_ap = scipy.fft.next_fast_len(int(sr_ap.fs * duration), real=True)

    # Calculate the next fast length for the LF data
    ns_lf = scipy.fft.next_fast_len(int(sr_lf.fs * duration), real=True)

    # Check if requested time range is within bounds
    max_time_ap = sr_ap.ns / sr_ap.fs
    max_time_lf = sr_lf.ns / sr_lf.fs

    if t0 < 0:
        raise ValueError(f"Start time t0 ({t0}) cannot be negative")
    if t0 + duration > max_time_ap:
        raise ValueError(
            f"Requested time range ({t0} to {t0 + duration}) exceeds AP data duration ({max_time_ap})"
        )
    if t0 + duration > max_time_lf:
        raise ValueError(
            f"Requested time range ({t0} to {t0 + duration}) exceeds LF data duration ({max_time_lf})"
        )

    # Calculate start and end indices
    n0_ap = int(sr_ap.fs * t0)
    n0_lf = int(sr_lf.fs * t0 + 3)  # Add 3 to account for LF latency

    # Verify channel indices
    n_channels_ap = sr_ap.nc - sr_ap.nsync
    n_channels_lf = sr_lf.nc - sr_lf.nsync

    if n_channels_ap <= 0 or n_channels_lf <= 0:
        raise ValueError(
            f"Invalid number of channels: AP={n_channels_ap}, LF={n_channels_lf}"
        )

    # Ignore the columns which include the sync pulse data
    try:
        raw_ap = sr_ap[slice(n0_ap, n0_ap + ns_ap), :n_channels_ap].T
    except IndexError as e:
        raise IndexError(
            f"Failed to access AP data: {str(e)}. Check if time range or channel count is valid."
        )

    # Add 3 to n0 to account for the 3 samples of latency in the LF data
    try:
        raw_lf = sr_lf[slice(n0_lf, n0_lf + ns_lf), :n_channels_lf].T
    except IndexError as e:
        raise IndexError(
            f"Failed to access LF data: {str(e)}. Check if time range or channel count is valid."
        )

    if channels.get("labels") is None:
        # If we have access to the whole recording, then we can detect bad channels from the cbin file.
        if sr_ap.file_bin is not None:
            channel_labels = ibldsp.voltage.detect_bad_channels_cbin(sr_ap.file_bin)
        # Else we can detect bad channels fromm the snippet of data.
        else:
            channel_labels, _ = ibldsp.voltage.detect_bad_channels(raw_ap, fs=sr_ap.fs)
        # There is no need to update the channel labels, since we do it later on during aggregation.
    else:
        channel_labels = channels["labels"]

    return compute_features_from_raw(
        raw_ap=raw_ap,
        raw_lf=raw_lf,
        fs_ap=sr_ap.fs,
        fs_lf=sr_lf.fs,
        geometry=sr_ap.geometry,
        channel_labels=channel_labels,
        features_to_compute=features_to_compute,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        **kwargs,
    )


# TODO - Need to be clear here , if I want to check based on SDSC or not, VS pid as dict or pid as string.
# (Ask OW) Recomputing channels when launching multiple jobs.
def load_data_from_pid(
    pid, one, probe_level_dir, recompute_channels=False, eid=None, probe_name=None
):
    """
    Load data using a probe ID from the ONE database.

    Args:
        pid (str): Probe ID
        one (ONE): ONE client instance
        probe_level_dir (Path): Directory for probe-level data
        recompute_channels (bool, optional): Whether to recompute channels even if cached
        eid (str, optional): Session ID (required for OneSdsc)
        probe_name (str, optional): Probe name (required for OneSdsc)

    Returns:
        tuple: (sr_ap, sr_lf, channels) SpikeGLX readers and channel information
    """
    logger.info(f"Loading data using PID: {pid}")

    if one.__class__.__name__ == "OneSdsc":
        logger.info(f"Loading data using OneSdsc: {pid}")
        assert pid is not None and eid is not None and probe_name is not None, (
            "pid, eid, and probe_name are required for OneSdsc"
        )

        ssl = SpikeSortingLoader(pid=pid, eid=eid, pname=probe_name, one=one)
        stream = False

    else:
        assert pid is not None, "PID must be a string"
        ssl = SpikeSortingLoader(pid=pid, one=one)
        stream = True

    sr_ap = ssl.raw_electrophysiology(band="ap", stream=stream)
    sr_lf = ssl.raw_electrophysiology(band="lf", stream=stream)

    assert sr_ap is not None and sr_lf is not None, "Failed to load data"

    # Load the channels file
    if probe_level_dir is not None:
        file_channels = Path(probe_level_dir) / "channels.pqt"

    if (
        probe_level_dir is not None
        and file_channels.exists()
        and (not recompute_channels)
    ):
        logger.info(f"Loading channels from {file_channels}")
        lock_file = str(file_channels) + ".lock"
        lock = FileLock(lock_file, timeout=60)
        logger.info(f"{os.getpid()} : Acquiring lock for reading the channels dataset.")
        with lock:
            logger.info(f"{os.getpid()} : Acquired lock for reading the channels dataset.")
            channels = pd.read_parquet(file_channels)
            logger.info(f"{os.getpid()} : Finished reading the channels dataset.")
        channels = {col: channels[col].to_numpy() for col in channels.columns}
    else:
        logger.info("Getting channels from SpikeGLX reader")
        try:
            channels = ssl.load_channels()
            if ("axial_um" not in channels) and ("y" in sr_ap.geometry):
                channels["axial_um"] = sr_ap.geometry["y"]
            if ("lateral_um" not in channels) and ("x" in sr_ap.geometry):
                channels["lateral_um"] = sr_ap.geometry["x"]
        except KeyError as e:
            logger.info(f"Channels key was not found: {str(e)}")
            channels = {}
        except Exception as e:
            logger.error(f"Failed to load channels: {str(e)}")
            logger.debug("Exception details:", exc_info=True)
            channels = {}

    logger.info(f"Session path: {ssl.session_path}, probe name: {ssl.pname}")
    return sr_ap, sr_lf, channels


# TODO - Handle how the probe level directory and channels data is handled. (Similar to the load_data_from_pid case)
def load_data_from_files(ap_file, lf_file, probe_level_dir):
    """
    Load data from .cbin files.

    Args:
        ap_file (str): Path to AP .cbin file
        lf_file (str): Path to LF .cbin file
        probe_level_dir (Path): Directory for probe-level data

    Returns:
        tuple: (sr_ap, sr_lf, channels) SpikeGLX readers and channel information
    """
    logger.info(f"Loading data from files: AP={ap_file}, LF={lf_file}")
    try:
        from spikeglx import Reader

        sr_ap = Reader(ap_file)
        sr_lf = Reader(lf_file)
        # Todo here I have to add the channel information
        channels = {}
        channels["rawInd"] = np.arange(sr_ap.nc - sr_ap.nsync)
        channels["axial_um"] = sr_ap.geometry["y"]

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
    """
    Compute features from either PID or .cbin files.

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
        **kwargs: Additional keyword arguments

    Returns:
        pd.DataFrame: DataFrame containing computed features
    """
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


def compute_features_from_pid(
    pid=None,
    eid=None,
    probe_name=None,
    t_start=None,
    duration=None,
    one=None,
    features_to_compute=None,
    output_dir=None,
    recompute_channels=False,
    scratch_dir=None,
    **kwargs,
):
    """
    Compute features from a probe ID and ONE database.

    Args:
        pid (str): Probe ID (required)
        eid (str, optional): Session ID (required for OneSdsc)
        probe_name (str, optional): Probe name (required for OneSdsc)
        t_start (float, optional): Start time in seconds. Defaults to 0.0 if not specified.
        duration (float, optional): Duration in seconds. If None, will use the entire available duration.
        one (ONE): ONE client instance (required)
        features_to_compute (list, optional): List of feature sets to compute
        output_dir (Path, optional): Output directory for saving features. If None, will not save features.
        recompute_channels (bool, optional): Whether to recompute channels even if cached
        scratch_dir (Path, optional): Directory for temporary files (dartsort scratch)
        **kwargs: Additional keyword arguments

    Returns:
        pd.DataFrame: DataFrame containing computed features
    """
    # Create a dictionary with all the function arguments for setup_output_directory
    params = {
        "pid": pid,
        "t_start": t_start,
        "duration": duration,
        "output_dir": output_dir,
    }

    logger.info(f"ProcessID for the process: {os.getpid()}")

    # Setup the output directory
    probe_level_dir, snippet_level_dir = setup_output_directory(params)

    # Validate input
    if one is None:
        raise ValueError("ONE client instance is required when using PID")
    elif one.__class__.__name__ == "OneSdsc":
        assert pid is not None, "PID is required when using SDSC"
        assert eid is not None, "EID is required when using SDSC"
        assert probe_name is not None, "Probe name is required when using SDSC"

    # Load data using PID
    sr_ap, sr_lf, channels = load_data_from_pid(
        pid, one, probe_level_dir, recompute_channels, eid=eid, probe_name=probe_name
    )

    # Convert time parameters to float
    t_start = float(t_start) if t_start is not None else 0.0

    # If duration is None, use the entire available duration
    if duration is None:
        max_time_ap = sr_ap.ns / sr_ap.fs
        max_time_lf = sr_lf.ns / sr_lf.fs
        duration = min(max_time_ap, max_time_lf) - t_start
    else:
        duration = float(duration)

    # Update the channel file with target information.
    # Add xyz target information using Alyx database
    # Check if the target information is already present in the channels dataset, if yes then skip it.
    if "x_target" not in channels.keys() or "y_target" not in channels.keys() or "z_target" not in channels.keys():
        channels = add_target_coordinates(pid=pid, one=one, channels=channels)

    if ("rawInd" not in channels) and ("channel" not in channels):
        assert channels["axial_um"].size == 384
        channels["rawInd"] = np.arange(384)

    # Export the channels file
    if probe_level_dir is not None:
        file_channels = probe_level_dir / "channels.pqt"

    # TODO have another condition that checks if the existing channels file has all channels or if it matches the channels dict.
    # TODO Make a module channel computation function that takes probe_level_dir AND pid(because it should work for non pid case as well) as an input.
    if probe_level_dir is not None and (not file_channels.exists() or (recompute_channels)):
        try:
            lock_file = str(file_channels) + ".lock"
            lock = FileLock(lock_file, timeout=60)
            with lock:
                logger.info(f"{os.getpid()} Acquired lock for writing the channels dataset.")
                tmp_file = str(file_channels) + ".tmp"
                df_channels = pd.DataFrame(channels).rename(columns={"rawInd": "channel"})
                df_channels["pid"] = pid
                # Remove the labels columns from df_channels if it exists
                if "labels" in df_channels.columns:
                    df_channels = df_channels.drop(columns=["labels"])
                df_channels.to_parquet(tmp_file)
                os.replace(tmp_file, file_channels)
                logger.info(f"{os.getpid()} Finished writing the channels dataset.")
        except Exception as e:
            logger.error(f"Failed to export channels file: {str(e)}")
            logger.debug("Exception details:", exc_info=True)

    # Compute features
    df = online_feature_computation(
        sr_ap=sr_ap,
        sr_lf=sr_lf,
        t0=t_start,
        duration=duration,
        channels=channels,
        features_to_compute=features_to_compute,
        output_dir=snippet_level_dir,
        scratch_dir=scratch_dir,
        **kwargs,
    )

    # Add metadata to all parquet files in subdirectories
    if output_dir is not None:
        snippet_attrs = {
            "pid": pid,
            "t_start": t_start,
            "duration": duration,
            "base_level_dir": output_dir.as_posix(),
            "snippet_level_dir": snippet_level_dir.relative_to(output_dir).as_posix(),
        }

        add_metadata_to_parquet_files(**snippet_attrs)

    return df


def compute_features_from_file(
    ap_file,
    lf_file,
    t_start=None,
    duration=None,
    traj_dict=None,
    features_to_compute=None,
    output_dir=Path("."),
    scratch_dir=None,
    **kwargs,
):
    """
    Compute features from .cbin files.

    Args:
        ap_file (str): Path to AP .cbin file
        lf_file (str): Path to LF .cbin file
        t_start (float, optional): Start time in seconds. Defaults to 0.0 if not specified.
        duration (float, optional): Duration in seconds. If None, will use the entire available duration.
        traj_dict (dict, optional): Dictionary containing trajectory information with keys:
            - x, y, z: coordinates
            - depth, theta, phi: insertion parameters
            Required if want to add xyz target information.
        features_to_compute (list, optional): List of feature sets to compute
        output_dir (Path, optional): Output directory for saving features
        scratch_dir (Path, optional): Directory for temporary files (dartsort scratch)
        **kwargs: Additional keyword arguments

    Returns:
        pd.DataFrame: DataFrame containing computed features
    """
    # Create a dictionary with all the function arguments
    params = {
        "ap_file": ap_file,
        "t_start": t_start,
        "duration": duration,
        "output_dir": output_dir,
    }

    # Setup the output directory
    probe_level_dir, snippet_level_dir = setup_output_directory(params)

    # Validate input
    if ap_file is None or lf_file is None:
        raise ValueError("Both AP and LF .cbin files must be provided")

    # Load data from files
    sr_ap, sr_lf, channels = load_data_from_files(ap_file, lf_file, probe_level_dir)

    # Convert time parameters to float
    t_start = float(t_start) if t_start is not None else 0.0

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
        scratch_dir=scratch_dir,
        **kwargs,
    )

    # Add xyz target information if trajectory dictionary is provided
    if traj_dict is not None:
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
            logger.error(f"Failed to export channels file: {str(e)}")
            logger.debug("Exception details:", exc_info=True)

    return df


def compute_features_from_raw(
    raw_ap,
    raw_lf,
    fs_ap,
    fs_lf,
    geometry,
    channel_labels=None,
    features_to_compute=None,
    output_dir=Path("."),
    scratch_dir=None,
    **kwargs,
):
    """
    Compute features from raw numpy arrays of AP and LF data.

    Args:
        raw_ap (np.ndarray): Raw AP data array of shape (n_channels, n_samples)
        raw_lf (np.ndarray): Raw LF data array of shape (n_channels, n_samples)
        fs_ap (float): Sampling frequency of AP data
        fs_lf (float): Sampling frequency of LF data
        geometry (dict): Dictionary containing 'x' and 'y' coordinates for each channel
        channel_labels (np.ndarray, optional): Array of channel labels. If None, will be computed.
        features_to_compute (list, optional): List of feature sets to compute. If None, computes all features.
            Available options: ['lf', 'csd', 'ap', 'waveforms']
        output_dir (Path, optional): Directory to save individual feature sets. If None, features are not saved.
        **kwargs: Additional keyword arguments

    Returns:
        pd.DataFrame: DataFrame containing computed features
    """
    # Assert input shapes and parameters
    assert raw_ap.ndim == 2 and raw_lf.ndim == 2, "Input arrays must be 2D"
    assert raw_ap.shape[0] == raw_lf.shape[0], (
        "Number of channels must match between AP and LF data"
    )
    assert raw_ap.shape[0] == len(geometry["x"]) == len(geometry["y"]), (
        "Number of channels must match geometry"
    )
    assert fs_ap > 0 and fs_lf > 0, "Sampling frequencies must be positive"

    if channel_labels is None:
        channel_labels = np.zeros(raw_ap.shape[0])

    # Define available feature sets
    available_features = ["lf", "csd", "ap", "waveforms"]

    # If no specific features are requested, compute all
    if features_to_compute is None:
        features_to_compute = available_features
    else:
        # Validate requested features
        invalid_features = [
            f for f in features_to_compute if f not in available_features
        ]
        if invalid_features:
            raise ValueError(
                f"Invalid feature sets requested: {invalid_features}. Available options: {available_features}"
            )

    # Todo do I need to check the dtype of the raw_ap and raw_lf?
    # Destripe AP and LF data
    des_ap = ibldsp.voltage.destripe(
        raw_ap,
        fs=fs_ap,
        neuropixel_version=1,
        channel_labels=channel_labels,
        k_filter=False,
    )
    des_lf = ibldsp.voltage.destripe_lfp(
        raw_lf,
        fs=fs_lf,
        channel_labels=channel_labels,
    )
    logger.info("Destriped AP and LF data")

    df = {}

    # TODO - Have consistent use of either Pathlib or os.path.join.
    # Function to save features to file
    def save_features(feature_name, feature_df):
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / f"{feature_name}_features.pqt"
            feature_df.to_parquet(file_path)
            logger.info(f"Saved {feature_name} features to {file_path}")

    # Function to load features from file
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

    # Define feature computation configurations
    feature_configs = {
        "lf": {
            "func": features.lf,
            "args": {"data": des_lf, "fs": fs_lf},
            "kwargs": {},
        },
        "csd": {
            "func": features.csd,
            "args": {"data": des_lf, "fs": fs_lf, "geometry": geometry},
            "kwargs": {"decimate": 10},
        },
        "ap": {
            "func": features.ap,
            "args": {
                "data": des_ap,
                "geometry": geometry,
                "channel_labels": channel_labels,
            },
            "kwargs": {},
        },
        "waveforms": {
            "func": features.spikes,
            "args": {"data": des_ap, "fs": fs_ap, "geometry": geometry},
            "kwargs": {"scratch_dir": scratch_dir},
        },
    }

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

        # Compute the feature
        if feature_name == "waveforms":
            # Special handling for waveforms which returns tuple
            df[feature_name], waveforms = config["func"](
                **config["args"], **config["kwargs"]
            )
            df[feature_name]["spike_count"] = df[feature_name]["spike_count"].astype(
                "Int64"
            )

            # Save waveform files if requested from the functoin call of compute_features_from_raw
            if (output_dir is not None) and kwargs.get("save_waveforms", False):
                waveforms_dir = output_dir / "waveforms"
                waveforms_dir.mkdir(parents=True, exist_ok=True)
                np.save(waveforms_dir / "raw.npy", waveforms["raw"].astype(np.float16))
                np.save(
                    waveforms_dir / "denoised.npy",
                    waveforms["denoised"].astype(np.float16),
                )
                np.save(
                    waveforms_dir / "waveform_channels.npy", waveforms["channel_index"]
                )
                waveforms["df_spikes"].to_parquet(waveforms_dir / "spikes.pqt")
        else:
            df[feature_name] = config["func"](**config["args"], **config["kwargs"])

        # Add package version metadata
        df[feature_name].attrs["ibleatools_version"] = ibleatools_version
        df[feature_name].attrs[f"{feature_name}_version"] = features_version

        # Save the feature
        save_features(feature_name, df[feature_name])

    # Compute each requested feature
    for feature_name in features_to_compute:
        if feature_name in feature_configs:
            compute_and_save_feature(feature_name, feature_configs[feature_name])
        else:
            logger.warning(f"Unknown feature type: {feature_name}")

    df_voltage = reduce(
        lambda left, right: pd.merge(left, right, on="channel", how="outer"),
        [df[k] for k in df.keys()],
    )

    return df_voltage


# TODO - Define a function to compute features for a single category.
def compute_features_for_category(df, category):
    """
    Compute features for a specific category from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing computed features
    """
    # TODO - Define the features to compute for the category.
