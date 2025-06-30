from functools import reduce
import logging
from deploy.iblsdsc import OneSdsc
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
from pathlib import Path
from ephysatlas.utils import setup_output_directory

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
    # (Ask OW)
    # aid_mm = needles.get_labels(xyz=xyz_mm, mode="clip")

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
        channels["labels"], _ = ibldsp.voltage.detect_bad_channels(raw_ap, fs=sr_ap.fs)

    return channels, compute_features_from_raw(
        raw_ap=raw_ap,
        raw_lf=raw_lf,
        fs_ap=sr_ap.fs,
        fs_lf=sr_lf.fs,
        geometry=sr_ap.geometry,
        channel_labels=channels.get("labels"),
        features_to_compute=features_to_compute,
        output_dir=output_dir,
        **kwargs,
    )


# TODO - Need to be clear here , if I want to check based on SDSC or not, VS pid as dict or pid as string.
# (Ask OW) Recomputing channels when launching multiple jobs.
def load_data_from_pid(pid, one, probe_level_dir, recompute_channels=False):
    """
    Load data using a probe ID from the ONE database.

    Args:
        pid (str or dict): Probe ID or dictionary containing probe information
        one (ONE): ONE client instance
        probe_level_dir (Path): Directory for probe-level data

    Returns:
        tuple: (sr_ap, sr_lf, channels) SpikeGLX readers and channel information
    """
    logger.info(f"Loading data using PID: {pid}")

    channel_labels = None
    # if pid is a dict, then extract eid and probe_name from it
    # TODO Check if the new version of iblscripts can work here to get the eid directly if not specified.
    # TODO - Check if the channel labels are being computed in all the cases.
    if isinstance(pid, dict):
        logger.info(
            f"Computing features for eid: {pid['eid']}, probe name: {pid['probe_name']}, pid: {pid['pid']}"
        )
        eid = pid["eid"]
        probe_name = pid["probe_name"]
        ssl = SpikeSortingLoader(pid=pid["pid"], eid=eid, pname=probe_name, one=one)
        if isinstance(one, OneSdsc):
            stream = False
        else:
            stream = True
        sr_ap = ssl.raw_electrophysiology(band="ap", stream=stream)
        sr_lf = ssl.raw_electrophysiology(band="lf", stream=stream)
        # TODO - Check is this is the best place to detect bad channels. I am doing it before loading the channels file.
        channel_labels = ibldsp.voltage.detect_bad_channels_cbin(sr_ap.file_bin)
    else:
        assert isinstance(pid, str), "PID must be a string"
        ssl = SpikeSortingLoader(pid=pid, one=one)
        sr_ap = ssl.raw_electrophysiology(band="ap", stream=True)
        sr_lf = ssl.raw_electrophysiology(band="lf", stream=True)

    # Load the channels file
    # TODO - Check if the channels file exists, if not, then create it.
    file_channels = Path(probe_level_dir) / "channels.parquet"
    if file_channels.exists() and (not recompute_channels):
        logger.info(f"Loading channels from {file_channels}")
        channels = pd.read_parquet(file_channels)
        channels = {col: channels[col].to_numpy() for col in channels.columns}
        if channels.get("labels") is None:
            channels["labels"] = channel_labels
    else:
        logger.info("Recomputing channels")
        try:
            channels = ssl.load_channels()
            if channels.get("labels") is None:
                channels["labels"] = channel_labels
        except KeyError as e:
            logger.info(f"Channels key was not found: {str(e)}")
        except Exception as e:
            logger.info(f"Failed to load channels: {str(e)}")

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

    channels, df = online_feature_computation(
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
    file_channels = probe_level_dir / "channels.parquet"
    if not file_channels.exists():
        try:
            df_channels = pd.DataFrame(channels).rename(columns={"rawInd": "channel"})
            df_channels.to_parquet(file_channels)
        except Exception as e:
            logger.info(f"Failed to export channels file: {str(e)}")

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
    if channel_labels is None:
        channel_labels, _ = ibldsp.voltage.detect_bad_channels(raw_ap, fs=fs_ap)

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
            file_path = output_dir / f"{feature_name}_features.parquet"
            feature_df.to_parquet(file_path)
            logger.info(f"Saved {feature_name} features to {file_path}")

    # Function to load features from file
    def load_features(feature_name):
        if output_dir is not None:
            file_path = output_dir / f"{feature_name}_features.parquet"
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
    # TODO - Write a loop here or a function to compute different features.

    if "lf" in features_to_compute:
        logger.info("Starting LF computation")
        df["lf"] = features.lf(des_lf, fs=fs_lf)
        # Add package version metadata to the DataFrame
        df["lf"].attrs["ibleatools_version"] = ibleatools_version
        save_features("lf", df["lf"])
    else:
        logger.info("Loading LF features from save directory")
        lf_data = load_features("lf")
        if lf_data is not None:
            df["lf"] = lf_data
        else:
            logger.warning(
                "LF features not found in save directory. LF features will not be computed."
            )
            # raise ValueError("LF features not found in save directory")

    if "csd" in features_to_compute:
        logger.info("Starting CSD computation")
        df["csd"] = features.csd(des_lf, fs=fs_lf, geometry=geometry, decimate=10)
        # Add package version metadata to the DataFrame
        df["csd"].attrs["ibleatools_version"] = ibleatools_version
        save_features("csd", df["csd"])
    else:
        logger.info("Loading CSD features from save directory")
        csd_data = load_features("csd")
        if csd_data is not None:
            df["csd"] = csd_data
        else:
            logger.warning(
                "CSD features not found in save directory. CSD features will not be computed."
            )
            # raise ValueError("CSD features not found in save directory")

    if "ap" in features_to_compute:
        logger.info("Starting AP computation")
        df["ap"] = features.ap(des_ap, geometry=geometry)
        # Add package version metadata to the DataFrame
        df["ap"].attrs["ibleatools_version"] = ibleatools_version
        save_features("ap", df["ap"])
    else:
        logger.info("Loading AP features from save directory")
        ap_data = load_features("ap")
        if ap_data is not None:
            df["ap"] = ap_data
        else:
            logger.warning(
                "AP features not found in save directory. AP features will not be computed."
            )
            # raise ValueError("AP features not found in save directory")

    # this takes a long time !
    if "waveforms" in features_to_compute:
        logger.info("Starting waveforms computation")
        df["waveforms"], waveforms = features.spikes(
            des_ap, fs=fs_ap, geometry=geometry
        )
        df["waveforms"]["spike_count"] = df["waveforms"]["spike_count"].astype("Int64")
        # Add package version metadata to the DataFrame
        df["waveforms"].attrs["ibleatools_version"] = ibleatools_version
        save_features("waveforms", df["waveforms"])

        if kwargs.get("save_waveforms", True):
            # Save other waveform features in waveform directory
            waveforms_dir = output_dir / "waveforms"
            waveforms_dir.mkdir(parents=True, exist_ok=True)
            # Save the waveforms to files
            np.save(waveforms_dir / "raw.npy", waveforms["raw"].astype(np.float16))
            np.save(
                waveforms_dir / "denoised.npy", waveforms["denoised"].astype(np.float16)
            )
            np.save(waveforms_dir / "waveform_channels.npy", waveforms["channel_index"])
            waveforms["df_spikes"].to_parquet(waveforms_dir / "spikes.pqt")
    else:
        waveforms_data = load_features("waveforms")
        if waveforms_data is not None:
            df["waveforms"] = waveforms_data
        else:
            logger.warning(
                "Waveforms features not found in save directory. Waveform features will not be computed."
            )
            # raise ValueError("Waveforms features not found in save directory")

    # TODO - Should I output the features dataset here??
    df_voltage = reduce(
        lambda left, right: pd.merge(left, right, on="channel", how="outer"),
        [df[k] for k in df.keys()],
    )
    # TODO - Check whether the dropna is needed or not. (Ask OW)
    original_index = df_voltage.index.copy()
    # TODO Do the dropping when loading the data.
    # df_voltage.dropna(inplace=True)
    dropped_indices = original_index.difference(df_voltage.index)
    if len(dropped_indices) > 0:
        logger.info(f"Dropped row indices: {dropped_indices.tolist()}")

    return df_voltage


# TODO - Define a function to compute features for a single category.
def compute_features_for_category(df, category):
    """
    Compute features for a specific category from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing computed features
    """
    # TODO - Define the features to compute for the category.
