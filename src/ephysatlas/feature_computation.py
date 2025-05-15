from functools import reduce
import logging

import scipy.signal
import scipy.fft
import pandas as pd
import numpy as np

from brainbox.io.one import SpikeSortingLoader
import ibldsp.voltage

from iblatlas.atlas import Insertion, NeedlesAtlas, AllenAtlas
from ibllib.pipes.histology import interpolate_along_track

from ephysatlas import features

# Set up logger
logger = logging.getLogger(__name__)

def get_target_coordinates(pid=None, one=None, channels=None, traj_dict=None):
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
        pd.DataFrame: DataFrame containing target coordinates for each channel
    """
    from pathlib import Path
    needles = NeedlesAtlas()
    allen = AllenAtlas()
    needles.compute_surface()
    
    # Validate input combinations
    if pid is not None and one is not None:
        # Mode 1: Using Alyx database
        trajs = one.alyx.rest("trajectories", "list", probe_insertion=pid, django='provenance__lte,30')
        traj = next((t for t in trajs if t['provenance'] == 'Micro-manipulator'), trajs[0])
    elif traj_dict is not None:
        # Mode 2: Using direct trajectory dictionary
        traj = traj_dict
    else:
        raise ValueError("Either provide (pid, one) or traj_dict")
    
    ins = Insertion.from_dict(traj, brain_atlas=needles)
    # TODO: apply the pitch correction by using iblatlas.atlas.tilt_spherical()
    txyz = np.flipud(ins.xyz)
    # we convert the coordinates from in-vivo to the Allen coordinate system
    txyz = (allen.bc.i2xyz(needles.bc.xyz2i(txyz / 1e6, round=False, mode="clip")) * 1e6)
    xyz_mm = interpolate_along_track(txyz, channels["axial_um"] / 1e6)
    # aid_mm = needles.get_labels(xyz=xyz_mm, mode="clip")
    # we interpolate the channels from the deepest point up. The neuropixel y coordinate is from the bottom of the probe
    dfc = pd.DataFrame({
        "x_target": xyz_mm[:, 0],
        "y_target": xyz_mm[:, 1],
        "z_target": xyz_mm[:, 2]}, index=channels['rawInd'])
    return dfc

def online_feature_computation(sr_lf, sr_ap, t0, duration, channel_labels=None):
    """
    Compute features from SpikeGLX reader objects.
    
    Args:
        sr_lf: SpikeGLX reader for LF data
        sr_ap: SpikeGLX reader for AP data
        t0 (float): Start time in seconds
        duration (float): Duration in seconds
        channel_labels (np.ndarray, optional): Array of channel labels
    
    Returns:
        pd.DataFrame: DataFrame containing computed features
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
        raise ValueError(f"Requested time range ({t0} to {t0 + duration}) exceeds AP data duration ({max_time_ap})")
    if t0 + duration > max_time_lf:
        raise ValueError(f"Requested time range ({t0} to {t0 + duration}) exceeds LF data duration ({max_time_lf})")

    # Calculate start and end indices
    n0_ap = int(sr_ap.fs * t0)
    n0_lf = int(sr_lf.fs * t0 + 3)  # Add 3 to account for LF latency
    
    # Verify channel indices
    n_channels_ap = sr_ap.nc - sr_ap.nsync
    n_channels_lf = sr_lf.nc - sr_lf.nsync
    
    if n_channels_ap <= 0 or n_channels_lf <= 0:
        raise ValueError(f"Invalid number of channels: AP={n_channels_ap}, LF={n_channels_lf}")

    # Ignore the columns which include the sync pulse data.
    try:
        raw_ap = sr_ap[slice(n0_ap, n0_ap + ns_ap), :n_channels_ap].T
    except IndexError as e:
        raise IndexError(f"Failed to access AP data: {str(e)}. Check if time range or channel count is valid.")

    # Add 3 to n0 to account for the 3 samples of latency in the LF data
    try:
        raw_lf = sr_lf[slice(n0_lf, n0_lf + ns_lf), :n_channels_lf].T
    except IndexError as e:
        raise IndexError(f"Failed to access LF data: {str(e)}. Check if time range or channel count is valid.")

    if channel_labels is None:
        channel_labels, _ = ibldsp.voltage.detect_bad_channels(raw_ap, fs=sr_ap.fs)

    return compute_features_from_raw(
        raw_ap=raw_ap,
        raw_lf=raw_lf,
        fs_ap=sr_ap.fs,
        fs_lf=sr_lf.fs,
        geometry=sr_ap.geometry,
        channel_labels=channel_labels
    )

def load_data_from_pid(pid, one):
    """
    Load data using a probe ID from the ONE database.
    
    Args:
        pid (str): Probe ID
        one (ONE): ONE client instance
    
    Returns:
        tuple: (sr_ap, sr_lf, channels) SpikeGLX readers and channel information
    """
    logger.info(f"Loading data using PID: {pid}")
    ssl = SpikeSortingLoader(pid=pid, one=one)
    channels = ssl.load_channels()
    sr_ap = ssl.raw_electrophysiology(band='ap', stream=True)
    sr_lf = ssl.raw_electrophysiology(band='lf', stream=True)
    logger.info(f"Session path: {ssl.session_path}, probe name: {ssl.pname}")
    return sr_ap, sr_lf, channels

def load_data_from_files(ap_file, lf_file):
    """
    Load data from .cbin files.
    
    Args:
        ap_file (str): Path to AP .cbin file
        lf_file (str): Path to LF .cbin file
    
    Returns:
        tuple: (sr_ap, sr_lf, None) SpikeGLX readers and None for channels
    """
    logger.info(f"Loading data from files: AP={ap_file}, LF={lf_file}")
    try:
        from spikeglx import Reader
        sr_ap = Reader(ap_file)
        sr_lf = Reader(lf_file)
        #Todo here I have to add the channel information
        channels = {}
        channels['rawInd'] = np.arange(sr_ap.nc - sr_ap.nsync)
        channels['axial_um'] = sr_ap.geometry['y']
        return sr_ap, sr_lf, channels
    except ImportError:
        raise ImportError("spikeglx package is required to read .cbin files")
    except Exception as e:
        raise RuntimeError(f"Failed to load .cbin files: {str(e)}")

def compute_features(pid=None, t_start=None, duration=None, one=None, ap_file=None, lf_file=None, traj_dict=None):
    """
    Compute features from either PID or .cbin files.
    
    Args:
        pid (str, optional): Probe ID. Required if ap_file and lf_file are not provided.
        t_start (float): Start time in seconds. Defaults to 0.0 if not specified.
        duration (float, optional): Duration in seconds. If None, will use the entire available duration.
        one (ONE, optional): ONE client instance. Required if pid is provided.
        ap_file (str, optional): Path to AP .cbin file. Required if pid is not provided.
        lf_file (str, optional): Path to LF .cbin file. Required if pid is not provided.
        traj_dict (dict, optional): Dictionary containing trajectory information with keys:
            - x, y, z: coordinates
            - depth, theta, phi: insertion parameters
            Required if using .cbin files and want to add xyz target information.
    
    Returns:
        pd.DataFrame: DataFrame containing computed features
    """
    # Validate input combinations
    if pid is not None:
        if one is None:
            raise ValueError("ONE client instance is required when using PID")
        if ap_file is not None or lf_file is not None:
            raise ValueError("Cannot provide both PID and .cbin files")
        sr_ap, sr_lf, channels = load_data_from_pid(pid, one)
    else:
        if ap_file is None or lf_file is None:
            raise ValueError("Both AP and LF .cbin files must be provided when not using PID")
        sr_ap, sr_lf, channels = load_data_from_files(ap_file, lf_file)

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
    df = online_feature_computation(sr_ap=sr_ap, sr_lf=sr_lf, t0=t_start, duration=duration)
    # df.to_parquet("/Users/pranavrai/Work/int-brain-lab/temp/features/features_wo_target.parquet",index=True)
    # df = pd.read_parquet("/Users/pranavrai/Work/int-brain-lab/temp/features/features_wo_target.parquet")
    
    # Add xyz target information if available
    if pid is not None and one is not None:
        # Mode 1: Using Alyx database
        xyz_target = get_target_coordinates(pid=pid, one=one, channels=channels)
        df = df.merge(xyz_target, left_index=True, right_index=True, how='left')
    elif traj_dict is not None:
        # Mode 2: Using direct trajectory dictionary
        xyz_target = get_target_coordinates(traj_dict=traj_dict, channels=channels)
        df = df.merge(xyz_target, left_index=True, right_index=True, how='left')
    else:
        logger.warning("No trajectory information available, skipping xyz target addition")
    
    return df

def compute_features_from_raw(raw_ap, raw_lf, fs_ap, fs_lf, geometry, channel_labels=None):
    """
    Compute features from raw numpy arrays of AP and LF data.
    
    Args:
        raw_ap (np.ndarray): Raw AP data array of shape (n_channels, n_samples)
        raw_lf (np.ndarray): Raw LF data array of shape (n_channels, n_samples)
        fs_ap (float): Sampling frequency of AP data
        fs_lf (float): Sampling frequency of LF data
        geometry (dict): Dictionary containing 'x' and 'y' coordinates for each channel
        channel_labels (np.ndarray, optional): Array of channel labels. If None, will be computed.
    
    Returns:
        pd.DataFrame: DataFrame containing computed features
    """
    # Assert input shapes and parameters
    assert raw_ap.ndim == 2 and raw_lf.ndim == 2, "Input arrays must be 2D"
    assert raw_ap.shape[0] == raw_lf.shape[0], "Number of channels must match between AP and LF data"
    assert raw_ap.shape[0] == len(geometry['x']) == len(geometry['y']), "Number of channels must match geometry"
    assert fs_ap > 0 and fs_lf > 0, "Sampling frequencies must be positive"
    
    #Todo do I need to check the dtype of the raw_ap and raw_lf?
    if channel_labels is None:
        channel_labels, _ = ibldsp.voltage.detect_bad_channels(raw_ap, fs=fs_ap)
    
    # destripe AP and LF data
    des_ap = ibldsp.voltage.destripe(
        raw_ap, fs=fs_ap, neuropixel_version=1, channel_labels=channel_labels,
        k_filter=False,
    )
    des_lf = ibldsp.voltage.destripe_lfp(
        raw_lf, fs=fs_lf, channel_labels=channel_labels,
    )
    logger.info(f"Destriped AP and LF data")
    
    df = {}
    df['channels'] = pd.DataFrame({
        'lateral_um': geometry['x'],
        'axial_um': geometry['y'],
        'labels': channel_labels,
        'channel': np.arange(len(channel_labels)),
    })
    
    logger.info(f"Starting LF, CSD and AP computation")
    df['lf'] = features.lf(des_lf, fs=fs_lf)
    df['csd'] = features.csd(des_lf, fs=fs_lf, geometry=geometry, decimate=10)
    df['ap'] = features.ap(des_ap, geometry=geometry)
    logger.info(f"LF, CSD and AP computation completed")
    
    # this takes a long time !
    logger.info(f"Starting waveforms computation")
    df['waveforms'], waveforms = features.spikes(des_ap, fs=fs_ap, geometry=geometry)
    df['waveforms']['spike_count'] = df['waveforms']['spike_count'].astype('Int64')

    df_voltage = reduce(lambda left, right: pd.merge(left, right, on='channel', how='outer'),
                        [df[k] for k in df.keys()])
    df_voltage.dropna(inplace=True)
    
    return df_voltage
