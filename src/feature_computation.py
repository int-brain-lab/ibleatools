from pathlib import Path
from functools import reduce

import scipy.signal
from scipy import fft
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('qt5agg')

from one.api import ONE
import iblatlas.atlas
from brainbox.io.one import SpikeSortingLoader
import ibldsp.voltage
from ibl_style.style import figure_style
import pandas as pd
import numpy as np
from iblatlas.atlas import Insertion, NeedlesAtlas, AllenAtlas
from ibllib.pipes.histology import interpolate_along_track

from .plots import plot_cumulative_probas
from . import features
from . import data
from . import decoding
from typing import Dict, Any, Tuple
from src.logger_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

def get_target_coordinates(pid, one, channels):
    """
    Get the micro-manipulator target coordinates from Alyx database
    """
    from pathlib import Path
    needles = NeedlesAtlas()
    allen = AllenAtlas()
    needles.compute_surface()
    trajs = one.alyx.rest("trajectories", "list", probe_insertion=pid, django='provenance__lte,30')
    # if len(trajs) == 0:
    #     trajs = one.alyx.rest("trajectories", "list", probe_insertion=pid)
    traj = next((t for t in trajs if t['provenance'] == 'Micro-manipulator'), trajs[0])
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
    :param sr_lf:
    :param sr_ap:
    :param t0:
    :return:
    """


    #Calculate the next fast length for the AP data
    ns_ap = fft.next_fast_len(int(sr_ap.fs * duration), real=True)

    #Calculate the next fast length for the LF data
    ns_lf = fft.next_fast_len(int(sr_lf.fs * duration), real=True)

    #Ignore the columns which include the sync pulse data.
    raw_ap = sr_ap[slice(n0 := int(sr_ap.fs * t0), n0 + ns_ap), :(sr_ap.nc - sr_ap.nsync)].T
    if channel_labels is None:
        channel_labels, _ = ibldsp.voltage.detect_bad_channels(raw_ap, fs=sr_ap.fs)

    # destripe AP and LF data
    des_ap = ibldsp.voltage.destripe(
        raw_ap, fs=sr_ap.fs, neuropixel_version=sr_ap.major_version, channel_labels=channel_labels,
        k_filter=False,
    )
    #Add 3 to n0 to account for the 3 samples of latency in the LF data
    raw_lf = sr_lf[slice(n0 := int(sr_lf.fs * t0 + 3), n0 + ns_lf), :(sr_lf.nc - sr_lf.nsync)].T
    des_lf = ibldsp.voltage.destripe_lfp(
        raw_lf, fs=sr_lf.fs, channel_labels=channel_labels,
    )
    logger.info(f"Destriped AP and LF data")
    df = {}
    df['channels'] = pd.DataFrame({
        'lateral_um': sr_ap.geometry['x'],
        'axial_um': sr_ap.geometry['y'],
        'labels': channel_labels,
        'channel': np.arange(len(channel_labels)),
    })
    logger.info(f"Starting LF, CSD and AP computation")
    df['lf'] = features.lf(des_lf, fs=sr_lf.fs)
    df['csd'] = features.csd(des_lf, fs=sr_lf.fs, geometry=sr_ap.geometry, decimate=10)
    df['ap'] = features.ap(des_ap, geometry=sr_ap.geometry)
    logger.info(f"LF, CSD and AP computation completed")
    # this takes a long time !
    logger.info(f"Starting waveforms computation")
    df['waveforms'], waveforms = features.spikes(des_ap, fs=sr_ap.fs, geometry=sr_ap.geometry)
    df['waveforms']['spike_count'] = df['waveforms']['spike_count'].astype('Int64')

    df_voltage = reduce(lambda left, right: pd.merge(left, right, on='channel', how='outer'),
                        [df[k] for k in df.keys()])
    df_voltage.dropna(inplace=True)
    # df_voltage = features.ModelRawFeatures.validate(df_voltage)

    return df_voltage


def compute_features(pid, t_start, duration, one):
    # Loads the data for a pid
    ssl = SpikeSortingLoader(pid=pid, one=one)  # this is the canonical way to load data
    channels = ssl.load_channels()
    # here for any kind of electrophysiology data, use spikeglx.Reader
    sr_ap = ssl.raw_electrophysiology(band='ap', stream=True)
    sr_lf = ssl.raw_electrophysiology(band='lf', stream=True)

    # Get the start and stop times from the command line arguments
    start_time = float(t_start)
    duration = float(duration)
    
    logger.info(f"Session path: {ssl.session_path}, probe name: {ssl.pname}")

    df = online_feature_computation(sr_ap=sr_ap, sr_lf=sr_lf, t0=start_time, duration=duration)

    #Merge the xyz target information
    xyz_target = get_target_coordinates(pid, one, channels)
    df = df.merge(xyz_target, left_index=True, right_index=True, how='left')

    return df