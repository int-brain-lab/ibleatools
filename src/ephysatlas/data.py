import logging
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
from ibldsp.waveforms import peak_to_trough_ratio
import neuropixel
from one.remote import aws

import iblatlas.atlas
import ephysatlas.features
import ephysatlas.anatomy

_logger = logging.getLogger("ibllib")

SPIKES_ATTRIBUTES = ["clusters", "times", "depths", "amps"]
CLUSTERS_ATTRIBUTES = ["channels", "depths", "metrics"]

EXTRACT_RADIUS_UM = 200  # for localisation , the default extraction radius in um


def get_waveforms_coordinates(
    trace_indices,
    xy=None,
    extract_radius_um=EXTRACT_RADIUS_UM,
    return_complex=False,
    return_indices=False,
):
    """
    Reproduces the localisation code channel selection when extracting waveforms from raw data.
    Args:
        trace_indices: np.array (nspikes,): index of the trace of the detected spike )
        xy: (optional)
        extract_radius_um: radius from peak trace: all traces within this radius will be included
        return_complex: if True, returns the complex coordinates, otherwise returns a 3D x, y, z array
        return_indices: if True, returns the indices of the channels within the radius
    Returns: (np.array, np.array) (nspikes, ntraces, n_coordinates) of axial and transverse coordinates, (nspikes, ntraces) of indices
    """
    if xy is None:
        th = neuropixel.trace_header(version=1)
        xy = th["x"] + 1j * th["y"]
    channel_lookups = _get_channel_distances_indices(
        xy, extract_radius_um=extract_radius_um
    )
    inds = channel_lookups[trace_indices.astype(np.int32)]
    # add a dummy channel to have nans in the coordinates
    inds[np.isnan(inds)] = xy.size - 1
    wxy = np.r_[xy, np.nan][inds.astype(np.int32)]
    if not return_complex:
        wxy = np.stack(
            (np.real(wxy), np.imag(wxy), np.zeros_like(np.imag(wxy))), axis=2
        )
    if return_indices:
        return wxy, inds.astype(int)
    else:
        return wxy


def _get_channel_distances_indices(xy, extract_radius_um=EXTRACT_RADIUS_UM):
    """
    params: xy: ntr complex array of x and y coordinates of each channel relative to the probe
    Computes the distance between each channel and all the other channels, and find the
    indices of the channels that are within the radius.
    For each row the indices of the channels within the radius are returned.
    returns: channel_dist: ntr x ntr_wav matrix of channel indices within the radius., where ntr_wav is the
    """
    ntr = xy.shape[0]
    channel_dist = np.zeros((ntr, ntr)) * np.nan
    for i in np.arange(ntr):
        cind = np.where(np.abs(xy[i] - xy) <= extract_radius_um)[0]
        channel_dist[i, : cind.size] = cind
    # prune the matrix: only so many channels are within the radius
    channel_dist = channel_dist[:, ~np.all(np.isnan(channel_dist), axis=0)]
    return channel_dist


def atlas_pids_autism(one):
    """
    Get autism data from JP
    fmr1 mouse line
    """
    project = "angelaki_mouseASD"
    # Get all insertions for this project
    str_query = (
        f"session__projects__name__icontains,{project},"
        "session__qc__lt,50,"
        "~json__qc,CRITICAL"
    )
    insertions = one.alyx.rest("insertions", "list", django=str_query)
    # Restrict to only those with subject starting with FMR
    ins_keep = [
        item for item in insertions if item["session_info"]["subject"][0:3] == "FMR"
    ]
    return [item["id"] for item in ins_keep], ins_keep


def atlas_pids(one, tracing=False):
    django_strg = [
        "session__projects__name__icontains,ibl_neuropixel_brainwide_01",
        "session__qc__lt,50",
        "~json__qc,CRITICAL",
        # 'session__extended_qc__behavior,1',
        "session__json__IS_MOCK,False",
    ]
    if tracing:
        django_strg.append("json__extended_qc__tracing_exists,True")

    insertions = one.alyx.rest("insertions", "list", django=django_strg)
    return [item["id"] for item in insertions], insertions


def read_correlogram(file_correlogram, nclusters):
    nbins = int(Path(file_correlogram).stat().st_size / nclusters / 4)
    mmap_correlogram = np.memmap(
        file_correlogram, dtype="int32", shape=(nclusters, nbins)
    )
    return mmap_correlogram


def download_tables(
    local_path,
    label="2024_W50",
    one=None,
    verify=False,
    overwrite=False,
    extended=False,
):
    """
    :param local_path: pathlib.Path() where the data will be stored locally
    :param label: revision string "2024_W04"
    :param one:
    :param verify: checks the indices and consistency of the dataframes and raise an error if not consistent
    :param overwrite: force redownloading if file exists
    :param extended: if True, will download also extended datasets, such as cross-correlograms that take up
    more space than just the tables (coople hundreds Mb for the table, several GB with extended data)
    :return:
    """
    # The AWS private credentials are stored in Alyx, so that only one authentication is required
    local_path = Path(local_path).joinpath(label)
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    local_files = aws.s3_download_folder(
        f"aggregates/atlas/{label}",
        local_path,
        s3=s3,
        bucket_name=bucket_name,
        overwrite=overwrite,
    )
    if extended:
        local_files = aws.s3_download_folder(
            f"aggregates/atlas/{label}_extended",
            local_path,
            s3=s3,
            bucket_name=bucket_name,
            overwrite=overwrite,
        )
    assert len(local_files), f"aggregates/atlas/{label} not found on AWS"
    return local_path


def read_features_from_disk(
    path_features: Path,
    brain_atlas: "iblatlas.atlas.BrainAtlas" = None,
    mappings: list[str] = None,
) -> pd.DataFrame:
    """
    Read electrophysiology features from disk and merge with channel information.

    This function loads raw electrophysiology features, channel information, and channel labels
    from parquet files, merges them into a single dataframe, and adds brain region mapping
    information using the provided brain atlas.

    Parameters
    ----------
    path_features : pathlib.Path
        Path to the directory containing the feature parquet files.
    brain_atlas : iblatlas.atlas.BrainAtlas
        Brain atlas object used to map coordinates to brain regions.
        Must be provided to enable region mapping.
    mappings : list, optional
        List of brain region mapping ontologies to include.
        Default is ['Cosmos', 'Beryl'].

    Returns
    -------
    pandas.DataFrame
        DataFrame containing merged electrophysiology features with channel information
        and brain region mappings.
    """
    mappings = ["Cosmos", "Beryl"] if mappings is None else mappings
    brain_atlas = (
        brain_atlas if brain_atlas is not None else ephysatlas.anatomy.ClassifierAtlas()
    )
    assert brain_atlas is not None, "Brain atlas is required to map labels to regions"
    assert all(mapping in brain_atlas.regions.mappings for mapping in mappings), (
        f"Unknown mapping: {mappings}"
    )
    # merge the channel information with the features
    df_features = pd.read_parquet(path_features / "raw_ephys_features_denoised.pqt")
    df_channels = pd.read_parquet(path_features / "channels.pqt")
    duplicate_cols = set(df_features.columns).intersection(set(df_channels.columns))
    df_channels = df_channels.drop(columns=duplicate_cols)
    df_features = df_features.merge(
        df_channels,
        how="inner",
        right_index=True,
        left_index=True,
    )
    if "labels" not in df_features.columns:
        df_features = df_features.merge(
            pd.read_parquet(path_features / "channels_labels.pqt").fillna(0),
            how="inner",
            right_index=True,
            left_index=True,
        )
    df_features["outside"] = df_features["labels"] == 3

    aids = brain_atlas.get_labels(
        df_features.loc[:, ["x", "y", "z"]].values, mode="clip"
    )
    df_features["Allen_id"] = aids
    for mapping in mappings:
        df_features[f"{mapping}_id"] = brain_atlas.regions.remap(aids, "Allen", mapping)

    # this will make sure that the features dataframe is compatible and healthy
    return pd.DataFrame(ephysatlas.features.ModelRawFeatures(df_features))


def compute_depth_dataframe(df_raw_features, df_clusters, df_channels):
    """
    Compute a features dataframe for each pid and depth along the probe,
    merging the raw voltage features, and the clusters features
    :param df_voltage:
    :param df_clusters:
    :param df_channels:
    :return:
    """
    df_depth_clusters = df_clusters.groupby(["pid", "axial_um"]).agg(
        amp_max=pd.NamedAgg(column="amp_max", aggfunc="mean"),
        amp_min=pd.NamedAgg(column="amp_min", aggfunc="mean"),
        amp_median=pd.NamedAgg(column="amp_median", aggfunc="mean"),
        amp_std_dB=pd.NamedAgg(column="amp_std_dB", aggfunc="mean"),
        contamination=pd.NamedAgg(column="contamination", aggfunc="mean"),
        contamination_alt=pd.NamedAgg(column="contamination_alt", aggfunc="mean"),
        drift=pd.NamedAgg(column="drift", aggfunc="mean"),
        missed_spikes_est=pd.NamedAgg(column="missed_spikes_est", aggfunc="mean"),
        noise_cutoff=pd.NamedAgg(column="noise_cutoff", aggfunc="mean"),
        presence_ratio=pd.NamedAgg(column="presence_ratio", aggfunc="mean"),
        presence_ratio_std=pd.NamedAgg(column="presence_ratio_std", aggfunc="mean"),
        slidingRP_viol=pd.NamedAgg(column="slidingRP_viol", aggfunc="mean"),
        spike_count=pd.NamedAgg(column="spike_count", aggfunc="mean"),
        firing_rate=pd.NamedAgg(column="firing_rate", aggfunc="mean"),
        label=pd.NamedAgg(column="label", aggfunc="mean"),
        x=pd.NamedAgg(column="x", aggfunc="mean"),
        y=pd.NamedAgg(column="y", aggfunc="mean"),
        z=pd.NamedAgg(column="z", aggfunc="mean"),
        acronym=pd.NamedAgg(column="acronym", aggfunc="first"),
        atlas_id=pd.NamedAgg(column="atlas_id", aggfunc="first"),
    )

    df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)
    df_depth_raw = df_voltage.groupby(["pid", "axial_um"]).agg(
        alpha_mean=pd.NamedAgg(column="alpha_mean", aggfunc="mean"),
        alpha_std=pd.NamedAgg(column="alpha_std", aggfunc="mean"),
        spike_count=pd.NamedAgg(column="spike_count", aggfunc="mean"),
        cloud_x_std=pd.NamedAgg(column="cloud_x_std", aggfunc="mean"),
        cloud_y_std=pd.NamedAgg(column="cloud_y_std", aggfunc="mean"),
        cloud_z_std=pd.NamedAgg(column="cloud_z_std", aggfunc="mean"),
        peak_trace_idx=pd.NamedAgg(column="peak_trace_idx", aggfunc="mean"),
        peak_time_idx=pd.NamedAgg(column="peak_time_idx", aggfunc="mean"),
        peak_val=pd.NamedAgg(column="peak_val", aggfunc="mean"),
        trough_time_idx=pd.NamedAgg(column="trough_time_idx", aggfunc="mean"),
        trough_val=pd.NamedAgg(column="trough_val", aggfunc="mean"),
        tip_time_idx=pd.NamedAgg(column="tip_time_idx", aggfunc="mean"),
        tip_val=pd.NamedAgg(column="tip_val", aggfunc="mean"),
        rms_ap=pd.NamedAgg(column="rms_ap", aggfunc="mean"),
        rms_lf=pd.NamedAgg(column="rms_lf", aggfunc="mean"),
        psd_delta=pd.NamedAgg(column="psd_delta", aggfunc="mean"),
        psd_theta=pd.NamedAgg(column="psd_theta", aggfunc="mean"),
        psd_alpha=pd.NamedAgg(column="psd_alpha", aggfunc="mean"),
        psd_beta=pd.NamedAgg(column="psd_beta", aggfunc="mean"),
        psd_gamma=pd.NamedAgg(column="psd_gamma", aggfunc="mean"),
        x=pd.NamedAgg(column="x", aggfunc="mean"),
        y=pd.NamedAgg(column="y", aggfunc="mean"),
        z=pd.NamedAgg(column="z", aggfunc="mean"),
        acronym=pd.NamedAgg(column="acronym", aggfunc="first"),
        atlas_id=pd.NamedAgg(column="atlas_id", aggfunc="first"),
        histology=pd.NamedAgg(column="histology", aggfunc="first"),
    )
    df_depth = df_depth_raw.merge(df_depth_clusters, left_index=True, right_index=True)
    return df_depth


def get_config():
    file_yaml = Path(__file__).parents[2].joinpath("config-ephys-atlas.yaml")
    with open(file_yaml, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def compute_summary_stat(df_voltage, features):
    """
    Summary statistics
    :param df_voltage:
    :param features:
    :return:
    """
    # The behavior of loc is inconsistent
    # If you input a str instead of a list, it returns a Series instead of a dataframe
    if not isinstance(features, list):  # Make sure input is a list
        features = [features]

    summary = (
        df_voltage.loc[:, features]
        .agg(["median", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)])
        .T
    )
    summary.columns = ["median", "q05", "q95"]
    summary["dq"] = summary["q95"] - summary["q05"]
    return summary


def sort_feature(values, features, ascending=True):
    """
    Sort the value (metrics being p-value, or else)
    :param values:
    :param features:
    :param ascending:
    :return:
    """
    id_sort = np.argsort(values)
    if not ascending:
        id_sort = np.flip(id_sort)
    features_sort = features[id_sort]
    values_sort = values[id_sort]
    return values_sort, features_sort


def prepare_mat_plot(array_in, id_feat, diag_val=0):
    """
    From the matrix storing the results of brain-to-brain regions comparison in the upper triangle for all features,
    select a feature and create a matrix with transpose for plotting in 2D
    :param array_in: array of N region x N region x N feature
    :param id_feat: index of feature that will be displayed
    :return:
    """
    mat_plot = np.squeeze(array_in[:, :, id_feat].copy())
    mat_plot[np.tril_indices_from(mat_plot)] = diag_val  # replace Nan by 0
    mat_plot = mat_plot + mat_plot.T  # add transpose for display
    return mat_plot


def prepare_df_voltage(df_voltage, df_channels, br=None):
    if br is None:
        br = iblatlas.atlas.BrainRegions()
    df_voltage = pd.merge(
        df_voltage, df_channels, left_index=True, right_index=True
    ).dropna()
    df_voltage["cosmos_id"] = br.remap(
        df_voltage["atlas_id"], source_map="Allen", target_map="Cosmos"
    )
    df_voltage["beryl_id"] = br.remap(
        df_voltage["atlas_id"], source_map="Allen", target_map="Beryl"
    )

    df_voltage = df_voltage.loc[
        ~df_voltage["cosmos_id"].isin(br.acronym2id(["void", "root"]))
    ]
    for feat in ["rms_ap", "rms_lf"]:
        df_voltage[feat] = 20 * np.log10(df_voltage[feat])
    df_voltage["spike_count_log"] = np.log10(df_voltage["spike_count"] + 1)

    # Add in peak_to_trough_ratio + peak_to_trough_duration
    df_voltage = peak_to_trough_ratio(df_voltage)
    df_voltage["peak_to_trough_duration"] = (
        df_voltage["trough_time_secs"] - df_voltage["peak_time_secs"]
    )
    return df_voltage
