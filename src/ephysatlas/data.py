import logging
from pathlib import Path
import yaml
import re

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
    """Reproduces the localisation code channel selection when extracting waveforms from raw data.

    Args:
        trace_indices (np.array): Index of the trace of the detected spike (nspikes,).
        xy (np.array, optional): Complex coordinates of channels. If None, uses neuropixel trace header.
        extract_radius_um (float, optional): Radius from peak trace. All traces within this radius will be included. Defaults to EXTRACT_RADIUS_UM.
        return_complex (bool, optional): If True, returns the complex coordinates, otherwise returns a 3D x, y, z array. Defaults to False.
        return_indices (bool, optional): If True, returns the indices of the channels within the radius. Defaults to False.

    Returns:
        np.array or tuple: 
            - If return_indices is False: (nspikes, ntraces, n_coordinates) array of axial and transverse coordinates
            - If return_indices is True: tuple of (coordinates, indices) where indices is (nspikes, ntraces) array
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
    """Compute the distance between each channel and all other channels, and find indices within radius.

    For each row, the indices of the channels within the specified radius are returned.

    Args:
        xy (np.array): ntr complex array of x and y coordinates of each channel relative to the probe.
        extract_radius_um (float, optional): Extraction radius in micrometers. Defaults to EXTRACT_RADIUS_UM.

    Returns:
        np.array: ntr x ntr_wav matrix of channel indices within the radius, where ntr_wav is the maximum number of channels within radius.
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
    """Get autism data from JP fmr1 mouse line.

    Args:
        one: ONE client instance for accessing the database.

    Returns:
        tuple: A tuple containing:
            - List of insertion IDs for FMR subjects
            - List of full insertion information for FMR subjects
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
    """Get atlas PIDs from the IBL neuropixel brainwide project.

    Args:
        one: ONE client instance for accessing the database.
        tracing (bool, optional): If True, only return insertions with existing tracing. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - List of insertion IDs
            - List of full insertion information
    """
    django_strg = [
        "session__projects__name__icontains,ibl_neuropixel_brainwide_01",
        "~json__qc,CRITICAL",
        # 'session__extended_qc__behavior,1',
        "session__json__IS_MOCK,False",
    ]
    if tracing:
        django_strg.append("json__extended_qc__tracing_exists,True")

    insertions = one.alyx.rest("insertions", "list", django=django_strg)
    return [item["id"] for item in insertions], insertions


def read_correlogram(file_correlogram, nclusters):
    """Read correlogram data from a binary file.

    Args:
        file_correlogram (str or Path): Path to the correlogram binary file.
        nclusters (int): Number of clusters in the data.

    Returns:
        np.ndarray: Memory-mapped correlogram array of shape (nclusters, nbins).
    """
    nbins = int(Path(file_correlogram).stat().st_size / nclusters / 4)
    mmap_correlogram = np.memmap(
        file_correlogram, dtype="int32", shape=(nclusters, nbins)
    )
    return mmap_correlogram

def _get_immediate_children(bucket, prefix=None, delimiter='/'):
    """Base function to get the immediate children of a prefix on AWS S3 as a list.
    
    Args:
        bucket: AWS S3 bucket object
        prefix (str, optional): S3 prefix to search under
        delimiter (str, optional): Delimiter to use for splitting keys. Defaults to '/'
        
    Returns:
        list: List of immediate child prefixes
    """
    immediate_children = set()

    for obj in bucket.objects.filter(Prefix=prefix):
        key = obj.key[len(prefix):]  # Remove the base prefix from key
        if delimiter in key:
            # Extract the immediate child prefix up to the first delimiter
            child_prefix = key.split(delimiter)[0]
            immediate_children.add(child_prefix)
        else:
            # This is an object directly inside the prefix (not a folder)
            # Optional: include or ignore
            pass

    return list(immediate_children)


def get_immediate_labels(bucket, prefix=None, delimiter='/', limit=None):
    """Get immediate children under a prefix that match the label format (YYYY_WXX) from AWS S3.
    
    Args:
        bucket: AWS S3 bucket object
        prefix (str, optional): S3 prefix to search under
        delimiter (str, optional): Delimiter to use for splitting keys. Defaults to '/'
        limit (int, optional): Maximum number of results to return
        
    Returns:
        list: List of immediate child prefixes that match the label format
    """
    immediate_children = _get_immediate_children(bucket, prefix, delimiter)
    
    # Filter for label format (YYYY_WXX)
    filtered_children = []
    for child_prefix in immediate_children:
        if re.match(r'^\d{4}_W\d{2}$', child_prefix):
            filtered_children.append(child_prefix)
        else:
            pass
            # print(f"Skipping {child_prefix} as it does not match the expected format")
    
    return sorted(filtered_children, reverse=True)[:limit]


def list_available_projects(one=None):
    """Get the list of available projects."""
    assert one is not None, "ONE client instance is required"
    _logger.info("Listing available projects.")
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    bucket = s3.Bucket(bucket_name)    
    return _get_immediate_children(bucket, prefix="aggregates/atlas/features/", delimiter='/')

def list_available_labels(one=None, project = None, limit=None):
    """List available labels on AWS S3."""
    assert one is not None, "ONE client instance is required"
    assert project is not None, "First get list of available projects using list_available_projects(one=one)"
    _logger.info(f"Listing available labels for project: {project}")
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    bucket = s3.Bucket(bucket_name)
    return get_immediate_labels(bucket, prefix=f"aggregates/atlas/features/{project}/", delimiter='/', limit=limit)

def get_latest_label(one=None, project = None):
    """Get the latest label on AWS S3."""
    return list_available_labels(one=one, project = project, limit=1)[0]

def download_tables(
    local_path,
    label="2024_W50",
    project=None,
    agg_level="agg_full",
    one=None,
    verify=False,
    overwrite=False,
    extended=False,
):
    """Download electrophysiology data tables from AWS S3.

    Downloads aggregated electrophysiology data from the IBL AWS S3 bucket to a local directory.
    For "agg_full", first tries the new path structure then falls back to the original path for
    backward compatibility. For other agg_levels, uses the direct path structure.

    Args:
        local_path (Path): Path where the data will be stored locally.
        label (str, optional): Revision string (e.g., "2024_W04"). Defaults to "2024_W50".
        project (str, optional): Project name. Defaults to "ea_active" if None.
        agg_level (str, optional): Aggregation level for the path structure. Defaults to "agg_full".
            For "agg_full": tries f"aggregates/atlas/features/{project}/{label}/{agg_level}" first,
            then falls back to f"aggregates/atlas/features/{project}/{label}" for backward compatibility.
            For other values: uses f"aggregates/atlas/features/{project}/{label}/{agg_level}" directly.
        one: ONE client instance for AWS authentication.
        verify (bool, optional): Checks the indices and consistency of the dataframes and raises an error if not consistent. Defaults to False.
        overwrite (bool, optional): Force redownloading if file exists. Defaults to False.
        extended (bool, optional): If True, will download also extended datasets, such as cross-correlograms that take up
            more space than just the tables (couple hundreds Mb for the table, several GB with extended data). Defaults to False.

    Returns:
        Path: Local path where the data was downloaded.

    Raises:
        AssertionError: If the specified label is not found on AWS.
    """
    # Set default project if None
    if project is None:
        project = "ea_active"
        _logger.warning(f"Project is None, using default project: {project}")
    
    # Create local directory structure
    local_path = Path(local_path).joinpath(project).joinpath(label).joinpath(agg_level)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Get AWS credentials
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    
    # Download main data with backward compatibility for agg_full
    if agg_level == "agg_full":
        # Try the new path with agg_level first, then fall back to backward compatibility
        primary_path = f"aggregates/atlas/features/{project}/{label}/{agg_level}"
        fallback_path = f"aggregates/atlas/features/{project}/{label}"
        
        try:
            local_files = aws.s3_download_folder(
                primary_path,
                local_path,
                s3=s3,
                bucket_name=bucket_name,
                overwrite=overwrite,
            )
            if len(local_files) == 0:
                # Primary path doesn't exist, try fallback
                local_files = aws.s3_download_folder(
                    fallback_path,
                    local_path,
                    s3=s3,
                    bucket_name=bucket_name,
                    overwrite=overwrite,
                )
        except Exception:
            # If primary path fails, try fallback
            local_files = aws.s3_download_folder(
                fallback_path,
                local_path,
                s3=s3,
                bucket_name=bucket_name,
                overwrite=overwrite,
            )
    else:
        # For other agg_levels, use the direct path
        local_files = aws.s3_download_folder(
            f"aggregates/atlas/features/{project}/{label}/{agg_level}",
            local_path,
            s3=s3,
            bucket_name=bucket_name,
            overwrite=overwrite,
        )
    
    # Download extended data if requested
    if extended:
        local_files = aws.s3_download_folder(
            f"aggregates/atlas/features/{project}/{label}_extended",
            local_path,
            s3=s3,
            bucket_name=bucket_name,
            overwrite=overwrite,
        )
    
    assert len(local_files), f"aggregates/atlas/{project}/{label} not found on AWS"
    return local_path


def read_features_from_disk(
    path_features: Path,
    brain_atlas: "iblatlas.atlas.BrainAtlas" = None,
    mappings: list[str] = None,
    strict: bool = True,
    load_denoised: bool = True,
) -> pd.DataFrame:
    """Read electrophysiology features from disk and merge with channel information.

    This function loads raw electrophysiology features, channel information, and channel labels
    from parquet files, merges them into a single dataframe, and adds brain region mapping
    information using the provided brain atlas.

    Args:
        path_features (Path): Path to the directory containing the feature parquet files.
        brain_atlas (iblatlas.atlas.BrainAtlas, optional): Brain atlas object used to map coordinates to brain regions.
            Must be provided to enable region mapping.
        mappings (list, optional): List of brain region mapping ontologies to include.
            Default is ['Cosmos', 'Beryl'].
        strict (bool, optional): Whether to raise an error on panderas validation. Default is True.
        load_denoised (bool, optional): Whether to return the denoised or raw features. Default is True.

    Returns:
        pd.DataFrame: DataFrame containing merged electrophysiology features with channel information
            and brain region mappings.

    Raises:
        AssertionError: If brain atlas is not provided or if unknown mappings are specified.
    """
    mappings = ["Cosmos", "Beryl"] if mappings is None else mappings
    brain_atlas = (
        brain_atlas if brain_atlas is not None else ephysatlas.anatomy.ClassifierAtlas()
    )
    assert brain_atlas is not None, "Brain atlas is required to map labels to regions"
    assert all(
        mapping in brain_atlas.regions.mappings for mapping in mappings
    ), f"Unknown mapping: {mappings}"
    # merge the channel information with the features
    if load_denoised:  # load the denoised features
        df_features = pd.read_parquet(path_features / "raw_ephys_features_denoised.pqt")
    else:  # load the raw features
        df_features = pd.read_parquet(path_features / "raw_ephys_features.pqt")
    df_channels = pd.read_parquet(path_features / "channels.pqt")
    duplicate_cols = set(df_features.columns).intersection(set(df_channels.columns))
    df_channels = df_channels.drop(columns=duplicate_cols)
    df_features = df_features.merge(
        df_channels,
        how="inner",
        right_index=True,
        left_index=True,
    )
    if ("channel_labels" not in df_features.columns) and (
        "labels" not in df_features.columns
    ):
        df_features = df_features.merge(
            pd.read_parquet(path_features / "channels_labels.pqt").fillna(0),
            how="inner",
            right_index=True,
            left_index=True,
        )
        df_features.rename(columns={"labels": "channel_labels"}, inplace=True)
    if "channel_labels" in df_features.columns:
        df_features["outside"] = df_features["channel_labels"] == 3
    elif "labels" in df_features.columns:
        df_features["outside"] = df_features["labels"] == 3
    else:
        raise ValueError("channel_labels or labels not found in the features dataframe")

    aids = brain_atlas.get_labels(
        df_features.loc[:, ["x", "y", "z"]].values, mode="clip"
    )
    df_features["Allen_id"] = aids
    for mapping in mappings:
        df_features[f"{mapping}_id"] = brain_atlas.regions.remap(aids, "Allen", mapping)

    # this will make sure that the features dataframe is compatible and healthy
    if strict:
        df_features = pd.DataFrame(ephysatlas.features.ModelRawFeatures(df_features))

    return df_features


def compute_depth_dataframe(df_raw_features, df_clusters, df_channels):
    """Compute a features dataframe for each pid and depth along the probe.

    Merges the raw voltage features and the clusters features, aggregating by depth.

    Args:
        df_raw_features (pd.DataFrame): DataFrame containing raw voltage features.
        df_clusters (pd.DataFrame): DataFrame containing cluster information.
        df_channels (pd.DataFrame): DataFrame containing channel information.

    Returns:
        pd.DataFrame: DataFrame with features aggregated by (pid, axial_um) groups.
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
    """Load configuration from the config-ephys-atlas.yaml file.

    Returns:
        dict: Configuration dictionary loaded from the YAML file.
    """
    file_yaml = Path(__file__).parents[2].joinpath("config-ephys-atlas.yaml")
    with open(file_yaml, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def compute_summary_stat(df_voltage, features):
    """Compute summary statistics for specified features.

    Args:
        df_voltage (pd.DataFrame): DataFrame containing voltage features.
        features (str or list): Feature name(s) to compute statistics for.

    Returns:
        pd.DataFrame: Summary statistics including median, q05, q95, and dq (q95 - q05).
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
    """Sort features based on their values.

    Args:
        values (np.array): Array of values to sort by (e.g., p-values, metrics).
        features (np.array): Array of feature names corresponding to the values.
        ascending (bool, optional): If True, sort in ascending order. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - values_sort (np.array): Sorted values
            - features_sort (np.array): Features sorted according to the values
    """
    id_sort = np.argsort(values)
    if not ascending:
        id_sort = np.flip(id_sort)
    features_sort = features[id_sort]
    values_sort = values[id_sort]
    return values_sort, features_sort


def prepare_mat_plot(array_in, id_feat, diag_val=0):
    """Prepare matrix for plotting from brain-to-brain regions comparison results.

    From the matrix storing the results of brain-to-brain regions comparison in the upper triangle for all features,
    select a feature and create a matrix with transpose for plotting in 2D.

    Args:
        array_in (np.array): Array of N region x N region x N feature.
        id_feat (int): Index of feature that will be displayed.
        diag_val (float, optional): Value to fill the lower triangle. Defaults to 0.

    Returns:
        np.array: Matrix prepared for plotting with symmetric values.
    """
    mat_plot = np.squeeze(array_in[:, :, id_feat].copy())
    mat_plot[np.tril_indices_from(mat_plot)] = diag_val  # replace Nan by 0
    mat_plot = mat_plot + mat_plot.T  # add transpose for display
    return mat_plot


def prepare_df_voltage(df_voltage, df_channels, br=None):
    """Prepare voltage DataFrame by merging with channel information and adding derived features.

    Args:
        df_voltage (pd.DataFrame): DataFrame containing voltage features.
        df_channels (pd.DataFrame): DataFrame containing channel information.
        br (iblatlas.atlas.BrainRegions, optional): Brain regions object for mapping. If None, creates a new one.

    Returns:
        pd.DataFrame: Prepared DataFrame with merged channel information, region mappings, and derived features.
    """
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
