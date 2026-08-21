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

# Files in cells_aggregates/ that are always downloaded (standard set, ~1 GB)
_CELLS_AGGREGATES_FILES = [
    "clusters.table.pqt",
    "clusters_good.table.pqt",
    "clusters.acgs_log.npy",
    "acgs_log.times.npy",
    "clusters.waveforms_peak.npy",
    "clusters_good.stpc.npy",
    "clusters_good.stlfp.npy",
]
# Large neighbourhood-waveform files (~8 GB combined) — opt-in only
_WAVEFORMS_FILES = [
    "waveforms.voltage.npy",
    "waveforms.table.pqt",
]
# 3D (firing-rate-decile x log-time-lag) ACGs (~3.5 GB) — opt-in, not every
# historical dataset has these (see ephysatlas.cells.compute_3d_acgs)
_ACG3D_FILES = [
    "clusters.acgs_3d.npy",
    "acgs_3d.times.npy",
]
# Merged multi-recording LFP archives in lfp_aggregates/, keyed by compression level.
_LFP_AGGREGATES_FILES = {
    "default": "lf_compressed_all.h5",  # epsilon=150, alpha=28, ~23 GB
    "aggressive": "lf_compressed_aggressive_all.h5",  # epsilon=450, alpha=96, ~12 GB
}


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


def atlas_pids(one, tracing=True, project="ibl_neuropixel_brainwide_01"):
    """Get atlas PIDs from the IBL neuropixel brainwide project.

    Args:
        one: ONE client instance for accessing the database.
        tracing (bool, optional): If True, only return insertions with existing tracing. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - List of insertion IDs
            - List of full insertion information
    """
    django_strg = [
        f"session__projects__name__icontains,{project}",
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


def _get_immediate_children(bucket, prefix=None, delimiter="/"):
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
        key = obj.key[len(prefix) :]  # Remove the base prefix from key
        if delimiter in key:
            # Extract the immediate child prefix up to the first delimiter
            child_prefix = key.split(delimiter)[0]
            immediate_children.add(child_prefix)
        else:
            # This is an object directly inside the prefix (not a folder)
            # Optional: include or ignore
            pass

    return list(immediate_children)


def get_immediate_labels(bucket, prefix=None, delimiter="/", limit=None):
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
        if re.match(r"^\d{4}_W\d{2}$", child_prefix):
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
    return _get_immediate_children(
        bucket, prefix="aggregates/atlas/features/", delimiter="/"
    )


def list_available_labels(one=None, project=None, limit=None):
    """List available labels on AWS S3."""
    assert one is not None, "ONE client instance is required"
    assert project is not None, (
        "First get list of available projects using list_available_projects(one=one)"
    )
    _logger.info(f"Listing available labels for project: {project}")
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    bucket = s3.Bucket(bucket_name)
    return get_immediate_labels(
        bucket,
        prefix=f"aggregates/atlas/features/{project}/",
        delimiter="/",
        limit=limit,
    )


def get_latest_label(one=None, project=None):
    """Get the latest label on AWS S3."""
    return list_available_labels(one=one, project=project, limit=1)[0]


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

    assert len(local_files), (
        f"aggregates/atlas/features/{project}/{label} not found on AWS"
    )
    return local_path


_ENCODING_VOLUME_FILENAME_RE = re.compile(r"brainwide_ephys_atlas_(\d+)um\.npz$")


def _list_encoding_volume_resolutions(s3, bucket_name, project, label):
    """List the voxel resolutions (µm) available on S3 for an encoding volume vintage."""
    prefix = f"aggregates/atlas/encoding_volumes/{project}/{label}/"
    objects = s3.Bucket(name=bucket_name).objects.filter(Prefix=prefix)
    resolutions = (
        int(match.group(1))
        for match in (_ENCODING_VOLUME_FILENAME_RE.search(obj.key) for obj in objects)
        if match is not None
    )
    return sorted(resolutions)


def download_encoding_volume(
    local_path, label="2026_W12", project=None, res_um=None, one=None, overwrite=False
):
    """Download a pre-computed ephys atlas encoding volume from AWS S3.

    The encoding volume is a 4-D volumetric representation of electrophysiological
    features on the Allen Common Coordinate Framework (CCF), stored as a .npz file.
    Load the result with ``np.load(file_path, allow_pickle=True)``.

    Parameters
    ----------
    local_path : Path
        Local directory where the file will be saved.
    label : str, optional
        Vintage label, e.g. "2026_W12". Defaults to "2026_W12".
    project : str, optional
        Project name. Defaults to "ea_active".
    res_um : int, optional
        CCF voxel resolution in µm of the requested volume, e.g. 25 or 50.
        If not specified, automatically picks the finest (smallest) resolution
        available on S3 for this ``project``/``label``.
    one : ONE
        ONE client instance for AWS authentication.
    overwrite : bool, optional
        Force re-download even if the file already exists. Defaults to False.

    Returns
    -------
    Path
        Local path to the downloaded .npz file.
    """
    if project is None:
        project = "ea_active"
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    if res_um is None:
        resolutions = _list_encoding_volume_resolutions(s3, bucket_name, project, label)
        if not resolutions:
            raise FileNotFoundError(
                f"No encoding volume found on S3 for project={project!r}, label={label!r}"
            )
        res_um = resolutions[0]
        _logger.info(
            f"res_um not specified, using finest available resolution: {res_um} um"
        )
    filename = f"brainwide_ephys_atlas_{res_um}um.npz"
    local_file = Path(local_path).joinpath(filename)
    s3_key = f"aggregates/atlas/encoding_volumes/{project}/{label}/{filename}"
    return aws.s3_download_file(
        s3_key, local_file, s3=s3, bucket_name=bucket_name, overwrite=overwrite
    )


def outlier_treatment(df_features, columns=None, replace_with_nan=False):
    # TODO can make it more general by allowing for different detection and replacement functions.
    if columns is None:
        return df_features
    bad_index = False
    for column in columns:
        # Threshold based on a factor of the median value.
        bad = df_features[column].values >= (1e3 * np.nanmedian(df_features[column]))
        bad_index = np.logical_or(bad_index, bad)
    if np.sum(bad) > 0:
        _logger.warning(
            f"Number of bad channels: {np.sum(bad)},"
            f" those will be replaced with replace_with_nan = {replace_with_nan} strategy."
        )
    for column in columns:
        if replace_with_nan:
            df_features.loc[bad_index, column] = np.nan
        else:
            df_features.loc[bad_index, column] = np.nanmedian(df_features[column])

    return df_features


def replace_nan(df_features, columns=None):
    if columns is None:
        return df_features
    for column in columns:
        if np.isnan(df_features[column]).sum() > 0:
            _logger.warning(
                f"Number of nan values in column {column}: {np.isnan(df_features[column]).sum()}"
            )
            df_features.loc[np.isnan(df_features[column]), column] = np.nanmedian(
                df_features[column]
            )
        else:
            _logger.info(f"No nan values in column {column}")
    return df_features


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
    assert all(mapping in brain_atlas.regions.mappings for mapping in mappings), (
        f"Unknown mapping: {mappings}"
    )
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

    # Do the outlier treatment for the alpha features.
    df_features = outlier_treatment(df_features, columns=["alpha_mean", "alpha_std"])

    return df_features


def _project_s3(local_path, project, one):
    """Set up ONE/S3 connection and return (s3, bucket_name, local_project_path)."""
    from one.api import ONE

    if one is None:
        one = ONE()
    local_project_path = Path(local_path) / project
    local_project_path.mkdir(parents=True, exist_ok=True)
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    return s3, bucket_name, local_project_path


def download_probe_details(
    local_path, project="ibl_neuropixel_brainwide_01", one=None, overwrite=False
):
    """Download probe insertion metadata (df_probe_details.pqt) for a project from S3.

    Args:
        local_path (Path): Local root; file is placed under local_path/project/.
        project (str): Project name.
        one (one.api.ONE, optional): ONE instance for AWS credentials.
        overwrite (bool): Re-download if file already exists locally.

    Returns:
        Path: Path to the downloaded df_probe_details.pqt file.
    """
    s3, bucket_name, local_project_path = _project_s3(local_path, project, one)
    local_file = local_project_path / "df_probe_details.pqt"
    if overwrite or not local_file.exists():
        s3.Bucket(bucket_name).download_file(
            f"aggregates/atlas/projects/{project}/df_probe_details.pqt",
            str(local_file),
        )
    return local_file


def download_cells_features(
    local_path,
    project="ibl_neuropixel_brainwide_01",
    one=None,
    overwrite=False,
    large_files=False,
    acg3d=False,
):
    """Download cluster-level aggregates from S3 (``cells_aggregates/`` subfolder, ~1 GB).

    Downloads the standard cluster files. The large waveform files
    (``waveforms.voltage.npy`` and ``waveforms.table.pqt``, ~8 GB combined) are only
    downloaded when ``large_files=True``. The 3D ACG files (``clusters.acgs_3d.npy``
    and ``acgs_3d.times.npy``, ~3.5 GB) are only downloaded when ``acg3d=True``.

    Use :func:`download_probe_details` if you only need probe metadata, or
    :func:`download_project_data` to get both in one call.

    Args:
        local_path (Path): Local root; files are placed under local_path/project/cells_aggregates/.
        project (str): Project name.
        one (one.api.ONE, optional): ONE instance for AWS credentials.
        overwrite (bool): Re-download files that already exist locally.
        large_files (bool): If True, also download waveforms.voltage.npy and waveforms.table.pqt
            (~8 GB). Defaults to False.
        acg3d (bool): If True, also download clusters.acgs_3d.npy and acgs_3d.times.npy
            (~3.5 GB). Not every historical dataset has these. Defaults to False.

    Returns:
        Path: local_path/project/cells_aggregates/ directory.
    """
    s3, bucket_name, local_project_path = _project_s3(local_path, project, one)
    dest = local_project_path / "cells_aggregates"
    dest.mkdir(parents=True, exist_ok=True)
    s3_prefix = f"aggregates/atlas/projects/{project}/cells_aggregates"
    files_to_download = (
        _CELLS_AGGREGATES_FILES
        + (_WAVEFORMS_FILES if large_files else [])
        + (_ACG3D_FILES if acg3d else [])
    )
    for fname in files_to_download:
        aws.s3_download_file(
            f"{s3_prefix}/{fname}",
            dest / fname,
            s3=s3,
            bucket_name=bucket_name,
            overwrite=overwrite,
        )
    return dest


def download_project_data(
    local_path,
    project="ibl_neuropixel_brainwide_01",
    one=None,
    overwrite=False,
    large_files=False,
    acg3d=False,
):
    """Download all project data (probe details + cell aggregates) from S3.

    Convenience wrapper that calls download_probe_details() and download_cells_features().

    Args:
        local_path (Path): Local root; files are placed under local_path/project/.
        project (str): Project name.
        one (one.api.ONE, optional): ONE instance for AWS credentials.
        overwrite (bool): Re-download files that already exist locally.
        large_files (bool): If True, also download the large waveform files (~8 GB). Defaults to False.
        acg3d (bool): If True, also download the 3D ACG files (~3.5 GB). Defaults to False.

    Returns:
        Path: local_path/project directory.
    """
    download_probe_details(local_path, project=project, one=one, overwrite=overwrite)
    download_cells_features(
        local_path,
        project=project,
        one=one,
        overwrite=overwrite,
        large_files=large_files,
        acg3d=acg3d,
    )
    return Path(local_path) / project


def read_probe_details(path_project, strict=True):
    """Read probe insertion metadata from disk.

    Args:
        path_project (Path): Path to the project folder (containing df_probe_details.pqt).
        strict (bool): Validate against ModelProbeDetails schema if True.

    Returns:
        pd.DataFrame: One row per probe insertion.
    """
    import ephysatlas.features

    df = pd.read_parquet(Path(path_project) / "df_probe_details.pqt")
    if strict:
        df = pd.DataFrame(ephysatlas.features.ModelProbeDetails.validate(df))
    return df


def read_cells_features(path_project):
    """Read cluster-level features and associated arrays from cells_aggregates/.

    Parameters
    ----------
    path_project : Path
        Path to the project folder (parent of cells_aggregates/).

    Returns
    -------
    dict
        Always present: df_clusters, df_clusters_good, acgs_log, acgs_log_times,
        waveforms_peak, stpc, stlfp.
        Present only if downloaded with ``large_files=True``: waveforms, df_waveforms.
        Present only if downloaded with ``acg3d=True``: acgs_3d, acgs_3d_times.

    See Also
    --------
    docs/source/how-to/load-cells-features.rst : full file reference, shapes and dtypes.
    """
    path = Path(path_project) / "cells_aggregates"
    result = {
        "df_clusters": pd.read_parquet(path / "clusters.table.pqt"),
        "df_clusters_good": pd.read_parquet(path / "clusters_good.table.pqt"),
        "acgs_log": np.load(path / "clusters.acgs_log.npy", mmap_mode="r").astype(
            np.float32
        ),
        "acgs_log_times": np.load(path / "acgs_log.times.npy"),
        "waveforms_peak": np.load(
            path / "clusters.waveforms_peak.npy", mmap_mode="r"
        ).astype(np.float32),
        "stpc": np.load(path / "clusters_good.stpc.npy", mmap_mode="r"),
        "stlfp": np.load(path / "clusters_good.stlfp.npy", mmap_mode="r"),
    }
    if (path / "waveforms.voltage.npy").exists():
        result["waveforms"] = np.load(path / "waveforms.voltage.npy", mmap_mode="r")
        result["df_waveforms"] = pd.read_parquet(path / "waveforms.table.pqt")
    if (path / "clusters.acgs_3d.npy").exists():
        # stays a float16 memmap rather than eager-cast to float32 (as acgs_log
        # does): at (n_clusters, 10, 201) this array is ~3.5 GB even at float16.
        result["acgs_3d"] = np.load(path / "clusters.acgs_3d.npy", mmap_mode="r")
        result["acgs_3d_times"] = np.load(path / "acgs_3d.times.npy")
    return result


def download_lfp_features(
    local_path,
    project="ibl_neuropixel_brainwide_01",
    one=None,
    overwrite=False,
    level="default",
):
    """Download the merged LFP-compressed HDF5 archive from AWS S3.

    The archive is a single multi-recording HDF5 file (produced by ``lfpack.merge_h5``)
    containing one top-level group per insertion (pid). Use :func:`read_lfp_features`
    to open a reader for a specific pid.

    Parameters
    ----------
    local_path : Path
        Local root; file is placed under ``local_path/project/lfp_aggregates/``.
    project : str, optional
        Project name. Defaults to "ibl_neuropixel_brainwide_01".
    one : one.api.ONE, optional
        ONE client instance for AWS authentication.
    overwrite : bool, optional
        Force re-download if the file already exists locally. Defaults to False.
    level : {"default", "aggressive"}, optional
        Compression level to download. "default" (epsilon=150, alpha=28, ~23 GB) is
        higher fidelity; "aggressive" (epsilon=450, alpha=96, ~12 GB) trades fidelity
        for size. Defaults to "default".

    Returns
    -------
    Path
        Local path to the downloaded HDF5 file.
    """
    assert level in _LFP_AGGREGATES_FILES, (
        f"level must be one of {list(_LFP_AGGREGATES_FILES)}, got {level!r}"
    )
    s3, bucket_name, local_project_path = _project_s3(local_path, project, one)
    dest = local_project_path.joinpath("lfp_aggregates")
    dest.mkdir(parents=True, exist_ok=True)
    fname = _LFP_AGGREGATES_FILES[level]
    local_file = dest.joinpath(fname)
    aws.s3_download_file(
        f"aggregates/atlas/projects/{project}/lfp_aggregates/{fname}",
        local_file,
        s3=s3,
        bucket_name=bucket_name,
        overwrite=overwrite,
    )
    return local_file


def read_lfp_features(path_project, pid, level="default", scale=0, bin_channels=1):
    """Open a decompressed LFP reader for one insertion from the merged HDF5 archive.

    Parameters
    ----------
    path_project : Path
        Path to the project folder (parent of ``lfp_aggregates/``).
    pid : str
        Insertion ID; matches the top-level recording key in the merged HDF5 file.
    level : {"default", "aggressive"}, optional
        Compression level to read, matching the file downloaded by
        :func:`download_lfp_features`. Defaults to "default".
    scale : int, optional
        Pyramidal resolution level to open, 0 = base full LFP rate. Defaults to 0.
    bin_channels : int, optional
        Number of adjacent channels to sum together on every read. Defaults to 1
        (no binning).

    Returns
    -------
    lfpack.LFPackReader
        Drop-in ``spikeglx.Reader``-like object; chunks are decompressed on demand.
        ``sr[0:2500, :]`` returns an ``(n_samples, n_channels)`` float32 array in volts.
    """
    import lfpack

    assert level in _LFP_AGGREGATES_FILES, (
        f"level must be one of {list(_LFP_AGGREGATES_FILES)}, got {level!r}"
    )
    h5_file = Path(path_project).joinpath(
        "lfp_aggregates", _LFP_AGGREGATES_FILES[level]
    )
    return lfpack.LFPackReader(
        h5_file, recording=pid, scale=scale, bin_channels=bin_channels
    )


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
