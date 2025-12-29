from pathlib import Path
from typing import List
import logging
import pandas as pd
from functools import reduce
from joblib import Parallel, delayed
from ephysatlas.utils import get_aggregated_snippets_df
from ephysatlas.data import outlier_treatment, replace_nan
from ephysatlas.features import ChannelDataFrameSchema, ModelRawFeatures
from ephysatlas.features import denoise_dataframe
import tqdm
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)

# TODO - Test at the end - ephysatlas.data.read_features_from_disk(path_features, brain_atlas=brain_atlas, strict=True)


# TODO - There can be a better way to specify both of these arguments - maybe only one is needed.
def aggregate_all_probes(
    path_list: List[Path], base_level_dir: Path | None | str = None, n_jobs: int = -1, verbose: int = 1
):
    """Aggregate snippet-level data from multiple probe directories into a single DataFrame.

    This function processes a list of probe directory paths in parallel using joblib, calling `get_aggregated_snippets_df` on each to extract and concatenate their snippet-level data into a unified DataFrame. Optionally, it can annotate the result with a `base_level_dir` column in case the base_directory is moved.

    Args:
        path_list (List[pathlib.Path]): List of Path objects, each pointing to a probe directory containing snippet-level data to aggregate. Each directory must be compatible with `get_aggregated_snippets_df`.
        base_level_dir (Path or str or None, optional): If provided, this value will update the column ('base_level_dir') to the resulting DataFrame for all rows.

    Returns:
        pandas.DataFrame: A DataFrame containing the concatenated snippet-level data from all probes in `path_list`.

    Example:
        >>> from pathlib import Path
        >>> probe_dirs = [Path('/data/probe1'), Path('/data/probe2')]
        >>> df = aggregate_all_probes(probe_dirs, base_level_dir='/data')
        >>> print(df.head())

    Note:
        - Each path in `path_list` should point to a directory structure compatible with `get_aggregated_snippets_df`.
        - The function processes all paths in parallel using all available CPU cores (n_jobs=-1).
        - The function ignores index continuity and resets the index in the returned DataFrame.
        - If `base_level_dir` is provided, it is stored as a string in the 'base_level_dir' column for all rows.
        - This function does not perform validation on the contents of the aggregated DataFrames beyond concatenation.
    """
    # Process all paths in parallel using joblib
    dfs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(get_aggregated_snippets_df)(path) for path in path_list
    )
    
    # Concatenate all results into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    if base_level_dir is not None:
        df["base_level_dir"] = Path(base_level_dir).as_posix()

    return df


# Function to aggregate channels dataframe
def concatenate_channels_data(
    parquet_files_channels: List[Path], output_dir: Path | None = None
):
    """Aggregate and validate channel-level metadata from multiple probe directories into a single DataFrame.

    This function reads channel information from a list of Parquet files (one per probe), validates each DataFrame against the `ChannelDataFrameSchema`, and concatenates them into a single DataFrame. The result is grouped by ('pid', 'channel') and the first occurrence is kept for each group. If a 'channel_labels' column is present, it is dropped with a warning. Optionally, the aggregated DataFrame can be saved to a Parquet file in the specified output directory.

    Args:
        parquet_files_channels (List[pathlib.Path]): List of Path objects, each pointing to a Parquet file containing channel-level metadata for a probe. Each file must be compatible with `ChannelDataFrameSchema`.
        output_dir (Path or None, optional): If provided, the aggregated DataFrame is saved as 'channels.pqt' in this directory. The directory is created if it does not exist. Default is None (no file is written).

    Returns:
        pandas.DataFrame: A DataFrame containing the concatenated and grouped channel metadata from all input files. Indexed by ('pid', 'channel').

    Example:
        >>> from pathlib import Path
        >>> parquet_files = [Path('probe1/channels.pqt'), Path('probe2/channels.pqt')]
        >>> df_channels = concatenate_channels_data(parquet_files, output_dir=Path('output'))
        >>> print(df_channels.head())

    Note:
        - Each input file is validated using `ChannelDataFrameSchema.validate`.
        - If a file lacks a 'channel' column, it is created from the DataFrame index.
        - The result is grouped by ('pid', 'channel') and the first entry is kept for each group.
        - If a 'channel_labels' column is present, it is dropped with a warning.
        - If `output_dir` is provided, the result is saved as 'channels.pqt' in that directory.
    """
    df_channels = []
    for i, fc in enumerate(parquet_files_channels):
        dfd = pd.read_parquet(fc)
        if "channel" not in dfd.columns:
            dfd["channel"] = dfd.index
        ChannelDataFrameSchema.validate(dfd)
        df_channels.append(dfd)
    logger.debug(f"Length of df_channels = {len(df_channels)}")
    logger.debug(f"Shape of df_channels items = {[d.shape for d in df_channels]}")
    df_channels = pd.concat(df_channels)
    df_channels = df_channels.groupby(["pid", "channel"]).first()

    if "channel_labels" in df_channels.columns:
        logger.warning(
            "channel_labels column found in channels.pqt file. "
            "This column will be dropped."
        )
        df_channels = df_channels.drop(columns=["channel_labels"])

    if output_dir is not None:
        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        df_channels.to_parquet(output_dir / "channels.pqt")
    return df_channels


def get_features_from_snippets(snippet_level_dir: Path):
    """Load and merge all feature tables from a snippet-level directory into a single DataFrame.

    This function scans the given directory for all Parquet files (*.pqt), loads each as a DataFrame,
    and merges them on the 'channel' column using an outer join. The result is a wide DataFrame
    containing all available features for each channel in the snippet.

    Args:
        snippet_level_dir (Path): Path to the directory containing snippet-level feature Parquet files. Each file should have a 'channel' column.

    Returns:
        pandas.DataFrame: A DataFrame where each row corresponds to a channel, and columns are the union of all features
            found in the Parquet files in the directory. Channels missing from some files will have NaNs for those features.

    Example:
        >>> from pathlib import Path
        >>> df = get_features_from_snippets(Path('probe1/snippet_0001'))
        >>> print(df.head())

    Note:
        - All Parquet files in the directory are loaded and merged. Files must have a 'channel' column.
        - The merge is performed as an outer join, so all channels present in any file are included.
        - If no Parquet files are found, the function returns an empty DataFrame.
    """
    # Ensure the input is a Path object
    snippet_level_dir = Path(snippet_level_dir)
    # Find all Parquet files in the directory
    feature_files = list(snippet_level_dir.glob("*.pqt"))
    df = {}
    # Load each Parquet file into a DataFrame, keyed by the file stem
    for file in feature_files:
        df[file.stem] = pd.read_parquet(file)

    # Merge all DataFrames on the 'channel' column using an outer join
    df_voltage = (
        reduce(
            lambda left, right: pd.merge(left, right, on="channel", how="outer"),
            [df[k] for k in df.keys()],
        )
        if df
        else pd.DataFrame()
    )

    return df_voltage


def concat_raw_features(input_df: pd.DataFrame):
    """Concatenate raw features from multiple snippet directories into a single DataFrame.

    This function iterates over each row in the input DataFrame, constructs the path to each snippet directory,
    loads the raw features from that directory using `get_features_from_snippets`, and concatenates all results
    into a single DataFrame. This is typically used to combine features from multiple snippets of a pid.

    Args:
        input_df (pandas.DataFrame): DataFrame containing at least two columns:
            - 'base_level_dir': Base directory path for each snippet
            - 'snippet_level_dir': Relative path to the snippet directory within the base directory

    Returns:
        pandas.DataFrame: A DataFrame containing all raw features from all snippet directories, concatenated vertically.
            Each row represents features for a single channel from a single snippet.

    Example:
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> input_df = pd.DataFrame({
        ...     'base_level_dir': ['/data/probe1', '/data/probe2'],
        ...     'snippet_level_dir': ['snippet_001', 'snippet_002']
        ... })
        >>> df = concat_raw_features(input_df)
        >>> print(df.head())

    Note:
        - The function expects each row in `input_df` to have 'base_level_dir' and 'snippet_level_dir' columns.
        - Snippet directories are constructed as `base_level_dir / snippet_level_dir`.
        - Features are loaded using `get_features_from_snippets` for each snippet directory.
        - The index is reset during concatenation to ensure continuous indexing.
        - If a snippet directory is empty or invalid, it will be skipped (resulting in an empty DataFrame for that row).
    """
    # Initialize an empty DataFrame to store all concatenated features
    df_final = pd.DataFrame()
    # Iterate over each row in the input DataFrame
    for _, row in input_df.iterrows():
        # Construct the full path to the snippet directory
        snippet_dir = Path(row["base_level_dir"]) / row["snippet_level_dir"]
        # Load raw features from the snippet directory
        df_voltage = get_features_from_snippets(snippet_dir)
        # Concatenate the features from this snippet with the accumulated results
        df_final = pd.concat([df_final, df_voltage], ignore_index=True)

    return df_final


def aggregate_raw_features(concatenated_df: pd.DataFrame):
    """Aggregate raw electrophysiological features by grouping on probe ID and channel.

    This function takes a DataFrame containing raw features from multiple snippets and aggregates them
    by ('pid', 'channel') groups. It uses different aggregation strategies for different feature types:
    - 'spike_count': Mean of non-null values (with zeros for nulls)
    - 'channel_labels': Mode (most frequent value)
    - All other features: Median of non-null values

    Args:
        concatenated_df (pandas.DataFrame): DataFrame containing raw electrophysiological features with at least 'pid' and 'channel' columns.
            Should contain features that match the ModelRawFeatures schema.

    Returns:
        pandas.DataFrame: A DataFrame with one row per unique ('pid', 'channel') combination, containing aggregated
            feature values. The DataFrame is indexed by ('pid', 'channel').

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'pid': ['probe1', 'probe1', 'probe2'],
        ...     'channel': [1, 1, 2],
        ...     'spike_count': [10, 15, 20],
        ...     'amplitude': [0.5, 0.6, 0.4]
        ... })
        >>> agg_df = aggregate_raw_features(df)
        >>> print(agg_df)

    Note:
        - Only features that exist in both the input DataFrame and ModelRawFeatures schema are aggregated.
        - The function handles missing values appropriately for each feature type.
        - For 'spike_count', null values are treated as zeros before taking the mean.
        - For 'channel_labels', if no mode exists, the result is NaN.
        - The result maintains the multi-index structure ('pid', 'channel').
    """
    # Get the list of valid raw feature columns from the schema
    raw_features_columns = ModelRawFeatures.to_schema().columns.keys()

    # Get all columns present in the input DataFrame
    columns_in_df = concatenated_df.columns.tolist()

    # Find the intersection of schema columns and DataFrame columns for aggregation
    agg_columns = set(columns_in_df) & set(raw_features_columns)

    # Define custom aggregation functions for specific feature types
    agg_func_dict = {
        "spike_count": lambda x: np.mean(x.fillna(0)),  # Mean with nulls as zeros
        "channel_labels": lambda x: x.mode().iloc[0]
        if not x.mode().empty
        else np.nan,  # Mode or NaN
    }

    # Helper function to get the appropriate aggregation function for a column
    def get_agg_func(k):
        return agg_func_dict.get(k, lambda x: np.nanmedian(x))

    # Create the aggregation dictionary with NamedAgg for each column
    dagg = {k: pd.NamedAgg(column=k, aggfunc=get_agg_func(k)) for k in agg_columns}
    # Perform the aggregation by grouping on ('pid', 'channel')
    aggregated_df = concatenated_df.groupby(["pid", "channel"]).agg(**dagg)

    return aggregated_df


def get_aggregated_features_per_pid(snippet_df_per_pid: pd.DataFrame):
    """Process and aggregate raw features for a single probe ID (PID) across all its snippets.

    This function takes a DataFrame containing snippet information for a single PID, concatenates
    all raw features from those snippets, aggregates them by channel, and enriches the result with
    channel metadata (axial and lateral positions). The function ensures that only one PID is
    processed at a time and that the required channel metadata file exists.

    Args:
        snippet_df_per_pid (pandas.DataFrame): DataFrame containing snippet information for a single PID. Must have columns:
            - 'pid': Probe ID (must be unique across all rows)
            - 'base_level_dir': Base directory path for each snippet
            - 'snippet_level_dir': Relative path to the snippet directory

    Returns:
        pandas.DataFrame: A DataFrame with one row per channel for the PID, containing:
            - Aggregated raw electrophysiological features
            - Channel metadata (axial_um, lateral_um)
            - pid and channel columns as regular columns (not index)

    Raises:
        AssertionError: If the DataFrame contains more than one unique PID.
        FileNotFoundError: If the channels.pqt file is not found in the snippet directory's parent.

    Example:
        >>> import pandas as pd
        >>> snippet_df = pd.DataFrame({
        ...     'pid': ['probe1', 'probe1'],
        ...     'base_level_dir': ['/data/probe1', '/data/probe1'],
        ...     'snippet_level_dir': ['snippet_001', 'snippet_002']
        ... })
        >>> agg_df = get_aggregated_features_per_pid(snippet_df)
        >>> print(agg_df.head())

    Note:
        - The function requires that all rows have the same PID value.
        - Channel metadata is loaded from channels.pqt in the snippet directory's parent.
        - Channels with bad alpha are filtered out. And those values are set to NaN.
        - The result includes both aggregated features and channel position information.
    """
    # Ensure only one PID is present in the DataFrame
    assert (
        snippet_df_per_pid["pid"].nunique() == 1
    ), "There should be only one pid in the dataframe"

    # Concatenate raw features from all snippets for this PID
    df_concat = concat_raw_features(snippet_df_per_pid)

    # Add the PID information to the concatenated DataFrame
    df_concat["pid"] = snippet_df_per_pid["pid"].iloc[0]

    # Aggregate the raw features by (pid, channel)
    agg_df_per_pid = aggregate_raw_features(df_concat)

    # Reset the index to make pid and channel regular columns
    agg_df_per_pid = agg_df_per_pid.reset_index()

    agg_df_per_pid = outlier_treatment(agg_df_per_pid, columns = ['alpha_mean','alpha_std'], replace_with_nan=True)


    # then we join with the channel information to get coordinates and anatomical information
    # Load channel metadata (axial and lateral positions) from channels.pqt
    # Construct the path to the snippet directory to find its parent
    snippet_level_dir = Path(snippet_df_per_pid["base_level_dir"].iloc[0]) / Path(
        snippet_df_per_pid["snippet_level_dir"].iloc[0]
    )
    # Check if the channels.pqt file exists in the parent directory
    if not (snippet_level_dir.parent / "channels.pqt").exists():
        raise FileNotFoundError(
            f"channels.pqt file not found in {snippet_level_dir.parent}"
        )
    # Load the channel metadata
    df_channels = pd.read_parquet(snippet_level_dir.parent / "channels.pqt")
    # Merge the channel position information with the aggregated features
    agg_df_per_pid = agg_df_per_pid.merge(
        df_channels[["channel", "axial_um", "lateral_um"]], on="channel", how="left"
    )

    return agg_df_per_pid


def get_aggregated_raw_features(
    snippet_df: pd.DataFrame, output_dir: Path | None = None, n_jobs: int = -1, verbose: int = 1
):
    """Process and aggregate raw features for multiple probe IDs (PIDs) across all their snippets.

    This function takes a DataFrame containing snippet information for multiple PIDs, processes each PID
    individually in parallel using `get_aggregated_features_per_pid`, and combines all results into a single DataFrame.
    The final result is indexed by ('pid', 'channel') and optionally saved to a Parquet file.

    Args:
        snippet_df (pandas.DataFrame): DataFrame containing snippet information for multiple PIDs. Must have columns:
            - 'pid': Probe ID
            - 'base_level_dir': Base directory path for each snippet
            - 'snippet_level_dir': Relative path to the snippet directory
        output_dir (Path or None, optional): If provided, the aggregated DataFrame is saved as 'raw_ephys_features.pqt' in this directory.
            The directory is created if it does not exist. Default is None (no file is written).
        n_jobs (int, optional): Number of parallel jobs to run. -1 means using all processors. Default is -1.
        verbose (int, optional): Verbosity level for joblib.Parallel. 0 means no messages, 1 means progress messages, >1 means more detailed messages. Default is 1.

    Returns:
        pandas.DataFrame: A DataFrame indexed by ('pid', 'channel') containing aggregated raw electrophysiological features
            and channel metadata for all PIDs. Each row represents one channel from one PID.

    Example:
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> snippet_df = pd.DataFrame({
        ...     'pid': ['probe1', 'probe1', 'probe2', 'probe2'],
        ...     'base_level_dir': ['/data/probe1', '/data/probe1', '/data/probe2', '/data/probe2'],
        ...     'snippet_level_dir': ['snippet_001', 'snippet_002', 'snippet_001', 'snippet_002']
        ... })
        >>> agg_df = get_aggregated_raw_features(snippet_df, output_dir=Path('output'))
        >>> print(agg_df.head())

    Note:
        - Each PID is processed independently using `get_aggregated_features_per_pid`.
        - The function groups the input DataFrame by 'pid' and processes each group in parallel using joblib.
        - If `output_dir` is provided, the result is saved as 'raw_ephys_features.pqt'.
        - The function handles multiple PIDs efficiently by aggregating the features for a PID before concatting across multiple pids.
    """
    # Collect all PID groups
    pid_groups = [pid_df for _, pid_df in snippet_df.groupby("pid")]
    
    # Process all PIDs in parallel using joblib
    agg_dfs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(get_aggregated_features_per_pid)(pid_df) for pid_df in pid_groups
    )
    
    # Concatenate all results into a single DataFrame
    agg_df = pd.concat(agg_dfs, ignore_index=True)
    # Set the multi-index to (pid, channel)
    agg_df = agg_df.set_index(["pid", "channel"])

    # Optionally save the aggregated features to a Parquet file
    if output_dir is not None:
        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save the DataFrame to a Parquet file
        agg_df.to_parquet(output_dir / "raw_ephys_features.pqt")

    return agg_df


def denoise_raw_features_data(
    agg_raw_ephys_features: pd.DataFrame, output_dir: Path | None = None, n_jobs: int = -1, verbose: int = 1
):
    """Apply denoising to aggregated raw electrophysiological features for each probe ID (PID).

    This function takes aggregated raw features and applies denoising algorithms to reduce noise
    and improve signal quality. The denoising is performed PID-by-PID in parallel using the `denoise_dataframe`
    function, which requires channel labels for each PID. The process preserves the original column
    structure while handling nan and noisy results.

    Args:
        agg_raw_ephys_features (pandas.DataFrame): DataFrame containing aggregated raw electrophysiological features, typically indexed by
            ('pid', 'channel'). Must contain a 'channel_labels' column for each PID.
        output_dir (Path or None, optional): If provided, the denoised DataFrame is saved as 'raw_ephys_features_denoised.pqt' in this directory.
            The directory is created if it does not exist. Default is None (no file is written).
        n_jobs (int, optional): Number of parallel jobs to run. -1 means using all processors. Default is -1.
        verbose (int, optional): Verbosity level for joblib.Parallel. 0 means no messages, 1 means progress messages, >1 means more detailed messages. Default is 1.

    Returns:
        pandas.DataFrame: A DataFrame with the same structure as the input but with denoised feature values.
            The denoising process reduces noise while preserving the original column structure.

    Example:
        >>> import pandas as pd
        >>> # Assuming agg_df is a DataFrame with aggregated raw features
        >>> denoised_df = denoise_raw_features_data(agg_df, output_dir=Path('output'))
        >>> print(denoised_df.head())

    Note:
        - The function processes each PID separately in parallel to apply PID-specific denoising.
        - Denoising requires channel labels, which must be present in the 'channel_labels' column.
        - The denoising factor (fac) is set to 1, which can be adjusted in the denoise_dataframe function.
        - If `output_dir` is provided, the result is saved as 'raw_ephys_features_denoised.pqt'.
    """
    # Store the original column names to preserve structure
    original_columns = agg_raw_ephys_features.columns.tolist()
    
    # Helper function to process a single PID group
    def denoise_pid(pid_df_tuple):
        pid, df_pid = pid_df_tuple
        logger.info(f"Denoising for PID: {pid}")
        df_denoised = denoise_dataframe(
            df_pid, fac=1, channel_labels=df_pid["channel_labels"].to_numpy()
        )
        # Keep only the original columns to maintain structure
        return df_denoised.loc[:, original_columns]
    
    # Collect all PID groups
    pid_groups = list(agg_raw_ephys_features.groupby("pid"))
    
    # Process all PIDs in parallel using joblib
    df_pids = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(denoise_pid)(pid_group) for pid_group in pid_groups
    )
    
    # Concatenate all denoised PID DataFrames
    df_features_denoise = pd.concat(df_pids)

    df_features_denoise = outlier_treatment(df_features_denoise, columns = ['alpha_mean','alpha_std'])

    df_features_denoise = replace_nan(df_features_denoise, columns = df_features_denoise.columns)

    # Optionally save the denoised features to a Parquet file
    if output_dir is not None:
        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save the denoised DataFrame to a Parquet file
        df_features_denoise.to_parquet(output_dir / "raw_ephys_features_denoised.pqt")

    return df_features_denoise


#  TODO: Information in input_dir, can be compiled in the snippet_df
def produce_output_dataframes(
    snippets_df: pd.DataFrame, input_dir: Path, output_dir: Path | None = None
):
    """Orchestrate the complete pipeline to produce aggregated and denoised electrophysiological data.

    This function serves as the main entry point for processing electrophysiological data from multiple
    probes. It coordinates the entire pipeline: aggregating channel metadata, processing raw features,
    and applying denoising. The function handles both the data processing and optional file output.

    Args:
        snippets_df (pandas.DataFrame): DataFrame containing snippet information for multiple PIDs. Must have columns:
            - 'pid': Probe ID
            - 'base_level_dir': Base directory path for each snippet
            - 'snippet_level_dir': Relative path to the snippet directory
        input_dir (Path): Root directory containing probe data. The function searches for 'channels.pqt' files
            in subdirectories of this path.
        output_dir (Path or None, optional): If provided, all output DataFrames are saved as Parquet files in this directory:
            - 'snippets_df.pqt': Input snippets DataFrame
            - 'channels.pqt': Aggregated channel metadata
            - 'raw_ephys_features.pqt': Aggregated raw features
            - 'raw_ephys_features_denoised.pqt': Denoised features
            The directory is created if it does not exist. Default is None (no files are written).

    Returns:
        tuple: A tuple containing three DataFrames:
            - df_channels: Aggregated channel metadata with position information
            - df_raw_ephys: Aggregated raw electrophysiological features
            - df_features_denoise: Denoised electrophysiological features

    Example:
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> snippets_df = pd.DataFrame({
        ...     'pid': ['probe1', 'probe2'],
        ...     'base_level_dir': ['/data/probe1', '/data/probe2'],
        ...     'snippet_level_dir': ['snippet_001', 'snippet_001']
        ... })
        >>> channels, raw_features, denoised_features = produce_output_dataframes(
        ...     snippets_df, Path('/data'), Path('output')
        ... )

    Note:
        - The function searches for 'channels.pqt' files recursively in the input directory.
        - All processing steps are performed sequentially: channel aggregation, feature aggregation, then denoising.
        - If output_dir is provided, all intermediate and final results are saved as Parquet files.
        - This function serves as a high-level wrapper around the individual processing functions.
    """
    # Ensure output_dir is a Path object and create it if needed
    output_dir = Path(output_dir)
    if output_dir is not None:
        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save the input snippets DataFrame for reference
        snippets_df.to_parquet(output_dir / "snippets_df.pqt")

    # Find all channels.pqt files in the input directory structure
    # TODO - Get the channels information only for the pids found in the snippets_df
    channels_pqt_files = list(Path(input_dir).glob("*/channels.pqt"))
    logger.info(f"channels_pqt_files = {channels_pqt_files}")
    # Aggregate channel metadata from all found files
    df_channels = concatenate_channels_data(channels_pqt_files, output_dir=output_dir)

    # Process and aggregate raw electrophysiological features
    df_raw_ephys = get_aggregated_raw_features(snippets_df, output_dir=output_dir)

    # Apply denoising to the aggregated raw features
    df_features_denoise = denoise_raw_features_data(df_raw_ephys, output_dir=output_dir)

    return df_channels, df_raw_ephys, df_features_denoise


# # %% Eventually upload to S3
# print(f'aws --profile ibl s3 sync "{path_features}" s3://ibl-brain-wide-map-private/aggregates/atlas')

# import ephys_atlas.data
# from one.api import ONE
# one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
# df_voltage, _, df_channels, df_probes = ephys_atlas.data.download_tables(local_path='/home/olivier/scratch', label='2024_W50', one=one)
