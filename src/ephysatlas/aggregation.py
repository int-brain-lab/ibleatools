from pathlib import Path
from typing import List
import logging
import pandas as pd
from functools import reduce
from ephysatlas.utils import get_aggregated_snippets_df
from ephysatlas.features import ChannelDataFrameSchema, ModelRawFeatures
from ephysatlas.features import denoise_dataframe
import tqdm
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)

#TODO - Test at the end - ephysatlas.data.read_features_from_disk(path_features, brain_atlas=brain_atlas, strict=True)


# TODO - There can be a better way to specify both of these arguments - maybe only one is needed.
def aggregate_all_probes(path_list: List[Path], base_level_dir: Path | None | str = None):
    """
    Aggregates data from multiple probe directories into a single DataFrame.

    For each path in the provided list, this function calls `get_aggregated_snippets_df(path)`
    to retrieve a DataFrame of aggregated snippet data, and concatenates the results into
    a single DataFrame.

    Parameters
    ----------
    path_list : List[pathlib.Path]
        A list of Path objects, each pointing to a probe directory containing data
        to be aggregated.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the concatenated results from all probes in the path list.

    Notes
    -----
    - Each path in `path_list` should be compatible with `get_aggregated_snippets_df`.
    - The function ignores index continuity and resets the index in the returned DataFrame.
    """
    df = pd.DataFrame()
    for path in path_list:
        df = pd.concat([df, get_aggregated_snippets_df(path)], ignore_index=True)

    if base_level_dir is not None:
        df["base_level_dir"] = Path(base_level_dir).as_posix()

    return df




# Function to aggregate channels dataframe
# Make sure that the channels.pqt file is updated with the channel labels.
def concatenate_channels_data(
    parquet_files_channels: List[Path], output_dir: Path | None = None
):
    """
    Aggregate channels data from the path list.
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
    """
    Get features from the snippets.
    """
    snippet_level_dir = Path(snippet_level_dir)
    feature_files = list(snippet_level_dir.glob("*.pqt"))
    df = {}
    for file in feature_files:
        df[file.stem] = pd.read_parquet(file)

    df_voltage = reduce(
        lambda left, right: pd.merge(left, right, on="channel", how="outer"),
        [df[k] for k in df.keys()],
    )


    return df_voltage

def concat_raw_features(input_df: pd.DataFrame):
    """
    Aggregate raw features from the input dataframe.
    """
    df_final = pd.DataFrame()
    for _, row  in input_df.iterrows():
        snippet_dir = Path(row["base_level_dir"]) / row["snippet_level_dir"]
        df_voltage = get_features_from_snippets(snippet_dir)
        df_final = pd.concat([df_final, df_voltage], ignore_index=True)

    return df_final


def aggregate_raw_features(concatenated_df: pd.DataFrame):
    """
    Aggregate raw features from the concatenated dataframe.
    This function is being used to aggregate features for one pid for efficienct purpose
    But this should work even if there are multiple pids in the dataframe.
    """
    # Group by pid and channel
    raw_features_columns = ModelRawFeatures.to_schema().columns.keys()

    #Columns in the dataframe
    columns_in_df = concatenated_df.columns.tolist()

    # Take intersection of the columns in the dataframe and the raw features columns
    agg_columns = set(columns_in_df) & set(raw_features_columns)

    # Define aggregation functions for different column types
    agg_func_dict = {
        "spike_count": lambda x: np.mean(x.fillna(0)),
        "channel_labels": lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    }
    
    # Get aggregation function for a column, defaulting to np.nanmedian
    get_agg_func = lambda k: agg_func_dict.get(k, lambda x: np.nanmedian(x))
    
    # Create the aggregation dictionary
    dagg = {k: pd.NamedAgg(column=k, aggfunc=get_agg_func(k)) for k in agg_columns}
    aggregated_df = concatenated_df.groupby(["pid","channel"]).agg(**dagg)


    return aggregated_df


def get_aggregated_features_per_pid(snippet_df_per_pid: pd.DataFrame):
    """"
    Should return a dataframe with one pid, and number of rows equal to the number of channels.
    The output is not multi-indexed at this stage
    """
    assert snippet_df_per_pid["pid"].nunique() == 1, "There should be only one pid in the dataframe"

    # Get the concatenated version of all the raw features for each snippet
    df_concat = concat_raw_features(snippet_df_per_pid)

    # Add the pid to the dataframe
    df_concat["pid"] = snippet_df_per_pid["pid"].iloc[0]

    # Aggregate the raw features based
    agg_df_per_pid = aggregate_raw_features(df_concat)

    agg_df_per_pid = agg_df_per_pid.reset_index()

    #Add axial_um and lateral_um information to the dataframe
    #Assert that the parent of the snippet level dir has a channels.pqt file
    snippet_level_dir = Path(snippet_df_per_pid["base_level_dir"].iloc[0]) / Path(snippet_df_per_pid["snippet_level_dir"].iloc[0])
    if not (snippet_level_dir.parent / "channels.pqt").exists():
        raise FileNotFoundError(
            f"channels.pqt file not found in {snippet_level_dir.parent}"
        )
    df_channels = pd.read_parquet(snippet_level_dir.parent / "channels.pqt")
    #Merge the 'axial_um', 'lateral_um' from df_channels with df_voltage on the channel column
    agg_df_per_pid = agg_df_per_pid.merge(df_channels[["channel","axial_um", "lateral_um"]], on="channel", how="left")

    return agg_df_per_pid


def get_aggregated_raw_features(snippet_df: pd.DataFrame, output_dir: Path | None = None):
    
    agg_df = pd.DataFrame()
    for idx, pid_df in snippet_df.groupby("pid"):
        agg_df_per_pid = get_aggregated_features_per_pid(pid_df)
        agg_df = pd.concat([agg_df, agg_df_per_pid], ignore_index=True)
    # Set the multi-index to pid and channel
    agg_df = agg_df.set_index(["pid", "channel"])

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        agg_df.to_parquet(output_dir / "raw_ephys_features.pqt")

    return agg_df



def denoise_raw_features_data(agg_raw_ephys_features: pd.DataFrame, output_dir: Path | None = None):
    # At this both channels and raw_ephys_features should be calculated.
    # # %% Denoise the features
    #
    original_columns = agg_raw_ephys_features.columns.tolist()
    df_pids = []
    for pid, df_pid in tqdm.tqdm(agg_raw_ephys_features.groupby("pid")):
        df_denoised = denoise_dataframe(df_pid, fac=1, channel_labels=df_pid["channel_labels"].to_numpy())
        df_pids.append(df_denoised.loc[:, original_columns])
    df_features_denoise = pd.concat(df_pids)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df_features_denoise.to_parquet(output_dir / "raw_ephys_features_denoised.pqt")

    return df_features_denoise

#  TODO: Information in input_dir, can be compiled in the snippet_df
# I should add a probe_level_dir in snippets_df as well. to get the channel labels.
def produce_output_dataframes(snippets_df: pd.DataFrame, input_dir: Path, output_dir: Path | None = None):
    output_dir = Path(output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        snippets_df.to_parquet(output_dir / "snippets_df.pqt")

    channels_pqt_files = list(Path(input_dir).glob("*/channels.pqt"))
    logger.info(f"channels_pqt_files = {channels_pqt_files}")
    df_channels = concatenate_channels_data(channels_pqt_files, output_dir=output_dir)

    # Get the raw ephys features
    df_raw_ephys = get_aggregated_raw_features(snippets_df, output_dir=output_dir)

    # Denoise the raw ephys features
    df_features_denoise = denoise_raw_features_data(df_raw_ephys, output_dir=output_dir)

    return df_channels, df_raw_ephys, df_features_denoise
    
    



# Write a function that does all the three things together and outputs channel , raw ephys and raw_ephys denoised.


    # # %% Eventually upload to S3
    # print(f'aws --profile ibl s3 sync "{path_features}" s3://ibl-brain-wide-map-private/aggregates/atlas')

    # import ephys_atlas.data
    # from one.api import ONE
    # one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    # df_voltage, _, df_channels, df_probes = ephys_atlas.data.download_tables(local_path='/home/olivier/scratch', label='2024_W50', one=one)
