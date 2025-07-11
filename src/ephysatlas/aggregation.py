from pathlib import Path
from typing import List
import logging
import pandas as pd
from functools import reduce
from ephysatlas.utils import get_aggregated_snippets_df
from ephysatlas.features import ChannelDataFrameSchema
from ephysatlas.features import denoise_dataframe
import tqdm

# Set up logger
logger = logging.getLogger(__name__)


def aggregate_all_probes(path_list: List[Path]):
    """
    Aggregate all probes in the path list.
    """
    df = pd.DataFrame()
    for path in path_list:
        df = pd.concat([df, get_aggregated_snippets_df(path)])
    return df


# Function to aggregate channels from ap_features.parquet file and add it to the channels.pqt file.


def aggregate_channel_labels(probe_level_dir: Path) -> pd.Series | None:
    """
    Read all ap_features.parquet files, extract channel_labels column,
    and get the mode across the column axis.
    """
    ap_files = list(probe_level_dir.glob("ap_features.parquet"))

    if not ap_files:
        print(f"No ap_features.parquet files found in {probe_level_dir}")
        return None

    # Read all ap_files and extract channel_labels column
    channel_labels_dfs = []
    for ap_file in ap_files:
        df = pd.read_parquet(ap_file)
        if "channel_labels" in df.columns:
            channel_labels_dfs.append(df["channel_labels"])
        else:
            print(f"Warning: 'channel_labels' column not found in {ap_file}")

    if not channel_labels_dfs:
        print("No channel_labels columns found in any ap_files")
        return None

    # Combine all channel_labels columns into a single dataframe
    combined_channel_labels = pd.concat(channel_labels_dfs, axis=1)

    # Get the mode across the column axis (axis=1)
    mode_result = combined_channel_labels.mode(axis=1)

    # In case of multiple modes, take the first one
    if mode_result.shape[1] > 1:
        mode_result = mode_result.iloc[:, 0]

    return mode_result


def update_channel_pqt_with_channel_labels(probe_level_dir: Path):
    """
    Update the channels.pqt file with the channel labels.
    """
    channel_labels = aggregate_channel_labels(probe_level_dir)
    if channel_labels is not None:
        # Read the channels.pqt file
        channels_df = pd.read_parquet(probe_level_dir / "channels.pqt")
        # Update the channel_labels column
        channels_df["channel_labels"] = channel_labels
        # Write the updated channels.pqt file
        channels_df.to_parquet(probe_level_dir / "channels.pqt")
    else:
        logger.warning(
            f"No channel labels found in {probe_level_dir}. "
            "Channel labels will not be updated."
        )


# Function to aggregate channels dataframe
# Make sure that the channels.pqt file is updated with the channel labels.
def aggregate_channels_data(
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
    df_channels = pd.concat(df_channels)
    df_channels = df_channels.groupby(["pid", "channel"]).first()
    if output_dir is not None:
        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        df_channels.to_parquet(output_dir / "channels.parquet")
    return df_channels


def get_features_from_snippets(snippet_level_dir: Path):
    """
    Get features from the snippets.
    """
    feature_files = list(snippet_level_dir.glob("*.parquet"))
    df = {}
    for file in feature_files:
        df[file.stem] = pd.read_parquet(file)

    df_voltage = reduce(
        lambda left, right: pd.merge(left, right, on="channel", how="outer"),
        [df[k] for k in df.keys()],
    )

    return df_voltage


def aggregate_raw_features(input_df: pd.DataFrame, output_dir: Path | None = None):
    """
    Aggregate raw features from the input dataframe.
    """
    # Get the concatenate version here and then reduce them using some aggregation method.
    # Use dask if it is needed.
    df_final = pd.DataFrame()
    for snippet_dir in input_df["snippet_level_dir"]:
        df_voltage = get_features_from_snippets(snippet_dir)
        df_final = pd.concat([df_final, df_voltage])

    if output_dir is not None:
        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(output_dir / "raw_ephys_features.parquet")

    return df_final


def denoise_raw_features_data(aggregation_path: Path):
    # Reimplement this
    aggregation_path = Path(aggregation_path)
    df_features = pd.read_parquet(aggregation_path / "raw_ephys_features.parquet")

    # # %% Denoise the features
    #
    original_columns = df_features.columns.tolist()
    df_features_merged = df_features.merge(
        pd.read_parquet(aggregation_path / "channels.parquet"),
        how="inner",
        right_index=True,
        left_index=True,
    )
    # df_features_merged = df_features_merged.merge(pd.read_parquet(path_features / 'channels_labels.pqt').fillna(0), how='inner', right_index=True, left_index=True)

    df_pids = []
    for pid, df_pid in tqdm.tqdm(df_features_merged.groupby("pid")):
        df_denoised = denoise_dataframe(df_pid, fac=1)
        df_pids.append(df_denoised.loc[:, original_columns])
    df_features_denoise = pd.concat(df_pids)

    df_features_denoise.to_parquet(aggregation_path / "raw_ephys_features_denoised.pqt")

    return df_features_denoise

    # # %% Eventually upload to S3
    # print(f'aws --profile ibl s3 sync "{path_features}" s3://ibl-brain-wide-map-private/aggregates/atlas')

    # import ephys_atlas.data
    # from one.api import ONE
    # one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    # df_voltage, _, df_channels, df_probes = ephys_atlas.data.download_tables(local_path='/home/olivier/scratch', label='2024_W50', one=one)
