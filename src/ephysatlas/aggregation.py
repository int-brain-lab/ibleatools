from pathlib import Path
from typing import List
from ephysatlas.utils import get_aggregated_snippets_df
import pandas as pd


def aggregate_all_probes(path_list: List[Path]):
    """
    Aggregate all probes in the path list.
    """
    df = pd.DataFrame()
    for path in path_list:
        df = pd.concat([df, get_aggregated_snippets_df(path)])
    return df


def function_to_aggregate_raw_features(input_df: pd.DataFrame):
    """
    Aggregate raw features from the input dataframe.
    """
    pass


def function_to_denoise_raw_features_data():
    pass


def function_to_aggreagte_channels_data():
    pass
