# TODO -  Remove the requirement for .pqt and .pqt separately. Or handle it by defining a repo wide VARIABLE.

import hashlib
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import logging

# Set up logger
logger = logging.getLogger(__name__)


def setup_output_directory(params: Dict[str, Any]) -> Path:
    """Set up the output directory structure and change to it.

    The function creates a hierarchical directory structure:
    - Probe level subdirectory (using pid or hash of ap_file)
    - Snippet level subdirectory (using probe info, t_start, and duration)

    Example output structure:
    |-- 76ed566f-59dd-47ff-8ba7-59b11d09b67c
    |   |-- probe_76ed566f-59dd-47ff-8ba7-59b11d09b67c_000300.0_05.0
    |   `-- probe_76ed566f-59dd-47ff-8ba7-59b11d09b67c_003000.0_05.0
    `-- af0a0534-9cdc-4a29-93c0-1342891d74ec
        |-- probe_af0a0534-9cdc-4a29-93c0-1342891d74ec_000300.0_05.0
        `-- probe_af0a0534-9cdc-4a29-93c0-1342891d74ec_003000.0_05.0
    """

    if params.get("output_dir") is None:
        return None, None

    # Create base output directory if specified, otherwise use current directory
    base_dir = Path(params.get("output_dir"))
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create probe level subdirectory (pid or hash of ap_file)
    if params["pid"] is not None:
        probe_level_dir = base_dir / params["pid"]
    elif params["ap_file"] is not None:
        # For file-based processing, create a hash of just the AP filename
        ap_file = Path(params["ap_file"]).name
        # Create a hash of the filename
        ap_file_hash = hashlib.md5(ap_file.encode()).hexdigest()[:12]
        probe_level_dir = base_dir / ap_file_hash
    else:
        raise ValueError("Either pid or ap_file must be provided")

    probe_level_dir.mkdir(parents=True, exist_ok=True)

    # Create SNippet level subdirectory
    # Pad t_start and duration
    t_start_padded = f"{params['t_start']:08.1f}"  # 8 digits with 1 decimal place
    duration_padded = f"{params['duration']:04.1f}"  # 4 digits with 1 decimal place
    snippet_level_dir = (
        probe_level_dir / f"probe_{params['pid']}_{t_start_padded}_{duration_padded}"
    )
    snippet_level_dir.mkdir(parents=True, exist_ok=True)

    return probe_level_dir, snippet_level_dir


def get_aggregated_snippets_df(probe_level_dir: Path):
    """
    Get a dataframe of metadata info for all snippets in the probe level directory.
    """

    data = []

    def get_metadata_for_snippet(snippet_dir: Path):
        """
        Get the metadata for a snippet from a parquet file.
        """
        parquet_files = list(snippet_dir.glob("*.parquet")) + list(
            snippet_dir.glob("*.pqt")
        )

        result = {}
        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            result.update(df.attrs)
        return result

    for subdir in probe_level_dir.iterdir():
        if subdir.is_dir():
            data.append(get_metadata_for_snippet(subdir))

    df = pd.DataFrame(data)
    return df


def add_metadata_to_parquet_files(**snippet_attrs: Dict[str, Any]):
    """
    Add metadata attributes to all .parquet and .pqt files in subdirectories of the given directory.

    Args:
        **snippet_attrs: Additional key-value pairs to add as metadata attributes

    Returns:
        None
    """
    snippet_level_dir = Path(snippet_attrs["base_level_dir"]) / Path(
        snippet_attrs["snippet_level_dir"]
    )
    if not snippet_level_dir.exists() or not snippet_level_dir.is_dir():
        logger.warning(
            f"Directory {snippet_level_dir} does not exist or is not a directory"
        )

    # Look for both .parquet and .pqt files in the subdirectory
    for file_path in list(snippet_level_dir.glob("*.parquet")) + list(
        snippet_level_dir.glob("*.pqt")
    ):
        _update_parquet_metadata(file_path, **snippet_attrs)

    logger.info(f"Updated metadata for {snippet_level_dir}")


def _update_parquet_metadata(file_path: Path, **snippet_attrs: Dict[str, Any]):
    """
    Helper function to update metadata for a single parquet file.

    Args:
        file_path (Path): Path to the parquet file
        **snippet_attrs: Additional key-value pairs to add as metadata attributes

    Returns:
        None
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)

        # Add the required metadata attributes
        for key, value in snippet_attrs.items():
            df.attrs[key] = value

        # Write the file back with updated metadata
        df.to_parquet(file_path)
        logger.debug(f"Updated metadata for {file_path}")

    except Exception as e:
        logger.warning(f"Failed to update metadata for {file_path}: {str(e)}")
