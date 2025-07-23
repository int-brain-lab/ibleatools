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
    Add metadata attributes to all Parquet files in a snippet-level directory.

    This function takes snippet attributes and adds them as metadata to all .parquet and .pqt files
    found in the specified snippet directory. The metadata is useful for tracking provenance,
    parameters, and other contextual information.

    Parameters
    ----------
    **snippet_attrs : Dict[str, Any]
        Keyword arguments containing metadata to add to the Parquet files. Must include:
        - 'base_level_dir': Base directory path
        - 'snippet_level_dir': Snippet directory name
        Additional key-value pairs will be added as metadata attributes.

    Returns
    -------
    None
        The function modifies files in place and does not return any values.

    Example
    -------
    >>> add_metadata_to_parquet_files(
    ...     base_level_dir='/data/probe1',
    ...     snippet_level_dir='snippet_001',
    ...     pid='probe1',
    ...     t_start=100.5,
    ...     duration=30.0
    ... )

    Notes
    -----
    - The function constructs the full snippet directory path from base_level_dir and snippet_level_dir.
    - Both .parquet and .pqt file extensions are supported.
    - If the directory doesn't exist, a warning is logged but no error is raised.
    - Each file is processed individually using _update_parquet_metadata.
    - The function logs the number of files processed for debugging.
    """
    # Construct the full path to the snippet directory
    snippet_level_dir = Path(snippet_attrs["base_level_dir"]) / Path(
        snippet_attrs["snippet_level_dir"]
    )
    # Check if the directory exists and is actually a directory
    if not snippet_level_dir.exists() or not snippet_level_dir.is_dir():
        logger.warning(
            f"Directory {snippet_level_dir} does not exist or is not a directory"
        )

    # Find all Parquet files (both .parquet and .pqt extensions) in the snippet directory
    for file_path in list(snippet_level_dir.glob("*.parquet")) + list(
        snippet_level_dir.glob("*.pqt")
    ):
        # Update metadata for each individual file
        _update_parquet_metadata(file_path, **snippet_attrs)

    # Log completion of metadata update for the entire directory
    logger.info(f"Updated metadata for {snippet_level_dir}")


def _update_parquet_metadata(file_path: Path, **snippet_attrs: Dict[str, Any]):
    """
    Update metadata attributes for a single Parquet file.

    This helper function reads a Parquet file, adds the provided metadata attributes to the
    DataFrame's attrs dictionary, and writes the file back to disk.

    Parameters
    ----------
    file_path : Path
        Path to the Parquet file to be updated.
    **snippet_attrs : Dict[str, Any]
        Keyword arguments containing metadata attributes to add to the file.
        These will be stored in the DataFrame's attrs dictionary.

    Returns
    -------
    None
        The function modifies the file in place and does not return any values.

    Example
    -------
    >>> _update_parquet_metadata(
    ...     Path('data.pqt'),
    ...     pid='probe1',
    ...     t_start=100.5,
    ...     duration=30.0
    ... )

    Notes
    -----
    - The function reads the entire Parquet file into memory, modifies it, and writes it back.
    - All provided snippet_attrs are added to the DataFrame's attrs dictionary.
    - If an error occurs during processing, it is logged as a warning but doesn't stop execution.
    - The function uses debug-level logging for successful updates and warning-level for errors.
    - This function is designed to be called by add_metadata_to_parquet_files for batch processing.
    """
    try:
        # Read the Parquet file into a DataFrame
        df = pd.read_parquet(file_path)

        # Add each metadata attribute to the DataFrame's attrs dictionary
        for key, value in snippet_attrs.items():
            df.attrs[key] = value

        # Write the DataFrame back to the same file with updated metadata
        df.to_parquet(file_path)
        # Log successful metadata update at debug level
        logger.debug(f"Updated metadata for {file_path}")

    except Exception as e:
        # Log any errors that occur during the update process
        logger.warning(f"Failed to update metadata for {file_path}: {str(e)}")
