import hashlib
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import os
import logging

# Set up logger
logger = logging.getLogger(__name__)

def setup_output_directory(params: Dict[str, Any]) -> Path:
    """Set up the output directory structure and change to it."""

    # Create base output directory if specified, otherwise use current directory
    base_dir = Path(params.get("output_dir", "."))
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
        parquet_files = list(snippet_dir.glob("*.parquet")) + list(snippet_dir.glob("*.pqt"))

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


                    

def add_metadata_to_parquet_files(directory: Path, **snippet_attrs: Dict[str, Any]):
    """
    Add metadata attributes to all .parquet and .pqt files in subdirectories of the given directory.
    
    Args:
        directory (Path): Base directory to search for subdirectories
        **snippet_attrs: Additional key-value pairs to add as metadata attributes
    
    Returns:
        None
    """
    if not directory.exists() or not directory.is_dir():
        logger.warning(f"Directory {directory} does not exist or is not a directory")

    
    # Iterate through all subdirectories (not the directory itself)
    for subdir in directory.iterdir():
        if not subdir.is_dir():
            continue
            
        # Look for both .parquet and .pqt files in the subdirectory
        for file_path in (list(subdir.glob("*.parquet")) + list(subdir.glob("*.pqt"))):
            _update_parquet_metadata(file_path, **snippet_attrs)


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