import hashlib
import os
from pathlib import Path
from typing import Any, Dict


def setup_output_directory(params: Dict[str, Any]) -> Path:
    """Set up the output directory structure and change to it."""
    
    # Create base output directory if specified, otherwise use current directory
    base_dir = Path(params.get("output_dir", "."))
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create probe level subdirectory (pid or hash of ap_file)
    if params["pid"] is not None:
        probe_level_dir = base_dir / params["pid"]
    else:
        # For file-based processing, create a hash of just the AP filename
        ap_file = Path(params["ap_file"]).name
        # Create a hash of the filename
        ap_file_hash = hashlib.md5(ap_file.encode()).hexdigest()[:12]
        probe_level_dir = base_dir / ap_file_hash
    
    probe_level_dir.mkdir(parents=True, exist_ok=True)

    # Create SNippet level subdirectory
    # Pad t_start and duration
    t_start_padded = f"{params['t_start']:08.1f}"  # 8 digits with 1 decimal place
    duration_padded = f"{params['duration']:04.1f}"  # 4 digits with 1 decimal place
    snippet_level_dir = probe_level_dir / f"probe_{params['pid']}_{t_start_padded}_{duration_padded}"
    snippet_level_dir.mkdir(parents=True, exist_ok=True)

    return probe_level_dir, snippet_level_dir