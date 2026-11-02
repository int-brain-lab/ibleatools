"""Pack recordings from different subjects and dates into one HDF5 file.

Edit the inputs and run by # %% cells. Rerunning skips recordings already present in
an intact output file. See the how-to for recovery after an interruption during append.
"""

# %% Imports and configuration
import logging
from pathlib import Path

import h5py

from ephysatlas.snippet_extractor import extract_snippets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Each entry has a stable, unique name and its own AP file with adjacent metadata/LF.
# The AP filenames can repeat across subjects; the explicit recording names distinguish them.
recordings = [
    (
        "subjectA_2026-01-15_probe00",
        Path("/data/subjectA/2026-01-15/probe00/run_g0_t0.imec0.ap.cbin"),
    ),
    (
        "subjectB_2026-02-20_probe00",
        Path("/data/subjectB/2026-02-20/probe00/run_g0_t0.imec0.ap.cbin"),
    ),
]
# Keep the same output path, names and window settings when resuming a batch.
output_file = Path("snippet_exports/collection.h5")
t_starts = [600.0, 1200.0]
duration = 1.0

# %% Extract missing recordings; completed recordings are preserved on reruns
# This assumes an intact output. A name alone cannot prove a group survived an
# interruption during the final HDF5 copy; see the how-to before resuming that case.
for recording_name, ap_file in recordings:
    if output_file.exists():
        with h5py.File(output_file, "r") as file:
            if recording_name in file["recordings"]:
                stored = file[f"recordings/{recording_name}"]
                if stored.attrs["source_ap_file"] != str(ap_file.resolve()):
                    raise ValueError(
                        f"Recording name {recording_name!r} belongs to another source; "
                        "choose a unique name for this recording."
                    )
                logger.info("Skipping stored recording %s", recording_name)
                continue
    extract_snippets(
        ap_file,
        output_file,
        name=recording_name,
        t_starts=t_starts,
        duration=duration,
    )

# %% Inspect the collection, including recordings from earlier runs
logger.info("wrote %s (%.1f MB)", output_file, output_file.stat().st_size / 1e6)
with h5py.File(output_file) as file:
    logger.info("recordings in the file: %s", list(file["recordings"]))
