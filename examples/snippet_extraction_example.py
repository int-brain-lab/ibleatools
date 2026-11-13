"""Pack recordings from different subjects and dates, then compute features from the file.

Edit the inputs and run by # %% cells. Rerunning skips recordings already present in
an intact output file. See the how-to for recovery after an interruption during append.
Each recording stores AP and LF; NP2 LF is derived with NP2Converter while AP stays raw.
The extraction cells run at the partner lab; the final cell computes features at the IBL.
This example uses two 1 s snippets; the extraction default is five 5 s snippets.
"""

# %% Imports and configuration
import logging
from pathlib import Path

import h5py

from ephysatlas.feature_calculators import (
    FeatureComputationOptions,
    SnippetFileFeatureCalculator,
    list_recordings,
)
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
# Optional inputs belong to individual recordings. Omit an entry when unavailable.
histology = {
    "subjectA_2026-01-15_probe00": {
        "channel_locations": Path(
            "/data/subjectA/2026-01-15/alf/probe00/channel_locations.json"
        ),
        "xyz_picks": Path("/data/subjectA/2026-01-15/alf/probe00/xyz_picks.json"),
    },
}
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
    # Add histology_raw=[...] or trajectory={...} for this recording when needed.
    extract_snippets(
        ap_file,
        output_file,
        name=recording_name,
        t_starts=t_starts,
        duration=duration,
        **histology.get(recording_name, {}),
    )

# %% Inspect the collection, including recordings from earlier runs
logger.info("wrote %s (%.1f MB)", output_file, output_file.stat().st_size / 1e6)
logger.info("recordings in the file: %s", list_recordings(output_file))

# %% Compute features of one recording from the file (our side)
calc = SnippetFileFeatureCalculator(output_file, recording=recordings[0][0])
# Without a GPU, skip the waveforms family
options = FeatureComputationOptions(features_to_compute=["lf", "csd", "ap"])
df = calc.compute_denoised_features(options=options)
logger.info("features table: %d rows x %d columns", *df.shape)
logger.info("index: %s", df.index.names)

# %% Attach the histology stored in the file
# Run this cell when the selected recording has a channel table.
# The channel table is matched to the features on the index level "channel"
channels = calc.load_channel_metadata()[["channel", "x", "y", "z", "acronym"]]
df = df.join(channels.set_index("channel"), on="channel")
logger.info("channels per region:\n%s", df["acronym"].value_counts().to_string())
