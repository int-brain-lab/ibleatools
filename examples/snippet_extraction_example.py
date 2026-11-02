"""Extract two short raw snippets per probe into one HDF5 file. Run by # %% cells."""

# %% Imports and configuration
import logging
import tempfile
from pathlib import Path

import h5py

from ephysatlas.snippet_extractor import extract_snippets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# One session with two probes; each AP file has its metadata beside it.
session = Path("/mnt/s0/Data/2026_cazettes/2026_cazettes/Data/VF066/2025_12_04")
# The output goes to a temporary folder, never next to the recording
output_file = Path(tempfile.mkdtemp(prefix="snippets_")) / "vf066.h5"

# %% Extract the snippets of both probes into one file (partner side)
# The first call creates the file; the second adds a recording named after its AP file.
for probe in ("probe00", "probe01"):
    extract_snippets(
        session / f"Rec/{probe}/disabled_g0_t0.imec{probe[-1]}.ap.cbin",
        output_file,
        t_starts=[600.0, 1200.0],
        duration=1.0,
    )
logger.info("wrote %s (%.1f MB)", output_file, output_file.stat().st_size / 1e6)
with h5py.File(output_file) as file:
    logger.info("recordings in the file: %s", list(file["recordings"]))
