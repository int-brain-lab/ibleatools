"""Extract raw snippets of one probe into an HDF5 file and compute features from that file.

This is what a partner lab does (``extract_snippets``, or the ``ibleatools-snippets`` command)
followed by what we do with the file they send (``SnippetFileFeatureCalculator``). The example
uses two 1 s snippets to stay fast; the default is 5 snippets of 5 s. Run cell-by-cell
(``# %%`` cells).
"""

# %% Imports and configuration
import logging
import tempfile
from pathlib import Path

from ephysatlas.feature_calculators import FeatureComputationOptions
from ephysatlas.feature_calculators.snippet_file import SnippetFileFeatureCalculator
from ephysatlas.snippet_extractor import extract_snippets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The .ap.cbin of one probe; its .meta and .lf.cbin are found next to it
ap_file = Path(
    "/mnt/s0/Data/2026_spike_sorting_benchmarks/recordings/1a276285-8b0e-4cc9-9f0a-a3a002978724"
    "/_spikeglx_ephysData_g0_t0.imec0.ap.cbin"
)
# The output goes to a temporary folder, never next to the recording
output_file = Path(tempfile.mkdtemp(prefix="snippets_")) / "probe00.h5"

# %% Extract the snippets (partner side)
# Pass histology=[...] with the probe's channel_locations.json etc., and trajectory={...}
# with the planned insertion, to store them in the file as well
extract_snippets(ap_file, output_file, t_starts=[600.0, 1200.0], duration=1.0)
logger.info("wrote %s (%.1f MB)", output_file, output_file.stat().st_size / 1e6)

# %% Compute features from the file (our side)
calc = SnippetFileFeatureCalculator(output_file, name="example_probe00")
# Without a GPU, skip the waveforms family
options = FeatureComputationOptions(features_to_compute=["lf", "csd", "ap"])
df = calc.compute_denoised_features(options=options)
logger.info("features table: %d rows x %d columns", *df.shape)
logger.info("index: %s", df.index.names)
