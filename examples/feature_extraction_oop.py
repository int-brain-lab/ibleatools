"""Feature extraction using the object-oriented calculators directly.

``compute_features_from_pid`` / ``compute_features_from_file`` are thin wrappers
over the calculators in ``ephysatlas.feature_calculators``. Using a calculator
directly returns the same features and also exposes the intermediate steps
(channel metadata, the destriped snippet) — handy for inspection and plotting.

Two sources are shown: an IBL probe insertion (streamed through ONE) and a local
SpikeGLX AP/LF file pair. Run cell-by-cell (``# %%`` cells).
"""

# %% Imports and configuration
import logging
import tempfile
from pathlib import Path

from one.api import ONE

from ephysatlas.feature_calculators import (
    CsdParams,
    FeatureComputationOptions,
    FeatureParams,
    IBLPIDFeatureCalculator,
    SnippetWindow,
    SpikeGLXFileFeatureCalculator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_dir = Path(tempfile.mkdtemp(prefix="ephysatlas_oop_"))
# One window is reused for every source; AP/LF durations are independent.
window = SnippetWindow(t_start=300.0, duration_ap=1.0, duration_lf=1.0)

# %% 1. IBL probe insertion (streamed through ONE)
one = ONE()
pid = "4cb60c5c-d15b-4abd-8cfd-776bc5a81dbe"

calc = IBLPIDFeatureCalculator(pid=pid, one=one)
options = FeatureComputationOptions(
    features_to_compute=["lf", "csd", "ap"],
    output_dir=output_dir,
    # per-feature options: typed objects or a nested dict, e.g. {"csd": {"scale": False}}
    feature_params=FeatureParams(csd=CsdParams(scale=False)),
)
result = calc.compute_snippet(window, options)
logger.info("IBL features: shape=%s", result.features.shape)

# Inspect the intermediate destriped snippet (raw + destriped AP/LF + geometry +
# channel labels) without recomputing features — useful for plotting.
snippet = calc.get_destriped_snippet(window)
logger.info(
    "Destriped AP=%s, LF=%s",
    None if snippet.des_ap is None else snippet.des_ap.shape,
    None if snippet.des_lf is None else snippet.des_lf.shape,
)

# %% 2. Local SpikeGLX AP/LF files
# Point these at a local .cbin (or .bin) pair. An optional trajectory dict
# (keys x, y, z, depth, theta, phi) adds target coordinates to the channels.
ap_file = "/path/to/probe.ap.cbin"
lf_file = "/path/to/probe.lf.cbin"
traj_dict = None

file_calc = SpikeGLXFileFeatureCalculator(
    ap_file=ap_file, lf_file=lf_file, traj_dict=traj_dict
)
file_options = FeatureComputationOptions(
    features_to_compute=["lf", "csd", "ap"],
    output_dir=output_dir,
    include_trajectory=traj_dict is not None,
)
file_result = file_calc.compute_snippet(window, file_options)
logger.info("File features: shape=%s", file_result.features.shape)
