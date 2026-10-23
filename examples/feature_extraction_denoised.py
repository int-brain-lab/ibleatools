"""Denoised features of one probe, aggregated over several snippets of the recording.

A single snippet gives noisy features. ``compute_denoised_features`` computes the features on
several snippets spread over the recording (by default 5 snippets of 5 s, the first and last 5
minutes excluded), takes their per-channel median, denoises it along the probe and remaps the
waveform features, in one call. It works with every feature calculator.

The example is a probe recorded with SpikeGLX. Run cell-by-cell (``# %%`` cells).
"""

# %% Imports and configuration
import logging
from pathlib import Path

from ephysatlas.feature_calculators import (
    SnippetWindow,
    SpikeGLXFileFeatureCalculator,
    compute_aggregated_features,
    get_snippet_start_times,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Point this at a session folder holding rec/<probe>/*.ap.cbin and *.lf.cbin
session_path = Path("/path/to/VF087/2026_07_15")
ap_file = next(session_path.joinpath("rec", "probe00").glob("*.ap.cbin"))
lf_file = ap_file.with_name(ap_file.name.replace(".ap.cbin", ".lf.cbin"))
# The pid names the outputs: AP file names repeat across sessions (myRun_g0_t0.imec0.ap)
pid = "VF087_2026_07_15_probe00"
calc = SpikeGLXFileFeatureCalculator(ap_file=ap_file, lf_file=lf_file, name=pid)

# %% 1. One call: 5 snippets of 5 s, computed in parallel (the waveform features need the GPU)
df = calc.compute_denoised_features(n_jobs=5)
# Indexed by (pid, channel): the alignment GUI reads it as the session yaml's features
df.to_parquet(f"features_{pid}.pqt")

# %% 2. The snippet windows and code versions are stored with the features (and in the file)
logger.info("Snippet windows: %s", df.attrs["snippet_windows"])
logger.info(
    "Features version %s, ibleatools commit %s",
    df.attrs["features_version"],
    df.attrs.get("ibleatools_git_commit_hash"),
)

# %% 3. Or choose the snippets: e.g. 10 evenly spaced snippets with 10 min margins
duration = min(calc.available_duration())
t_starts = get_snippet_start_times(
    duration, n_snippets=10, snippet_duration=5, margin=600, method="linspace"
)
windows = [SnippetWindow(t, duration_ap=5, duration_lf=5) for t in t_starts]
df = compute_aggregated_features(calc, windows, n_jobs=5)
