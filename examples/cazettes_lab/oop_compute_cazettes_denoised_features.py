"""Compute and denoise OOP features for the VF066 probe00 recording.

The script computes five 5-second snippets, aggregates the snippet-level
features by channel, and saves one denoised feature table. Edit the constants
below to use another local SpikeGLX recording.
"""

# %% Imports
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ephysatlas.aggregation import (
    denoise_raw_features_data,
    get_aggregated_features_per_pid,
)
from ephysatlas.feature_calculators import (
    FeatureComputationOptions,
    SnippetWindow,
    SpikeGLXFileFeatureCalculator,
)

LOGGER = logging.getLogger(__name__)


# %% Recording and output paths
DATA_ROOT = Path("/mnt/s0/Data/2026_cazettes/2026_cazettes/Data")
RECORDING_NAME = "VF066_2025_12_04_probe00"
AP_FILE = DATA_ROOT / "VF066/2025_12_04/Rec/probe00/disabled_g0_t0.imec0.ap.cbin"
LF_FILE = DATA_ROOT / "VF066/2025_12_04/Rec/probe00/disabled_g0_t0.imec0.lf.cbin"

EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLE_DIR / "output" / RECORDING_NAME
FEATURE_CACHE_DIR = OUTPUT_DIR / "feature_cache"
SCRATCH_DIR = OUTPUT_DIR / "scratch"


# %% Feature settings
SNIPPET_T_STARTS = (300.0, 600.0, 900.0, 1200.0, 1500.0)
DURATION_AP = 5.0
DURATION_LF = 5.0
FEATURES_TO_COMPUTE = ("lf", "csd", "ap", "waveforms")


# %% Compute, aggregate, and denoise
def main() -> pd.DataFrame:
    """Compute five snippets and save the denoised per-channel feature table."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for input_file in (AP_FILE, LF_FILE):
        if not input_file.exists():
            raise FileNotFoundError(input_file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calculator = SpikeGLXFileFeatureCalculator(
        ap_file=AP_FILE,
        lf_file=LF_FILE,
        name=RECORDING_NAME,
    )

    # Compute one feature directory for each requested snippet.
    manifest_records = []
    for t_start in SNIPPET_T_STARTS:
        LOGGER.info("Computing %s features at %.1f seconds", RECORDING_NAME, t_start)
        result = calculator.compute_snippet(
            SnippetWindow(
                t_start=t_start,
                duration_ap=DURATION_AP,
                duration_lf=DURATION_LF,
            ),
            FeatureComputationOptions(
                features_to_compute=FEATURES_TO_COMPUTE,
                output_dir=FEATURE_CACHE_DIR,
                scratch_dir=SCRATCH_DIR / f"{int(t_start)}s",
                include_trajectory=False,
            ),
        )
        manifest_records.append(
            {**dict(result.manifest_record), "filename": str(AP_FILE)}
        )

    # The manifest tells the aggregator where each snippet was written.
    df_manifest = pd.DataFrame.from_records(manifest_records)
    manifest_file = OUTPUT_DIR / f"{RECORDING_NAME}_snippet_manifest.pqt"
    df_manifest.to_parquet(manifest_file)

    df_aggregated = get_aggregated_features_per_pid(df_manifest)
    df_aggregated["pid"] = RECORDING_NAME
    df_denoised = denoise_raw_features_data(
        df_aggregated.set_index(["pid", "channel"]).sort_index(),
        n_jobs=1,
        verbose=0,
    ).reset_index()
    df_denoised = df_denoised.sort_values("channel").reset_index(drop=True)

    features_file = OUTPUT_DIR / f"{RECORDING_NAME}_denoised_features.pqt"
    df_denoised.to_parquet(features_file)
    LOGGER.info("Saved denoised features to %s", features_file)
    return df_denoised


if __name__ == "__main__":
    main()
