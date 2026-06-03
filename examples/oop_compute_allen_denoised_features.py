"""Compute denoised Allen features with the OOP Zarr feature calculator.

This example mirrors the feature-generation part of
``allen_sample_data_check/analysis/run_allen_aggregated_inference.py``. It
computes five 5-second snippets with :class:`NWBZarrFeatureCalculator`,
aggregates the snippet-level feature files, denoises the aggregated features,
and writes the same final per-probe parquet artifact used by the legacy
inference script.

The script is split into ``# %%`` cells so it can be run interactively.
"""

# %% Imports
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ephysatlas.aggregation import (
    denoise_raw_features_data,
    get_aggregated_features_per_pid,
)
from ephysatlas.feature_calculators import (
    FeatureComputationOptions,
    NWBZarrFeatureCalculator,
    SnippetWindow,
)

LOGGER = logging.getLogger(__name__)


# %% Module-level constants
def _examples_dir() -> Path:
    """Return the examples directory when run as a script or from a cell."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


DEFAULT_INPUT_PATH = Path("/mnt/s0/Data/2026_allen_data/ecephys/ecephys_compressed")
OUTPUT_DIR = _examples_dir() / "output" / "allen_oop_denoised_features"
DEFAULT_FEATURES_TO_COMPUTE: tuple[str, ...] = ("lf", "csd", "ap", "waveforms")
SNIPPET_T_STARTS: tuple[float, ...] = (300.0, 600.0, 900.0, 1200.0, 1500.0)
DURATION_AP = 5.0
DURATION_LF = 5.0


# %% Probe configuration
@dataclass(frozen=True)
class ProbeConfig:
    """Identifying name and AP/LFP Zarr paths for one Allen probe."""

    name: str
    ap_zarr: Path
    lf_zarr: Path


def _zarr(record_node: int, probe_letter: str, band: str) -> Path:
    """Build the path to one Allen ecephys compressed Zarr folder."""
    suffix = "AP" if band == "ap" else "LFP"
    folder = (
        f"experiment1_Record Node {record_node}#Neuropix-PXI-100."
        f"Probe{probe_letter}-{suffix}.zarr"
    )
    return DEFAULT_INPUT_PATH / folder


def _probe_config(record_node: int, probe_letter: str) -> ProbeConfig:
    """Create a probe config for one Allen record node/probe pair."""
    return ProbeConfig(
        name=f"Node{record_node}_Probe{probe_letter}",
        ap_zarr=_zarr(record_node, probe_letter, "ap"),
        lf_zarr=_zarr(record_node, probe_letter, "lf"),
    )


DATASETS: list[ProbeConfig] = [
    _probe_config(108, "A"),
    _probe_config(108, "B"),
    _probe_config(108, "C"),
    _probe_config(104, "D"),
    _probe_config(104, "E"),
    _probe_config(104, "F"),
]


# %% Small helpers
def _seconds_slug(seconds: float) -> str:
    """Return a compact filename-safe seconds label, for example ``300s``."""
    seconds = float(seconds)
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{str(seconds).replace('.', 'p')}s"


# %% OOP feature computation and denoising
def _compute_snippet_manifest(
    calculator: NWBZarrFeatureCalculator,
    config: ProbeConfig,
    t_start: float,
    duration_ap: float,
    duration_lf: float,
    features_to_compute: Sequence[str],
    probe_output_dir: Path,
    skip_saved_computation: bool,
) -> dict:
    """Compute one snippet with the OOP Zarr calculator and return a manifest row."""
    feature_cache_dir = probe_output_dir / "oop_feature_cache"
    scratch_dir = probe_output_dir / "oop_scratch" / _seconds_slug(t_start)

    LOGGER.info("Computing OOP Allen Zarr features for %s at %.1fs", config.name, t_start)
    result = calculator.compute_snippet(
        SnippetWindow(t_start=t_start, duration_ap=duration_ap, duration_lf=duration_lf),
        FeatureComputationOptions(
            features_to_compute=tuple(features_to_compute),
            output_dir=feature_cache_dir,
            scratch_dir=scratch_dir,
            skip_saved_computation=skip_saved_computation,
            include_trajectory=False,
        ),
    )
    return {**dict(result.manifest_record), "filename": str(config.ap_zarr)}


def _aggregate_and_denoise(
    df_manifest: pd.DataFrame,
    probe_name: str,
) -> pd.DataFrame:
    """Aggregate snippet features and denoise them with the legacy aggregator."""
    df_aggregated = get_aggregated_features_per_pid(df_manifest)
    df_aggregated["pid"] = probe_name
    df_features = denoise_raw_features_data(
        df_aggregated.set_index(["pid", "channel"]).sort_index(),
        n_jobs=1,
        verbose=0,
    ).reset_index()
    df_features["channel"] = pd.to_numeric(
        df_features["channel"], errors="coerce"
    ).astype("Int64")
    return df_features.sort_values("channel").reset_index(drop=True)


def run_probe(
    config: ProbeConfig,
    output_dir: Path,
    t_starts: Sequence[float] = SNIPPET_T_STARTS,
    duration_ap: float = DURATION_AP,
    duration_lf: float = DURATION_LF,
    features_to_compute: Sequence[str] = DEFAULT_FEATURES_TO_COMPUTE,
    skip_saved_computation: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute and denoise OOP features for one Allen probe.

    Args:
        config (ProbeConfig): Probe paths and output name.
        output_dir (Path): Root directory where probe outputs are written.
        t_starts (Sequence[float]): Snippet start times in seconds.
        duration_ap (float): AP snippet duration in seconds.
        duration_lf (float): LFP snippet duration in seconds.
        features_to_compute (Sequence[str]): Feature families to compute.
        skip_saved_computation (bool): Reuse existing snippet feature files when
            they are already present.

    Returns:
        tuple[dict[str, object], pd.DataFrame]: Run summary row and denoised
            per-channel feature table.
    """
    if not config.ap_zarr.exists():
        raise FileNotFoundError(config.ap_zarr)
    if not config.lf_zarr.exists():
        raise FileNotFoundError(config.lf_zarr)

    probe_output_dir = Path(output_dir) / config.name
    probe_output_dir.mkdir(parents=True, exist_ok=True)
    calculator = NWBZarrFeatureCalculator(
        ap_zarr_path=config.ap_zarr,
        lf_zarr_path=config.lf_zarr,
        name=config.name,
    )

    records = [
        _compute_snippet_manifest(
            calculator=calculator,
            config=config,
            t_start=t_start,
            duration_ap=duration_ap,
            duration_lf=duration_lf,
            features_to_compute=features_to_compute,
            probe_output_dir=probe_output_dir,
            skip_saved_computation=skip_saved_computation,
        )
        for t_start in t_starts
    ]
    df_manifest = pd.DataFrame.from_records(records)
    manifest_file = probe_output_dir / f"{config.name}_snippet_manifest.pqt"
    df_manifest.to_parquet(manifest_file)

    df_features = _aggregate_and_denoise(
        df_manifest=df_manifest,
        probe_name=config.name,
    )
    features_file = probe_output_dir / f"{config.name}_denoised_features.pqt"
    df_features.to_parquet(features_file)
    LOGGER.info("Saved %s denoised features to %s", config.name, features_file)

    return (
        {
            "name": config.name,
            "manifest": manifest_file.as_posix(),
            "features": features_file.as_posix(),
            "n_channels": len(df_features),
        },
        df_features,
    )


# %% Script entry point
def _configure_logging(level: str = "INFO") -> None:
    """Configure concise console logging for script runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(
    datasets: Sequence[ProbeConfig] = DATASETS,
    t_starts: Sequence[float] = SNIPPET_T_STARTS,
    duration_ap: float = DURATION_AP,
    duration_lf: float = DURATION_LF,
    features_to_compute: Sequence[str] = DEFAULT_FEATURES_TO_COMPUTE,
    output_dir: Path = OUTPUT_DIR,
    run_id: str | None = None,
    skip_saved_computation: bool = False,
) -> pd.DataFrame:
    """Compute Allen OOP features and write per-probe plus combined parquet files."""
    _configure_logging("INFO")
    run_output_dir = Path(output_dir) / run_id if run_id else Path(output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing Allen OOP feature outputs to %s", run_output_dir)

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    feature_frames: list[pd.DataFrame] = []
    for config in datasets:
        try:
            summary_row, df_features = run_probe(
                config=config,
                output_dir=run_output_dir,
                t_starts=t_starts,
                duration_ap=duration_ap,
                duration_lf=duration_lf,
                features_to_compute=features_to_compute,
                skip_saved_computation=skip_saved_computation,
            )
            results.append(summary_row)
            feature_frames.append(df_features)
        except Exception as exc:
            LOGGER.exception("Failed Allen OOP feature computation for %s", config.name)
            failures.append({"name": config.name, "error": repr(exc)})

    combined_file = run_output_dir / "allen_oop_denoised_features.pqt"
    if feature_frames:
        pd.concat(feature_frames, ignore_index=True).to_parquet(combined_file)
        LOGGER.info("Saved combined Allen denoised features to %s", combined_file)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results["combined_features"] = combined_file.as_posix()
    df_results.to_parquet(run_output_dir / "run_summary.pqt")
    if failures:
        pd.DataFrame(failures).to_parquet(run_output_dir / "run_failures.pqt")
    return df_results


if __name__ == "__main__":
    main()
