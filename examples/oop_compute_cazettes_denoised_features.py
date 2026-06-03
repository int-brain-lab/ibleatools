"""Compute denoised Cazettes features with the OOP feature calculators.

This example mirrors the feature-generation part of
``cazettes_sample_data_check/analysis/run_aggregated_inference.py``. It computes
five 5-second snippets with :class:`SpikeGLXFileFeatureCalculator`, aggregates
the snippet-level feature files, denoises the aggregated features, and writes the
same final per-probe parquet artifact used by the legacy inference script.

The script is split into ``# %%`` cells so it can be run interactively.
"""

# %% Imports
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

import ephysatlas.anatomy
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


# %% Module-level constants
def _examples_dir() -> Path:
    """Return the examples directory when run as a script or from a cell."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


DEFAULT_INPUT_PATH = Path("/mnt/s0/Data/2026_cazettes/2026_cazettes/Data")
OUTPUT_DIR = _examples_dir() / "output" / "cazettes_oop_denoised_features"
DEFAULT_FEATURES_TO_COMPUTE: tuple[str, ...] = ("lf", "csd", "ap", "waveforms")
SNIPPET_T_STARTS: tuple[float, ...] = (300.0, 600.0, 900.0, 1200.0, 1500.0)
DURATION_AP = 5.0
DURATION_LF = 5.0


# %% Probe configuration
@dataclass(frozen=True)
class ProbeConfig:
    """Identifying name, raw paths, and optional ALF metadata for one probe."""

    name: str
    ap_file: Path
    lf_file: Path
    alf_probe_path: Path | None = None


DATASETS: list[ProbeConfig] = [
    ProbeConfig(
        name="VF065_2025_12_17_probe00",
        ap_file=DEFAULT_INPUT_PATH
        / "VF065/2025_12_17/Rec/probe00/disabled_g0_t0.imec0.ap.cbin",
        lf_file=DEFAULT_INPUT_PATH
        / "VF065/2025_12_17/Rec/probe00/disabled_g0_t0.imec0.lf.cbin",
        alf_probe_path=DEFAULT_INPUT_PATH / "VF065/2025_12_17/alf/probe00",
    ),
    ProbeConfig(
        name="VF066_2025_12_04_probe00",
        ap_file=DEFAULT_INPUT_PATH
        / "VF066/2025_12_04/Rec/probe00/disabled_g0_t0.imec0.ap.cbin",
        lf_file=DEFAULT_INPUT_PATH
        / "VF066/2025_12_04/Rec/probe00/disabled_g0_t0.imec0.lf.cbin",
        alf_probe_path=DEFAULT_INPUT_PATH / "VF066/2025_12_04/alf/probe00",
    ),
    ProbeConfig(
        name="VF066_2025_12_04_probe01",
        ap_file=DEFAULT_INPUT_PATH
        / "VF066/2025_12_04/Rec/probe01/disabled_g0_t0.imec1.ap.cbin",
        lf_file=DEFAULT_INPUT_PATH
        / "VF066/2025_12_04/Rec/probe01/disabled_g0_t0.imec1.lf.cbin",
        alf_probe_path=DEFAULT_INPUT_PATH / "VF066/2025_12_04/alf/probe01",
    ),
]


# %% Small helpers
def _seconds_slug(seconds: float) -> str:
    """Return a compact filename-safe seconds label, for example ``300s``."""
    seconds = float(seconds)
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{str(seconds).replace('.', 'p')}s"


def _load_channel_locations(alf_probe_path: Path | None) -> pd.DataFrame | None:
    """Load Cazettes per-channel ALF labels from ``channel_locations.json``.

    Args:
        alf_probe_path (Path | None): ALF probe directory containing
            ``channel_locations.json``.

    Returns:
        pd.DataFrame | None: Channel metadata with Allen/Cosmos labels, or
            ``None`` when no ALF folder or JSON file is available.

    Raises:
        ValueError: If the JSON exists but does not describe 96 or 384 channels.
    """
    if alf_probe_path is None:
        return None

    import json

    locations_file = Path(alf_probe_path) / "channel_locations.json"
    if not locations_file.exists():
        LOGGER.info("No channel_locations.json found at %s", locations_file)
        return None

    with locations_file.open("r", encoding="utf-8") as stream:
        channel_locations = json.load(stream)

    records: list[dict] = []
    for key, values in channel_locations.items():
        if not isinstance(values, dict) or not key.startswith("channel_"):
            continue
        channel = int(values.get("original_channel_idx", key.split("_")[-1]))
        records.append(
            {
                "channel": channel,
                "original_channel_idx": channel,
                "channel_plot_index": channel,
                "x": values.get("x") / 1e6,
                "y": values.get("y") / 1e6,
                "z": values.get("z") / 1e6,
                "axial_um": values.get("axial"),
                "lateral_um": values.get("lateral"),
            }
        )

    if len(records) not in (96, 384):
        raise ValueError(
            f"Expected 96 or 384 channels but got {len(records)} from "
            f"{locations_file}"
        )

    df_locations = (
        pd.DataFrame.from_records(records).sort_values("channel").reset_index(drop=True)
    )

    # Match the legacy script: derive atlas ids/acronyms from the ALF xyz columns.
    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    xyz_chans = df_locations[list("xyz")].to_numpy()
    aids = brain_atlas.get_labels(xyz_chans)
    brain_regions = brain_atlas.regions.get(aids)
    df_locations["atlas_id"] = brain_regions["id"]
    df_locations["acronym"] = brain_regions["acronym"]
    df_locations["Allen_id"] = aids
    df_locations["Cosmos_id"] = brain_atlas.regions.remap(aids, "Allen", "Cosmos")
    return df_locations


def _enrich_features_with_channel_locations(
    df_features: pd.DataFrame,
    df_locations: pd.DataFrame | None,
    probe_name: str,
) -> pd.DataFrame:
    """Merge optional Cazettes channel-location columns onto feature rows."""
    if df_locations is None:
        LOGGER.info("No channel locations for %s; returning features unchanged", probe_name)
        return df_features

    df_features = df_features.copy()
    if "channel" not in df_features.columns:
        df_features["channel"] = df_features.index.to_numpy()
    df_features["channel"] = pd.to_numeric(
        df_features["channel"], errors="coerce"
    ).astype("Int64")

    overlap = [
        column
        for column in df_locations.columns
        if column in df_features.columns and column != "channel"
    ]
    if overlap:
        df_features = df_features.drop(columns=overlap)

    df_features = df_features.merge(
        df_locations, on="channel", how="left", validate="many_to_one"
    )
    missing = (
        df_features["atlas_id"].isna().sum()
        if "atlas_id" in df_features.columns
        else len(df_features)
    )
    if missing:
        LOGGER.warning("%s rows lack channel-location labels for %s", missing, probe_name)
    return df_features


# %% OOP feature computation and denoising
def _compute_snippet_manifest(
    calculator: SpikeGLXFileFeatureCalculator,
    config: ProbeConfig,
    t_start: float,
    duration_ap: float,
    duration_lf: float,
    features_to_compute: Sequence[str],
    probe_output_dir: Path,
    skip_saved_computation: bool,
) -> dict:
    """Compute one snippet with the OOP calculator and return a manifest row."""
    feature_cache_dir = probe_output_dir / "oop_feature_cache"
    scratch_dir = probe_output_dir / "oop_scratch" / _seconds_slug(t_start)

    LOGGER.info("Computing OOP Cazettes features for %s at %.1fs", config.name, t_start)
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
    return {**dict(result.manifest_record), "filename": str(config.ap_file)}


def _aggregate_and_denoise(
    df_manifest: pd.DataFrame,
    probe_name: str,
    df_channel_locations: pd.DataFrame | None,
) -> pd.DataFrame:
    """Aggregate snippet features, denoise them, and merge ALF locations."""
    df_aggregated = get_aggregated_features_per_pid(df_manifest)
    df_aggregated["pid"] = probe_name
    df_denoised = denoise_raw_features_data(
        df_aggregated.set_index(["pid", "channel"]).sort_index(),
        n_jobs=1,
        verbose=0,
    ).reset_index()

    df_features = _enrich_features_with_channel_locations(
        df_denoised, df_channel_locations, probe_name
    )
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
    """Compute and denoise OOP features for one Cazettes probe.

    Args:
        config (ProbeConfig): Probe paths and output name.
        output_dir (Path): Root directory where probe outputs are written.
        t_starts (Sequence[float]): Snippet start times in seconds.
        duration_ap (float): AP snippet duration in seconds.
        duration_lf (float): LF snippet duration in seconds.
        features_to_compute (Sequence[str]): Feature families to compute.
        skip_saved_computation (bool): Reuse existing snippet feature files when
            they are already present.

    Returns:
        tuple[dict[str, object], pd.DataFrame]: Run summary row and denoised
            per-channel feature table.
    """
    if not config.ap_file.exists():
        raise FileNotFoundError(config.ap_file)
    if not config.lf_file.exists():
        raise FileNotFoundError(config.lf_file)

    probe_output_dir = Path(output_dir) / config.name
    probe_output_dir.mkdir(parents=True, exist_ok=True)
    df_channel_locations = _load_channel_locations(config.alf_probe_path)
    calculator = SpikeGLXFileFeatureCalculator(
        ap_file=config.ap_file,
        lf_file=config.lf_file,
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
        df_channel_locations=df_channel_locations,
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
    """Compute Cazettes OOP features and write per-probe plus combined parquet files."""
    _configure_logging("INFO")
    run_output_dir = Path(output_dir) / run_id if run_id else Path(output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing Cazettes OOP feature outputs to %s", run_output_dir)

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
            LOGGER.exception("Failed Cazettes OOP feature computation for %s", config.name)
            failures.append({"name": config.name, "error": repr(exc)})

    combined_file = run_output_dir / "cazettes_oop_denoised_features.pqt"
    if feature_frames:
        pd.concat(feature_frames, ignore_index=True).to_parquet(combined_file)
        LOGGER.info("Saved combined Cazettes denoised features to %s", combined_file)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results["combined_features"] = combined_file.as_posix()
    df_results.to_parquet(run_output_dir / "run_summary.pqt")
    if failures:
        pd.DataFrame(failures).to_parquet(run_output_dir / "run_failures.pqt")
    return df_results


if __name__ == "__main__":
    main()
