"""Typed containers for object-oriented feature computation.

This module contains small immutable value objects shared by the feature
calculator classes. The objects describe one snippet request, raw voltage
payloads, destriping options, and the result of one feature computation.

Classes
-------
SnippetWindow
    Start time and AP/LF durations for one snippet.
RawSnippet
    Raw AP/LF arrays and sampling frequencies.
DestripeOptions
    Parameters used for intermediate AP/LF destriping.
DestripedSnippet
    Raw and destriped arrays returned for debugging.
FeatureComputationOptions
    Options for one feature computation call.
FeatureComputationResult
    Feature DataFrame, provenance, paths, and manifest metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SnippetWindow:
    """Time window used for one feature-computation snippet.

    Attributes:
        t_start (float): Start time in seconds.
        duration_ap (float): AP-band duration in seconds.
        duration_lf (float): LF-band duration in seconds.

    Raises:
        ValueError: If any time value is negative.
    """

    t_start: float = 0.0
    duration_ap: float = 5.0
    duration_lf: float = 25.0

    def __post_init__(self) -> None:
        """Validate that snippet times are non-negative."""
        if self.t_start < 0:
            raise ValueError(f"t_start ({self.t_start}) cannot be negative")
        if self.duration_ap < 0:
            raise ValueError(f"duration_ap ({self.duration_ap}) cannot be negative")
        if self.duration_lf < 0:
            raise ValueError(f"duration_lf ({self.duration_lf}) cannot be negative")


@dataclass(frozen=True)
class RawSnippet:
    """Raw AP/LF voltage arrays and sampling frequencies.

    Attributes:
        raw_ap (np.ndarray | None): AP voltage shaped ``(channels, samples)`` in
            volts. ``None`` means AP data is unavailable.
        raw_lf (np.ndarray | None): LF voltage shaped ``(channels, samples)`` in
            volts. ``None`` means LF data is unavailable.
        fs_ap (float | None): AP sampling frequency in Hz.
        fs_lf (float | None): LF sampling frequency in Hz.

    Raises:
        ValueError: If neither AP nor LF data is provided.
    """

    raw_ap: np.ndarray | None
    raw_lf: np.ndarray | None
    fs_ap: float | None
    fs_lf: float | None

    def __post_init__(self) -> None:
        """Validate that the snippet contains at least one stream."""
        if self.raw_ap is None and self.raw_lf is None:
            raise ValueError("One of raw_ap or raw_lf must be provided")


@dataclass(frozen=True)
class DestripeOptions:
    """Options used when exposing intermediate destriped snippets.

    Attributes:
        lf_k_filter (bool | None): Spatial filter mode passed to
            ``ibldsp.voltage.destripe_lfp``. ``None`` disables LF spatial
            filtering.
        ap_k_filter (bool): Spatial filter mode passed to
            ``ibldsp.voltage.destripe`` for AP data.
        neuropixel_version (int | None): Neuropixels version. When ``None``, the
            calculator's default version is used.
        nshank (int): Number of probe shanks.
    """

    lf_k_filter: bool | None = False
    ap_k_filter: bool = False
    neuropixel_version: int | None = None
    nshank: int = 1


@dataclass(frozen=True)
class DestripedSnippet:
    """Raw and destriped AP/LF data for debugging and visual inspection.

    Attributes:
        raw (RawSnippet): Raw voltage snippet used as the destriping input.
        des_ap (np.ndarray | None): Destriped AP data, or ``None`` if AP data is
            unavailable.
        des_lf (np.ndarray | None): Destriped LF data, or ``None`` if LF data is
            unavailable.
        geometry (Mapping[str, np.ndarray]): Channel geometry used by ibldsp.
        channel_labels (np.ndarray | None): Channel labels used by the destriper.
    """

    raw: RawSnippet
    des_ap: np.ndarray | None
    des_lf: np.ndarray | None
    geometry: Mapping[str, np.ndarray]
    channel_labels: np.ndarray | None


@dataclass(frozen=True)
class FeatureComputationOptions:
    """Options controlling one OOP feature-computation call.

    Attributes:
        features_to_compute (Sequence[str] | None): Feature families to compute.
            ``None`` computes every supported feature for the available streams.
        output_dir (Path | None): Root directory for probe/snippet outputs. When
            ``None``, no feature files or channel files are written.
        scratch_dir (Path | None): Scratch directory for downstream feature
            routines such as waveform extraction.
        skip_saved_computation (bool): Reuse existing feature files in the
            snippet directory when possible.
        save_waveforms (bool): Save waveform arrays when waveform features are
            computed.
        recompute_channels (bool): Rewrite ``channels.pqt`` even if it exists.
        include_trajectory (bool): Try to add trajectory-derived coordinates.
        require_trajectory (bool): Raise an error when trajectory metadata is
            missing.
        lf_k_filter (bool | None): Spatial filter mode forwarded to LF
            destriping in ``compute_features_from_raw``.
        extra_kwargs (Mapping[str, Any]): Extra keyword arguments forwarded to
            ``compute_features_from_raw``.
    """

    features_to_compute: Sequence[str] | None = None
    output_dir: Path | None = None
    scratch_dir: Path | None = None
    skip_saved_computation: bool = False
    save_waveforms: bool = False
    recompute_channels: bool = False
    include_trajectory: bool = True
    require_trajectory: bool = False
    lf_k_filter: bool | None = False
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureComputationResult:
    """Result returned by one OOP feature computation.

    Attributes:
        features (pd.DataFrame): Feature table returned by
            ``compute_features_from_raw`` and enriched with channel metadata when
            possible.
        provenance (Mapping[str, Any]): Provenance stored on newly computed
            feature files.
        computed_features (tuple[str, ...]): Feature families recalculated in
            this call.
        cached_features (tuple[str, ...]): Feature families loaded from existing
            files.
        probe_level_dir (Path | None): Probe-level output directory.
        snippet_level_dir (Path | None): Snippet-level output directory.
        manifest_record (Mapping[str, Any]): Aggregation-ready manifest row.
    """

    features: pd.DataFrame
    provenance: Mapping[str, Any]
    computed_features: tuple[str, ...]
    cached_features: tuple[str, ...]
    probe_level_dir: Path | None
    snippet_level_dir: Path | None
    manifest_record: Mapping[str, Any]
