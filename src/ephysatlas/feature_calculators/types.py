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

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
class LfParams:
    """Per-feature parameters for the LF feature family.

    Attributes:
        bands (Mapping | None): Frequency bands passed to
            :func:`ephysatlas.features.lf`. ``None`` uses the function default.
        decay_features (bool): Whether to compute the PSD-decay features.
        compute_rms_no_car (bool): Also compute the no-CAR LF RMS
            (``rms_lf_no_car``).
    """

    bands: Mapping | None = None
    decay_features: bool = True
    compute_rms_no_car: bool = False


@dataclass(frozen=True)
class CsdParams:
    """Per-feature parameters for the CSD feature family.

    Attributes:
        bands (Mapping | None): Frequency bands for the CSD spectral step.
            ``None`` uses the default bands.
        decimate (int): Temporal decimation factor applied before the CSD.
        scale (bool): Whether ``ibldsp.voltage.current_source_density`` scales the
            CSD.
    """

    bands: Mapping | None = None
    decimate: int = 10
    scale: bool = True


@dataclass(frozen=True)
class WaveformParams:
    """Per-feature parameters for the waveforms (spike) feature family.

    Attributes:
        n_jobs (int): dartsort worker mode. ``0`` runs in the main process (CUDA
            initialized in-process, no multiprocessing; GPU memory is not freed
            between calls). ``1`` runs in a single worker subprocess (frees GPU
            memory when it exits, but requires a working multiprocessing
            environment). Defaults to ``0``.
    """

    n_jobs: int = 0


@dataclass(frozen=True)
class FeatureParams:
    """Typed per-feature parameters forwarded to ``compute_features_from_raw``.

    The engine reads these attributes duck-typed (via ``getattr``) so that
    ``feature_computation`` never imports this package (avoids a circular import).
    A sub-config left as ``None`` falls back to the engine's default behavior for
    that family, so the default ``FeatureParams()`` reproduces today's output.

    Attributes:
        lf (LfParams | None): LF feature-family parameters.
        csd (CsdParams | None): CSD feature-family parameters.
        waveforms (WaveformParams | None): Waveforms/spike feature-family
            parameters (e.g. dartsort ``n_jobs``).
    """

    lf: LfParams | None = None
    csd: CsdParams | None = None
    waveforms: WaveformParams | None = None

    @classmethod
    def from_dict(cls, data: Mapping) -> FeatureParams:
        """Build ``FeatureParams`` from a nested dict, e.g. ``{"csd": {"scale": False}}``.

        Each family value may be a params dict or an already-built
        ``LfParams``/``CsdParams``. Unknown family names or sub-parameters raise,
        so a typo fails loudly instead of being silently ignored.

        Args:
            data (Mapping): Mapping of feature family -> params dict/object.

        Returns:
            FeatureParams: The typed, validated equivalent.

        Raises:
            ValueError: If a key is not a recognized feature family.
            TypeError: If a family dict contains an unknown sub-parameter.
        """
        families = {"lf": LfParams, "csd": CsdParams, "waveforms": WaveformParams}
        kwargs = {}
        for key, value in data.items():
            if key not in families:
                raise ValueError(
                    f"unknown feature family {key!r}; expected one of {list(families)}"
                )
            param_cls = families[key]
            kwargs[key] = value if isinstance(value, param_cls) else param_cls(**value)
        return cls(**kwargs)


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
        channel_labels (np.ndarray | None): Explicit per-channel bad-channel
            labels. When provided, this overrides the calculator's automatic
            label resolution (stored labels / cbin / snippet detection) -- e.g.
            pass ``np.zeros(n_channels)`` to disable bad-channel handling. ``None``
            keeps the automatic behavior.
        feature_params (FeatureParams | Mapping | None): Per-feature parameters
            forwarded to ``compute_features_from_raw``. Accepts a ``FeatureParams``
            or a nested dict (e.g. ``{"csd": {"scale": False}}``), which is
            normalized to ``FeatureParams`` on init. ``None`` uses the engine
            defaults for every family.
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
    channel_labels: np.ndarray | None = None
    feature_params: FeatureParams | Mapping | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize a nested-dict ``feature_params`` into typed ``FeatureParams``."""
        if isinstance(self.feature_params, Mapping):
            # Frozen dataclass: assign through object.__setattr__.
            object.__setattr__(
                self, "feature_params", FeatureParams.from_dict(self.feature_params)
            )


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
