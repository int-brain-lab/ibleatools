"""Abstract base classes for OOP feature calculators.

This module defines the shared interface and template method used by all
source-specific calculators. Concrete subclasses only need to load source data
and metadata; the base class handles output paths, channel cache writes,
destriped snippet inspection, provenance, and the call into
``compute_features_from_raw``.

Classes
-------
BaseFeatureCalculator
    Abstract interface and shared implementation for one recording source.
"""

from __future__ import annotations

import abc
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import ibldsp.voltage
import numpy as np
import pandas as pd
from filelock import FileLock

from ephysatlas.feature_computation import compute_features_from_raw, destripe_ap_lf
from ephysatlas.utils import setup_output_directory

from .provenance import collect_ibleatools_provenance, log_reproduction_command
from .types import (
    DestripeOptions,
    DestripedSnippet,
    FeatureComputationOptions,
    FeatureComputationResult,
    RawSnippet,
    SnippetWindow,
)

LOGGER = logging.getLogger(__name__)

AVAILABLE_FEATURES = ("lf", "csd", "ap", "waveforms")


class BaseFeatureCalculator(abc.ABC):
    """Abstract base class for source-specific feature calculators.

    Concrete subclasses implement the source-dependent loaders: raw AP/LF
    snippets, geometry, channel metadata, and available durations. The base
    class then performs the shared feature-computation workflow.

    Args:
        name (str): Human-readable recording identifier. It is used as ``pid`` in
            new OOP output manifests.
        neuropixel_version (int): Neuropixels version passed to ibldsp
            destriping and ``compute_features_from_raw``.

    Attributes:
        name (str): Human-readable recording identifier.
        neuropixel_version (int): Neuropixels version used for destriping.

    Note:
        It owns live reader objects, caches, and arrays; validation is
        handled at method boundaries
        where the relevant data are available.
    """

    def __init__(self, name: str, neuropixel_version: int = 1) -> None:
        self.name = name
        self.neuropixel_version = neuropixel_version

    @abc.abstractmethod
    def load_raw_snippet(self, window: SnippetWindow) -> RawSnippet:
        """Load raw AP/LF voltage for one snippet.

        Args:
            window (SnippetWindow): Time window to read.

        Returns:
            RawSnippet: Raw AP/LF arrays shaped ``(channels, samples)`` in volts
            and their sampling frequencies.

        Raises:
            ValueError: If the requested window cannot be read.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_geometry(self) -> dict[str, np.ndarray]:
        """Load channel geometry for ibldsp.

        Returns:
            dict[str, np.ndarray]: Geometry with at least ``"x"`` and ``"y"``
            arrays. Subclasses should include ``"sample_shift"``, ``"shank"``,
            ``"row"``, and ``"col"`` when available.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_channel_metadata(self) -> pd.DataFrame:
        """Load channel metadata for aggregation and feature-table enrichment.

        Returns:
            pd.DataFrame: Channel metadata keyed by a ``"channel"`` column. At
            minimum, this should include ``"axial_um"`` and ``"lateral_um"``
            when geometry is available.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def available_duration(self) -> tuple[float | None, float | None]:
        """Return available AP and LF recording durations in seconds.

        Returns:
            tuple[float | None, float | None]: ``(duration_ap, duration_lf)``.
            A value is ``None`` when that stream is absent or unknown.
        """
        raise NotImplementedError

    def output_directory_params(
        self, window: SnippetWindow, output_dir: Path | None
    ) -> dict[str, Any]:
        """Build parameters passed to ``setup_output_directory``.

        Args:
            window (SnippetWindow): Snippet window being computed.
            output_dir (Path | None): Output root directory.

        Returns:
            dict[str, Any]: Parameters using ``self.name`` as the OOP ``pid``.
        """
        return {
            "pid": self.name,
            "t_start": float(window.t_start),
            "duration_ap": float(window.duration_ap),
            "duration_lf": float(window.duration_lf),
            "output_dir": output_dir,
        }

    def enrich_channel_metadata(
        self, channels: pd.DataFrame, options: FeatureComputationOptions
    ) -> pd.DataFrame:
        """Optionally enrich channel metadata before writing ``channels.pqt``.

        Args:
            channels (pd.DataFrame): Channel metadata loaded by the subclass.
            options (FeatureComputationOptions): Current computation options.

        Returns:
            pd.DataFrame: Enriched channel metadata. The base implementation
            returns ``channels`` unchanged.

        Raises:
            ValueError: If trajectory data are required but unavailable.
        """
        if options.require_trajectory:
            raise ValueError(
                f"{self.__class__.__name__} does not provide trajectory metadata"
            )
        if options.include_trajectory:
            LOGGER.info("No trajectory metadata available for %s", self.name)
        return channels

    def get_destriped_snippet(
        self,
        window: SnippetWindow,
        options: DestripeOptions | None = None,
        channel_labels: np.ndarray | None = None,
    ) -> DestripedSnippet:
        """Return raw and destriped AP/LF data for debugging.

        Args:
            window (SnippetWindow): Time window to read and destripe.
            options (DestripeOptions, optional): Destriping options. Defaults to
                the calculator's standard options.
            channel_labels (np.ndarray, optional): Precomputed bad-channel
                labels. When omitted, labels are loaded from channel metadata or
                detected from the AP snippet.

        Returns:
            DestripedSnippet: Raw snippet, destriped arrays, geometry, and labels.
        """
        options = options or DestripeOptions()
        raw = self.load_raw_snippet(window)
        geometry = self.load_geometry()
        channels = self.load_channel_metadata()
        labels = self._resolve_channel_labels(raw, channels, channel_labels)
        np_version = options.neuropixel_version or self.neuropixel_version

        # Delegate to the shared destriping primitive so this debug/inspection path
        # cannot diverge from the production feature engine.
        des_ap, des_lf = destripe_ap_lf(
            raw_ap=raw.raw_ap,
            raw_lf=raw.raw_lf,
            fs_ap=raw.fs_ap,
            fs_lf=raw.fs_lf,
            geometry=geometry,
            channel_labels=labels,
            neuropixel_version=np_version,
            ap_k_filter=options.ap_k_filter,
            lf_k_filter=options.lf_k_filter,
            nshank=options.nshank,
        )

        return DestripedSnippet(
            raw=raw,
            des_ap=des_ap,
            des_lf=des_lf,
            geometry=geometry,
            channel_labels=labels,
        )

    def compute_snippet(
        self, window: SnippetWindow, options: FeatureComputationOptions | None = None
    ) -> FeatureComputationResult:
        """Compute features for one snippet using the shared raw-array engine.

        Args:
            window (SnippetWindow): Snippet window to compute.
            options (FeatureComputationOptions, optional): Feature, output, cache,
                trajectory, and scratch options.

        Returns:
            FeatureComputationResult: Feature table, computed/cached feature
            families, provenance, output paths, and aggregation manifest metadata.

        Raises:
            ValueError: If the requested window or feature family is invalid.
        """
        options = options or FeatureComputationOptions()
        output_dir = Path(options.output_dir) if options.output_dir else None
        self._validate_window(window)

        probe_level_dir, snippet_level_dir = setup_output_directory(
            self.output_directory_params(window, output_dir)
        )

        raw = self.load_raw_snippet(window)
        geometry = self.load_geometry()
        channels = self.enrich_channel_metadata(
            self._normalize_channel_metadata(self.load_channel_metadata()),
            options,
        )
        self._write_channels_file(
            channels,
            probe_level_dir=probe_level_dir,
            recompute=options.recompute_channels,
        )

        feature_names = self._resolve_feature_names(
            options.features_to_compute, raw=raw
        )
        preexisting = self._feature_file_existence(snippet_level_dir, feature_names)
        computed_features, cached_features = self._classify_computed_features(
            feature_names=feature_names,
            preexisting=preexisting,
            skip_saved=options.skip_saved_computation,
        )
        # An explicit ``channel_labels`` in the options overrides automatic
        # resolution (stored labels / cbin / snippet detection).
        if options.channel_labels is not None:
            channel_labels = np.asarray(options.channel_labels)
        else:
            channel_labels = self._resolve_channel_labels(raw, channels)

        provenance = collect_ibleatools_provenance(
            calculator_name=self.__class__.__name__,
            feature_names=tuple(feature_names),
            extra={
                "recording_name": self.name,
                "t_start": float(window.t_start),
                "duration_ap": float(window.duration_ap),
                "duration_lf": float(window.duration_lf),
                "neuropixel_version": self.neuropixel_version,
                "lf_k_filter": options.lf_k_filter,
            },
        )
        log_reproduction_command(provenance)

        LOGGER.info(
            "Computing %s for %s at t=%.1fs",
            feature_names,
            self.name,
            window.t_start,
        )
        kwargs = {
            "skip_saved_computation": options.skip_saved_computation,
            "save_waveforms": options.save_waveforms,
            **dict(options.extra_kwargs),
        }
        features = compute_features_from_raw(
            raw_ap=raw.raw_ap,
            raw_lf=raw.raw_lf,
            fs_ap=raw.fs_ap,
            fs_lf=raw.fs_lf,
            geometry=geometry,
            channel_labels=channel_labels,
            neuropixel_version=self.neuropixel_version,
            features_to_compute=list(feature_names),
            output_dir=snippet_level_dir,
            scratch_dir=options.scratch_dir,
            lf_k_filter=options.lf_k_filter,
            feature_params=options.feature_params,
            **kwargs,
        )

        self._apply_selective_provenance(
            snippet_level_dir=snippet_level_dir,
            computed_features=computed_features,
            provenance=provenance,
        )
        # Stamp physical coordinates from geometry, then merge channel metadata on
        # the physical site (axial_um, lateral_um, shank)
        features = self._attach_physical_coordinates(features, geometry)
        features = self._merge_channel_metadata(features, channels)
        manifest_record = self._manifest_record(
            window=window,
            output_dir=output_dir,
            snippet_level_dir=snippet_level_dir,
        )

        return FeatureComputationResult(
            features=features,
            provenance=provenance,
            computed_features=tuple(computed_features),
            cached_features=tuple(cached_features),
            probe_level_dir=probe_level_dir,
            snippet_level_dir=snippet_level_dir,
            manifest_record=manifest_record,
        )

    def _validate_window(self, window: SnippetWindow) -> None:
        """Validate a snippet window against available stream durations."""
        max_ap, max_lf = self.available_duration()
        if max_ap is not None and window.t_start + window.duration_ap > max_ap:
            raise ValueError(
                f"Requested AP range ({window.t_start} to "
                f"{window.t_start + window.duration_ap}) exceeds AP duration "
                f"({max_ap})"
            )
        if max_lf is not None and window.t_start + window.duration_lf > max_lf:
            raise ValueError(
                f"Requested LF range ({window.t_start} to "
                f"{window.t_start + window.duration_lf}) exceeds LF duration "
                f"({max_lf})"
            )

    def _resolve_channel_labels(
        self,
        raw: RawSnippet,
        channels: pd.DataFrame,
        channel_labels: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return channel labels from metadata or AP bad-channel detection."""
        if channel_labels is not None:
            return channel_labels
        if "labels" in channels.columns:
            return channels["labels"].to_numpy()
        if "channel_labels" in channels.columns:
            return channels["channel_labels"].to_numpy()
        if raw.raw_ap is None:
            return None
        labels, _ = ibldsp.voltage.detect_bad_channels(raw.raw_ap, fs=raw.fs_ap)
        return labels

    def _resolve_feature_names(
        self, requested: Sequence[str] | None, raw: RawSnippet
    ) -> tuple[str, ...]:
        """Resolve requested feature names against available AP/LF streams."""
        available = list(AVAILABLE_FEATURES)
        if raw.raw_ap is None:
            available = [feature for feature in available if feature in ("lf", "csd")]
        if raw.raw_lf is None:
            available = [
                feature for feature in available if feature in ("ap", "waveforms")
            ]

        if requested is None:
            feature_names = available
        else:
            invalid = [
                feature for feature in requested if feature not in AVAILABLE_FEATURES
            ]
            if invalid:
                raise ValueError(
                    f"Invalid feature sets requested: {invalid}. "
                    f"Available options: {list(AVAILABLE_FEATURES)}"
                )
            feature_names = [feature for feature in requested if feature in available]

        if not feature_names:
            raise ValueError(
                "No requested feature families match the available streams"
            )
        return tuple(feature_names)

    def _feature_file_existence(
        self, snippet_level_dir: Path | None, feature_names: Sequence[str]
    ) -> dict[str, bool]:
        """Return whether each requested feature file existed before computing."""
        if snippet_level_dir is None:
            return {feature_name: False for feature_name in feature_names}
        return {
            feature_name: (snippet_level_dir / f"{feature_name}_features.pqt").exists()
            for feature_name in feature_names
        }

    def _classify_computed_features(
        self,
        feature_names: Sequence[str],
        preexisting: Mapping[str, bool],
        skip_saved: bool,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Classify feature families as recomputed or cached for provenance."""
        cached = tuple(
            feature_name
            for feature_name in feature_names
            if skip_saved and preexisting.get(feature_name, False)
        )
        computed = tuple(
            feature_name for feature_name in feature_names if feature_name not in cached
        )
        return computed, cached

    def _apply_selective_provenance(
        self,
        snippet_level_dir: Path | None,
        computed_features: Sequence[str],
        provenance: Mapping[str, Any],
    ) -> None:
        """Attach provenance only to feature files recomputed in this call."""
        if snippet_level_dir is None:
            return
        for feature_name in computed_features:
            file_path = snippet_level_dir / f"{feature_name}_features.pqt"
            if not file_path.exists():
                LOGGER.warning("Computed feature file was not found: %s", file_path)
                continue

            lock = FileLock(str(file_path) + ".lock", timeout=60)
            with lock:
                df_feature = pd.read_parquet(file_path)
                df_feature.attrs.update(dict(provenance))
                df_feature.attrs["feature_calculator_feature_name"] = feature_name
                tmp_file = file_path.with_suffix(
                    file_path.suffix + f".{os.getpid()}.tmp"
                )
                df_feature.to_parquet(tmp_file)
                os.replace(tmp_file, file_path)
            LOGGER.debug("Updated provenance for %s", file_path)

    def _normalize_channel_metadata(self, channels: pd.DataFrame) -> pd.DataFrame:
        """Ensure channel metadata has a numeric ``channel`` column.
        This function is needed because sometimes channels are
        loaded from the probe_directory, which has the column
        "channel" and if it is loaded using ONE, then it
        contains rawInd.
        """
        channels = channels.copy()
        if "channel" not in channels.columns:
            if "rawInd" in channels.columns:
                channels["channel"] = channels["rawInd"]
            else:
                channels["channel"] = np.arange(len(channels))
        channels["channel"] = pd.to_numeric(
            channels["channel"], errors="coerce"
        ).astype("Int64")
        return channels

    def _write_channels_file(
        self,
        channels: pd.DataFrame,
        probe_level_dir: Path | None,
        recompute: bool,
    ) -> Path | None:
        """Write ``channels.pqt`` with a file lock and atomic replace."""
        if probe_level_dir is None:
            return None

        file_channels = Path(probe_level_dir) / "channels.pqt"
        if file_channels.exists() and not recompute:
            LOGGER.info("Keeping existing channels file at %s", file_channels)
            return file_channels

        lock = FileLock(str(file_channels) + ".lock", timeout=60)
        with lock:
            if file_channels.exists() and not recompute:
                return file_channels
            channels_to_save = channels.copy()
            if "pid" not in channels_to_save.columns:
                channels_to_save["pid"] = self.name
            tmp_file = file_channels.with_suffix(
                file_channels.suffix + f".{os.getpid()}.tmp"
            )
            channels_to_save.to_parquet(tmp_file)
            os.replace(tmp_file, file_channels)
        LOGGER.info("Wrote channels file to %s", file_channels)
        return file_channels

    def _attach_physical_coordinates(
        self, features: pd.DataFrame, geometry: dict
    ) -> pd.DataFrame:
        """Stamp ``axial_um``/``lateral_um``/``shank`` onto the feature table.

        The feature table's ``channel`` column is the positional row index into
        the destriped snippet (``np.arange`` in ``ephysatlas.features``), which is
        the same ordering as ``geometry``. We look each row's coordinates up *by
        its channel value* (not by row position), so a reordered or subset table --
        e.g. the waveforms family, which returns one row per spiking channel --
        still receives the correct site.

        Args:
            features (pd.DataFrame): Feature table with a numeric ``channel`` column.
            geometry (dict): Reader geometry; ``x``/``y``/``shank`` are indexed by
                channel.

        Returns:
            pd.DataFrame: ``features`` with ``axial_um``/``lateral_um``/``shank``.

        Raises:
            ValueError: If ``features`` has no ``channel`` column, or a ``channel``
                value cannot index ``geometry``.
        """
        # compute_features_from_raw always provides "channel" (it is the merge key
        # joining the feature families). A missing column means the engine contract
        # is broken; fail loud rather than silently dropping all channel metadata
        # downstream (the merge would also no-op).
        if "channel" not in features.columns:
            raise ValueError(
                "feature table is missing the 'channel' column required to map "
                "onto geometry (compute_features_from_raw should always provide it)"
            )

        n_channels = len(np.asarray(geometry["x"]))
        channel = pd.to_numeric(features["channel"], errors="coerce")
        if channel.isna().any() or ((channel < 0) | (channel >= n_channels)).any():
            raise ValueError(
                "feature 'channel' values must be integers in "
                f"[0, {n_channels}) to map onto geometry"
            )

        idx = channel.to_numpy(dtype=int)
        shank = np.asarray(geometry.get("shank", np.zeros(n_channels)))
        features = features.copy()
        features["axial_um"] = np.asarray(geometry["y"])[idx]
        features["lateral_um"] = np.asarray(geometry["x"])[idx]
        features["shank"] = shank[idx]
        return features

    @staticmethod
    def _physical_site_key(channels: pd.DataFrame) -> pd.Series:
        """Build an integer-um rounded ``(axial_um, lateral_um, shank)`` join key.

        Coordinates from the feature side (reader geometry) and the metadata side
        (ONE/ALF or geometry) originate from the same probe geometry but may carry
        tiny float differences; rounding to the nearest micrometre absorbs that
        without collapsing distinct sites (which are >= ~16 um apart).
        """
        axial = np.round(np.asarray(channels["axial_um"], dtype=float)).astype(int)
        lateral = np.round(np.asarray(channels["lateral_um"], dtype=float)).astype(int)
        shank = np.asarray(channels["shank"], dtype=float).astype(int)
        parts = pd.DataFrame(
            {"axial": axial, "lateral": lateral, "shank": shank}, index=channels.index
        )
        return (
            parts["axial"].astype(str)
            + "_"
            + parts["lateral"].astype(str)
            + "_"
            + parts["shank"].astype(str)
        )

    def _merge_channel_metadata(
        self, features: pd.DataFrame, channels: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge channel metadata onto a feature table on the physical channel site.

        Channels are matched on ``(axial_um, lateral_um, shank)`` -- the physical
        identity of a recording site -- instead of on ``channel``/``rawInd``, whose
        numbering is unreliable across data sources. Feature columns win over
        metadata columns of the same name; the positional ``channel`` from the
        feature table is preserved and ``rawInd`` is carried as descriptive
        metadata only.
        """
        keys = ["axial_um", "lateral_um", "shank"]
        if not all(key in features.columns for key in keys) or not all(
            key in channels.columns for key in keys
        ):
            # Physical coordinates missing on one side: nothing to join on.
            return features

        features = features.copy()
        metadata = channels.copy()
        features["_site_key"] = self._physical_site_key(features)
        metadata["_site_key"] = self._physical_site_key(metadata)

        # Drop metadata columns that duplicate feature columns (keep the feature
        # values); the "_site_key" join column is preserved on both sides.
        overlap = [
            column
            for column in metadata.columns
            if column in features.columns and column != "_site_key"
        ]
        metadata = metadata.drop(columns=overlap)

        merged = features.merge(metadata, on="_site_key", how="left")
        # A left merge keeps every feature row; a changed row count means the
        # channel metadata had duplicate physical sites that fanned out rows.
        if len(merged) != len(features):
            raise ValueError(
                f"channel merge changed row count {len(features)} -> {len(merged)}; "
                "channel metadata likely has duplicate physical sites "
                "(axial_um, lateral_um, shank)"
            )
        return merged.drop(columns="_site_key")

    def _manifest_record(
        self,
        window: SnippetWindow,
        output_dir: Path | None,
        snippet_level_dir: Path | None,
    ) -> dict[str, Any]:
        """Build an aggregation-ready manifest row for this snippet."""
        record: dict[str, Any] = {
            "pid": self.name,
            "t_start": float(window.t_start),
            "duration_ap": float(window.duration_ap),
            "duration_lf": float(window.duration_lf),
        }
        if output_dir is not None and snippet_level_dir is not None:
            record["base_level_dir"] = output_dir.as_posix()
            record["snippet_level_dir"] = snippet_level_dir.relative_to(
                output_dir
            ).as_posix()
        else:
            record["base_level_dir"] = None
            record["snippet_level_dir"] = None
        return record
