"""Object-oriented feature calculators for ephys-atlas recordings.

This package provides an OOP layer on top of
``ephysatlas.feature_computation.compute_features_from_raw``. Each calculator
normalizes a source recording into raw AP/LF numpy arrays and channel metadata,
then delegates numerical feature extraction to the existing implementation.

Classes
-------
BaseFeatureCalculator
    Abstract source interface and shared compute workflow.
SpikeGlxLikeFeatureCalculator
    Shared reader-contract logic for spikeglx.Reader-backed sources.
SpikeGLXFileFeatureCalculator
    Calculator for local AP/LF SpikeGLX files.
IBLPIDFeatureCalculator
    Calculator for IBL probe insertions loaded through ONE.
SpikeInterfaceFeatureCalculator
    Shared reader-contract logic for SpikeInterface BaseRecording-backed sources.
NwbFeatureCalculator
    Calculator for NWB recordings (local, streamed, or DANDI).
NwbSource
    Describes how to open one NWB-backed SpikeInterface recording.
LFPackFeatureCalculator
    Calculator for an lfpack-compressed LF archive.
NumpyArrayFeatureCalculator
    Calculator for an LF array already held in memory.
SnippetFileFeatureCalculator
    Calculator for an HDF5 snippet file written by ``ephysatlas.snippet_extractor``.

Examples
--------
>>> from ephysatlas.feature_calculators import (
...     FeatureComputationOptions,
...     SnippetWindow,
...     IBLPIDFeatureCalculator,
... )
>>> calculator = IBLPIDFeatureCalculator(pid="probe00", one=one)
>>> result = calculator.compute_snippet(
...     SnippetWindow(t_start=300.0, duration_ap=5.0, duration_lf=25.0),
...     FeatureComputationOptions(features_to_compute=["lf", "csd"]),
... )

Features aggregated over several snippets (per-channel median, denoised):

>>> df = calculator.compute_denoised_features(n_snippets=5, duration_ap=5.0, duration_lf=5.0)
>>> # or choose the windows
>>> t_starts = get_snippet_start_times(3600.0, n_snippets=5, snippet_duration=5.0)
>>> windows = [SnippetWindow(t, 5.0, 5.0) for t in t_starts]
>>> df = compute_aggregated_features(calculator, windows)
>>> df.attrs["snippet_windows"]  # the windows the features were computed on
"""

from .aggregate import (
    aggregate_snippets,
    check_snippet_overlap,
    compute_aggregated_features,
    get_snippet_start_times,
)
from .base import BaseFeatureCalculator
from .ibl import IBLPIDFeatureCalculator
from .lfpack import LFPackFeatureCalculator
from .numpy_array import NumpyArrayFeatureCalculator
from .nwb import NwbFeatureCalculator, NwbSource
from .snippet_file import SnippetFileFeatureCalculator
from .spikeglx import SpikeGLXFileFeatureCalculator
from .spikeglx_like import SpikeGlxLikeFeatureCalculator
from .spikeinterface_like import SpikeInterfaceFeatureCalculator
from .spikeinterface_zarr import SpikeInterfaceZarrFeatureCalculator
from .types import (
    CsdParams,
    DestripeOptions,
    DestripedSnippet,
    FeatureComputationOptions,
    FeatureComputationResult,
    FeatureParams,
    LfParams,
    RawSnippet,
    SnippetWindow,
    WaveformParams,
)

__all__ = [
    "BaseFeatureCalculator",
    "CsdParams",
    "DestripeOptions",
    "DestripedSnippet",
    "FeatureComputationOptions",
    "FeatureComputationResult",
    "FeatureParams",
    "IBLPIDFeatureCalculator",
    "LFPackFeatureCalculator",
    "LfParams",
    "NumpyArrayFeatureCalculator",
    "NwbFeatureCalculator",
    "NwbSource",
    "RawSnippet",
    "SnippetFileFeatureCalculator",
    "SnippetWindow",
    "SpikeGLXFileFeatureCalculator",
    "SpikeGlxLikeFeatureCalculator",
    "SpikeInterfaceFeatureCalculator",
    "SpikeInterfaceZarrFeatureCalculator",
    "WaveformParams",
    "aggregate_snippets",
    "check_snippet_overlap",
    "compute_aggregated_features",
    "get_snippet_start_times",
]
