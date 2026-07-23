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
"""

from .base import BaseFeatureCalculator
from .ibl import IBLPIDFeatureCalculator
from .nwb import NwbFeatureCalculator, NwbSource
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
    "LfParams",
    "NwbFeatureCalculator",
    "NwbSource",
    "RawSnippet",
    "SnippetWindow",
    "SpikeGLXFileFeatureCalculator",
    "SpikeGlxLikeFeatureCalculator",
    "SpikeInterfaceFeatureCalculator",
    "SpikeInterfaceZarrFeatureCalculator",
    "WaveformParams",
]
