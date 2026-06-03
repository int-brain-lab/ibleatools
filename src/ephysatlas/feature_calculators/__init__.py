"""Object-oriented feature calculators for ephys-atlas recordings.

This package provides an OOP layer on top of
``ephysatlas.feature_computation.compute_features_from_raw``. Each calculator
normalizes a source recording into raw AP/LF numpy arrays and channel metadata,
then delegates numerical feature extraction to the existing implementation.

Classes
-------
BaseFeatureCalculator
    Abstract source interface and shared compute workflow.
SpikeGLXFileFeatureCalculator
    Calculator for local AP/LF SpikeGLX files.
IBLPIDFeatureCalculator
    Calculator for IBL probe insertions loaded through ONE.
NWBZarrFeatureCalculator
    Calculator for SpikeInterface-readable AP/LFP Zarr recordings.

Examples
--------
>>> from ephysatlas.feature_calculators import (
...     FeatureComputationOptions,
...     SnippetWindow,
...     SpikeGLXFileFeatureCalculator,
... )
>>> calculator = SpikeGLXFileFeatureCalculator(
...     ap_file="probe.ap.cbin",
...     lf_file="probe.lf.cbin",
...     name="example_probe",
... )
>>> result = calculator.compute_snippet(
...     SnippetWindow(t_start=300.0, duration_ap=5.0, duration_lf=5.0),
...     FeatureComputationOptions(features_to_compute=["lf", "csd"]),
... )
"""

from .base import BaseFeatureCalculator
from .ibl import IBLPIDFeatureCalculator
from .nwb import NWBZarrFeatureCalculator
from .spikeglx import SpikeGLXFileFeatureCalculator
from .types import (
    DestripeOptions,
    DestripedSnippet,
    FeatureComputationOptions,
    FeatureComputationResult,
    RawSnippet,
    SnippetWindow,
)

__all__ = [
    "BaseFeatureCalculator",
    "DestripeOptions",
    "DestripedSnippet",
    "FeatureComputationOptions",
    "FeatureComputationResult",
    "IBLPIDFeatureCalculator",
    "NWBZarrFeatureCalculator",
    "RawSnippet",
    "SnippetWindow",
    "SpikeGLXFileFeatureCalculator",
]
