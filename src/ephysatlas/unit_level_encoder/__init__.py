"""Unit-level Ephys Atlas model.

Multimodal waveform/ACG/population-coupling autoencoder -> standardized 60-D joint latent ->
hierarchical Gaussian mixture of putative cell types and their sub-types (top-down assignment),
with sub-type weights conditioned on the molecular context of the position -> phenotypes read out
through each sub-type's TRAIN members.
"""

from .config import Config
from .pipeline import (
    UnitModelBundle,
    load_unit_model,
    prepare_unit_data,
    train_unit_model,
)

__all__ = [
    "Config",
    "UnitModelBundle",
    "load_unit_model",
    "prepare_unit_data",
    "train_unit_model",
]
