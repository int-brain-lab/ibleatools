"""Unit-level Ephys Atlas model.

Released model: multimodal waveform/ACG/stPC autoencoder -> standardized 60-D
joint latent -> K=25 full-covariance GMM with global component geometry and
molecular-context-conditioned mixture weights -> k=20 empirical TRAIN-neighbor
projection for phenotype features.
"""

from .config import Config
from .pipeline import UnitModelBundle, load_unit_model, prepare_unit_data, train_unit_model

__all__ = [
    "Config",
    "UnitModelBundle",
    "load_unit_model",
    "prepare_unit_data",
    "train_unit_model",
]
