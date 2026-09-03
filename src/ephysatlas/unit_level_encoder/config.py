from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from ephysatlas.model_registry import UNIT_AE_FILE, UNIT_GMM_FILE


@dataclass
class Config:
    """Configuration for the released waveform+ACG unit-level atlas model.

    Runtime paths are deliberately not part of the scientific configuration.
    The runner assigns a temporary output directory and all persistent artifacts
    are published to / loaded from Hugging Face.
    """

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    waveform_shape: Tuple[int, int] = (20, 128)
    acg_shape: Tuple[int, int] = (10, 201)
    xyz_in_meters: bool = True
    mirror_x_to_left_hemisphere: bool = True
    voxel_size_um: float = 200.0

    # Canonical unit phenotype latent: waveform + ACG only.
    shared_latent_dim: int = 32
    projector_hidden_dim: int = 192
    projector_dim: int = 32

    ae_epochs: int = 60
    ae_batch_size: int = 256
    validation_batch_size: int = 512
    num_workers: int = 0
    ae_learning_rate: float = 3e-4
    ae_weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = 12
    min_delta: float = 1e-4
    amp: bool = True

    waveform_noise_std: float = 0.01
    waveform_amplitude_jitter: float = 0.05
    waveform_time_shift: int = 2
    waveform_channel_mask_probability: float = 0.05
    waveform_time_mask_probability: float = 0.02
    acg_noise_std: float = 0.01
    acg_mask_probability: float = 0.02

    lambda_waveform_reconstruction: float = 1.0
    lambda_acg_reconstruction: float = 1.0
    lambda_waveform_morphology: float = 0.05
    lambda_raw_latent_scale: float = 0.01
    feature_softmax_temperature: float = 0.05
    raw_latent_std_target: float = 1.0
    acg_activity_weight: float = 4.0
    lambda_vicreg_invariance: float = 1.0
    lambda_vicreg_variance: float = 1.0
    lambda_vicreg_covariance: float = 0.04
    vicreg_std_target: float = 1.0
    vicreg_eps: float = 1e-4

    gmm_components: int = 16
    gmm_reg_covar: float = 1e-4
    gmm_sklearn_max_iter: int = 300
    gmm_sklearn_n_init: int = 5
    pt_hidden_dim: int = 160
    pt_heads: int = 5
    pt_layers: int = 3
    pt_dropout: float = 0.10
    neighbor_token_dropout_probability: float = 0.20
    full_neighbor_dropout_probability: float = 0.20
    pt_epochs: int = 100
    pt_learning_rate: float = 3e-4
    pt_weight_decay: float = 1e-5
    pt_batch_size: int = 64
    pt_patience: int = 15
    pt_min_delta: float = 1e-4
    max_neighbor_units: int = 64
    max_neighbor_distance_um: float = 500.0
    min_target_units_per_voxel: int = 3
    sigma_min: float = 0.05

    waveform_sampling_rate_hz: float = 30_000.0
    cosmos_region_names: Tuple[str, ...] = (
        "CB",
        "CNU",
        "CTXsp",
        "HB",
        "HPF",
        "HY",
        "Isocortex",
        "MB",
        "OLF",
        "TH",
        "root",
    )

    # Runtime-only fields populated by the training runner.
    output_dir: Path | str = Path(".")
    # The trainers write the canonical release filenames directly into output_dir (the model-dir
    # root), so the training output IS the publish-ready layout -- no working-name -> rename stage.
    ae_checkpoint_name: str = UNIT_AE_FILE
    pt_checkpoint_name: str = UNIT_GMM_FILE
    summary_name: str = "summary.json"
