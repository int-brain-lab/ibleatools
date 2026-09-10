from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import torch


@dataclass
class Config:
    """Configuration of the released K=25 / context / kNN=20 unit model."""

    seed: int = 0
    project: str = "ibl_neuropixel_brainwide_01"
    repo_id: str = "AlonSaguy/ephys-atlas-models"
    vintage: str = "2026_W26"

    prepared_data_dir: Path = Path("unit_level_model_data/prepared_data")
    model_dir: Path = Path("unit_level_model_checkpoints")
    output_dir: Path = Path("unit_level_model_results")
    release_staging_dir: Path = Path("unit_level_release_staging")

    force_reprepare_data: bool = False
    prepare_data_if_missing: bool = True

    use_acg: bool = True
    use_stpc: bool = True
    modality_latent_dim: int = 20
    waveform_shape: tuple[int, ...] = (20, 128)
    acg_shape: tuple[int, ...] = (10, 201)
    stpc_shape: tuple[int, ...] = (161,)

    ae_batch_size: int = 512
    eval_batch_size: int = 1024
    ae_epochs: int = 60
    ae_learning_rate: float = 2e-4
    ae_weight_decay: float = 1e-4
    ae_patience: int = 8
    ae_min_delta: float = 1e-5
    grad_clip: float = 5.0
    num_workers: int = 0
    latent_std_target: float = 0.5
    lambda_latent_variance: float = 0.05
    lambda_latent_covariance: float = 0.002

    # Retained for checkpoint compatibility; disabled in the released model.
    feature_fidelity: bool = False
    feature_fidelity_hidden_dim: int = 128
    lambda_feature_continuous: float = 0.25
    lambda_feature_polarity: float = 0.10

    # Final density model.
    gmm_components: int = 25
    gmm_covariance_type: str = "full"
    gmm_reg_covar: float = 1e-4
    gmm_n_init: int = 10
    gmm_max_iter: int = 500
    gmm_min_component_fraction: float = 0.001

    # Molecular context conditions only the mixture weights gamma_k(x).
    n_cell_pcs: int = 50
    n_gene_pcs: int = 50
    context_atlas_subdir: str = "context_atlas"
    context_hidden_dim: int = 128
    context_layers: int = 2
    context_dropout: float = 0.1
    context_weight_epochs: int = 80
    context_weight_batch_size: int = 2048
    context_weight_lr: float = 3e-4
    context_weight_decay: float = 1e-4
    context_weight_patience: int = 10
    rare_component_power: float = 0.5
    rare_component_weight_cap: float = 10.0

    # Final empirical projection.
    knn_decoder_k: int = 20

    mirror_x_to_single_hemisphere: bool = True
    mirror_x_sign: float = -1.0

    # Baseline settings are used only by publication figures, not by main training.
    region_gaussian_variance_floor: float = 1e-3
    kde_neighbors: int = 128
    kde_spatial_bandwidth_um: float = 350.0
    kde_latent_bandwidth: float = 0.55
    kde_chunk_size: int = 256

    # Publication/visualization settings. These are not used by the main runner.
    waveform_sampling_rate_hz: float = 30_000.0
    diagnostic_examples: int = 16
    diagnostic_hist_bins: int = 40
    diagnostic_samples_per_region: int = 3000
    diagnostic_slice_quantiles: tuple[float, ...] = (0.2, 0.5, 0.8)
    diagnostic_spatial_pcs: int = 3
    diagnostic_voxel_size_um: float = 100.0
    diagnostic_slice_cmap: str = "seismic"
    diagnostic_feature_slice_cmap: str = "seismic"
    diagnostic_save_voxel_predictions: bool = False
    max_regions_in_distribution_plot: int = 16

    feature_slice_count: int = 5
    feature_slice_seed: int = 20260831
    feature_slice_reference_max_units: int = 20_000
    feature_slice_component_mc_samples: int = 128
    feature_slice_regional_mc_samples: int = 128
    feature_slice_display_quantiles: tuple[float, float] = (0.02, 0.98)
    feature_slice_display_padding_fraction: float = 0.03
    feature_slice_center: str = "median"
    diagnostic_voxel_size_um: float = 100.0
    diagnostic_feature_slice_cmap: str = "seismic"
    feature_nll_samples_per_test_unit: int = 4
    feature_nll_min_bandwidth_fraction: float = 0.02
    feature_categorical_alpha: float = 1.0

    latent_fidelity_test_units: int = 8000
    latent_fidelity_sliced_wasserstein_projections: int = 128

    merfish_correlation_voxel_um: int = 200
    merfish_correlation_cmap: str = "inferno"
    merfish_correlation_vmin: float = 0.0
    merfish_correlation_vmax: float = 0.5

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def active_modalities(self) -> tuple[str, ...]:
        names = ["waveform"]
        if self.use_acg:
            names.append("acg")
        if self.use_stpc:
            names.append("stpc")
        return tuple(names)

    def latent_dim(self) -> int:
        return self.modality_latent_dim * len(self.active_modalities())

    def to_json_dict(self) -> dict:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, Path):
                out[key] = str(value)
        out["device"] = str(self.device)
        return out

    @classmethod
    def from_json(cls, path: Path | str, *, device: str | None = None) -> "Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in payload.items() if k in valid}
        for key in ("prepared_data_dir", "model_dir", "output_dir", "release_staging_dir"):
            if key in kwargs:
                kwargs[key] = Path(kwargs[key])
        for key in ("waveform_shape", "acg_shape", "stpc_shape"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        if device is not None:
            kwargs["device"] = device
        return cls(**kwargs)
