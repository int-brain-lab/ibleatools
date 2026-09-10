from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import fit_context_transform, load_prepared_data, set_seed
from .gmm_models import (
    GlobalWeightModel,
    conditional_log_prob,
    fit_context_weight_model,
    fit_global_gmm,
    fit_latent_scaler,
    load_context_weight_bundle,
    responsibilities,
    save_context_weight_bundle,
)
from .knn_decoder import EmpiricalKNNDecoder
from .prepare_data import prepare_latest_cells_encoder_data
from .train import checkpoint_name, encode_all, load_autoencoder_file, train_autoencoder
from .waveform_features import extract_generated_waveform_features


@dataclass
class UnitModelBundle:
    cfg: Config
    autoencoder: object
    latent_scaler: object
    gmm: object
    context_model: object
    knn_decoder: EmpiricalKNNDecoder
    data: object | None = None
    latents: dict[str, np.ndarray] | None = None
    z_scaled: np.ndarray | None = None
    model_root: Path | None = None


def prepare_unit_data(cfg: Config):
    """Create/load the prepared unit arrays required by training and evaluation."""
    data_dir = Path(cfg.prepared_data_dir)
    expected_ctx_dim = int(cfg.n_cell_pcs) + int(cfg.n_gene_pcs)
    required = [
        "waveforms.npy", "acgs.npy", "stpc.npy", "ctx.npy", "xyz.npy",
        "pids.npy", "cosmos.npy", "allen.npy", "waveform_features.npy",
        "waveform_feature_names.json", "latest_cells_encoder_manifest.json",
    ]
    have_all = all((data_dir / name).exists() for name in required)
    current = False
    if have_all:
        try:
            manifest = json.loads((data_dir / "latest_cells_encoder_manifest.json").read_text(encoding="utf-8"))
            ctx_shape = np.load(data_dir / "ctx.npy", mmap_mode="r").shape
            current = (
                len(ctx_shape) == 2
                and int(ctx_shape[1]) == expected_ctx_dim
                and manifest.get("context_type") == "merfish_agea_pca"
                and int(manifest.get("n_cell_pcs", -1)) == int(cfg.n_cell_pcs)
                and int(manifest.get("n_gene_pcs", -1)) == int(cfg.n_gene_pcs)
                and str(manifest.get("context_vintage")) == str(cfg.vintage)
            )
        except Exception:
            current = False

    if not (have_all and current and not cfg.force_reprepare_data):
        if not cfg.prepare_data_if_missing and not have_all:
            raise FileNotFoundError(f"Missing prepared arrays in {data_dir}")
        prepare_latest_cells_encoder_data(
            root_path=Path.cwd(),
            out_dir=data_dir,
            project=cfg.project,
            download=True,
            target_channels=20,
            overwrite_multichannel_cache=False,
            use_acg3d=True,
            use_stpc=True,
            stpc_window_ms=80.0,
            allow_peak_fallback=False,
            context_repo_id=cfg.repo_id,
            context_vintage=cfg.vintage,
            n_cell_pcs=cfg.n_cell_pcs,
            n_gene_pcs=cfg.n_gene_pcs,
            mirror_x_to_single_hemisphere=cfg.mirror_x_to_single_hemisphere,
            mirror_x_sign=cfg.mirror_x_sign,
        )

    data = load_prepared_data(data_dir, cfg)
    cfg.waveform_shape = tuple(data.waveforms.shape[1:])
    cfg.acg_shape = tuple(data.acgs.shape[1:])
    cfg.stpc_shape = tuple(data.stpc.shape[1:])
    return data


def _model_paths(cfg: Config) -> dict[str, Path]:
    root = Path(cfg.model_dir)
    return {
        "root": root,
        "ae_dir": root / "autoencoder",
        "scaler": root / "shared_latent_scaler.joblib",
        "gmm_dir": root / "gmm_k25_full",
        "context_dir": root / "context_weights_k25",
        "knn": root / "knn_bank_k20.npz",
    }


def _model_space_waveform_features(data, cfg) -> np.ndarray:
    """Features in the same normalized-waveform convention as the AE/kNN model."""
    path = Path(cfg.prepared_data_dir) / "waveform_features_model_space.npy"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if cached.shape == (len(data.waveforms), len(data.waveform_feature_names)):
            return cached.astype(np.float32, copy=False)
    features, names = extract_generated_waveform_features(
        data.waveforms, sampling_rate_hz=cfg.waveform_sampling_rate_hz
    )
    if tuple(names) != tuple(data.waveform_feature_names):
        raise RuntimeError("Waveform feature order mismatch between prepared data and model-space extractor")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features.astype(np.float32), allow_pickle=False)
    return features.astype(np.float32, copy=False)


def _build_knn(data, z_scaled, cfg) -> EmpiricalKNNDecoder:
    model_space_features = _model_space_waveform_features(data, cfg)
    return EmpiricalKNNDecoder(
        z_scaled,
        data.split == 0,
        model_space_features,
        k=int(cfg.knn_decoder_k),
        feature_names=data.waveform_feature_names,
    )


def train_unit_model(cfg: Config, data=None) -> UnitModelBundle:
    """Train only the final unit-level model selected for release."""
    set_seed(cfg.seed)
    data = prepare_unit_data(cfg) if data is None else data
    paths = _model_paths(cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)

    ae_cfg = copy.deepcopy(cfg)
    ae_cfg.feature_fidelity = False
    ae_cfg.use_acg = True
    ae_cfg.use_stpc = True
    ae, _, ae_path = train_autoencoder(data, ae_cfg, paths["ae_dir"])
    latents = encode_all(ae, data, ae_cfg)

    train_mask = data.split == 0
    val_mask = data.split == 1
    scaler = fit_latent_scaler(latents["joint"], train_mask, paths["scaler"])

    gcfg = copy.deepcopy(cfg)
    gcfg.gmm_components = 25
    gcfg.gmm_covariance_type = "full"
    gmm, _, z_scaled, resp_train, _ = fit_global_gmm(
        latents["joint"], train_mask, gcfg, paths["gmm_dir"], scaler=scaler
    )

    context_transform = fit_context_transform(data, cfg)
    context_pc = context_transform.transform(data.context)
    resp_val = responsibilities(gmm, z_scaled[val_mask])
    component_mass = resp_train.mean(axis=0)
    context_model, context_info = fit_context_weight_model(
        context_pc,
        train_mask,
        val_mask,
        resp_train,
        resp_val,
        component_mass,
        cfg,
        paths["context_dir"],
    )
    context_model.transform = context_transform
    save_context_weight_bundle(context_model, context_transform, paths["context_dir"], cfg)
    (paths["context_dir"] / "conditioning_summary.json").write_text(
        json.dumps(
            {
                **context_info,
                "context_definition": (
                    f"{cfg.n_cell_pcs} MERFISH PCs + {cfg.n_gene_pcs} AGEA PCs; "
                    "TRAIN-only StandardScaler; mixture weights gamma only"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    knn = _build_knn(data, z_scaled, cfg)
    knn.save_bank(paths["knn"])

    return UnitModelBundle(
        cfg=cfg,
        autoencoder=ae,
        latent_scaler=scaler,
        gmm=gmm,
        context_model=context_model,
        knn_decoder=knn,
        data=data,
        latents=latents,
        z_scaled=z_scaled,
        model_root=paths["root"],
    )


def load_unit_model(
    cfg: Config,
    *,
    source: str = "hub",
    data=None,
    token: str | None = None,
    revision: str = "main",
) -> UnitModelBundle:
    """Load the final pretrained unit model from the Hub or a local model directory."""
    set_seed(cfg.seed)
    if source not in {"hub", "local"}:
        raise ValueError("source must be 'hub' or 'local'")

    if source == "hub":
        from .release import download_unit_release
        requested_data_dir = Path(cfg.prepared_data_dir)
        requested_output_dir = Path(cfg.output_dir)
        requested_model_dir = Path(cfg.model_dir)
        device = cfg.device
        release_root = download_unit_release(cfg.repo_id, revision=revision, token=token)
        model_dir = release_root / "models" / "unit"
        cfg = Config.from_json(model_dir / "config.json", device=device)
        cfg.prepared_data_dir = requested_data_dir
        cfg.output_dir = requested_output_dir
        cfg.model_dir = requested_model_dir
    else:
        paths = _model_paths(cfg)
        model_dir = paths["root"]
        release_root = None

    if data is None:
        data = prepare_unit_data(cfg)
    else:
        cfg.waveform_shape = tuple(data.waveforms.shape[1:])
        cfg.acg_shape = tuple(data.acgs.shape[1:])
        cfg.stpc_shape = tuple(data.stpc.shape[1:])

    if source == "hub":
        ae_path = model_dir / "autoencoder.pt"
        scaler_path = model_dir / "shared_latent_scaler.joblib"
        gmm_path = model_dir / "global_gmm.joblib"
        context_dir = model_dir
        knn_path = model_dir / "knn_bank.npz"
    else:
        paths = _model_paths(cfg)
        ae_path = paths["ae_dir"] / checkpoint_name(cfg)
        scaler_path = paths["scaler"]
        gmm_path = paths["gmm_dir"] / "global_gmm.joblib"
        context_dir = paths["context_dir"]
        knn_path = paths["knn"]

    ae, _, _ = load_autoencoder_file(ae_path, cfg)
    scaler = joblib.load(scaler_path)
    gmm = joblib.load(gmm_path)
    context_model, _ = load_context_weight_bundle(data.context, context_dir, cfg)
    knn = EmpiricalKNNDecoder.load_bank(knn_path, k=cfg.knn_decoder_k)
    latents = encode_all(ae, data, cfg)
    z_scaled = scaler.transform(latents["joint"]).astype(np.float32)

    return UnitModelBundle(
        cfg=cfg,
        autoencoder=ae,
        latent_scaler=scaler,
        gmm=gmm,
        context_model=context_model,
        knn_decoder=knn,
        data=data,
        latents=latents,
        z_scaled=z_scaled,
        model_root=model_dir,
    )


@torch.no_grad()
def basic_test(bundle: UnitModelBundle) -> dict:
    """Small held-out sanity report with no publication diagnostics.

    Reports AE reconstruction error, conditional-vs-unconditional latent NLL,
    and distances from held-out latents to the released TRAIN kNN bank.
    """
    data = bundle.data
    cfg = bundle.cfg
    z = bundle.z_scaled
    test = np.flatnonzero(data.split == 2)
    if len(test) == 0:
        raise RuntimeError("No held-out TEST units are available")

    ids = test[: min(len(test), 4096)]
    w = torch.from_numpy(data.waveforms[ids]).to(cfg.device)
    a = torch.from_numpy(data.acgs[ids]).to(cfg.device)
    s = torch.from_numpy(data.stpc[ids]).to(cfg.device)
    lat = bundle.autoencoder.encode(w, a, s)
    rec = bundle.autoencoder.decode(lat)

    reconstruction = {
        "waveform_mse": float(F.mse_loss(rec["waveform"], w).cpu()),
        "acg_mse": float(F.mse_loss(rec["acg"], a).cpu()),
        "stpc_mse": float(F.mse_loss(rec["stpc"], s).cpu()),
        "n_test_units": int(len(ids)),
    }

    conditional_nll = float(-np.mean(conditional_log_prob(z, test, bundle.gmm, bundle.context_model)))
    global_model = GlobalWeightModel(bundle.gmm.weights_, len(data.waveforms))
    unconditional_nll = float(-np.mean(conditional_log_prob(z, test, bundle.gmm, global_model)))
    knn_summary = bundle.knn_decoder.distance_summary(z[ids])

    return {
        "model": "context_weights_k25_knn20",
        "split": "held-out test PIDs",
        "reconstruction": reconstruction,
        "latent_nll": {
            "conditional_k25": conditional_nll,
            "unconditional_k25": unconditional_nll,
            "conditional_improvement": unconditional_nll - conditional_nll,
            "lower_is_better": True,
        },
        "knn20": knn_summary,
        "sanity_checks": {
            "all_metrics_finite": bool(
                np.isfinite(list(reconstruction.values())[:-1]).all()
                and np.isfinite(conditional_nll)
                and np.isfinite(unconditional_nll)
                and np.isfinite(knn_summary["nearest_distance_median"])
            ),
            "conditional_beats_unconditional": bool(conditional_nll < unconditional_nll),
        },
    }
