from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from ephysatlas import model_registry

from .config import UNIT_MODEL_REPO_ID, Config
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
from .knn_decoder import (
    COMPONENT_FEATURE_SEED_OFFSET,
    EmpiricalKNNDecoder,
    component_feature_expectations,
)
from .prepare_data import channel_context_sha1, prepare_latest_cells_encoder_data
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
    # [n_components, n_features] E[kNN phenotype feature | component]; see compute_component_features.
    component_features: np.ndarray | None = None


def prepare_unit_data(cfg: Config, split_manifest: dict | None = None):
    """Create/load the prepared unit arrays required by training and evaluation.

    Raw IBL cell aggregates are downloaded under ``cfg.data_dir`` (needs ONE/S3 access) and the
    prepared arrays are cached in ``cfg.prepared_data_dir``.

    Args:
        cfg: Unit-model configuration.
        split_manifest: Probe split to use. Defaults to the channel release's ``split.json`` for
            ``cfg.vintage``; pass a released unit model's own split to reproduce it exactly.
    """
    data_dir = Path(cfg.prepared_data_dir)
    expected_ctx_dim = int(cfg.n_cell_pcs) + int(cfg.n_gene_pcs)
    required = [
        "waveforms.npy",
        "acgs.npy",
        "stpc.npy",
        "ctx.npy",
        "xyz.npy",
        "pids.npy",
        "cosmos.npy",
        "allen.npy",
        "waveform_features.npy",
        "waveform_feature_names.json",
        "latest_cells_encoder_manifest.json",
    ]
    have_all = all((data_dir / name).exists() for name in required)
    current = False
    if have_all:
        try:
            manifest = json.loads(
                (data_dir / "latest_cells_encoder_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            ctx_shape = np.load(data_dir / "ctx.npy", mmap_mode="r").shape
            current = (
                len(ctx_shape) == 2
                and int(ctx_shape[1]) == expected_ctx_dim
                and manifest.get("context_type") == "merfish_agea_pca"
                and int(manifest.get("n_cell_pcs", -1)) == int(cfg.n_cell_pcs)
                and int(manifest.get("n_gene_pcs", -1)) == int(cfg.n_gene_pcs)
                and str(manifest.get("context_vintage")) == str(cfg.vintage)
                # The contexts must come from exactly the volumes of cfg.channel_model.
                and manifest.get("context_volumes_sha1")
                == channel_context_sha1(cfg.channel_model, cfg.vintage)
            )
        except Exception:
            current = False

    if not (have_all and current and not cfg.force_reprepare_data):
        if not cfg.prepare_data_if_missing and not have_all:
            raise FileNotFoundError(f"Missing prepared arrays in {data_dir}")
        prepare_latest_cells_encoder_data(
            root_path=Path(cfg.data_dir),
            out_dir=data_dir,
            project=cfg.project,
            download=True,
            target_channels=20,
            overwrite_multichannel_cache=False,
            use_acg3d=True,
            use_stpc=True,
            stpc_window_ms=80.0,
            allow_peak_fallback=False,
            channel_model=cfg.channel_model,
            context_vintage=cfg.vintage,
            n_cell_pcs=cfg.n_cell_pcs,
            n_gene_pcs=cfg.n_gene_pcs,
            mirror_x_to_single_hemisphere=cfg.mirror_x_to_single_hemisphere,
            mirror_x_sign=cfg.mirror_x_sign,
        )

    data = load_prepared_data(data_dir, cfg, split_manifest=split_manifest)
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
        "component_features": root / model_registry.UNIT_COMPONENT_FEATURES_FILE,
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
        raise RuntimeError(
            "Waveform feature order mismatch between prepared data and model-space extractor"
        )
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


def compute_component_features(
    gmm, knn: EmpiricalKNNDecoder, cfg: Config
) -> np.ndarray:
    """E[kNN phenotype feature | GMM component], with the settings the published figures use."""
    return component_feature_expectations(
        gmm,
        knn,
        n_samples=int(cfg.feature_slice_component_mc_samples),
        seed=int(cfg.feature_slice_seed) + COMPONENT_FEATURE_SEED_OFFSET,
    )


def save_component_features(
    path: Path, features: np.ndarray, feature_names, cfg: Config
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(features, np.float32),
        feature_names=np.asarray(list(feature_names), dtype="U"),
        n_samples=np.asarray(int(cfg.feature_slice_component_mc_samples), np.int64),
        seed=np.asarray(
            int(cfg.feature_slice_seed) + COMPONENT_FEATURE_SEED_OFFSET, np.int64
        ),
    )
    return path


def load_component_features(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        return (
            payload["features"].astype(np.float32),
            payload["feature_names"].astype(str).tolist(),
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

    # The released model: K=cfg.gmm_components (25) full-covariance components.
    gmm, _, z_scaled, resp_train, _ = fit_global_gmm(
        latents["joint"], train_mask, cfg, paths["gmm_dir"], scaler=scaler
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
    save_context_weight_bundle(
        context_model, context_transform, paths["context_dir"], cfg
    )
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
    component_features = compute_component_features(gmm, knn, cfg)
    save_component_features(
        paths["component_features"], component_features, knn.feature_names, cfg
    )

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
        component_features=component_features,
    )


def load_unit_model(
    cfg: Config,
    *,
    source: str = "hub",
    data=None,
    revision: str | None = None,
    repo_id: str = UNIT_MODEL_REPO_ID,
    release_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> UnitModelBundle:
    """Load the unit model together with its dataset, latents and standardized latents.

    Args:
        cfg: Runtime configuration. Its paths (``data_dir``, ``prepared_data_dir``, ...) and
            ``device`` are kept; for a published model every scientific setting is replaced by
            the release's own ``config.json``.
        source: ``"hub"`` downloads ``repo_id`` at ``revision`` from the Hugging Face Hub;
            ``"release"`` reads a local published release directory (``release_dir``);
            ``"local"`` reads the training layout written by :func:`train_unit_model` under
            ``cfg.model_dir``.
        data: Prepared :class:`UnitData`. Prepared (downloaded from IBL S3 when missing) if None.
        revision: Hub tag to pin; defaults to ``cfg.vintage``.
        repo_id: Hugging Face repository of the unit model.
        release_dir: Local release directory, for ``source="release"``.
        cache_dir: Hub download cache.
    """
    set_seed(cfg.seed)
    if source not in {"hub", "release", "local"}:
        raise ValueError("source must be 'hub', 'release' or 'local'")

    if source in {"hub", "release"}:
        from ephysatlas.models.unit_encoder import UnitEncoder

        if source == "hub":
            model_dir = model_registry.resolve_model(
                repo_id, revision=revision or cfg.vintage, cache_dir=cache_dir
            )
        else:
            if release_dir is None:
                raise ValueError("source='release' requires release_dir")
            model_dir = Path(release_dir)
            model_registry.verify_checksums(model_dir, missing_ok=True)
        encoder = UnitEncoder(model_dir, device=cfg.device)
        released = encoder.cfg
        # Keep the caller's runtime locations; everything scientific comes from the release.
        for key in ("data_dir", "prepared_data_dir", "model_dir", "output_dir"):
            setattr(released, key, getattr(cfg, key))
        # Sample the unit contexts from the volumes this release ships -- the ones it was trained
        # with -- rather than from whatever the channel repository holds today.
        released.channel_model = str(encoder.context_dir)
        if data is None:
            data = prepare_unit_data(released, split_manifest=encoder.split())
        return encoder.bundle(data)

    paths = _model_paths(cfg)
    if data is None:
        data = prepare_unit_data(cfg)
    else:
        cfg.waveform_shape = tuple(data.waveforms.shape[1:])
        cfg.acg_shape = tuple(data.acgs.shape[1:])
        cfg.stpc_shape = tuple(data.stpc.shape[1:])

    ae, _, _ = load_autoencoder_file(paths["ae_dir"] / checkpoint_name(cfg), cfg)
    scaler = joblib.load(paths["scaler"])
    gmm = joblib.load(paths["gmm_dir"] / "global_gmm.joblib")
    context_model, _ = load_context_weight_bundle(
        data.context, paths["context_dir"], cfg
    )
    knn = EmpiricalKNNDecoder.load_bank(paths["knn"], k=cfg.knn_decoder_k)
    if paths["component_features"].exists():
        component_features, _ = load_component_features(paths["component_features"])
    else:
        component_features = compute_component_features(gmm, knn, cfg)
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
        model_root=paths["root"],
        component_features=component_features,
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

    conditional_nll = float(
        -np.mean(conditional_log_prob(z, test, bundle.gmm, bundle.context_model))
    )
    global_model = GlobalWeightModel(bundle.gmm.weights_, len(data.waveforms))
    unconditional_nll = float(
        -np.mean(conditional_log_prob(z, test, bundle.gmm, global_model))
    )
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
            "conditional_beats_unconditional": bool(
                conditional_nll < unconditional_nll
            ),
        },
    }
