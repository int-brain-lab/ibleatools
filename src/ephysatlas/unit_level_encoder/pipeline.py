"""Training, loading and checking the unit-level model.

The model has four fitted stages, all on the TRAIN units:

1. a multimodal autoencoder embeds a unit's waveform, 3D ACG and population coupling into a joint
   latent, which a train-only scaler standardizes (:mod:`.train`);
2. a hierarchical Gaussian mixture of the standardized latents: putative cell types and their
   sub-types, with top-down assignment (:mod:`.hierarchy`);
3. two context networks give the sub-type weights at a position from its molecular context
   (:mod:`.context_weights`);
4. the members readout represents each sub-type by its TRAIN units
   (:class:`.hierarchy.MemberReadout`).
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from ephysatlas import model_registry

from .config import UNIT_MODEL_REPO_ID, UNIT_MODEL_REVISION, Config
from .context_weights import (
    SUBTYPE_WEIGHTS_FILE,
    TRANSFORM_FILE,
    TYPE_WEIGHTS_FILE,
    HierarchicalWeightModel,
    fit_context_weight_model,
)
from .data import fit_context_transform, load_prepared_data, set_seed
from .hierarchy import HierarchicalMixture, MemberReadout, fit_hierarchical_mixture
from .prepare_data import channel_context_sha1, prepare_latest_cells_encoder_data
from .train import checkpoint_name, encode_all, load_autoencoder_file, train_autoencoder
from .waveform_features import FEATURE_NAMES, extract_generated_waveform_features

MODEL_SPACE_FEATURES_FILE = "waveform_features_model_space.npy"
TRAINING_SUMMARY_FILE = "training_summary.json"


@dataclass
class UnitModelBundle:
    """The fitted unit-level model, optionally over a prepared unit dataset.

    Attributes:
        mixture: The types and sub-types (:class:`~.hierarchy.HierarchicalMixture`).
        context_model: The sub-type weights at a position
            (:class:`~.context_weights.HierarchicalWeightModel`); ``weights(indices)`` index the
            units of ``data``.
        readout: The phenotypes of the sub-types (:class:`~.hierarchy.MemberReadout`).
        z_scaled: ``[N, latent_dim]`` standardized latents of the units of ``data``.
    """

    cfg: Config
    autoencoder: object
    latent_scaler: object
    mixture: HierarchicalMixture
    context_model: HierarchicalWeightModel
    readout: MemberReadout
    data: object | None = None
    latents: dict[str, np.ndarray] | None = None
    z_scaled: np.ndarray | None = None
    model_root: Path | None = None


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
        "waveform_channel_xy_um.npy",
        "cluster_ids.npy",
        "unit_axial_um.npy",
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
                # Prepared with the current unit waveform features.
                and list(manifest.get("waveform_feature_names", []))
                == list(FEATURE_NAMES)
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
    _set_shapes(cfg, data)
    return data


def _set_shapes(cfg: Config, data) -> None:
    cfg.waveform_shape = tuple(data.waveforms.shape[1:])
    cfg.acg_shape = tuple(data.acgs.shape[1:])
    cfg.stpc_shape = tuple(data.stpc.shape[1:])


def _model_paths(cfg: Config) -> dict[str, Path]:
    root = Path(cfg.model_dir)
    return {
        "root": root,
        "ae_dir": root / "autoencoder",
        "scaler": root / "shared_latent_scaler.joblib",
        "mixture_dir": root / "hierarchy",
        "context_dir": root / "context_weights",
        "members": root / "members.npz",
        "summary": root / TRAINING_SUMMARY_FILE,
    }


def save_model_space_waveform_features(
    cfg: Config, features, feature_names, **info
) -> Path:
    """Cache model-space waveform features under ``cfg.prepared_data_dir``, with their names."""
    path = Path(cfg.prepared_data_dir) / MODEL_SPACE_FEATURES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(features, np.float32), allow_pickle=False)
    path.with_suffix(".json").write_text(
        json.dumps({"feature_names": list(feature_names), **info}, indent=2),
        encoding="utf-8",
    )
    return path


def model_space_waveform_features(data, cfg: Config) -> np.ndarray:
    """Waveform features of the exact max-abs normalized waveforms the autoencoder sees.

    The prepared ``data.waveform_features`` come mostly from the IBL cluster table, in the
    original amplitude convention; the autoencoder, the members readout and every decoded waveform
    live in the normalized convention, so the model and the figures use these instead. Extracted
    once and cached in ``cfg.prepared_data_dir``; the cache is used only for the same units and the
    same feature names (data preparation removes it whenever it rewrites the waveforms).
    """
    path = Path(cfg.prepared_data_dir) / MODEL_SPACE_FEATURES_FILE
    names = list(data.waveform_feature_names)
    if path.exists() and path.with_suffix(".json").exists():
        cached = np.load(path, allow_pickle=False)
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (
            cached.shape == (len(data.waveforms), len(names))
            and info.get("feature_names") == names
        ):
            return cached.astype(np.float32, copy=False)
    if data.channel_xy_um is None:
        raise RuntimeError(
            "The prepared unit data has no channel positions (waveform_channel_xy_um.npy); "
            "re-prepare it with prepare_unit_data."
        )
    print("[features] extracting model-space waveform features once ...")
    features, extracted, report = extract_generated_waveform_features(
        data.waveforms,
        data.channel_xy_um,
        sampling_rate_hz=cfg.waveform_sampling_rate_hz,
        return_report=True,
    )
    if list(extracted) != names:
        raise RuntimeError(
            f"Waveform feature order mismatch between prepared data ({names}) and the "
            f"model-space extractor ({list(extracted)}); re-prepare the unit data."
        )
    save_model_space_waveform_features(
        cfg,
        features,
        extracted,
        definition="features of the max-abs normalized waveforms.npy, extracted like decoded waveforms",
        extractor_report=report,
    )
    return features.astype(np.float32, copy=False)


def fit_latent_scaler(
    z_joint: np.ndarray, train_mask: np.ndarray, path: Path | None = None
):
    scaler = StandardScaler().fit(np.asarray(z_joint)[train_mask])
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, path)
    return scaler


def train_unit_model(
    cfg: Config, data=None, *, autoencoder_checkpoint: Path | str | None = None
) -> UnitModelBundle:
    """Train the unit-level model and write it under ``cfg.model_dir``.

    Args:
        cfg: Unit-model configuration.
        data: Prepared :class:`UnitData`; prepared from ``cfg`` when None.
        autoencoder_checkpoint: An autoencoder checkpoint to reuse instead of training one, e.g. the
            ``autoencoder.pt`` of a release trained on the same units and split. Every later
            stage is fitted anew.
    """
    set_seed(cfg.seed)
    data = prepare_unit_data(cfg) if data is None else data
    _set_shapes(cfg, data)
    paths = _model_paths(cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)
    train, val = data.split == 0, data.split == 1

    ae_cfg = copy.deepcopy(cfg)
    ae_cfg.feature_fidelity = False
    ae_cfg.use_acg = True
    ae_cfg.use_stpc = True
    if autoencoder_checkpoint is None:
        ae, _, _ = train_autoencoder(data, ae_cfg, paths["ae_dir"])
    else:
        ae, _, _ = load_autoencoder_file(Path(autoencoder_checkpoint), ae_cfg)
        paths["ae_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(autoencoder_checkpoint, paths["ae_dir"] / checkpoint_name(cfg))
    latents = encode_all(ae, data, ae_cfg)
    scaler = fit_latent_scaler(latents["joint"], train, paths["scaler"])
    z_scaled = scaler.transform(latents["joint"]).astype(np.float32)

    mixture = fit_hierarchical_mixture(z_scaled[train], cfg, paths["mixture_dir"])
    p_type, p_subtype = mixture.posterior(z_scaled)

    transform = fit_context_transform(data, cfg)
    context_pc = transform.transform(data.context)
    nets, infos = [], {}
    for name, resp in (("types", p_type), ("subtypes", p_subtype)):
        net, info = fit_context_weight_model(
            context_pc,
            train,
            val,
            resp[train],
            resp[val],
            resp[train].mean(axis=0),
            cfg,
        )
        net.transform = transform
        nets.append(net)
        infos[name] = info
    context_model = HierarchicalWeightModel(*nets, mixture.parent)
    context_model.save(paths["context_dir"], cfg)

    features = model_space_waveform_features(data, cfg)
    readout = MemberReadout.fit(
        p_subtype[train], features[train], data.waveform_feature_names
    )
    readout.save(paths["members"])

    types, subtypes = mixture.assign(z_scaled[train])
    paths["summary"].write_text(
        json.dumps(
            {
                "autoencoder": str(autoencoder_checkpoint or "trained"),
                "n_types": mixture.n_types,
                "n_subtypes": mixture.n_subtypes,
                "train_units_per_type": np.bincount(
                    types, minlength=mixture.n_types
                ).tolist(),
                "train_units_per_subtype": np.bincount(
                    subtypes, minlength=mixture.n_subtypes
                ).tolist(),
                "context_definition": (
                    f"{cfg.n_cell_pcs} MERFISH PCs + {cfg.n_gene_pcs} AGEA PCs; TRAIN-only "
                    "StandardScaler; mixture weights only"
                ),
                "context_networks": infos,
                "readout": {
                    "method": "members",
                    "features": list(readout.feature_names),
                    "stored_members_per_subtype": np.diff(readout.offsets).tolist(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return UnitModelBundle(
        cfg=cfg,
        autoencoder=ae,
        latent_scaler=scaler,
        mixture=mixture,
        context_model=context_model,
        readout=readout,
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
        revision: Hub tag to pin; defaults to :data:`~.config.UNIT_MODEL_REVISION`.
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
                repo_id, revision=revision or UNIT_MODEL_REVISION, cache_dir=cache_dir
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
        _set_shapes(cfg, data)
    ae, _, _ = load_autoencoder_file(paths["ae_dir"] / checkpoint_name(cfg), cfg)
    scaler = joblib.load(paths["scaler"])
    mixture = HierarchicalMixture.load(paths["mixture_dir"])
    context_model = HierarchicalWeightModel.load(
        mixture.parent,
        cfg,
        data.context,
        transform=paths["context_dir"] / TRANSFORM_FILE,
        types=paths["context_dir"] / TYPE_WEIGHTS_FILE,
        subtypes=paths["context_dir"] / SUBTYPE_WEIGHTS_FILE,
    )
    readout = MemberReadout.load(paths["members"])
    if list(readout.feature_names) != list(data.waveform_feature_names):
        raise ValueError(
            f"the members readout describes units by {readout.feature_names}, the prepared data "
            f"by {list(data.waveform_feature_names)}; retrain the model on the current data"
        )
    latents = encode_all(ae, data, cfg)
    z_scaled = scaler.transform(latents["joint"]).astype(np.float32)
    return UnitModelBundle(
        cfg=cfg,
        autoencoder=ae,
        latent_scaler=scaler,
        mixture=mixture,
        context_model=context_model,
        readout=readout,
        data=data,
        latents=latents,
        z_scaled=z_scaled,
        model_root=paths["root"],
    )


@torch.no_grad()
def basic_test(bundle: UnitModelBundle) -> dict:
    """Small held-out sanity report on the TEST units.

    Reports the autoencoder reconstruction error, the latent negative log-likelihood under the
    context-dependent sub-type weights and under the brain-wide weights, and checks that the
    top-down assignment keeps every sub-type inside its type.
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

    mixture = bundle.mixture
    conditional_nll = float(
        -np.mean(mixture.log_prob(z[test], bundle.context_model.weights(test)))
    )
    brain_wide = np.repeat(mixture.subtypes.weights_[None], len(test), axis=0)
    unconditional_nll = float(-np.mean(mixture.log_prob(z[test], brain_wide)))
    types, subtypes = mixture.assign(z[test])

    return {
        "model": f"hierarchical_gmm_{mixture.n_types}_types_{mixture.n_subtypes}_subtypes",
        "split": "held-out test PIDs",
        "reconstruction": reconstruction,
        "latent_nll": {
            "context_weights": conditional_nll,
            "brain_wide_weights": unconditional_nll,
            "context_improvement": unconditional_nll - conditional_nll,
            "n_test_units": int(len(test)),
            "lower_is_better": True,
        },
        "sanity_checks": {
            "all_metrics_finite": bool(
                np.isfinite(list(reconstruction.values())[:-1]).all()
                and np.isfinite(conditional_nll)
                and np.isfinite(unconditional_nll)
            ),
            "context_beats_brain_wide": bool(conditional_nll < unconditional_nll),
            "subtypes_inside_types": bool(np.all(mixture.parent[subtypes] == types)),
        },
    }
