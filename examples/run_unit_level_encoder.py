from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import torch
from sklearn.metrics import r2_score

from ephysatlas.unit_level_encoder.config import Config
from ephysatlas.unit_level_encoder.data import prepare_data, set_seed
from ephysatlas.unit_level_encoder.gmm_models import load_point_transformer_gmm, evaluate_nll
from ephysatlas.unit_level_encoder.hf_io import (
    download_json,
    download_unit_artifacts,
    download_unit_data,
    publish_stage,
    stage_unit_release,
)
from ephysatlas.unit_level_encoder.model import MultimodalAutoencoder
from ephysatlas.unit_level_encoder.prepare_latest_cells_encoder_data import prepare_latest_cells_encoder_data
from ephysatlas.unit_level_encoder.train import collect_mean_predictions, encode_all, fit_and_evaluate, train_autoencoder


@dataclass
class RunConfig:
    repo_id: str = "AlonSaguy/ephys-atlas-models"
    vintage: str = "2026_W26"
    token: Optional[str] = None
    private_repo: bool = False

    mode: Literal["train_all", "train_ae", "train_gmm", "evaluate"] = "train_all"

    # A published prepared dataset is the default input. Set True only when
    # intentionally regenerating it from IBL aggregates and publishing it.
    prepare_and_publish_data: bool = False
    publish_after_run: bool = True


def _load_ae(checkpoint: Path, cfg: Config) -> MultimodalAutoencoder:
    payload = torch.load(checkpoint, map_location=cfg.device, weights_only=False)
    model = MultimodalAutoencoder(cfg).to(cfg.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def _prepare_or_download_arrays(run_cfg: RunConfig, temp_data_dir: Path):
    if not run_cfg.prepare_and_publish_data:
        return download_unit_data(run_cfg.repo_id, run_cfg.vintage, run_cfg.token)

    prepare_latest_cells_encoder_data(
        root_path=Path.cwd(),
        out_dir=temp_data_dir,
        download=True,
        target_channels=20,
        overwrite_multichannel_cache=False,
        use_acg3d=True,
        use_stpc=False,
    )
    return {
        "waveforms": np.load(temp_data_dir / "waveforms.npy", allow_pickle=False),
        "acgs": np.load(temp_data_dir / "acgs.npy", allow_pickle=False),
        "context": np.load(temp_data_dir / "ctx.npy", allow_pickle=False),
        "xyz": np.load(temp_data_dir / "xyz.npy", allow_pickle=False),
        "pids": np.load(temp_data_dir / "pids.npy", allow_pickle=True),
    }


def _minimal_evaluate_existing(model_ae, data, cfg, artifacts):
    shared = encode_all(model_ae, data, cfg)
    scaler = joblib.load(artifacts["scaler"])
    standardized = scaler.transform(shared).astype(np.float32)
    pt, _, loaders, payload = load_point_transformer_gmm(artifacts["pt_gmm"], data, standardized, cfg)
    observed, predicted = collect_mean_predictions(pt, loaders[2], cfg)
    return {
        "pt_gmm": {
            "test_nll": float(evaluate_nll(pt, loaders[2], cfg)),
            "checkpoint_history": payload.get("history", {}),
        },
        "test_posterior_mean_r2_variance_weighted": float(
            r2_score(observed, predicted, multioutput="variance_weighted")
        ),
        "config": {k: v for k, v in vars(cfg).items() if k != "output_dir"},
    }


def main() -> None:
    run_cfg = RunConfig(
        mode="train_all",
        prepare_and_publish_data=True,
        publish_after_run=True,
    )
    cfg = Config()
    set_seed(cfg.seed)

    # The top-level split.json belongs to the channel/spatial encoder and is
    # authoritative for the unit model as well.
    split_manifest = download_json(run_cfg.repo_id, "split.json", run_cfg.vintage, run_cfg.token)
    base_metadata = download_json(run_cfg.repo_id, "metadata.json", run_cfg.vintage, run_cfg.token)

    with tempfile.TemporaryDirectory(prefix="ephysatlas_unit_") as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "data"
        run_dir = tmp / "run"
        stage_dir = tmp / "stage"
        data_dir.mkdir(); run_dir.mkdir()
        cfg.output_dir = run_dir

        arrays = _prepare_or_download_arrays(run_cfg, data_dir)
        arrays = {k: (v.astype(np.float32) if k != "pids" else v) for k, v in arrays.items()}
        cfg.waveform_shape = tuple(arrays["waveforms"].shape[1:])
        cfg.acg_shape = tuple(arrays["acgs"].shape[1:])
        data = prepare_data(
            arrays["waveforms"], arrays["acgs"], arrays["context"], arrays["xyz"], arrays["pids"], cfg,
            split_manifest=split_manifest,
        )
        print(
            f"units={len(data.waveforms):,} | train={(data.split==0).sum():,} "
            f"validation={(data.split==1).sum():,} test={(data.split==2).sum():,}"
        )

        artifacts = None
        if run_cfg.mode in {"train_gmm", "evaluate"}:
            artifacts = download_unit_artifacts(run_cfg.repo_id, run_cfg.vintage, run_cfg.token)
            # Released config is authoritative for architecture/hyperparameters.
            saved_cfg = json.loads(artifacts["config"].read_text(encoding="utf-8"))
            for key, value in saved_cfg.items():
                if hasattr(cfg, key) and key not in {"device", "output_dir"}:
                    setattr(cfg, key, tuple(value) if key in {"waveform_shape", "acg_shape", "cosmos_region_names"} else value)

        if run_cfg.mode in {"train_all", "train_ae"}:
            model_ae, ae_info = train_autoencoder(data, cfg)
        else:
            model_ae = _load_ae(artifacts["autoencoder"], cfg)
            ae_info = {"loaded_from_hugging_face": True}

        if run_cfg.mode == "train_ae":
            summary = {"autoencoder": ae_info, "config": {k: v for k, v in vars(cfg).items() if k != "output_dir"}}
            (run_dir / cfg.summary_name).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            print("Autoencoder training complete. PT-GMM was not trained.")
            return

        if run_cfg.mode in {"train_all", "train_gmm"}:
            if run_cfg.mode == "train_gmm":
                # Copy the downloaded AE into the temporary run so one staged
                # unit release always contains a complete model pair.
                (run_dir / cfg.ae_checkpoint_name).write_bytes(artifacts["autoencoder"].read_bytes())
            model_gmm, scaler, summary = fit_and_evaluate(model_ae, data, cfg, ae_info)
        else:
            summary = _minimal_evaluate_existing(model_ae, data, cfg, artifacts)
            print(json.dumps(summary, indent=2, default=str))
            return

        if run_cfg.publish_after_run:
            stage_unit_release(
                stage_dir, run_dir, data_dir, data, cfg, summary,
                include_data=run_cfg.prepare_and_publish_data,
                base_metadata=base_metadata,
            )
            oid = publish_stage(
                run_cfg.repo_id, run_cfg.vintage, stage_dir,
                token=run_cfg.token, private=run_cfg.private_repo,
            )
            print(f"Published unit-level release at {run_cfg.repo_id}@{run_cfg.vintage} ({oid})")


if __name__ == "__main__":
    main()
