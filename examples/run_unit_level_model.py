"""Train, publish, and sanity-check the unit-level Ephys Atlas model.

Default behavior
----------------
Running this file directly will:

1. Prepare the unit-level dataset if needed.
2. Retrain the selected unit-level model from scratch.
3. Run a basic held-out TEST-set sanity check.
4. Stage the Hugging Face unit-level release.
5. Publish it directly to:
       AlonSaguy/ephys-atlas-models
6. Replace the existing files under ``models/unit``.
7. Update:
       preprocessing/unit_stats.npz
       results/unit/summary.json
       metadata.json
8. Move the ``2026_W26`` tag to the newly updated Hub revision.

To load the pretrained or local model instead, change ``MODE`` below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ephysatlas.unit_level_encoder import (
    Config,
    load_unit_model,
    prepare_unit_data,
    train_unit_model,
)
from ephysatlas.unit_level_encoder.pipeline import basic_test
from ephysatlas.unit_level_encoder.release import (
    publish_unit_release,
    stage_unit_release,
)


# ============================================================================
# USER SETTINGS
# ============================================================================

# "train":
#     Retrain the model from scratch.
#
# "pretrained":
#     Download/load the released model from Hugging Face.
#
# "local":
#     Load an already-trained model from MODEL_DIR.
MODE = "train"


# ---------------------------------------------------------------------------
# Hugging Face release
# ---------------------------------------------------------------------------

HF_REPO_ID = "AlonSaguy/ephys-atlas-models"
VINTAGE = "2026_W26"

# Used only when loading MODE="pretrained".
HF_REVISION = "main"

# Usually leave as None and authenticate once with:
#
#     hf auth login
#
HF_TOKEN = None


# ---------------------------------------------------------------------------
# Publishing behavior
# ---------------------------------------------------------------------------

# Default is intentionally True:
# every successful training run will publish the staged release.
PUBLISH_TO_HF = True

# False = commit directly to the repository.
# True  = create a Hugging Face PR instead.
CREATE_HF_PR = False

# Move the existing VINTAGE tag to the newly published commit.
#
# This is useful because other ephys-atlas code can then continue loading:
#
#     revision="2026_W26"
#
# and receive the new unit-level model rather than the previous release.
RETAG_VINTAGE = True


# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------

PREPARED_DATA_DIR = Path("unit_level_model_data/prepared_data")
MODEL_DIR = Path("unit_level_model_checkpoints")
OUTPUT_DIR = Path("unit_level_model_results")
RELEASE_STAGING_DIR = Path("unit_level_release_staging")

# Root of the local ibleatools checkout.
# Used to record the git commit in release metadata.
CODE_REPO_DIR = Path(".")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

# False reuses the prepared data when it is already compatible.
FORCE_REPREPARE_DATA = False


def _summary(bundle, test_metrics, mode: str) -> dict:
    """Build the compact released unit-model summary."""
    data = bundle.data

    return {
        "model_name": "context_weights_k25_knn20",
        "mode": mode,
        "vintage": bundle.cfg.vintage,
        "architecture": {
            "modalities": list(bundle.cfg.active_modalities()),
            "modality_latent_dim": int(bundle.cfg.modality_latent_dim),
            "joint_latent_dim": int(bundle.cfg.latent_dim()),
            "gmm_components": 25,
            "gmm_covariance_type": "full",
            "gmm_geometry": "global means and covariance matrices",
            "context_conditioning": "mixture weights gamma only",
            "context_dim": int(
                bundle.cfg.n_cell_pcs
                + bundle.cfg.n_gene_pcs
            ),
            "context_definition": (
                f"{bundle.cfg.n_cell_pcs} MERFISH PCs + "
                f"{bundle.cfg.n_gene_pcs} AGEA PCs"
            ),
            "knn_k": 20,
            "knn_policy": (
                "distance-weighted empirical TRAIN-exemplar projection "
                "in standardized 60-D latent space"
            ),
        },
        "split_counts": {
            "train_units": int(
                np.sum(data.split == 0)
            ),
            "validation_units": int(
                np.sum(data.split == 1)
            ),
            "test_units": int(
                np.sum(data.split == 2)
            ),
            "train_pids": int(
                len(
                    np.unique(
                        data.pids[data.split == 0]
                    )
                )
            ),
            "validation_pids": int(
                len(
                    np.unique(
                        data.pids[data.split == 1]
                    )
                )
            ),
            "test_pids": int(
                len(
                    np.unique(
                        data.pids[data.split == 2]
                    )
                )
            ),
        },
        "heldout_basic_test": test_metrics,
        "invariants": [
            (
                "TEST PIDs are never used for AE/GMM/context training "
                "or as kNN retrieval exemplars."
            ),
            (
                "The K=25 full-covariance GMM means and covariance "
                "matrices are global."
            ),
            (
                "Molecular context changes only the GMM mixture "
                "weights gamma_k(x)."
            ),
            (
                "The kNN=20 stage changes phenotype projection only; "
                "it does not change the conditional latent density."
            ),
            (
                "All spatial modeling uses the configured "
                "single-hemisphere mirroring convention."
            ),
        ],
    }


def _make_config() -> Config:
    """Create the unit-model configuration used by this run."""
    return Config(
        repo_id=HF_REPO_ID,
        vintage=VINTAGE,
        prepared_data_dir=PREPARED_DATA_DIR,
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR,
        release_staging_dir=RELEASE_STAGING_DIR,
        force_reprepare_data=FORCE_REPREPARE_DATA,
    )


def _run_model(cfg: Config, data):
    """Train or load the model according to MODE."""
    if MODE == "train":
        print("\n" + "=" * 80)
        print("TRAINING UNIT-LEVEL MODEL FROM SCRATCH")
        print("=" * 80)

        return train_unit_model(
            cfg,
            data=data,
        )

    if MODE == "pretrained":
        print("\n" + "=" * 80)
        print("LOADING UNIT-LEVEL MODEL FROM HUGGING FACE")
        print("=" * 80)

        return load_unit_model(
            cfg,
            source="hub",
            data=data,
            token=HF_TOKEN,
            revision=HF_REVISION,
        )

    if MODE == "local":
        print("\n" + "=" * 80)
        print("LOADING LOCAL UNIT-LEVEL MODEL")
        print("=" * 80)

        return load_unit_model(
            cfg,
            source="local",
            data=data,
        )

    raise ValueError(
        "MODE must be one of: "
        "'train', 'pretrained', 'local'"
    )


def _save_summary(
    summary: dict,
) -> Path:
    """Save the local copy of the released summary."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT_DIR / "summary.json"

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def _stage_and_publish(
    bundle,
    summary: dict,
) -> None:
    """Stage the unit release and optionally replace the Hub release."""
    staging = stage_unit_release(
        bundle,
        summary,
        code_repo_dir=CODE_REPO_DIR,
        token=HF_TOKEN,
        staging_dir=RELEASE_STAGING_DIR,
    )

    print(
        "\nStaged Hugging Face replacement tree:"
    )
    print(staging)

    if not PUBLISH_TO_HF:
        print(
            "\nPUBLISH_TO_HF=False: "
            "release was staged but not uploaded."
        )
        return

    if MODE != "train":
        raise RuntimeError(
            "Publishing is restricted to MODE='train'. "
            "This prevents accidentally republishing a model "
            "that was merely downloaded or loaded locally."
        )

    print("\n" + "=" * 80)
    print("PUBLISHING UNIT-LEVEL RELEASE TO HUGGING FACE")
    print("=" * 80)

    print(f"Repository : {HF_REPO_ID}")
    print(f"Vintage    : {VINTAGE}")
    print(f"Create PR  : {CREATE_HF_PR}")
    print(f"Retag      : {RETAG_VINTAGE}")

    publish_unit_release(
        staging,
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        create_pr=CREATE_HF_PR,
        retag=(
            VINTAGE
            if RETAG_VINTAGE
            else None
        ),
    )

    print(
        "\nPublished unit-level release successfully."
    )

    print(
        "\nThe release publisher should have replaced:"
    )
    print("  models/unit/*")

    print(
        "\nAnd updated:"
    )
    print("  preprocessing/unit_stats.npz")
    print("  results/unit/summary.json")
    print("  metadata.json")

    if RETAG_VINTAGE:
        print(
            f"\nHugging Face tag '{VINTAGE}' now points "
            "to the updated release."
        )


def main() -> None:
    """Run the complete unit-model workflow."""
    cfg = _make_config()

    print("=" * 80)
    print("EPHYS ATLAS UNIT-LEVEL MODEL")
    print("=" * 80)
    print(f"Mode        : {MODE}")
    print(f"HF repo     : {HF_REPO_ID}")
    print(f"Vintage     : {VINTAGE}")
    print(f"Publish     : {PUBLISH_TO_HF}")
    print(f"Retag       : {RETAG_VINTAGE}")
    print(f"Model dir   : {MODEL_DIR}")
    print(f"Results dir : {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Prepared data
    # ------------------------------------------------------------------

    data = prepare_unit_data(
        cfg,
    )

    # ------------------------------------------------------------------
    # Train / load
    # ------------------------------------------------------------------

    bundle = _run_model(
        cfg,
        data,
    )

    # ------------------------------------------------------------------
    # Minimal held-out evaluation
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("HELD-OUT BASIC TEST")
    print("=" * 80)

    test_metrics = basic_test(
        bundle,
    )

    print(
        json.dumps(
            test_metrics,
            indent=2,
        )
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = _summary(
        bundle,
        test_metrics,
        MODE,
    )

    local_summary = _save_summary(
        summary,
    )

    print(
        f"\nSaved local summary: {local_summary}"
    )

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    if MODE == "train":
        _stage_and_publish(
            bundle,
            summary,
        )

    elif PUBLISH_TO_HF:
        print(
            "\nNOTE: PUBLISH_TO_HF=True, but publishing is intentionally "
            "skipped because MODE is not 'train'."
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
