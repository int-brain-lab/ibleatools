"""Train the unit-level encoder (Model family 3) from scratch and write a publish-ready model.

The output directory this writes **is** the release layout: the four canonical files at the
model-dir root and the manifest (``ephysatlas_model.json``) as the single source of truth. There is
no ``meta.yaml`` scaffold, no ``pt_gmm/`` subdirectory and no ``best_*`` working names -- so the
directory loads through ``load_pretrained`` unchanged and is ready for
``scripts/publish_model_to_hf.py`` (which only adds the card, the golden example and checksums).

Pipeline (three stages, matching the manifest's ``artifacts`` roles):

    prepare arrays  ->  train_autoencoder    ->  autoencoder.pt              [stage 1]
    (on disk)           fit_and_evaluate     ->  shared_latent_scaler.joblib
                        (== the GMM trainer)     unconditional_gmm...joblib   [stage 2]
                                                 point_transformer_gmm.pt

Stage 1 (:func:`ephysatlas.unit_level_encoder.train.train_autoencoder`) learns the per-unit
waveform+ACG phenotype. Stage 2 (:func:`ephysatlas.unit_level_encoder.train.fit_and_evaluate`,
which calls ``fit_point_transformer_gmm``) fits the Gaussian mixture over those latents. Both
trainers are used unchanged; they now write the canonical filenames straight into the model-dir root
(their ``Config`` defaults point at ``UNIT_AE_FILE`` / ``UNIT_GMM_FILE`` and ``fit_and_evaluate``
writes the GMM stage at the root), so this orchestrator only feeds them and writes the manifest.

Inputs come from :mod:`ephysatlas.unit_level_encoder.prepare_latest_cells_encoder_data`, which
writes ``waveforms.npy``, ``acgs.npy``, ``ctx.npy``, ``xyz.npy`` and ``pids.npy`` into a directory.
So the "update features or volumes, then retrain" loop is:

    1. python -m ephysatlas.unit_level_encoder.prepare_latest_cells_encoder_data --out-dir <arrays>
    2. python training/train_unit_encoder.py --data-dir <arrays> --out-dir <model_dir> --vintage 2026_W40
    3. python scripts/publish_model_to_hf.py --model-dir <model_dir> --features <arrays> --method unit [...]

This is a training script: it pulls torch (and, through the data prep, ONE/S3), so it lives under
``training/`` rather than in the installed package. Torch is imported lazily inside the functions
that need it: the region classifier imports xgboost at module scope and the two segfault together on
macOS arm64, so nothing here may pull torch at import time (same rule as the rest of the
unit-encoder code).
"""

# %%
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ephysatlas import model_registry

logger = logging.getLogger(__name__)

# Default S3/ONE project recorded in the manifest's data_source. The atlas arrays a released model
# reads at load come from here; training itself reads the local prepared arrays instead.
PROJECT_DEFAULT = "ibl_neuropixel_brainwide_01"

# The arrays the data-prep step writes and that prepare_data consumes, in order.
# `ctx` is the anatomical context matrix (the "volumes" side of the inputs); `xyz` the atlas
# coordinates. `pids` is object-typed (UUID strings), so it alone is pickled.
_ARRAY_FILES = {
    "waveforms": "waveforms.npy",
    "acgs": "acgs.npy",
    "context": "ctx.npy",
    "xyz": "xyz.npy",
    "pids": "pids.npy",
}


# %%
def load_prepared_arrays(data_dir: Path) -> dict:
    """Load the per-unit arrays written by ``prepare_latest_cells_encoder_data``.

    Args:
        data_dir (Path): Directory holding ``waveforms.npy``, ``acgs.npy``, ``ctx.npy``,
            ``xyz.npy`` and ``pids.npy``.

    Returns:
        dict: ``{"waveforms", "acgs", "context", "xyz", "pids"}`` as numpy arrays.

    Raises:
        FileNotFoundError: If any expected array is missing, naming the offender.
    """
    data_dir = Path(data_dir)
    arrays = {}
    for key, name in _ARRAY_FILES.items():
        path = data_dir.joinpath(name)
        if not path.exists():
            raise FileNotFoundError(
                f"{data_dir} is missing {name}. Run "
                f"`python -m ephysatlas.unit_level_encoder.prepare_latest_cells_encoder_data "
                f"--out-dir {data_dir}` first."
            )
        # Only pids is object-typed (UUID strings); loading the rest with allow_pickle would
        # silently accept a corrupted/pickled float array.
        arrays[key] = np.load(path, allow_pickle=(key == "pids"))
    logger.info(
        f"loaded {len(arrays['pids'])} units over "
        f"{len(np.unique(arrays['pids'].astype(str)))} probes from {data_dir}"
    )
    return arrays


# %%
def make_split_manifest(
    pids,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    """Derive a deterministic train/validation/test split over whole probes.

    Splitting on the *probe* (pid), not the unit, is what keeps every unit of a held-out
    insertion out of training -- ``prepare_data`` then enforces this with a hard leakage check.

    Prefer passing the spatial encoder's authoritative ``split.json`` via ``--split-json`` so the
    unit and channel models hold out the same insertions (``_release.check_split_agreement``
    relies on that). This fallback is for a standalone unit run with no channel model to defer to.

    Args:
        pids (Sequence[str]): Per-unit insertion pids (repeats expected).
        test_frac (float): Fraction of probes held out for test.
        val_frac (float): Fraction of probes held out for validation.
        seed (int): Shuffle seed, so the split is reproducible.

    Returns:
        dict: ``{"train_pids", "validation_pids", "test_pids"}`` -- the shape
        ``split_probes_from_manifest`` expects.
    """
    unique = sorted({str(p) for p in np.asarray(pids)})
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    # max(1, ...): a from-scratch split must have at least one test probe, or prepare_data refuses
    # to build the split (the held-out set must be non-empty).
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    manifest = {
        "test_pids": unique[:n_test],
        "validation_pids": unique[n_test : n_test + n_val],
        "train_pids": unique[n_test + n_val :],
    }
    logger.info(
        f"derived split: {len(manifest['train_pids'])} train / "
        f"{len(manifest['validation_pids'])} val / {len(manifest['test_pids'])} test probes"
    )
    return manifest


# %%
def _build_meta(cfg, vintage: str, project: str) -> dict:
    """Assemble the UPPER_CASE training metadata that ``write_manifest`` reads.

    This is the ``meta`` dict ``_blocks_unit_encoding`` consumes -- built in memory and handed to
    :func:`write_unit_release`, never written to a ``meta.yaml`` on disk. ``MODEL_CLASS`` is the
    bare class name because that is what ``model_registry.MODEL_CLASS_TASKS`` keys on for this
    family. Mirrors ``scripts/repackage_release_from_hf.build_unit_meta`` so a from-scratch run and
    a repackaged legacy release produce the same manifest.
    """
    return {
        "MODEL_CLASS": "MultimodalAutoencoder",
        "VINTAGE": str(vintage),
        "LATENT_DIM": int(cfg.shared_latent_dim),
        "GMM_COMPONENTS": int(cfg.gmm_components),
        "PROJECT": str(project),
        "RANDOM_SEED": int(cfg.seed),
    }


# %%
def write_unit_release(
    path_model: Path,
    meta: dict,
    *,
    split_manifest: dict = None,
    method: str = "gmm",
) -> dict:
    """Write the manifest (and ``split.json``) for a unit-encoder release, meta-free.

    The two-stage trainers already wrote the four canonical files at ``path_model`` root
    (``autoencoder.pt``, ``point_transformer_gmm.pt``, ``shared_latent_scaler.joblib``,
    ``unconditional_gmm_train_only.joblib``): the training output IS the release layout, so this
    finalize step only records what inference needs -- the manifest as the single source of truth,
    and the held-out split for provenance. There is no ``meta.yaml`` scaffold and no staging copy.

    Mirrors ``training/train_spatial_encoder.write_spatial_release``: the manifest is written
    straight from the in-memory ``meta`` dict via :func:`model_registry.write_manifest`, with no
    ``meta.yaml`` round trip.

    Args:
        path_model (Path): Model-dir root the trainers wrote into.
        meta (dict): UPPER_CASE training metadata (see :func:`_build_meta`) that
            ``_blocks_unit_encoding`` reads.
        split_manifest (dict, optional): ``{"train_pids", "validation_pids", "test_pids"}``,
            written as ``split.json`` when given so held-out status is checkable against the
            channel model (``_release.check_split_agreement``).
        method (str, optional): Semantic label recorded in the manifest. Defaults to ``"gmm"``,
            matching the published ``ea-encoder-unit``.

    Returns:
        dict: The manifest that was written.
    """
    path_model = Path(path_model)
    path_model.mkdir(parents=True, exist_ok=True)

    # split.json records the held-out insertions themselves (not just a hash), so held-out status
    # is checkable and _release.check_split_agreement can compare against the channel model.
    if split_manifest is not None:
        path_model.joinpath(model_registry.MODEL_SPLIT_FILE).write_text(
            json.dumps(split_manifest, indent=2) + "\n"
        )

    # The manifest is the single source of truth every loader reads, written straight from the
    # values in hand -- no meta.yaml round trip. write_manifest scans the root for the checkpoints
    # actually present, so only the stages that ran are recorded.
    index = model_registry.write_manifest(
        path_model, meta, task=model_registry.TASK_UNIT_ENCODING, method=method
    )
    logger.info(f"packaged unit encoder -> {path_model} (task={index.get('task')})")
    return index


# %%
def train_unit_encoder(
    data_dir: Path,
    out_dir: Path,
    *,
    vintage: str,
    cfg=None,
    split_manifest: dict = None,
    project: str = PROJECT_DEFAULT,
    device: str = None,
) -> Path:
    """Run the full from-scratch unit-encoder pipeline and return the release directory.

    The trainers write the four canonical files directly into ``out_dir`` (the model-dir root);
    this function then writes the manifest via :func:`write_unit_release`. The output is the
    publish-ready release layout -- no working-name staging, no ``meta.yaml``.

    Args:
        data_dir (Path): Prepared-arrays directory (see :func:`load_prepared_arrays`).
        out_dir (Path): Release directory to populate with the canonical artifacts.
        vintage (str): Release tag recorded in the manifest, e.g. ``"2026_W40"``.
        cfg (Config, optional): Training config. A default :class:`Config` is built when omitted;
            ``waveform_shape``/``acg_shape`` are always overwritten from the loaded arrays so the
            config cannot disagree with the data.
        split_manifest (dict, optional): ``{"train_pids", "validation_pids", "test_pids"}``. When
            omitted, a deterministic split is derived from the pids (see :func:`make_split_manifest`).
        project (str): ONE/S3 project recorded in the manifest's ``data_source``.
        device (str, optional): Torch device override (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        Path: ``out_dir``, now a canonical, manifest-recorded model directory.
    """
    from ephysatlas.unit_level_encoder.config import Config
    from ephysatlas.unit_level_encoder.data import prepare_data
    from ephysatlas.unit_level_encoder.train import fit_and_evaluate, train_autoencoder

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_prepared_arrays(data_dir)

    cfg = cfg or Config()
    if device is not None:
        cfg.device = device
    # The config must describe the data, not the other way round: prepare_data asserts these match.
    cfg.waveform_shape = tuple(arrays["waveforms"].shape[1:])
    cfg.acg_shape = tuple(arrays["acgs"].shape[1:])
    # The trainers write the canonical files straight into the release directory -- output IS the
    # release layout, so there is no separate work dir to stage from.
    cfg.output_dir = out_dir

    if split_manifest is None:
        split_manifest = make_split_manifest(arrays["pids"], seed=cfg.seed)

    data = prepare_data(
        arrays["waveforms"],
        arrays["acgs"],
        arrays["context"],
        arrays["xyz"],
        arrays["pids"],
        cfg,
        split_manifest=split_manifest,
    )

    logger.info(f"stage 1/2: training the multimodal autoencoder on device={cfg.device}")
    model_ae, ae_outputs = train_autoencoder(data, cfg)

    logger.info("stage 2/2: fitting the point-transformer GMM over the unit latents")
    fit_and_evaluate(model_ae, data, cfg, training_outputs=ae_outputs)

    meta = _build_meta(cfg, vintage=vintage, project=project)
    write_unit_release(out_dir, meta, split_manifest=split_manifest)
    return out_dir


# %%
def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory of prepared arrays (waveforms/acgs/ctx/xyz/pids.npy).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Release directory to write the canonical model files into.",
    )
    parser.add_argument("--vintage", type=str, required=True, help="Release tag, e.g. 2026_W40.")
    parser.add_argument("--project", type=str, default=PROJECT_DEFAULT)
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Authoritative split.json with train_pids/validation_pids/test_pids. "
        "Recommended: reuse the spatial encoder's split so both models hold out the same probes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (default: cuda if available, else cpu).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
    args = _parse_args(argv)
    split_manifest = None
    if args.split_json is not None:
        split_manifest = json.loads(Path(args.split_json).read_text())
        logger.info(f"using authoritative split from {args.split_json}")
    model_dir = train_unit_encoder(
        args.data_dir,
        args.out_dir,
        vintage=args.vintage,
        split_manifest=split_manifest,
        project=args.project,
        device=args.device,
    )
    logger.info(f"done. Model written to {model_dir}")
    logger.info(
        "next: publish with `python scripts/publish_model_to_hf.py "
        f"--model-dir {model_dir} --features {args.data_dir} --method unit ...`"
    )
    return model_dir


if __name__ == "__main__":
    main()
