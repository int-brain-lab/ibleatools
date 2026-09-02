"""Repackage a coworker's *combined* Ephys Atlas release into this branch's per-family format.

The paper-figure scripts load their models through :func:`ephysatlas.load_pretrained`, which
expects a per-family repository carrying an ``ephysatlas_model.json`` manifest. The published
release ``AlonSaguy/ephys-atlas-models`` (tag ``2026_W26``) predates that format: it is a single
*combined* repo (``models/channel/`` + ``models/unit/``) whose manifest is ``metadata.json``, so
``load_pretrained`` cannot read it as-is (``SpatialEncoder``/``UnitEncoder`` raise without their
manifest).

This tool bridges the two. It downloads the combined release and emits two **local** directories,
one per family, each in this branch's format:

    <out-root>/ea-encoder-channel/   # spatial encoder (figure1/2/5, supp_fig4/5)
    <out-root>/ea-encoder-unit/      # unit-level encoder (figure3, supp_fig2)

For each family it reconstructs a training-style metadata dict from the release's own
``config.json``/``features.json`` and hands it straight to the *real*
:func:`ephysatlas.model_registry.write_manifest` -- so the synthesized
``ephysatlas_model.json`` is correct by construction, identical in shape to what the training
pipeline writes, and the output is meta-free (no ``meta.yaml`` scaffold). Checksums are written
last.

``load_pretrained("<out-root>/ea-encoder-channel")`` then loads the model. Once verified locally,
the same directories upload to ``int-brain-lab/ea-encoder-channel`` / ``ea-encoder-unit`` with
``scripts/publish_model_to_hf.py --upload`` (or the ``--upload`` flag here).

Usage::

    # 1. download + repackage locally (no token needed for a public source repo)
    python scripts/repackage_release_from_hf.py --out-root /tmp/ea_release

    # 2. also stage the per-unit atlas arrays, so figure3/supp_fig2 run offline (no S3)
    python scripts/repackage_release_from_hf.py --out-root /tmp/ea_release --with-unit-data

Design note: two departures from ``publish_model_to_hf.py``. That script packages from a *training
output* directory (it already has a manifest + live feature data); this one re-wraps an
*existing release*, so it synthesizes the metadata dict and copies weights out of the combined repo.
The neighbour bank a channel ``predict`` needs is not in the release and needs the ``agg_full``
feature table to rebuild -- see ``--features``; ``figure2`` does not need it.
"""

# %%
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import ephysatlas.model_registry as model_registry

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
_logger = logging.getLogger("repackage_release_from_hf")

# The public source release, and the tag that pins the W26 vintage.
DEFAULT_SOURCE_REPO = "AlonSaguy/ephys-atlas-models"
DEFAULT_REVISION = "2026_W26"

# Per-family output directory names. write_manifest records `model_id = <dir name>`, so these
# double as the model ids and mirror the intended Hub repo names (int-brain-lab/<name>).
CHANNEL_DIR_NAME = "ea-encoder-channel"
UNIT_DIR_NAME = "ea-encoder-unit"

# The atlas arrays the unit encoder reads; shipped in the release under data/unit/ and normally
# fetched from S3 at load time. Staging them locally lets the unit figures run offline.
UNIT_ATLAS_ARRAYS = ("waveforms", "acgs", "ctx", "xyz", "pids")


# %%
# ----------------------------------------------------------------------------------------------
# Pure manifest-synthesis helpers (no I/O -- unit-tested directly).
# ----------------------------------------------------------------------------------------------
def build_channel_meta(config: dict, features: list, vintage: str) -> dict:
    """Reconstruct the metadata dict a spatial-encoder publish would have carried.

    Maps the release's ``config.json`` onto the ``UPPER_CASE`` keys
    :func:`model_registry.write_manifest` / :func:`_blocks_spatial_encoding` read. The
    architecture hints are advisory: the loader re-derives ``f_ctx``/``f_ephys`` from the
    checkpoint's normalisation buffers, so a wrong hint here cannot corrupt a load.

    Args:
        config (dict): The release ``config.json`` (expects a ``channel_level`` block).
        features (list): Ordered ephys feature names (from the release ``features.json``); this is
            the model's *output* list, and its order is hashed into the manifest.
        vintage (str): Release tag, e.g. ``"2026_W26"``.

    Returns:
        dict: An ``UPPER_CASE`` metadata dict, ready to hand to ``write_manifest``.
    """
    channel = config.get("channel_level") or {}
    arch = channel.get("architecture") or {}
    neighbors = channel.get("neighbors") or {}
    training = channel.get("training") or {}
    context = config.get("context") or {}
    n_cell = int(context.get("n_cell_pcs", 50))
    n_gene = int(context.get("n_gene_pcs", 50))
    return {
        # The bare class name is what MODEL_CLASS_TASKS keys on for this family.
        "MODEL_CLASS": "NeighborInpaintingModel",
        "VINTAGE": str(vintage),
        "FEATURES": [str(f) for f in features],
        # Context feature dimension = cell PCs + gene PCs (advisory; buffers are authoritative).
        "F_CTX": n_cell + n_gene,
        "D_MODEL": int(arch.get("d_model", 256)),
        "NHEAD": int(arch.get("nhead", 8)),
        "DEPTH": int(arch.get("depth", 2)),
        "DROP": float(arch.get("drop", 0.1)),
        "RADIUS_UM": float(neighbors.get("radius_um", 600.0)),
        "M_MAX": int(neighbors.get("m_max", 64)),
        "N_CELL_PCS": n_cell,
        "N_GENE_PCS": n_gene,
        "RANDOM_SEED": int(training.get("seed", 0)),
    }


def build_unit_meta(unit_config: dict, vintage: str, project: str) -> dict:
    """Reconstruct the metadata dict a unit-encoder publish would have carried.

    The unit wrapper reads the *real* latent dim and component count from the GMM checkpoint at
    load; the values here only populate the manifest's informational ``outputs``/``config``. The
    dataset is recorded as an S3 source (``data_source``) -- the release ships the arrays too, but
    the format's contract is weights-only-on-Hub, so the manifest keeps the S3 pointer.

    Args:
        unit_config (dict): The release ``models/unit/config.json`` (the training ``Config``).
        vintage (str): Release tag, e.g. ``"2026_W26"``.
        project (str): ONE/S3 project the atlas arrays come from.

    Returns:
        dict: An ``UPPER_CASE`` metadata dict, ready to hand to ``write_manifest``.
    """
    # The training Config may name the latent dim under either key; accept both.
    latent_dim = int(
        unit_config.get("shared_latent_dim")
        or unit_config.get("latent_dim")
        or 32
    )
    n_components = int(
        unit_config.get("gmm_components")
        or unit_config.get("n_components")
        or 16
    )
    return {
        # The bare class name is what MODEL_CLASS_TASKS keys on for this family.
        "MODEL_CLASS": "MultimodalAutoencoder",
        "VINTAGE": str(vintage),
        "LATENT_DIM": latent_dim,
        "GMM_COMPONENTS": n_components,
        "PROJECT": str(project),
        "RANDOM_SEED": int(unit_config.get("seed", 0)),
    }


# %%
# ----------------------------------------------------------------------------------------------
# I/O: download + per-family staging.
# ----------------------------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    """Read a JSON file into a dict."""
    import json

    return json.loads(Path(path).read_text())


def _copy_into(src: Path, dst_dir: Path, dst_name: str = None) -> Path:
    """Copy ``src`` into ``dst_dir`` (optionally renaming), returning the destination path."""
    src = Path(src)
    dst = Path(dst_dir).joinpath(dst_name or src.name)
    shutil.copy2(src, dst)
    _logger.info(f"   copied {src.name} -> {dst.relative_to(dst_dir.parent)}")
    return dst


def download_release(revision: str, dst: Path, source_repo: str, with_data: bool) -> Path:
    """Download the combined release into ``dst`` (classic transport, Xet disabled).

    Xet (HF's large-file transport) is unreliable in some sandboxes; disabling it routes every
    file through ``huggingface.co`` directly, which is slower but robust. The heavy per-unit atlas
    arrays under ``data/unit/`` are skipped unless ``with_data`` -- they are only needed to render
    the unit figures, not to repackage.

    Args:
        revision (str): Source tag/branch.
        dst (Path): Local directory to populate.
        source_repo (str): Source HF repo id.
        with_data (bool): Also fetch the ~1.5 GB ``data/unit/*.npy`` arrays.

    Returns:
        Path: ``dst``.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download

    allow = [
        "*.json", "README.md", ".gitattributes",
        "models/channel/*", "models/unit/*",
        "context/*.npy", "preprocessing/*.npz", "results/**",
    ]
    if with_data:
        allow.append("data/unit/*.npy")
    _logger.info(f"downloading {source_repo}@{revision} -> {dst} (with_data={with_data})")
    snapshot_download(
        source_repo, repo_type="model", revision=revision,
        local_dir=str(dst), allow_patterns=allow, max_workers=3,
    )
    return Path(dst)


def _finalize(out_dir: Path, meta: dict, method: str) -> dict:
    """Write the manifest, validate artifacts, and checksum -- in that order.

    The reconstructed ``meta`` dict is handed straight to :func:`model_registry.write_manifest`, the
    same assembler a from-scratch publish uses -- so the repackaged output is a meta-free release
    (the manifest is the single source of truth), identical in shape to a training-script output. No
    ``meta.yaml`` is written to disk.

    Args:
        out_dir (Path): The staged per-family directory (weights already copied in).
        meta (dict): The reconstructed UPPER_CASE metadata dict (see the ``build_*_meta`` helpers).
        method (str): Semantic method label recorded in the manifest.

    Returns:
        dict: The written manifest (``ephysatlas_model.json``).
    """
    # write_manifest scans the artifacts on disk and writes the manifest from the in-memory meta
    # dict -- no meta.yaml round trip, so the output carries only the manifest.
    index = model_registry.write_manifest(out_dir, meta, method=method)
    model_registry.validate_artifacts(out_dir, index)
    # Checksums last, so they cover the manifest too.
    model_registry.write_checksums(out_dir)
    model_registry.verify_checksums(out_dir, missing_ok=False)
    _logger.info(f"packaged {out_dir.name}: task={index.get('task')} model_class={index.get('model_class')}")
    return index


def stage_channel(src: Path, out_root: Path, vintage: str) -> Path:
    """Emit the channel spatial-encoder directory in this branch's format.

    Copies the weights, confidence model and context volumes to the directory root (where the
    wrappers expect bare filenames), reconstructs the metadata dict from the release config, and
    finalizes. The neighbour bank is *not* built here (it needs feature data); ``figure2`` does
    not require it, and ``predict``-based figures build it separately.

    Returns:
        Path: The staged directory.
    """
    src = Path(src)
    out = Path(out_root).joinpath(CHANNEL_DIR_NAME)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _logger.info(f"staging channel encoder -> {out}")

    _copy_into(src.joinpath("models/channel/spatial_encoder.pt"), out, model_registry.ENCODER_WEIGHTS_FILE)
    confidence = src.joinpath("models/channel/confidence_model.pt")
    if confidence.exists():
        _copy_into(confidence, out, model_registry.ENCODER_CONFIDENCE_FILE)
    # Context volumes live at the model-dir root (this branch's format, consistent with the
    # training/publish pipeline). Both the wrapper's predict() and the figures find them via the
    # manifest -- SpatialEncoder.context_dir -- so no `context/` subdir copy is needed.
    for name in model_registry.ENCODER_CONTEXT_FILES:
        _copy_into(src.joinpath("context", name), out, name)
    _copy_into(src.joinpath("split.json"), out, model_registry.MODEL_SPLIT_FILE)

    config = _read_json(src.joinpath("config.json"))
    features = _read_json(src.joinpath("features.json"))["features"]
    meta = build_channel_meta(config, features, vintage)
    _finalize(out, meta, method="transformer")
    return out


def stage_unit(src: Path, out_root: Path, vintage: str) -> Path:
    """Emit the unit-level encoder directory in this branch's format.

    Copies the autoencoder, GMM, scaler and unconditional-GMM baseline to the directory root,
    reconstructs the metadata dict from the release's unit config, and finalizes.

    Returns:
        Path: The staged directory.
    """
    src = Path(src)
    out = Path(out_root).joinpath(UNIT_DIR_NAME)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _logger.info(f"staging unit encoder -> {out}")

    for name in (
        model_registry.UNIT_AE_FILE,
        model_registry.UNIT_GMM_FILE,
        model_registry.UNIT_SCALER_FILE,
        model_registry.UNIT_UNCOND_GMM_FILE,
    ):
        source = src.joinpath("models/unit", name)
        if source.exists():
            _copy_into(source, out, name)
    # Prefer the unit-specific split (probe-level); fall back to the repo-root split.
    unit_split = src.joinpath("models/unit", model_registry.MODEL_SPLIT_FILE)
    _copy_into(unit_split if unit_split.exists() else src.joinpath("split.json"),
               out, model_registry.MODEL_SPLIT_FILE)

    unit_config = _read_json(src.joinpath("models/unit/config.json"))
    data_manifest = src.joinpath("models/unit/data_manifest.json")
    project = "ibl_neuropixel_brainwide_01"
    if data_manifest.exists():
        project = _read_json(data_manifest).get("project", project)
    meta = build_unit_meta(unit_config, vintage, project)
    _finalize(out, meta, method="gmm")
    return out


def stage_unit_data(src: Path, cache_dir: Path) -> Path:
    """Lay out the release's per-unit atlas arrays as the unit encoder's local cache.

    ``UnitEncoder._atlas_arrays`` reads ``<cache_dir>/arrays/<name>.npy`` and, when they are
    absent, falls back to a multi-GB S3 fetch via ONE. Copying the release's shipped ``data/unit``
    arrays here lets ``figure3``/``supp_fig2`` run offline. Requires the release to have been
    downloaded ``--with-unit-data``.

    Returns:
        Path: The ``arrays`` directory.
    """
    src = Path(src)
    arrays = Path(cache_dir).joinpath("arrays")
    arrays.mkdir(parents=True, exist_ok=True)
    for name in UNIT_ATLAS_ARRAYS:
        source = src.joinpath("data/unit", f"{name}.npy").resolve()
        if not source.exists():
            raise FileNotFoundError(
                f"{source} not found; re-run download with --with-unit-data to stage unit arrays"
            )
        # These arrays are large (~1.5 GB total); symlink rather than duplicate them. Fall back to
        # a copy on filesystems that cannot symlink.
        dst = arrays.joinpath(f"{name}.npy")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(source)
        except OSError:
            shutil.copy2(source, dst)
    _logger.info(f"staged unit atlas arrays under {arrays}")
    return arrays


# %%
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, required=True, help="where the per-family dirs are written")
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO, help="combined-release HF repo id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="source tag/branch to pin")
    parser.add_argument("--download-dir", type=Path, default=None,
                        help="where the source snapshot lives; default <out-root>/_source")
    parser.add_argument("--families", nargs="+", choices=["channel", "unit"], default=["channel", "unit"])
    parser.add_argument("--with-unit-data", action="store_true",
                        help="also download + stage the per-unit atlas arrays (offline unit figures)")
    parser.add_argument("--skip-download", action="store_true",
                        help="reuse an existing snapshot in --download-dir instead of fetching")
    args = parser.parse_args(argv)

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    download_dir = (args.download_dir or out_root.joinpath("_source")).resolve()

    if not args.skip_download:
        download_release(args.revision, download_dir, args.source_repo, args.with_unit_data)

    if "channel" in args.families:
        stage_channel(download_dir, out_root, args.revision)
    if "unit" in args.families:
        stage_unit(download_dir, out_root, args.revision)
        if args.with_unit_data:
            stage_unit_data(download_dir, out_root.joinpath(UNIT_DIR_NAME, "_unit_data_cache"))

    _logger.info(f"done. load with: load_pretrained('{out_root.joinpath(CHANNEL_DIR_NAME)}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
