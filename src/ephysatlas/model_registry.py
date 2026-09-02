"""Resolve trained ephys-atlas models from a model hub or from AWS S3, and package them.

Two backends fetch a model directory, both exposing
``fetch(model_id, revision, cache_dir) -> Path``:

* :class:`HFModelSource` -- Hugging Face Hub. Public repos need no authentication, which is
  what lets people outside the IBL use these models at all.
* :class:`S3ModelSource` -- the historical private bucket. Requires an authenticated ONE
  client to mint credentials.

:func:`resolve_model` picks between them. Making Hugging Face the only source later is a
change to the default of its ``source`` argument, not a rewrite of the callers.

This module also builds ``ephysatlas_model.json``, the *publication contract*. It is
deliberately separate from ``meta.yaml``: ``meta.yaml`` is a training artifact whose shape
follows whatever the trainer wrote, while the manifest normalises every model family into one
shape so a single load path can serve them all. It is a discriminated union -- a shared core
plus ``inputs``/``outputs``/``config`` blocks interpreted according to ``task``:

    {"task": "region-classification", "granularity": "channel", "model_class": ...,
     "artifacts": {...}, "inputs": {"index": [...], "features": [...]},
     "outputs": {"kind": "categorical", "columns": [...]},
     "config": {"classes": [...], "class_acronyms": [...]}}

Supporting a new family means registering it in three places -- :data:`MODEL_CLASS_TASKS` and
:data:`TASK_BUILDERS` here, plus ``regionclassifier.MODEL_LOADERS`` -- and writing the block
builder and loader they point at. The transport layer above is model-agnostic and does not
change. ``load_model`` works from the manifest alone, so a family that never goes through
``save_model`` (the torch spatial encoder) does not need to produce a ``meta.yaml``.
"""

import fnmatch
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# S3 prefix the models have always lived under.
S3_MODEL_PREFIX = "aggregates/atlas/models"
# Name of the publication manifest inside a model directory
MODEL_MANIFEST_FILE = "ephysatlas_model.json"
# Hugging Face organisation the models are published under.
DEFAULT_HF_ORG = "int-brain-lab"
# Files kept locally but never published. `predictions.pqt` is tens of MB of out-of-fold
# predictions over real IBL insertions -- useful for reproducibility work in-house, not part
# of what a user needs to run the model.
DEFAULT_UPLOAD_IGNORE = ("predictions.pqt", ".DS_Store", "*.tmp")
# Per-file digests of everything published, so a truncated download or a file silently dropped
# by DEFAULT_UPLOAD_IGNORE fails loudly instead of surfacing later as a confusing load error.
MODEL_CHECKSUM_FILE = "checksums.json"
# The cross-validation split, published so that "was this scored on held-out probes?" is a
# checkable fact rather than a matter of trusting two scripts seeded their RNG identically.
MODEL_SPLIT_FILE = "split.json"
# Spatial-encoder artifact names. The encoder is not reconstructible from weights alone: it also
# needs the frozen PCA context volumes and a bank of training-channel features to draw
# neighbours from, neither of which can be recomputed downstream.
ENCODER_WEIGHTS_FILE = "spatial_encoder.pt"
ENCODER_CONFIDENCE_FILE = "confidence_model.pt"
ENCODER_BANK_FILE = "neighbor_bank.npz"
ENCODER_CONTEXT_FILES = ("agea_vol_pca.npy", "merfish_vol_pca.npy")

# Unit-level encoder: the canonical filenames a published release stages its checkpoints under.
# Unlike the other families it ships no recorded data -- the atlas dataset is fetched from S3.
UNIT_AE_FILE = "autoencoder.pt"
UNIT_GMM_FILE = "point_transformer_gmm.pt"
UNIT_SCALER_FILE = "shared_latent_scaler.joblib"
UNIT_UNCOND_GMM_FILE = "unconditional_gmm_train_only.joblib"
# Never hashed: the checksum file cannot cover itself; the Hub adds .gitattributes and a
# .cache/ tree to a snapshot; and the upload already drops the rest, so hashing any of it would
# make every download fail verification.
#
DEFAULT_CHECKSUM_IGNORE = DEFAULT_UPLOAD_IGNORE + (
    MODEL_CHECKSUM_FILE,
    "README.md",
    "LICENSE",
    ".gitattributes",
    ".git/*",
    ".cache/*",
)

# Task discriminators used by the manifest's "task" field.
TASK_REGION_CLASSIFICATION = "region-classification"
TASK_SPATIAL_ENCODING = "spatial-encoding"
TASK_UNIT_ENCODING = "unit-encoding"

# Which task a saved model belongs to, keyed on the MODEL_CLASS recorded in the metadata.
#
# The encoder appears twice on purpose: `save_model` records a fully-qualified class, but the
# encoder's hand-written meta.yaml files carry the bare class name, and both are in circulation.
MODEL_CLASS_TASKS = {
    "xgboost.sklearn.XGBClassifier": TASK_REGION_CLASSIFICATION,
    "ephysatlas.spatial_encoder.model.NeighborInpaintingModel": TASK_SPATIAL_ENCODING,
    "NeighborInpaintingModel": TASK_SPATIAL_ENCODING,
    # The unit-level encoder is two torch classes (autoencoder + PT-GMM). One manifest carries one
    # model_class, so the autoencoder -- the entry checkpoint -- is the dispatch key; UnitEncoder
    # loads the PT-GMM and scaler beside it.
    "ephysatlas.unit_level_encoder.model.MultimodalAutoencoder": TASK_UNIT_ENCODING,
    "MultimodalAutoencoder": TASK_UNIT_ENCODING,
}


class HFModelSource:
    """Fetch (and optionally publish) a model on the Hugging Face Hub.

    Args:
        repo_id (str, optional): Repository to use. If omitted, a ``model_id`` of the form
            ``owner/name`` is used as the repo id; otherwise fetching raises, since there is
            no sensible default once more than one model family is published.
        token (str, optional): Access token. Not needed to read public repositories.
    """

    def __init__(self, repo_id: str = None, token: str = None):
        self.repo_id = repo_id
        self.token = token

    def _resolve_repo_id(self, model_id: str) -> str:
        repo_id = self.repo_id or (model_id if "/" in model_id else None)
        if repo_id is None:
            raise ValueError(
                f"no Hugging Face repo for {model_id!r}: pass repo_id, or use an "
                f"'owner/name' model_id"
            )
        return repo_id

    def fetch(self, model_id: str, revision: str = None, cache_dir: Path = None) -> Path:
        """Download a snapshot of the repository and return its local directory."""
        # Lazy import: huggingface_hub is an optional dependency ([hub] extra), so a lite
        # install that only uses S3 must neither pay for it nor fail at import time.
        from huggingface_hub import snapshot_download

        repo_id = self._resolve_repo_id(model_id)
        logger.info(f"fetching {repo_id}@{revision or 'main'} from Hugging Face")
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
                token=self.token,
            )
        )

    def set_visibility(self, private: bool):
        """Flip the repository between private and public. Requires a write-scoped token.

        Args:
            private (bool): True to make it private, False to publish it publicly.
        """
        from huggingface_hub import HfApi

        if self.repo_id is None:
            raise ValueError("repo_id is required; pass HFModelSource(repo_id=...)")
        HfApi(token=self.token).update_repo_settings(
            repo_id=self.repo_id, repo_type="model", private=private
        )
        logger.info(
            f"{self.repo_id} is now {'private' if private else 'PUBLIC'}"
        )

    def upload(
        self,
        path_model: Path,
        revision: str = None,
        message: str = "",
        private: bool = True,
        ignore_patterns=DEFAULT_UPLOAD_IGNORE,
    ):
        """Upload a local model directory to the hub. Requires a write-scoped token.

        Creates the repository if it does not exist, pushes the folder to ``main``, then tags
        that commit with ``revision``. Uploading to ``main`` and tagging on top is deliberate:
        ``main`` always holds the recommended model, while each vintage tag is an immutable
        revision a paper can cite. Pushing straight to a vintage branch would leave ``main``
        empty, so ``load_pretrained`` without an explicit revision would find nothing.

        Args:
            path_model (Path): Local model directory to upload.
            revision (str, optional): Vintage tag to create, e.g. ``"2026_W32"``.
            message (str, optional): Commit message.
            private (bool, optional): Create the repo private. **Defaults to True** so nothing
                becomes world-readable as a side effect of running a script; publish with
                :meth:`set_visibility` once the contents have been reviewed.
            ignore_patterns (Iterable[str], optional): Glob patterns kept out of the upload.
                Defaults to :data:`DEFAULT_UPLOAD_IGNORE`.

        Returns:
            The value returned by ``HfApi.upload_folder``.

        Raises:
            ValueError: If no ``repo_id`` was configured.
        """
        from huggingface_hub import HfApi

        if self.repo_id is None:
            raise ValueError("repo_id is required to upload; pass HFModelSource(repo_id=...)")
        api = HfApi(token=self.token)
        ignore_patterns = list(ignore_patterns or [])

        api.create_repo(
            repo_id=self.repo_id, repo_type="model", private=private, exist_ok=True
        )
        logger.info(
            f"repository {self.repo_id} ready ({'private' if private else 'PUBLIC'})"
        )

        if ignore_patterns:
            logger.info(f"not uploading: {ignore_patterns}")
        commit = api.upload_folder(
            repo_id=self.repo_id,
            folder_path=str(path_model),
            commit_message=message or f"Publish {path_model.name}",
            ignore_patterns=ignore_patterns,
        )
        logger.info(f"uploaded {path_model.name} to {self.repo_id}@main")

        if revision:
            try:
                api.create_tag(repo_id=self.repo_id, tag=revision, repo_type="model")
                logger.info(f"tagged {revision}")
            except Exception as e:  # noqa: BLE001 - an existing tag must not be moved silently
                logger.warning(
                    f"could not create tag {revision!r} ({type(e).__name__}: {e}). "
                    f"If it already exists it still points at its original commit -- tags are "
                    f"meant to be immutable, so publish under a new revision or delete the tag "
                    f"deliberately."
                )
        return commit


class S3ModelSource:
    """Fetch a model from the private IBL AWS S3 bucket.

    Args:
        one: ONE client instance, used to mint temporary S3 credentials.
        overwrite (bool, optional): Re-download files that already exist.
    """

    def __init__(self, one=None, overwrite: bool = False):
        self.one = one
        self.overwrite = overwrite

    def fetch(self, model_id: str, revision: str = None, cache_dir: Path = None) -> Path:
        """Download the model folder from S3 and return its local directory."""
        from one.remote import aws

        if self.one is None:
            raise ValueError("an authenticated ONE instance is required to fetch from S3")
        if revision:
            logger.warning(f"S3 has no revisions; ignoring revision={revision!r}")
        local = Path(cache_dir).joinpath(model_id)
        s3, bucket_name = aws.get_s3_from_alyx(alyx=self.one.alyx)
        aws.s3_download_folder(
            f"{S3_MODEL_PREFIX}/{model_id}",
            local,
            s3=s3,
            bucket_name=bucket_name,
            overwrite=self.overwrite,
        )
        return local


def resolve_model(
    model_id: str,
    revision: str = None,
    source: str = "auto",
    cache_dir: Path = None,
    one=None,
    overwrite: bool = False,
    repo_id: str = None,
) -> Path:
    """Resolve a model to a local directory, trying Hugging Face then S3.

    Args:
    # TODO: Add an example here for downloading the channel level encoder model from HF.
        model_id (str): A hub repo id (``owner/name``), or a bare S3 model folder name such
            as ``2024_W50_Cosmos_lid-basket-sense``.
        revision (str, optional): Hugging Face branch/tag. Ignored by the S3 backend.
        source (str, optional): ``"auto"`` (hub, then S3), ``"hf"``, or ``"s3"``.
        cache_dir (Path, optional): Where to place downloads. Defaults to
            ``~/.cache/ephysatlas/models``.
        one: ONE client instance, required for the S3 backend.
        overwrite (bool, optional): Passed to the S3 backend.
        repo_id (str, optional): Hugging Face repository, when it cannot be read off
            ``model_id``.

    Returns:
        Path: Local directory containing the model files.

    Raises:
        ValueError: If ``source`` is unknown, or if every attempted backend failed.
    """
    cache_dir = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home().joinpath(".cache", "ephysatlas", "models")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    backends = {
        "hf": lambda: [HFModelSource(repo_id=repo_id)],
        "s3": lambda: [S3ModelSource(one=one, overwrite=overwrite)],
        "auto": lambda: [
            HFModelSource(repo_id=repo_id),
            S3ModelSource(one=one, overwrite=overwrite),
        ],
    }
    if source not in backends:
        raise ValueError(f"unknown source {source!r}, expected one of {sorted(backends)}")

    errors = []
    for backend in backends[source]():
        try:
            path_model = backend.fetch(model_id, revision, cache_dir)
        except Exception as e:  # noqa: BLE001 - try the next backend, report all at the end
            logger.info(f"{type(backend).__name__} could not fetch {model_id}: {e}")
            errors.append(f"{type(backend).__name__}: {e}")
            continue
        verify_checksums(path_model)
        return path_model
    raise ValueError(f"could not resolve model {model_id!r}. Tried -- " + " | ".join(errors))


def class_acronyms(classes, region_map: str):
    """Translate classifier class ids to region acronyms.

    Args:
        classes (Sequence[int]): Allen region ids, as recorded in the model metadata.
        region_map (str): Mapping the ids belong to, e.g. ``"Cosmos"``.

    Returns:
        list[str] | None: One acronym per class id, or None if they could not be resolved.

    Raises:
        ValueError: If the number of acronyms returned does not match the number of class
            ids. ``id2acronym`` drops ids it does not know rather than raising, so an
            unchecked result would silently misalign every prediction downstream.
    """
    from ephysatlas import anatomy

    try:
        regions = anatomy.classifier_regions()
        acronyms = list(regions.id2acronym(np.asarray(classes), mapping=region_map))
    except Exception as e:  # noqa: BLE001 - the manifest is still useful without acronyms
        logger.warning(f"could not resolve class acronyms: {e}")
        return None
    if len(acronyms) != len(classes):
        unmapped = [
            int(c)
            for c in classes
            if len(regions.id2acronym(np.array([c]), mapping=region_map)) == 0
        ]
        raise ValueError(
            f"{len(classes)} class ids mapped to only {len(acronyms)} acronyms under "
            f"{region_map!r}; unmapped ids: {unmapped}"
        )
    return acronyms


def _blocks_region_classification(meta: dict, path_model: Path) -> dict:
    """Return the task-specific manifest blocks for a region-classification model.

    Returns the ``granularity``, ``artifacts``, ``inputs``, ``outputs`` and ``config`` entries,
    which :func:`write_manifest` merges into the shared core.

    ``artifacts`` belongs here rather than in the shared core because its *roles* are
    family-specific: ``folds`` is a cross-validation concept the spatial encoder has no
    analogue for, and that encoder needs roles (context volumes, a neighbour bank) this family
    has never heard of.
    """
    classes = [int(c) for c in meta["CLASSES"]]
    folds_root = path_model.joinpath("folds")
    fold_dirs = sorted(p.name for p in folds_root.glob("FOLD*")) if folds_root.exists() else []
    # Listing the fold directories outright removes the need for callers to guess a naming
    # pattern or a fold count, which infer_regions used to hardcode.
    artifacts = {"weights": "model.ubj", "folds": fold_dirs}
    # Only claim a published split when one is actually there: validate_artifacts would
    # otherwise go looking for a file that was never written.
    if path_model.joinpath(MODEL_SPLIT_FILE).exists():
        artifacts["split"] = MODEL_SPLIT_FILE
    return {
        "granularity": "channel",
        "artifacts": artifacts,
        "inputs": {
            "table": "raw_ephys_features_denoised.pqt",
            # Identity of a row. Differs by granularity: a cluster model would use
            # ["pid", "uuids"], and pid is a plain column in the cluster table.
            "index": ["pid", "channel"],
            "features": list(meta["FEATURES"]),
            # Ordered digest of the list above. The estimator consumes the feature matrix
            # positionally, so an edited or reordered list is a silent wrong-answer bug.
            "feature_order_sha256": feature_order_sha256(meta["FEATURES"]),
        },
        "outputs": {
            # `kind` lets a caller know what predict() returns before running anything.
            "kind": "categorical",
            "columns": [
                "predicted_acronym",
                "predicted_atlas_id",
                "prediction_probability",
                "fold_agreement",
            ],
        },
        "config": {
            "classes": classes,
            "class_acronyms": class_acronyms(classes, meta["REGION_MAP"]),
            "region_map": meta["REGION_MAP"],
            "accuracy": meta.get("ACCURACY"),
        },
    }


def _blocks_spatial_encoding(meta: dict, path_model: Path) -> dict:
    """Return the task-specific manifest blocks for a neighbour-inpainting spatial encoder.

    Note where the ordered feature list goes. The classifier's input *is* the feature matrix, so
    its list lives in ``inputs.features``. This model predicts ephys features *from* position and
    anatomical context, so its features are the **output** -- the input is ``x, y, z``. Putting
    them in ``inputs`` would tell a caller to supply columns the model never reads.

    Args:
        meta (dict): Training metadata. Reads ``FEATURES`` and, when present, the architecture
            keys ``D_MODEL``/``NHEAD``/``DEPTH``.
        path_model (Path): Model directory, scanned for the artifacts that are actually there.

    Returns:
        dict: The ``granularity``, ``artifacts``, ``inputs``, ``outputs`` and ``config`` entries.
    """
    features = [str(f) for f in meta["FEATURES"]]
    weights = ENCODER_WEIGHTS_FILE if path_model.joinpath(ENCODER_WEIGHTS_FILE).exists() else (
        "SE_model.pth" if path_model.joinpath("SE_model.pth").exists() else ENCODER_WEIGHTS_FILE
    )
    artifacts = {"weights": weights}
    # Only record what is present. A missing role must not be invented: validate_artifacts
    # would then hunt for a file nobody wrote.
    context = [n for n in ENCODER_CONTEXT_FILES if path_model.joinpath(n).exists()]
    if context:
        artifacts["context"] = context
    for role, name in (
        ("confidence", ENCODER_CONFIDENCE_FILE),
        ("neighbor_bank", ENCODER_BANK_FILE),
    ):
        if path_model.joinpath(name).exists():
            artifacts[role] = name

    architecture = {
        "f_ctx": int(meta.get("F_CTX") or 0) or None,
        "f_ephys": len(features),
        "f_out": len(features),
        "d_model": int(meta.get("D_MODEL", 256)),
        "nhead": int(meta.get("NHEAD", 8)),
        "depth": int(meta.get("DEPTH", 2)),
        "drop": float(meta.get("DROP", 0.1)),
    }
    return {
        "granularity": "channel",
        "artifacts": artifacts,
        "inputs": {
            "table": "channels.pqt",
            "index": ["pid", "channel"],
            # The real input: a position per channel, in the atlas frame. Metres.
            "columns": ["x", "y", "z"],
        },
        "outputs": {
            "kind": "continuous",
            "columns": features,
            # The digest sits beside the positional list, which for this family is the output.
            "feature_order_sha256": feature_order_sha256(features),
        },
        "config": {
            "architecture": architecture,
            "neighbourhood": {
                # Unit recorded explicitly: ChannelNN works in metres while axial_um is µm, and
                # a bare "radius: 600" would be ambiguous by a factor of a million.
                "radius_um": float(meta.get("RADIUS_UM", 600.0)),
                "m_max": int(meta.get("M_MAX", 64)),
                # Deterministic nearest-M, not the training collate's random subset: a published
                # predict() that returns different numbers each call is a defect.
                "selection": "nearest",
                "allow_same_probe": bool(meta.get("ALLOW_SAME_PROBE", False)),
            },
            "context": {
                "n_cell_pcs": int(meta.get("N_CELL_PCS", 50)),
                "n_gene_pcs": int(meta.get("N_GENE_PCS", 50)),
                "voxel_um": 200,
            },
        },
    }


def _blocks_unit_encoding(meta: dict, path_model: Path) -> dict:
    """Return the task-specific manifest blocks for the two-stage unit-level encoder.

    Two departures from the other families, both deliberate:

    - **The output is a latent, not named features.** ``outputs`` records ``kind: "latent"`` and a
      ``latent_dim`` rather than a column list -- the phenotype is a point in a learned space, so
      there is no ordered feature list to hash.
    - **The recorded dataset is not an artifact.** It is fetched from S3 via ONE at load and
      recorded under ``data_source``, so the published repo carries only weights. This is the one
      family that requires an IBL account to run its atlas-wide operations.

    Args:
        meta (dict): Training metadata -- ``VINTAGE``, ``LATENT_DIM``, ``GMM_COMPONENTS``, and the
            S3 ``PROJECT``.
        path_model (Path): Model directory, scanned for the checkpoints actually present.

    Returns:
        dict: The ``granularity``, ``artifacts``, ``inputs``, ``outputs``, ``config`` and
        ``data_source`` entries.
    """
    # Record only the checkpoints on disk; a missing role must not be invented.
    artifacts = {}
    for role, name in (
        ("autoencoder", UNIT_AE_FILE),
        ("pt_gmm", UNIT_GMM_FILE),
        ("scaler", UNIT_SCALER_FILE),
        ("unconditional_gmm", UNIT_UNCOND_GMM_FILE),
    ):
        if path_model.joinpath(name).exists():
            artifacts[role] = name
    latent_dim = int(meta.get("LATENT_DIM") or meta.get("SHARED_LATENT_DIM", 32))
    return {
        "granularity": "unit",
        "artifacts": artifacts,
        "inputs": {
            "index": ["pid", "cluster"],
            # Per-unit arrays, not a flat feature table.
            "modalities": ["waveform", "acg"],
        },
        "outputs": {
            # The one place the manifest leaves the feature-list contract: a latent phenotype.
            "kind": "latent",
            "latent_dim": latent_dim,
        },
        "config": {
            "architecture": {
                "shared_latent_dim": latent_dim,
                "gmm_components": int(meta.get("GMM_COMPONENTS", 16)),
            },
        },
        # The recorded dataset stays on S3 under IBL's access controls, fetched via ONE at load
        # and never republished on the Hub.
        "data_source": {
            "backend": "s3-ibl",
            "project": str(meta.get("PROJECT", "ibl_neuropixel_brainwide_01")),
            "requires_one": True,
        },
    }


# One entry per model family. The transport layer above never changes.
TASK_BUILDERS = {
    TASK_REGION_CLASSIFICATION: _blocks_region_classification,
    TASK_SPATIAL_ENCODING: _blocks_spatial_encoding,
    TASK_UNIT_ENCODING: _blocks_unit_encoding,
}


def read_manifest(path_model: Path):
    """Read a model directory's manifest, or None when it has none.

    Args:
        path_model (Path): Model directory.

    Returns:
        dict | None: The manifest, or None if the file is absent.
    """
    manifest_file = Path(path_model).joinpath(MODEL_MANIFEST_FILE)
    if not manifest_file.exists():
        return None
    return json.loads(manifest_file.read_text())


def write_manifest(
    path_model: Path,
    meta: dict,
    *,
    task: str = None,
    method: str = None,
    compatibility: dict = None,
) -> dict:
    """Assemble the publication manifest from an in-memory ``meta`` dict and write it.

    This is the pure assembler at the heart of packaging. It takes the training values
    already in hand (the ``meta`` dict) and adds what inference needs but training leaves
    implicit: which files hold the weights, which fold directories exist, the row identity
    and feature list, what ``predict`` returns, the training-time package versions, and (for
    classifiers) the region acronyms. It then writes ``ephysatlas_model.json``.

    It reads nothing off disk except to *scan the model directory for artifacts* -- the
    ``_blocks_*`` builders list the folds / checkpoints actually present. In particular it
    does **not** read ``meta.yaml``: every producer (the training scripts and the repackage
    tool) passes the values directly, so no ``meta.yaml`` scaffold is ever written.

    Args:
        path_model (Path): Model directory to write the manifest into, and to scan for
            artifacts. Optionally holds a ``folds/`` subdirectory.
        meta (dict): Training metadata in the UPPER_CASE ``meta.yaml``-style shape (``MODEL_CLASS``,
            ``VINTAGE``, ``FEATURES`` ...) that the ``_blocks_*`` task builders read from.
        task (str, optional): Task discriminator. Inferred from ``MODEL_CLASS`` if omitted.
        method (str, optional): Semantic label for the approach -- ``"xgboost"``,
            ``"transformer"``, ``"ridge"``, ``"regionmean"``, ``"gmm"``. Distinct from
            ``model_class``, which is the fully-qualified implementation class used for
            loader dispatch: two methods can share one ``model_class`` (the channel and
            cluster inpainting encoders are both ``NeighborInpaintingModel``, separated by
            ``granularity``), and ``model_class`` changes under refactoring while ``method``
            is the stable label used in listings and citations.
        compatibility (dict, optional): e.g. ``{"probe": ["NP1"], "species": ["mouse"]}``.
            Recorded when given — omitted rather than guessed, since applying a model outside
            its compatibility is a silent failure.

    Returns:
        dict: The manifest, also written to ``path_model/ephysatlas_model.json``.

    Raises:
        ValueError: If the task cannot be inferred, or has no registered builder.
    """
    path_model = Path(path_model)

    model_class = meta.get("MODEL_CLASS")
    task = task or MODEL_CLASS_TASKS.get(model_class)
    if task is None:
        raise ValueError(
            f"cannot infer task for MODEL_CLASS {model_class!r}; pass task= explicitly. "
            f"Known: {sorted(MODEL_CLASS_TASKS)}"
        )
    if task not in TASK_BUILDERS:
        raise ValueError(f"no builder for task {task!r}; known: {sorted(TASK_BUILDERS)}")

    index = {
        "model_id": path_model.name,
        "task": task,
        "model_class": model_class,
        "vintage": meta["VINTAGE"],
        # `artifacts` is supplied by the task builder below -- its roles differ per family.
        "training": {"random_seed": meta.get("RANDOM_SEED"), **(meta.get("TRAINING") or {})},
        "environment": _environment(),
    }
    if method is not None:
        index["method"] = method
    if compatibility is not None:
        index["compatibility"] = compatibility
    index.update(TASK_BUILDERS[task](meta, path_model))

    path_model.joinpath(MODEL_MANIFEST_FILE).write_text(json.dumps(index, indent=2) + "\n")
    logger.info(f"wrote {path_model.joinpath(MODEL_MANIFEST_FILE)}")
    return index


def meta_from_manifest(manifest: dict) -> dict:
    """Project a manifest onto the ``meta.yaml`` key shape.

    Lets a model published with a manifest but no ``meta.yaml`` still satisfy callers that
    expect the training-time dict, such as ``infer_regions`` reading ``model_info["FEATURES"]``.

    Args:
        manifest (dict): A parsed manifest.

    Returns:
        dict: meta-shaped view of the manifest.
    """
    config = manifest.get("config") or {}
    inputs = manifest.get("inputs") or {}
    return {
        "MODEL_CLASS": manifest.get("model_class"),
        "VINTAGE": manifest.get("vintage"),
        "FEATURES": list(inputs.get("features") or []),
        "CLASSES": list(config.get("classes") or []),
        "REGION_MAP": config.get("region_map"),
        "ACCURACY": config.get("accuracy"),
        "RANDOM_SEED": (manifest.get("training") or {}).get("random_seed"),
        "TRAINING": manifest.get("training") or {},
    }


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Digest a file's bytes in 1 MiB chunks.

    Written rather than reused: ``iblutil.io.hashfile`` exposes only blake2b/md5/sha1, and it
    allocates a 256 MB buffer plus a progress bar per call -- wrong for a loop over many small
    model files.

    Args:
        path (Path): File to hash.
        chunk_size (int, optional): Read size in bytes.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for block in iter(lambda: fid.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload) -> str:
    """Digest a JSON-serialisable payload in a form that is stable across runs.

    Sorted keys and no whitespace, so the same content always produces the same digest
    regardless of insertion order or formatting.
    """
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_ignored(relative_posix: str, patterns) -> bool:
    """True when a model-relative path matches any ignore pattern.

    Matches against both the full relative path and the bare filename, so ``"*.tmp"`` catches
    a nested ``folds/FOLD00/x.tmp`` and ``".cache/*"`` catches a whole subtree.
    """
    name = relative_posix.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(relative_posix, pattern) or fnmatch.fnmatch(name, pattern)
        for pattern in patterns
    )


def _checked_relative_path(path_model: Path, declared: str) -> Path:
    """Join a path declared inside ``checksums.json`` onto the model directory, safely.

    ``checksums.json`` arrives with a download and is not itself covered by any digest, so its
    contents are untrusted input. Without this, a published repository could list
    ``"../../../../etc/hosts"`` and make any caller of :func:`verify_checksums` stat and hash an
    arbitrary local file, with the path echoed back in the raised message.

    Validation is purely lexical -- no ``resolve()`` -- because a Hugging Face snapshot's files
    are symlinks into a sibling ``blobs/`` directory, and resolving them would legitimately
    leave the model directory.

    Args:
        path_model (Path): Model directory.
        declared (str): Path as recorded in ``checksums.json``.

    Returns:
        Path: ``path_model`` joined with the declared path.

    Raises:
        ValueError: If the declared path is absolute or escapes the model directory.
    """
    candidate = Path(declared)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(
            f"{MODEL_CHECKSUM_FILE} lists {declared!r}, which is absolute or escapes the model "
            f"directory. Refusing to read outside {path_model.name}."
        )
    return path_model.joinpath(candidate)


def _iter_model_files(path_model: Path, ignore_patterns) -> list:
    """Every publishable file in a model directory, sorted by relative path."""
    files = []
    for path in path_model.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(path_model).as_posix()
        if _is_ignored(relative, ignore_patterns):
            continue
        files.append((relative, path))
    return sorted(files, key=lambda item: item[0])


def write_checksums(path_model: Path, ignore_patterns=DEFAULT_CHECKSUM_IGNORE) -> Path:
    """Record a sha256 digest of every publishable file in a model directory.

    Write this **last**, after the manifest, model card and golden example exist: the digests
    cover whatever is on disk at the time, so anything written afterwards ships unverified.
    ``ephysatlas_model.json`` is deliberately covered -- it is the publication contract, so
    tampering with it has to be detectable.

    Args:
        path_model (Path): Model directory to scan.
        ignore_patterns (Iterable[str], optional): Globs to skip. Defaults to
            :data:`DEFAULT_CHECKSUM_IGNORE`, which excludes the checksum file itself and
            everything the upload drops or the Hub adds.

    Returns:
        Path: The written ``checksums.json``.
    """
    path_model = Path(path_model)
    files = [
        {"path": relative, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for relative, path in _iter_model_files(path_model, ignore_patterns)
    ]
    out = path_model.joinpath(MODEL_CHECKSUM_FILE)
    out.write_text(json.dumps({"algo": "sha256", "files": files}, indent=2) + "\n")
    logger.info(f"wrote {out} covering {len(files)} files")
    return out


def verify_checksums(path_model: Path, missing_ok: bool = True):
    # TODO - Can I use missing_ok to False, so that I verify everytime. I don't have to maintain backward compatibility.
    """Re-hash the files ``checksums.json`` lists and report every discrepancy at once.

    Only *listed* files are checked. Extra files are fine and expected: a Hub snapshot carries
    ``.gitattributes`` and a ``.cache/`` tree, and the model card is deliberately left out of
    the digest set (see :data:`DEFAULT_CHECKSUM_IGNORE`).

    Args:
        path_model (Path): Model directory to verify.
        missing_ok (bool, optional): When True (the default) a model with no ``checksums.json``
            returns None rather than raising, so models published before checksums existed keep
            loading. Pass False on the publish path, where their absence is a mistake.

    Returns:
        bool | None: True when everything matches; None when there is nothing to check and
        ``missing_ok`` is set.

    Raises:
        FileNotFoundError: If ``checksums.json`` is absent and ``missing_ok`` is False.
        ValueError: If any listed file is missing or its bytes have changed (the message names
            every offender, since fixing them one round trip at a time is miserable), or if
            ``checksums.json`` is itself unreadable or names a path outside the model directory.
    """
    path_model = Path(path_model)
    manifest_file = path_model.joinpath(MODEL_CHECKSUM_FILE)
    if not manifest_file.exists():
        if missing_ok:
            logger.debug(f"{path_model} ships no {MODEL_CHECKSUM_FILE}; skipping verification")
            return None
        raise FileNotFoundError(f"{path_model} has no {MODEL_CHECKSUM_FILE}")

    # The integrity manifest is itself untrusted input: it travels with the download and no
    # digest covers it. Say plainly when *it* is the damaged file, rather than letting a
    # JSONDecodeError or KeyError escape and read as though the model were corrupt.
    try:
        payload = json.loads(manifest_file.read_text())
        entries = payload["files"]
        listed = [(str(e["path"]), e["sha256"], e.get("bytes")) for e in entries]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(
            f"{path_model.name}: {MODEL_CHECKSUM_FILE} is unreadable or malformed "
            f"({type(e).__name__}: {e}). The integrity manifest is damaged, not necessarily "
            f"the model."
        ) from e

    problems = []
    for declared, expected_sha, expected_bytes in listed:
        target = _checked_relative_path(path_model, declared)
        if not target.exists():
            problems.append(f"{declared} (missing)")
            continue
        # Compare size first: it is free, and catches truncation without hashing the file.
        actual_bytes = target.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            problems.append(f"{declared} (size {actual_bytes} != {expected_bytes})")
            continue
        if _sha256_file(target) != expected_sha:
            problems.append(f"{declared} (sha256 mismatch)")
    if problems:
        raise ValueError(
            f"{path_model.name}: {len(problems)} file(s) do not match "
            f"{MODEL_CHECKSUM_FILE}: " + "; ".join(problems)
        )
    logger.debug(f"{path_model.name}: {len(listed)} files verified")
    return True


def _artifact_paths(path_model: Path, role: str, value) -> list:
    """Resolve one ``artifacts`` entry to the paths it names.

    Args:
        path_model (Path): Model directory the manifest belongs to.
        role (str): The artifacts key, used only for error messages.
        value: A path string, or a list of names.

    Returns:
        list[Path]: Absolute paths the entry refers to.

    Raises:
        TypeError: On any other shape, so a future nested layout fails loudly here rather than
            passing validation silently.
    """
    if isinstance(value, str):
        # rstrip("/") so a directory entry such as "atlas_pca/" resolves.
        return [path_model.joinpath(value.rstrip("/"))]
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        if role == "folds":
            # Fold entries are bare directory names, not model-relative paths: the manifest
            # stores "FOLD00" while the directory is folds/FOLD00. RegionClassifier._fold_dirs
            # re-derives the same prefix, and this must agree with it.
            folds_root = path_model.joinpath("folds")
            base = folds_root if folds_root.is_dir() else path_model
            # Agree with the loader's *predicate*, not just its path prefix. _fold_dirs keeps a
            # fold only when it holds its weights (folds ship weights only, no meta.yaml), so
            # checking mere directory existence would pass a fold the loader then silently
            # drops -- quietly averaging fewer folds than the manifest and card advertise, and
            # reporting a fold_agreement computed over them.
            return [base.joinpath(name, "model.ubj") for name in value]
        return [path_model.joinpath(name) for name in value]
    raise TypeError(
        f"artifacts[{role!r}] has unsupported type {type(value).__name__}; expected a path "
        f"string or a list of strings"
    )


def validate_artifacts(path_model: Path, index: dict = None) -> bool:
    """Check that every path the manifest's ``artifacts`` block names exists on disk.

    This is the one validation that works for any model family, whatever its roles are, and it
    is what catches a needed file being dropped by ``DEFAULT_UPLOAD_IGNORE`` or a botched
    upload -- a repository that looks complete and fails only when a stranger loads it.

    Deliberately **not** called on the load path. A fold-only publication legitimately has no
    root ``model.ubj``, and ``RegionClassifier._model_dirs`` already raises a better-targeted
    error for that case.

    Args:
        path_model (Path): Model directory.
        index (dict, optional): Parsed manifest. Read from disk when omitted.

    Returns:
        bool: True when everything named is present, and when there is no manifest or no
        ``artifacts`` block to check.

    Raises:
        FileNotFoundError: Naming each missing role and its path.
        TypeError: On an artifacts value that is neither a string nor a list of strings.
    """
    path_model = Path(path_model)
    index = index if index is not None else read_manifest(path_model)
    artifacts = (index or {}).get("artifacts") or {}
    if not artifacts:
        return True

    missing = []
    for role, value in artifacts.items():
        for target in _artifact_paths(path_model, role, value):
            if not target.exists():
                missing.append(f"{role} -> {target.relative_to(path_model).as_posix()}")
    if missing:
        raise FileNotFoundError(
            f"{path_model.name}: manifest artifacts missing on disk: " + "; ".join(missing)
        )
    return True


def write_split(
    path_model: Path,
    pids,
    ifold,
    random_seed: int = None,
    n_folds: int = None,
) -> Path:
    """Publish the cross-validation split as ``split.json``.

    The manifest's ``training.hash_training`` proves two runs used the same split but cannot
    say *which* insertions were held out. This records the pid lists themselves, so a reader
    can verify held-out status instead of trusting it.

    Args:
        path_model (Path): Model directory to write into.
        pids: Insertion pids in the order the training run used them -- i.e. **after** the
            seeded shuffle. Order is preserved rather than sorted, because
            ``iblutil.numerical.hash_uuids`` is order-sensitive and sorting here would make
            ``meta.yaml``'s hashes unreproducible from this file.
        ifold: Per-pid fold index, aligned with ``pids``.
        random_seed (int, optional): Seed that produced the shuffle.
        n_folds (int, optional): Fold count. Inferred from ``ifold`` when omitted.

    Returns:
        Path: The written ``split.json``.

    Raises:
        ValueError: If a pid is not a UUID string, naming the offender -- ``hash_uuids``
            otherwise raises a bare "badly formed hexadecimal UUID string" with no clue which
            pid caused it.
    """
    from iblutil.numerical import hash_uuids

    path_model = Path(path_model)
    pids = [str(p) for p in np.asarray(pids, dtype=object)]
    ifold = np.asarray(ifold)
    # Enforced, not merely documented: the fold masks below are built with zip(), which stops at
    # the shorter sequence. A caller whose ifold was derived from a filtered pid list would
    # otherwise get a split.json listing every pid while the folds account for only some of
    # them -- self-contradictory, yet stamped with a valid split_sha256.
    if len(pids) != ifold.size:
        raise ValueError(
            f"pids and ifold must be aligned: got {len(pids)} pids and {ifold.size} fold "
            f"indices. A split recorded from mismatched inputs silently omits insertions."
        )
    if not pids:
        raise ValueError("cannot record a split for zero insertions")
    n_folds = int(n_folds) if n_folds is not None else int(ifold.max()) + 1

    def _hash(subset):
        """hash_uuids over one pid list, re-raising with the offending pid named."""
        try:
            return hash_uuids(subset)
        except ValueError as e:
            offenders = [p for p in subset if not _looks_like_uuid(p)]
            raise ValueError(
                f"cannot hash the split: {len(offenders)} pid(s) are not UUIDs "
                f"({offenders[:3]}): {e}"
            ) from e

    folds = []
    for i in range(n_folds):
        is_test = ifold == i
        test_pids = [p for p, flag in zip(pids, is_test) if flag]
        train_pids = [p for p, flag in zip(pids, is_test) if not flag]
        folds.append(
            {
                "fold": i,
                "train_pids": train_pids,
                "test_pids": test_pids,
                "hash_training": _hash(train_pids),
                "hash_testing": _hash(test_pids),
            }
        )

    payload = {
        "n_folds": n_folds,
        "random_seed": random_seed,
        "split_unit": "probe_insertion_pid",
        "fold_assignment": "contiguous blocks of the seeded shuffle of pids",
        "pids": pids,
        "folds": folds,
    }
    payload["split_sha256"] = _canonical_sha256(payload)
    out = path_model.joinpath(MODEL_SPLIT_FILE)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"wrote {out} for {len(pids)} insertions over {n_folds} folds")
    return out


def read_split(path_model: Path):
    """Read ``split.json`` if the release ships one.

    Args:
        path_model (Path): Model directory.

    Returns:
        dict or None: The parsed split payload (``pids``, ``folds`` and the self-describing
        labels), or None when the model publishes no split.
    """
    path = Path(path_model).joinpath(MODEL_SPLIT_FILE)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _looks_like_uuid(value: str) -> bool:
    """True when a string parses as a UUID, used only to name offenders in an error."""
    import uuid

    try:
        uuid.UUID(hex=str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def feature_order_sha256(features) -> str:
    """Digest an ordered feature-name list.

    Args:
        features (Iterable[str]): Feature names, in the order the model consumes them.

    Returns:
        str: Hex digest, sensitive to order.
    """
    return _canonical_sha256([str(f) for f in features])


def validate_feature_order(features, recorded_digest=None, current_features=None) -> bool:
    """Check an ordered feature list against its recorded digest.

    The model consumes its feature matrix positionally, so a reordered list is a silent
    wrong-answer bug rather than a load error: every value would be read from the right number
    in the wrong slot. The digest turns that into a raised exception.

    Takes the list and the digest explicitly rather than a manifest block, because which block
    holds the positional list differs by family: it is ``inputs.features`` for the region
    classifier, whose input *is* the feature matrix, but ``outputs.columns`` for the spatial
    encoder, whose input is position and whose output is the features.

    Note this validates the *manifest*, not a caller's DataFrame -- ``predict`` selects columns
    by name, so the caller's column order has never mattered.

    Args:
        features (Iterable[str]): The published ordered feature list.
        recorded_digest (str, optional): The digest recorded alongside it. None for a manifest
            written before digests existed, which is not an error.
        current_features (Iterable[str], optional): A feature list from the installed code, to
            compare against the published one. The region classifier takes its list from the
            manifest so has nothing to compare; a family with a module-level feature constant
            (the spatial encoder's ``FEATURE_LIST``) does, and code-vs-release drift matters
            there.

    Returns:
        bool: True when consistent, and when the manifest predates the digest.

    Raises:
        ValueError: On a digest mismatch, or when ``current_features`` differs in order or
            membership from the published list.
    """
    features = [str(f) for f in (features or [])]
    recorded = recorded_digest
    if recorded is None:
        # Published before the digest existed; nothing to check against.
        logger.debug("manifest records no feature_order_sha256; skipping the order check")
    else:
        actual = feature_order_sha256(features)
        if actual != recorded:
            raise ValueError(
                f"feature order digest mismatch: the manifest lists {len(features)} features "
                f"hashing to {actual[:12]}… but records {str(recorded)[:12]}…. The published "
                f"feature list has been edited or reordered, which would silently corrupt "
                f"every prediction."
            )
    if current_features is not None:
        current = [str(f) for f in current_features]
        if current != features:
            raise ValueError(
                f"feature order mismatch between the installed code ({len(current)} features) "
                f"and this release ({len(features)}). The model consumes features positionally, "
                f"so they must agree exactly, in order."
            )
    return True


def _environment() -> dict:
    """Record the versions that matter for reloading a model faithfully.

    Read from distribution metadata rather than by importing each package. Importing them would
    make packaging *any* model load xgboost, and on macOS arm64 xgboost and torch cannot coexist
    in one process -- they bring incompatible OpenMP runtimes and segfault at the first torch
    tensor copy. Recording a version must not decide which model families can be packaged.
    """
    import importlib.metadata
    import platform

    def _v(distribution_name):
        try:
            return importlib.metadata.version(distribution_name)
        except Exception:  # noqa: BLE001 - absent packages are simply not recorded
            return None

    return {
        "python": platform.python_version(),
        "xgboost": _v("xgboost"),
        "scikit_learn": _v("scikit-learn"),
        "numpy": _v("numpy"),
        "pandas": _v("pandas"),
        "torch": _v("torch"),
        "ephysatlas": _v("ibleatools"),
    }
