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

import json
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# S3 prefix the models have always lived under.
S3_MODEL_PREFIX = "aggregates/atlas/models"
# Name of the publication manifest inside a model directory. Deliberately project-scoped:
# `model_index.json` is diffusers' pipeline manifest and `config.json` implies transformers.
MODEL_MANIFEST_FILE = "ephysatlas_model.json"
# Hugging Face organisation the models are published under.
DEFAULT_HF_ORG = "int-brain-lab"
# Files kept locally but never published. `predictions.pqt` is tens of MB of out-of-fold
# predictions over real IBL insertions -- useful for reproducibility work in-house, not part
# of what a user needs to run the model.
DEFAULT_UPLOAD_IGNORE = ("predictions.pqt", ".DS_Store", "*.tmp")

# Task discriminators used by the manifest's "task" field.
TASK_REGION_CLASSIFICATION = "region-classification"

# Which task a saved model belongs to, keyed on the MODEL_CLASS that save_model records.
MODEL_CLASS_TASKS = {
    "xgboost.sklearn.XGBClassifier": TASK_REGION_CLASSIFICATION,
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
            return backend.fetch(model_id, revision, cache_dir)
        except Exception as e:  # noqa: BLE001 - try the next backend, report all at the end
            logger.info(f"{type(backend).__name__} could not fetch {model_id}: {e}")
            errors.append(f"{type(backend).__name__}: {e}")
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

    Returns the ``granularity``, ``inputs``, ``outputs`` and ``config`` entries, which
    :func:`build_model_index` merges into the shared core.
    """
    classes = [int(c) for c in meta["CLASSES"]]
    return {
        "granularity": "channel",
        "inputs": {
            "table": "raw_ephys_features_denoised.pqt",
            # Identity of a row. Differs by granularity: a cluster model would use
            # ["pid", "uuids"], and pid is a plain column in the cluster table.
            "index": ["pid", "channel"],
            "features": list(meta["FEATURES"]),
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


# One entry per model family. Adding the spatial encoder means adding a builder here that
# records its architecture hyper-parameters and the normalisation stats it ships.
TASK_BUILDERS = {
    TASK_REGION_CLASSIFICATION: _blocks_region_classification,
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


def build_model_index(
    path_model: Path,
    task: str = None,
    method: str = None,
    compatibility: dict = None,
) -> dict:
    """Build and write the publication manifest (``ephysatlas_model.json``).

    Reads the training-time ``meta.yaml`` and adds what inference needs but ``meta.yaml``
    leaves implicit: which files hold the weights, which fold directories exist, the row
    identity and feature list, what ``predict`` returns, the training-time package versions,
    and (for classifiers) the region acronyms.

    Args:
        path_model (Path): Model directory containing ``meta.yaml``, and optionally a
            ``folds/`` subdirectory.
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
    meta = yaml.safe_load(path_model.joinpath("meta.yaml").read_text())

    model_class = meta.get("MODEL_CLASS")
    task = task or MODEL_CLASS_TASKS.get(model_class)
    if task is None:
        raise ValueError(
            f"cannot infer task for MODEL_CLASS {model_class!r}; pass task= explicitly. "
            f"Known: {sorted(MODEL_CLASS_TASKS)}"
        )
    if task not in TASK_BUILDERS:
        raise ValueError(f"no builder for task {task!r}; known: {sorted(TASK_BUILDERS)}")

    folds_root = path_model.joinpath("folds")
    fold_dirs = sorted(p.name for p in folds_root.glob("FOLD*")) if folds_root.exists() else []

    index = {
        "model_id": path_model.name,
        "task": task,
        "model_class": model_class,
        "vintage": meta["VINTAGE"],
        # Listing the fold directories outright removes the need for callers to guess a
        # naming pattern or a fold count, which infer_regions used to hardcode.
        "artifacts": {"weights": "model.ubj", "folds": fold_dirs},
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


def _environment() -> dict:
    """Record the versions that matter for reloading a model faithfully."""
    import platform

    def _v(module_name):
        try:
            module = __import__(module_name)
            return getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001 - absent packages are simply not recorded
            return None

    return {
        "python": platform.python_version(),
        "xgboost": _v("xgboost"),
        "scikit_learn": _v("sklearn"),
        "numpy": _v("numpy"),
        "pandas": _v("pandas"),
        "ephysatlas": _v("ephysatlas"),
    }
