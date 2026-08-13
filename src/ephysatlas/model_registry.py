"""Resolve trained ephys-atlas models from a model hub or from AWS S3, and package them.

Two backends fetch a model directory, both exposing
``fetch(model_id, revision, cache_dir) -> Path``:

* :class:`HFModelSource` -- Hugging Face Hub. Public repos need no authentication, which is
  what lets people outside the IBL use these models at all.
* :class:`S3ModelSource` -- the historical private bucket. Requires an authenticated ONE
  client to mint credentials.

:func:`resolve_model` picks between them. Making Hugging Face the only source later is a
change to the default of its ``source`` argument, not a rewrite of the callers.

This module also builds ``model_index.json``, the *publication contract*. It is deliberately
separate from ``meta.yaml``: ``meta.yaml`` is a training artifact whose shape follows whatever
the trainer wrote, while ``model_index.json`` normalises every model family into one shape so
a single load path can serve them all. It is a discriminated union -- a shared core plus a
``config`` block interpreted according to ``task``:

    {"task": "region-classification", "model_class": ..., "artifacts": {...},
     "config": {"features": [...], "classes": [...], "class_acronyms": [...]}}

Supporting a new family (e.g. the torch spatial encoder) means adding one entry to
:data:`TASK_CONFIG_BUILDERS` and one to ``regionclassifier.MODEL_LOADERS`` -- the transport
layer above is model-agnostic and does not change.
"""

import json
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# S3 prefix the models have always lived under.
S3_MODEL_PREFIX = "aggregates/atlas/models"
# Name of the publication manifest inside a model directory.
MODEL_INDEX_FILE = "model_index.json"

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

    def upload(self, path_model: Path, revision: str = None, message: str = ""):
        """Upload a local model directory to the hub. Requires a write-scoped token.

        Args:
            path_model (Path): Local model directory to upload.
            revision (str, optional): Branch/tag to push to. Created if it does not exist.
            message (str, optional): Commit message.

        Returns:
            The value returned by ``HfApi.upload_folder``.
        """
        from huggingface_hub import HfApi

        repo_id = self._resolve_repo_id(path_model.name)
        api = HfApi(token=self.token)
        if revision:
            api.create_branch(
                repo_id=repo_id, branch=revision, repo_type="model", exist_ok=True
            )
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=str(path_model),
            revision=revision,
            commit_message=message or f"Publish {path_model.name}",
        )


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


def _config_region_classification(meta: dict, path_model: Path) -> dict:
    """Build the ``config`` block for a region-classification model."""
    classes = [int(c) for c in meta["CLASSES"]]
    return {
        "features": list(meta["FEATURES"]),
        "classes": classes,
        "class_acronyms": class_acronyms(classes, meta["REGION_MAP"]),
        "region_map": meta["REGION_MAP"],
        "accuracy": meta.get("ACCURACY"),
    }


# One entry per model family. Adding the spatial encoder means adding a builder here that
# records its architecture hyper-parameters and the normalisation stats it ships.
TASK_CONFIG_BUILDERS = {
    TASK_REGION_CLASSIFICATION: _config_region_classification,
}


def build_model_index(path_model: Path, task: str = None) -> dict:
    """Build and write the ``model_index.json`` publication manifest.

    Reads the training-time ``meta.yaml`` and adds what inference needs but ``meta.yaml``
    leaves implicit: which files hold the weights, which fold directories exist, the
    training-time package versions, and (for classifiers) the region acronyms.

    Args:
        path_model (Path): Model directory containing ``meta.yaml``, and optionally a
            ``folds/`` subdirectory.
        task (str, optional): Task discriminator. Inferred from ``MODEL_CLASS`` if omitted.

    Returns:
        dict: The manifest, also written to ``path_model/model_index.json``.

    Raises:
        ValueError: If the task cannot be inferred, or has no registered config builder.
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
    if task not in TASK_CONFIG_BUILDERS:
        raise ValueError(
            f"no config builder for task {task!r}; known: {sorted(TASK_CONFIG_BUILDERS)}"
        )

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
        "config": TASK_CONFIG_BUILDERS[task](meta, path_model),
    }
    path_model.joinpath(MODEL_INDEX_FILE).write_text(json.dumps(index, indent=2) + "\n")
    logger.info(f"wrote {path_model.joinpath(MODEL_INDEX_FILE)}")
    return index


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
