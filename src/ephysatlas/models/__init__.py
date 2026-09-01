"""The public entry point for loading published ephysatlas models.

One function serves every model family:

    >>> from ephysatlas import load_pretrained
    >>> model = load_pretrained("int-brain-lab/ea-decoder-channel-xgboost", revision="2026_W32")
    >>> out = model.predict(df_features)

:func:`load_pretrained` resolves the id to a local directory, reads the publication manifest
(``ephysatlas_model.json``), and dispatches to the right wrapper class -- on ``model_class``
first, falling back to ``task``. Two families can share a task and still need different wrappers,
so the more specific key wins.

"""

import logging
from pathlib import Path

from ephysatlas import model_registry

logger = logging.getLogger(__name__)


def _region_classifier(path_model: Path, index: dict, **kwargs):
    """Build the region-decoding wrapper.

    Imported lazily so that ``import ephysatlas`` does not pull in xgboost, and so a future
    torch-based family does not pull in torch for users who only need this one.
    """
    from ephysatlas.regionclassifier import RegionClassifier

    return RegionClassifier(path_model, index=index)

def _spatial_encoder(path_model: Path, index: dict, **kwargs):
    """Build the spatial-encoder wrapper. torch is imported inside the module it comes from."""
    from ephysatlas.models.encoder_inpainting import SpatialEncoder

    return SpatialEncoder(path_model, index=index, device=kwargs.get("device"))


def _unit_encoder(path_model: Path, index: dict, **kwargs):
    """Build the unit-level encoder wrapper. torch is imported inside the module it comes from."""
    from ephysatlas.models.unit_encoder import UnitEncoder

    return UnitEncoder(path_model, index=index, device=kwargs.get("device"))


MODEL_WRAPPERS = {
    "ephysatlas.spatial_encoder.model.NeighborInpaintingModel": _spatial_encoder,
    # The encoder's hand-written meta.yaml files record the bare class name; both are in use.
    "NeighborInpaintingModel": _spatial_encoder,
    # The unit encoder dispatches on its entry checkpoint's class; UnitEncoder loads the rest.
    "ephysatlas.unit_level_encoder.model.MultimodalAutoencoder": _unit_encoder,
    "MultimodalAutoencoder": _unit_encoder,
}

# One entry per task: the fallback if MODEL_WRAPPERS does not find a match
TASK_WRAPPERS = {
    model_registry.TASK_REGION_CLASSIFICATION: _region_classifier,
    model_registry.TASK_SPATIAL_ENCODING: _spatial_encoder,
    model_registry.TASK_UNIT_ENCODING: _unit_encoder,
}


def _resolve_wrapper(path_model: Path, index: dict):
    """Pick the wrapper builder for a manifest, most specific first.

    Args:
        path_model (Path): Model directory, used only in the error message.
        index (dict): Parsed manifest, or None for a model published without one.

    Returns:
        callable: A builder taking ``(path_model, index, **kwargs)``.

    Raises:
        ValueError: If neither the model class nor the task is registered.
    """
    model_class = (index or {}).get("model_class")
    if model_class in MODEL_WRAPPERS:
        return MODEL_WRAPPERS[model_class]
    # By default, the task is region classification
    task = (index or {}).get("task", model_registry.TASK_REGION_CLASSIFICATION)
    if task in TASK_WRAPPERS:
        return TASK_WRAPPERS[task]
    raise ValueError(
        f"{path_model} declares task {task!r} and model_class {model_class!r}, which this "
        f"version of ephysatlas cannot load. Known tasks: {sorted(TASK_WRAPPERS)}; known model "
        f"classes: {sorted(MODEL_WRAPPERS)}. Try upgrading ephysatlas."
    )


def load_pretrained(
    model_id,
    revision: str = None,
    cache_dir: Path = None,
    one=None,
    source: str = "auto",
    repo_id: str = None,
    **kwargs,
):
    """Load a published model, from the Hugging Face Hub, S3, or a local directory.

    Args:
        model_id (str or Path): A Hugging Face repo id (``owner/name``), a bare S3 model
            folder name, or a path to an already-downloaded model directory.
        revision (str, optional): Hugging Face branch/tag to pin, e.g. ``"2026_W32"``. Omitted,
            it resolves to ``main``, which tracks the *currently recommended* model and moves
            when a new vintage is published
        cache_dir (Path, optional): Where downloads are placed.
        one (optional): ONE client instance, needed only for the private S3 route.
        source (str, optional): ``"auto"``, ``"hf"`` or ``"s3"``.
        repo_id (str, optional): Hugging Face repository, when it cannot be read off
            ``model_id``.

    Returns:
        A model wrapper exposing ``.predict(df)``, ``.selftest()`` and ``.index``. Which class
        depends on the manifest's ``task``.

    Raises:
        ValueError: If the manifest declares a task with no registered wrapper.
    """
    # If the model is locally available, use it diretly.
    local = Path(model_id)
    if local.is_dir():
        path_model = local
        logger.info(f"using local model directory {path_model}")
        # Verify the checksum for the local file, if it was edited after download.
        model_registry.verify_checksums(path_model)
    else:
        # If revision is None, raise a warning that user is using main branch.
        if revision is None and "/" in str(model_id):
            logger.warning(
                f"no revision pinned for {model_id}: using 'main', which tracks the currently "
                f"recommended model and will change when a new vintage is published. Pass "
                f"revision='<vintage>' for a reproducible result."
            )
        path_model = model_registry.resolve_model(
            str(model_id),
            revision=revision,
            source=source,
            cache_dir=cache_dir,
            one=one,
            repo_id=repo_id,
        )

    index = model_registry.read_manifest(path_model)
    build = _resolve_wrapper(path_model, index)  
    return build(path_model, index, **kwargs)
