"""The public entry point for loading published ephysatlas models.

One function serves every model family:

    >>> from ephysatlas import load_pretrained
    >>> model = load_pretrained("int-brain-lab/ea-decoder-channel-xgboost", revision="2026_W32")
    >>> out = model.predict(df_features)

:func:`load_pretrained` resolves the id to a local directory, reads the publication manifest
(``ephysatlas_model.json``), and dispatches to the right wrapper class -- on ``model_class``
first, falling back to ``task``. Two families can share a task and still need different wrappers,
so the more specific key wins.

This indirection is the point: it is the *only* API a published model card should name. The
concrete wrapper classes and the modules they live in can be reorganised without invalidating
any card already on the Hub, because users never import them directly.
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


# Keyed on the manifest's `model_class`, and consulted BEFORE TASK_WRAPPERS.
#
# Task alone cannot decide the wrapper. The design keeps `region-decoding` as one task with
# `method` separating xgboost from transformer, so a channel transformer for region prediction
# carries the *same* task as the XGBoost model -- and handing it to RegionClassifier would reach
# `classifier.predict_proba(x)`, which a torch module does not have. Keying on `model_class`
# mirrors MODEL_LOADERS, so loader and wrapper stay in step by construction.
def _spatial_encoder(path_model: Path, index: dict, **kwargs):
    """Build the spatial-encoder wrapper. torch is imported inside the module it comes from."""
    from ephysatlas.models.encoder_inpainting import SpatialEncoder

    return SpatialEncoder(path_model, index=index, device=kwargs.get("device"))


MODEL_WRAPPERS = {
    "ephysatlas.spatial_encoder.model.NeighborInpaintingModel": _spatial_encoder,
    # The encoder's hand-written meta.yaml files record the bare class name; both are in use.
    "NeighborInpaintingModel": _spatial_encoder,
}

# One entry per task: the fallback, and what every manifest-less legacy model resolves through.
TASK_WRAPPERS = {
    model_registry.TASK_REGION_CLASSIFICATION: _region_classifier,
    model_registry.TASK_SPATIAL_ENCODING: _spatial_encoder,
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
    # Models published before the manifest existed are region classifiers by construction:
    # they are the only family that ever shipped without one.
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
            when a new vintage is published -- convenient for exploration, not reproducible.
            Pass a tag for anything you intend to cite or re-run later.
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
    # A local directory short-circuits the transport layer entirely: no network, no
    # credentials. This is also what makes the package usable offline and in CI.
    local = Path(model_id)
    if local.is_dir():
        path_model = local
        logger.info(f"using local model directory {path_model}")
        # Downloads are verified inside resolve_model, which every fetch route passes through.
        # A local directory bypasses it, so check here too -- someone may have edited a file in
        # a packaged directory in place.
        model_registry.verify_checksums(path_model)
    else:
        # A hub id without a pinned revision resolves to `main`, which moves as new vintages
        # are published. Say so, rather than let a script silently change model one day.
        # S3 names are not warned about: the vintage is baked into the folder name.
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
    return _resolve_wrapper(path_model, index)(path_model, index, **kwargs)
