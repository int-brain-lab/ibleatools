"""The public entry point for loading published ephysatlas models.

One function serves every model family:

    >>> from ephysatlas import load_pretrained
    >>> model = load_pretrained("int-brain-lab/ea-decoder-channel-xgboost", revision="2026_W32")
    >>> out = model.predict(df_features)

:func:`load_pretrained` resolves the id to a local directory, reads the publication manifest
(``ephysatlas_model.json``), and dispatches to the right wrapper class on the manifest's
``model_class`` -- a single 1:1 lookup. Every published model ships a manifest, so a directory
without one is not loadable.

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


# One canonical model_class per family -> its wrapper builder. Dispatch is 1:1: every published
# model carries exactly one of these in its manifest. The encoders record the bare class name
# (what their trainers write); the region classifier records the fully-qualified XGBoost class.
MODEL_WRAPPERS = {
    "xgboost.sklearn.XGBClassifier": _region_classifier,
    "NeighborInpaintingModel": _spatial_encoder,
    # The unit encoder dispatches on its entry checkpoint's class; UnitEncoder loads the rest.
    "MultimodalAutoencoder": _unit_encoder,
}


def _resolve_wrapper(path_model: Path, index: dict):
    """Pick the wrapper builder for a manifest, from its ``model_class``.

    Args:
        path_model (Path): Model directory, used only in the error message.
        index (dict): Parsed manifest. A model without one is not loadable.

    Returns:
        callable: A builder taking ``(path_model, index, **kwargs)``.

    Raises:
        ValueError: If the manifest is absent or names an unregistered ``model_class``.
    """
    if index is None:
        raise ValueError(
            f"{path_model} has no {model_registry.MODEL_MANIFEST_FILE}; it is not a loadable model."
        )
    model_class = index.get("model_class")
    try:
        return MODEL_WRAPPERS[model_class]
    except KeyError:
        raise ValueError(
            f"{path_model} declares model_class {model_class!r}, which this version of ephysatlas "
            f"cannot load. Known: {sorted(MODEL_WRAPPERS)}. Try upgrading ephysatlas."
        ) from None


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
            when a new vintage is published.
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
    local = Path(model_id)
    if local.is_dir():
        path_model = local
        logger.info(f"using local model directory {path_model}")
        # Require checksums here too, so an edited or incomplete local directory is caught.
        model_registry.verify_checksums(path_model, missing_ok=False)
    else:
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
