"""Resolve trained ephys-atlas models from the Hugging Face Hub and read their manifests.

:class:`HFModelSource` downloads a published model directory from the Hub -- public repos need no
authentication, which is what lets people outside the IBL use these models -- and
:func:`resolve_model` wraps it with the on-load integrity check.

Every published model ships ``ephysatlas_model.json``, the *publication contract*: a discriminated
union keyed on ``model_class`` that normalises every model family into one shape so a single
load path
serves them all. This module only *reads* that manifest (:func:`read_manifest`,
:func:`meta_from_manifest`) and verifies the download against the published checksums
(:func:`verify_checksums`).
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from iblutil.io import hashfile

logger = logging.getLogger(__name__)

# Name of the publication manifest inside a model directory.
MODEL_MANIFEST_FILE = "ephysatlas_model.json"
# Per-file digests of everything published, verified on load so a truncated download or a file
# silently dropped at publish time fails.
MODEL_CHECKSUM_FILE = "checksums.json"

CHECKSUM_HASHERS = {
    "blake2b": hashfile.blake2b,
    "md5": hashfile.md5,
    "sha1": hashfile.sha1,
}
# Spatial-encoder artifact names. The encoder is not reconstructible from weights alone: it also
# needs the frozen PCA context volumes and a bank of training-channel features to draw
# neighbours from.
ENCODER_WEIGHTS_FILE = "spatial_encoder.pt"
ENCODER_CONFIDENCE_FILE = "confidence_model.pt"
ENCODER_BANK_FILE = "neighbor_bank.npz"
ENCODER_CONTEXT_FILES = ("agea_vol_pca.npy", "merfish_vol_pca.npy")

# Unit-level encoder: the canonical filenames a published release stages its checkpoints under.
# Unlike the other families it ships no recorded data -- only weights. The per-unit atlas
# arrays are read from a local cache prepared outside this package.
UNIT_AE_FILE = "autoencoder.pt"
UNIT_GMM_FILE = "point_transformer_gmm.pt"
UNIT_SCALER_FILE = "shared_latent_scaler.joblib"
UNIT_UNCOND_GMM_FILE = "unconditional_gmm_train_only.joblib"


class HFModelSource:
    """Fetch a published model from the Hugging Face Hub.

    Args:
        repo_id (str, optional): Repository to use. If omitted, a ``model_id`` of the form
            ``owner/name`` is used as the repo id; otherwise fetching raises
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

    def fetch(
        self, model_id: str, revision: str = None, cache_dir: Path = None
    ) -> Path:
        """Download a snapshot of the repository and return its local directory."""
        # Lazy import: keeps huggingface_hub off the ``import ephysatlas`` path, which callers
        # loading a model from a local directory never need.
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


def resolve_model(
    model_id: str,
    revision: str = None,
    cache_dir: Path = None,
    repo_id: str = None,
) -> Path:
    """Resolve a model to a local directory by downloading it from the Hugging Face Hub.

    Args:
        model_id (str): A Hugging Face repo id (``owner/name``).
        revision (str, optional): Hugging Face branch/tag to pin, e.g. ``"2026_W32"``. Omitted,
            it resolves to ``main``.
        cache_dir (Path, optional): Where to place downloads. Defaults to
            ``~/.cache/ephysatlas/models``.
        repo_id (str, optional): Hugging Face repository, when it cannot be read off
            ``model_id``.

    Returns:
        Path: Local directory containing the model files.
    """
    cache_dir = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home().joinpath(".cache", "ephysatlas", "models")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    path_model = HFModelSource(repo_id=repo_id).fetch(model_id, revision, cache_dir)
    # Every published model ships checksums.json, so require it on the load path: a fetch that
    # arrived without it is incomplete and must not be loaded.
    verify_checksums(path_model, missing_ok=False)
    return path_model


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


def meta_from_manifest(manifest: dict) -> dict:
    """Project a manifest onto the UPPER_CASE key shape callers expect.

    Keeps ``infer_regions``-style callers reading ``model_info["FEATURES"]`` working off the
    manifest alone.

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


def class_acronyms(classes, region_map: str):
    """Translate classifier class ids to region acronyms.

    Args:
        classes (Sequence[int]): Allen region ids, as recorded in the model metadata.
        region_map (str): Mapping the ids belong to, e.g. ``"Cosmos"``.

    Returns:
        list[str]: One acronym per class id.

    Raises:
        ValueError: If the number of acronyms returned does not match the number of class
            ids. ``id2acronym`` drops ids it does not know rather than raising, so an
            unchecked result would silently misalign every prediction downstream.
    """
    from ephysatlas.anatomy import ClassifierRegions, NEW_VOID

    # ClassifierRegions adds void_fluid (id 2000) so every class a trained classifier can
    # emit is resolvable; plain BrainRegions would silently drop it (see id2acronym below).
    regions = ClassifierRegions()
    regions.add_new_region(NEW_VOID)
    acronyms = list(regions.id2acronym(np.asarray(classes), mapping=region_map))
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


def verify_checksums(path_model: Path, missing_ok: bool = True):
    """Re-hash the files ``checksums.json`` lists and report every discrepancy at once.

    Only *listed* files are checked. Extra files are fine and expected: a Hub snapshot carries
    ``.gitattributes`` and a ``.cache/`` tree, and the model card is deliberately not hashed.

    Args:
        path_model (Path): Model directory to verify.
        missing_ok (bool, optional): When True (the default) a model with no ``checksums.json``
            returns None rather than raising. The **load path passes False** -- every published
            model ships checksums, so their absence there means an incomplete download. True
            remains the default for callers that only want to verify a directory when it happens
            to carry checksums.

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
            logger.debug(
                f"{path_model} ships no {MODEL_CHECKSUM_FILE}; skipping verification"
            )
            return None
        raise FileNotFoundError(f"{path_model} has no {MODEL_CHECKSUM_FILE}")

    # The integrity manifest is itself untrusted input: it travels with the download and no
    # digest covers it. Say plainly when *it* is the damaged file, rather than letting a
    # JSONDecodeError or KeyError escape and read as though the model were corrupt.
    try:
        payload = json.loads(manifest_file.read_text())
        algo = payload["algo"]
        entries = payload["files"]
        listed = [(str(e["path"]), e["hash"], e.get("bytes")) for e in entries]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(
            f"{path_model.name}: {MODEL_CHECKSUM_FILE} is unreadable or malformed "
            f"({type(e).__name__}: {e}). The integrity manifest is damaged, not necessarily "
            f"the model."
        ) from e

    hasher = CHECKSUM_HASHERS.get(algo)
    if hasher is None:
        raise ValueError(
            f"{path_model.name}: {MODEL_CHECKSUM_FILE} declares unknown checksum algo {algo!r}; "
            f"expected one of {sorted(CHECKSUM_HASHERS)}."
        )

    problems = []
    for declared, expected_hash, expected_bytes in listed:
        target = _checked_relative_path(path_model, declared)
        if not target.exists():
            problems.append(f"{declared} (missing)")
            continue
        # Compare size first: it is free, and catches truncation without hashing the file.
        actual_bytes = target.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            problems.append(f"{declared} (size {actual_bytes} != {expected_bytes})")
            continue
        # Hash with the algorithm the writer recorded, through the same iblutil function it used.
        if hasher(target) != expected_hash:
            problems.append(f"{declared} (hash mismatch)")
    if problems:
        raise ValueError(
            f"{path_model.name}: {len(problems)} file(s) do not match "
            f"{MODEL_CHECKSUM_FILE}: " + "; ".join(problems)
        )
    logger.debug(f"{path_model.name}: {len(listed)} files verified")
    return True


def feature_order_sha256(features) -> str:
    """Digest an ordered feature-name list, sensitive to order.

    Canonical JSON (no whitespace) so the same list always produces the same digest, whichever
    side computes it. The manifest records this over the model's positional feature list, and the
    load path recomputes it to catch a list edited or reordered after publication.

    Args:
        features (Iterable[str]): Feature names, in the order the model consumes them.

    Returns:
        str: Hex digest, sensitive to order.
    """
    raw = json.dumps([str(f) for f in features], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_feature_order(features, recorded_digest=None) -> bool:
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

    Returns:
        bool: True when consistent, and when the manifest predates the digest.

    Raises:
        ValueError: On a digest mismatch.
    """
    features = [str(f) for f in (features or [])]
    recorded = recorded_digest
    if recorded is None:
        # Published before the digest existed; nothing to check against.
        logger.debug(
            "manifest records no feature_order_sha256; skipping the order check"
        )
    else:
        actual = feature_order_sha256(features)
        if actual != recorded:
            raise ValueError(
                f"feature order digest mismatch: the manifest lists {len(features)} features "
                f"hashing to {actual[:12]}… but records {str(recorded)[:12]}…. The published "
                f"feature list has been edited or reordered, which would silently corrupt "
                f"every prediction."
            )
    return True
