"""Loader shims that let the paper-figure scripts run on *this* branch's published models.

The figures were written against the coworker's release registry
(``ephysatlas.spatial_encoder.model_registry.resolve_release`` for the channel model,
``figure_unit_common.load_released_unit_model`` for the unit model). These two functions are
drop-in replacements that return the *same* variables, but sourced through
:func:`ephysatlas.load_pretrained` -- so a figure body stays byte-for-byte his, and re-running it
against a newly published model reproduces the same figure with updated numbers.

This is model access, not style: nothing here touches how a figure looks.
"""

import logging
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)


def _held_out_pids(split):
    """The set of held-out (test) insertion pids from a split, or None if it has no single one.

    A single train/val/test split exposes ``test_pids`` directly. A k-fold split has no single
    held-out set (every pid is test in exactly one fold), so its identity is its ``split_sha256``
    instead, and this returns None.
    """
    if not split:
        return None
    if split.get("test_pids") is not None:
        return frozenset(str(p) for p in split["test_pids"])
    return None


def check_split_agreement(splits, names=None, on_mismatch="raise"):
    """Verify that models combined in one figure used the same data split.

    Per-model ``split.json`` makes each repo self-contained, but it turns "these models held out
    the same insertions" from a structural guarantee into a convention. A cross-model figure
    (plotting one model's metric beside another's) is only valid, and leak-free, if they agree.
    This checks it before plotting.

    Rules:
      - Models sharing ``split_sha256`` are identical -> pass.
      - Otherwise their held-out (test) insertion sets must match -> pass; if they differ it is a
        real mismatch -> raise (or warn).
      - A model shipping no split, or only k-fold splits whose digests differ (no single held-out
        set to compare), cannot be verified -> warn and pass. A *detectable* mismatch is never
        passed silently.

    Args:
        splits: iterable of split dicts (from ``model_registry.read_split`` or ``release.split``);
            None entries are allowed (a model that ships no split).
        names: optional labels, for readable messages.
        on_mismatch: ``"raise"`` (default) or ``"warn"``.

    Returns:
        bool: True when agreement is confirmed or there is nothing to compare; False on a detected
        mismatch when ``on_mismatch="warn"``.

    Raises:
        ValueError: On a detected mismatch when ``on_mismatch="raise"``.
    """
    splits = list(splits)
    names = list(names) if names else [f"model{i}" for i in range(len(splits))]
    present = [(n, s) for n, s in zip(names, splits) if s]
    missing = [n for n, s in zip(names, splits) if not s]
    if missing:
        logger.warning(f"cannot verify split agreement: no split.json for {missing}")
    if len(present) < 2:
        return True

    # Fast path: identical canonical digests => same split by construction.
    digests = [s.get("split_sha256") for _, s in present]
    if all(d is not None for d in digests) and len(set(digests)) == 1:
        return True

    # Otherwise compare the held-out insertion sets.
    held = {n: _held_out_pids(s) for n, s in present}
    comparable = {n: h for n, h in held.items() if h is not None}
    if len(comparable) < 2:
        logger.warning(
            "cannot verify split agreement: fewer than two models expose a held-out set "
            f"(digests differ; regime may be k-fold). models={list(held)}"
        )
        return True
    if len(set(comparable.values())) == 1:
        return True

    sizes = {n: len(h) for n, h in comparable.items()}
    message = (
        "models were evaluated on DIFFERENT held-out insertions -- combining them in one figure "
        f"is not comparable and may leak. held-out set sizes: {sizes}"
    )
    if on_mismatch == "raise":
        raise ValueError(message)
    logger.warning(message)
    return False


def _unit_split_manifest(pids, test_frac=0.2, val_frac=0.2, seed=0):
    """A deterministic train/val/test split over whole probes.

    His unit code splits on the spatial encoder's authoritative ``split.json``
    (``{train_pids, validation_pids, test_pids}``). A weights-only unit release ships no split, so
    for figure reproduction we derive a stable one from the atlas PIDs -- enough for the figures,
    which only need a valid held-out set to build the voxel-neighbourhood loaders.
    """
    uniq = sorted({str(p) for p in np.asarray(pids)})
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    return {
        "test_pids": uniq[:n_test],
        "validation_pids": uniq[n_test : n_test + n_val],
        "train_pids": uniq[n_test + n_val :],
    }


def unit_release(repo_id, vintage=None, *, token=None, device="cpu", cache_dir=None):
    """Reconstruct his ``load_released_unit_model`` tuple from a published unit encoder.

    Returns:
        tuple: ``(cfg, data, model_ae, model_gmm, scaler, standardized, datasets, loaders)`` --
        the same shape his figures unpack, minus the trailing ``artifacts`` they ignore.
    """
    from ephysatlas import load_pretrained
    from ephysatlas.unit_level_encoder.data import prepare_data
    from ephysatlas.unit_level_encoder.gmm_models import make_loaders, make_neighborhood_datasets

    enc = load_pretrained(repo_id, revision=vintage, device=device)
    cfg = enc.cfg
    atlas = enc.atlas_arrays(cache_dir)
    cfg.waveform_shape = tuple(atlas["waveforms"].shape[1:])
    cfg.acg_shape = tuple(atlas["acgs"].shape[1:])

    split_manifest = _unit_split_manifest(atlas["pids"])
    data = prepare_data(
        atlas["waveforms"], atlas["acgs"], atlas["ctx"], atlas["xyz"], atlas["pids"], cfg,
        split_manifest=split_manifest,
    )
    # Standardised latents, aligned row-for-row with `data` (same atlas arrays, same order).
    standardized = enc.latents(cache_dir)
    datasets = make_neighborhood_datasets(data, standardized, cfg)
    loaders = make_loaders(datasets, cfg)
    return cfg, data, enc.model_ae, enc.model_gmm, enc.scaler, standardized, datasets, loaders


# -- pure atlas helpers, copied verbatim from his figure_unit_common ------------------------
# (figure_unit_common itself is not vendored: it imports the unit hf_io publisher, whose
# spatial_encoder.model_registry import does not exist on this branch. These helpers are pure.)


def cosmos_ids_for_xyz(brain_atlas, xyz_m):
    return np.asarray(brain_atlas.get_labels(np.asarray(xyz_m), mapping="Cosmos"), dtype=np.int64)


def region_color(brain_atlas, acronym: str):
    hit = np.flatnonzero(np.asarray(brain_atlas.regions.acronym, dtype=object) == acronym)
    if len(hit) == 0:
        raise RuntimeError(f"Atlas acronym not found: {acronym}")
    rgb = np.asarray(brain_atlas.regions.rgb[hit[0]], dtype=float)
    return rgb / 255.0 if rgb.max() > 1 else rgb


def region_id(brain_atlas, acronym: str) -> int:
    hit = np.flatnonzero(np.asarray(brain_atlas.regions.acronym, dtype=object) == acronym)
    if len(hit) == 0:
        raise RuntimeError(f"Atlas acronym not found: {acronym}")
    return int(brain_atlas.regions.id[hit[0]])


def channel_release(repo_id, vintage=None, *, device="cpu"):
    """Reconstruct the variables his channel figures read off ``resolve_release``.

    Returns:
        SimpleNamespace: with ``dir`` (local model directory), ``context_dir`` (where the context
        volumes live, per the manifest), ``features``, ``config``, ``split``, ``stats``
        (standardisation buffers), ``model`` (raw torch encoder), ``wrapper`` (the
        :class:`SpatialEncoder`) and ``predict`` (its ``predict`` method).
    """
    from ephysatlas import load_pretrained
    from ephysatlas import model_registry

    wrapper = load_pretrained(repo_id, revision=vintage, device=device)
    return SimpleNamespace(
        dir=wrapper.path_model,
        context_dir=wrapper.context_dir,
        features=list(wrapper.outputs.get("columns") or []),
        config=wrapper.config,
        split=model_registry.read_split(wrapper.path_model),
        stats=wrapper.preprocessing_stats(),
        model=wrapper.model,
        wrapper=wrapper,
        predict=wrapper.predict,
    )
