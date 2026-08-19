"""Package a trained region-classifier directory for the Hugging Face Hub.

Takes a local model directory produced by :func:`ephysatlas.regionclassifier.save_model`
and makes it self-describing and self-testing:

1. ``ephysatlas_model.json`` -- the manifest (row identity, features, classes + acronyms,
   fold layout, what ``predict`` returns, training-time environment).
2. ``README.md`` -- Hugging Face model card with YAML frontmatter.
3. ``LICENSE`` -- CC-BY-4.0 notice.
4. ``example/features_sample.parquet`` + ``example/expected_predictions.parquet`` -- a small
   real sample and its golden output, so ``RegionClassifier.selftest()`` can detect drift.

Packaging is always safe to run. Uploading is opt-in and requires ``--upload`` plus a
write-scoped token in ``HF_TOKEN``; without both, this script only writes local files.
Uploads are **private by default** and `predictions.pqt` is never published.

Usage::

    # 1. package locally and self-test -- no network, no token
    python scripts/publish_model_to_hf.py --model-dir <dir> --features <agg_full dir>

    # 2. upload to a PRIVATE repo, pushing main and tagging the vintage
    HF_TOKEN=... python scripts/publish_model_to_hf.py --model-dir <dir> \\
        --features <agg_full dir> --method xgboost \\
        --repo-id int-brain-lab/ea-decoder-channel-xgboost --revision 2026_W32 --upload

    # 3. review on the Hub, then open it up
    HF_TOKEN=... python scripts/publish_model_to_hf.py \\
        --repo-id int-brain-lab/ea-decoder-channel-xgboost --make-public
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

import ephysatlas.anatomy
import ephysatlas.data
import ephysatlas.model_registry as model_registry

# RegionClassifier (and, for the encoder, SpatialEncoder) are imported lazily inside the
# functions that use them. RegionClassifier pulls xgboost and SpatialEncoder pulls torch; on
# macOS arm64 the two segfault if loaded into one process, and a single publish run only ever
# touches one family. Keeping both imports out of module scope means an encoder run never loads
# xgboost and a classifier run never loads torch.

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
_logger = logging.getLogger("publish_model_to_hf")

LICENSE_TEXT = """Creative Commons Attribution 4.0 International (CC-BY-4.0)

You are free to share and adapt this material for any purpose, including commercially,
provided you give appropriate credit.

Full licence text: https://creativecommons.org/licenses/by/4.0/legalcode
"""

CARD_TEMPLATE = """---
license: cc-by-4.0
library_name: xgboost
tags:
  - neuroscience
  - electrophysiology
  - neuropixels
  - brain-region-classification
  - international-brain-lab
---

# Ephys Atlas region classifier ({vintage}, {region_map})

Predicts the **brain region of each Neuropixels recording channel** from electrophysiological
features alone -- no histology required. Trained by the
[International Brain Laboratory](https://www.internationalbrainlab.com/) on the Ephys Atlas
feature release `{vintage}`.

> **What you can and cannot do without IBL access.** The *model* runs for anyone. Computing
> the *input features* from raw Neuropixels data needs `ibllib` / `ibl-neuropixel`, and the
> raw data itself is IBL-hosted. To try the model immediately, use the bundled sample under
> `example/` -- no account, no raw data, no S3.

## Quickstart

```python
import pandas as pd
from ephysatlas import load_pretrained

model = load_pretrained("{repo_id}", revision="{revision}")
df = pd.read_parquet("example/features_sample.parquet")   # or your own features
out = model.predict(df)
print(out[["predicted_acronym", "prediction_probability", "fold_agreement"]].head())
```

`load_pretrained` is the entry point for every ephysatlas model, whatever its family — it reads
`ephysatlas_model.json` and returns the right wrapper. Use it rather than importing a concrete
class, so your code keeps working as the package evolves.

`predict` returns one row per input channel, indexed identically to the input:
`predicted_acronym`, its Allen `predicted_atlas_id`, the fold-averaged
`prediction_probability`, a `fold_agreement` column (fraction of the {n_folds} folds voting
for the winner -- the natural uncertainty signal), and a `p_<acronym>` column per class.

The prediction columns are namespaced so that `df.join(out)` works: the feature table already
carries histology-derived `acronym` / `atlas_id` columns, and predictions must not shadow them.

### Which weights are used

By default `predict` **averages the {n_folds} fold models**; the single all-data `model.ubj` at
the repo root is not used. Pass `estimator="global"` to use it instead — one fifth of the
inference cost, but `fold_agreement` then comes back as NaN, since no folds were consulted. The
two modes disagree on a small fraction of channels, so pick one per analysis.

## Inputs

- **{n_features} features**, listed in `ephysatlas_model.json` under `inputs.features`. Every
  one must be present; `predict` raises and names anything missing.
- Indexed by `(pid, channel)`, one row per recording channel.
- **Must be the denoised aggregated features of vintage `{vintage}`** -- that is, the
  `raw_ephys_features_denoised.pqt` table produced by the Ephys Atlas aggregation pipeline,
  as loaded by `ephysatlas.data.read_features_from_disk`. Units are baked into that table by
  the pipeline (RMS features in dB, `spike_count` in log2), so feeding raw features, or
  features from a vintage whose units differ, produces confident nonsense. Run
  `model.selftest()` to confirm your install reproduces the shipped output before trusting it.

## Performance

Pooled out-of-fold accuracy: **{accuracy:.4f}** over {n_classes} {region_map} regions,
{training_size} insertions. Splits are by insertion (`pid`), so no channel from a test
insertion appears in training. See `confusion_matrix.png`.

Note this figure scores each channel with the one fold that held it out, whereas the default
`estimator="ensemble"` averages all {n_folds} folds. On genuinely unseen data the ensemble is
typically the marginally better estimator, so treat the number as slightly conservative.

## Limitations

- Trained on IBL Neuropixels 1.0 recordings in mouse. Transfer to NP2, other species or
  other rigs is untested.
- Coverage follows IBL brain-wide-map targeting; rare regions are under-represented.
- `{region_map}` is a coarse parcellation. Predictions are per-channel and spatially
  unregularised -- neighbouring channels can disagree.
- Known-misaligned insertions were excluded from training.

## Reproducibility

**Pin the revision.** `revision="{revision}"` is an immutable tag. Omitting `revision` resolves
to `main`, which tracks whichever model is currently recommended and *will* change when a new
feature vintage is published — fine for a first look, not for anything you publish or re-run.

`ephysatlas_model.json` records the training-time `environment` (xgboost, scikit-learn, numpy,
ephysatlas, python) and `random_seed`. Verify your install reproduces the shipped output:

```python
model.selftest()
```

Note `scikit-learn<1.9` is required (1.9 broke `OneToOneFeatureMixin.get_feature_names_out`,
which the feature transformer relies on).

## Citation

Please cite the International Brain Laboratory Ephys Atlas. Model id `{model_id}`,
feature vintage `{vintage}`.
"""


CARD_TEMPLATE_ENCODER = """---
license: cc-by-4.0
library_name: pytorch
tags:
  - neuroscience
  - electrophysiology
  - neuropixels
  - spatial-encoding
  - international-brain-lab
---

# Ephys Atlas spatial encoder ({vintage})

Predicts the **electrophysiological feature vector expected at a channel's position** -- from
that position plus anatomical context and the recorded features of nearby channels on other
insertions. Trained by the
[International Brain Laboratory](https://www.internationalbrainlab.com/) on the Ephys Atlas
feature release `{vintage}`.

> **The input is a position, not features.** This is the inverse of the region classifier: that
> model takes features and returns a region; this one takes a channel's `x, y, z` (Allen frame,
> metres) and returns the {n_features} ephys features it would expect there. Feeding it the
> feature columns instead of coordinates is the most common mistake -- `predict` raises and names
> the coordinates it needs.

## What ships in this repo, and why all of it is needed

A published encoder is not the weights alone. `predict` cannot run without every one of:

- `{weights}` -- the trained network.
- the context PCA volumes (`agea_vol_pca.npy`, `merfish_vol_pca.npy`) -- the anatomical context
  it samples at each position.
- `neighbor_bank.npz` -- the **neighbour bank**: the position, standardised feature vector and
  insertion id of every training channel. At inference the model gathers the nearest recorded
  channels from this bank and attends to them, so the bank *is* part of the model, the way the
  stored points are part of a k-nearest-neighbours model. It cannot be reconstructed from the
  weights, which is why it is shipped here rather than recomputed on your machine.

> **First run downloads the Allen volume.** Building the context sampler constructs an
> `AllenAtlas`, which fetches the Allen CCF volume from `download.alleninstitute.org` (public, no
> account, a few hundred MB) the first time it runs on a machine, then caches it.

## Quickstart

```python
import pandas as pd
from ephysatlas import load_pretrained

model = load_pretrained("{repo_id}", revision="{revision}")
df = pd.read_parquet("example/features_sample.parquet")   # channel positions (x, y, z)
out = model.predict(df)                                    # one pred_<feature> column each
print(out.head())
```

`load_pretrained` is the entry point for every ephysatlas model, whatever its family -- it reads
`ephysatlas_model.json` and returns the right wrapper. Use it rather than importing a concrete
class, so your code keeps working as the package evolves.

`predict` returns one row per input channel, indexed identically to the input, with a
`pred_<feature>` column per entry in `outputs.columns`. The `pred_` prefix keeps `df.join(out)`
from colliding with the ground-truth feature columns of the same names.

## Inputs

- Indexed by `(pid, channel)`, one row per recording channel.
- The coordinate columns `x, y, z` named in `ephysatlas_model.json` under `inputs.columns`, in
  the Allen atlas frame, **in metres**. The feature columns are *not* read -- they are the output.

## Neighbourhood selection

For each query position the model attends to recorded channels within **{radius_um} µm**, taking
the **nearest {m_max}** (`selection: nearest`). This is deterministic: two calls on the same
input return identical predictions.

> Training used a *random* subset of the in-radius neighbours; this published `predict` takes the
> nearest instead. The two agree exactly whenever a position has at most {m_max} neighbours in
> radius, and differ only in dense regions where the training-time subset would itself have varied
> run to run. Deterministic output is the right contract for a published model.

## Limitations

- Trained on IBL Neuropixels 1.0 recordings in mouse. Transfer to NP2, other species or other
  rigs is untested.
- Coverage follows IBL brain-wide-map targeting; predictions far from any recorded channel fall
  back on anatomical context alone and are correspondingly weaker.

## Reproducibility

**Pin the revision.** `revision="{revision}"` is an immutable tag. Omitting `revision` resolves to
`main`, which tracks whichever model is currently recommended and *will* change when a new feature
vintage is published -- fine for a first look, not for anything you publish or re-run.

`ephysatlas_model.json` records the training-time `environment` and `random_seed`. Verify your
install reproduces the shipped output:

```python
model.selftest()
```

## Citation

Please cite the International Brain Laboratory Ephys Atlas. Model id `{model_id}`,
feature vintage `{vintage}`.
"""


CARD_TEMPLATE_UNIT = """---
license: cc-by-4.0
library_name: pytorch
tags:
  - neuroscience
  - electrophysiology
  - neuropixels
  - unit-level-encoder
  - international-brain-lab
---

# Ephys Atlas unit-level encoder ({vintage})

A **per-unit** (per spike-sorted neuron) model, in two stages: a multimodal autoencoder embeds a
unit's multi-channel **waveform** and its **autocorrelogram** into a {latent_dim}-d latent -- its
phenotype -- and a Point-Transformer Gaussian mixture over those latents whose components read as
**putative cell types**. Trained by the
[International Brain Laboratory](https://www.internationalbrainlab.com/) on release `{vintage}`.

> **This model encodes; it does not `predict(df)`.** Its input is a unit's waveform + ACG arrays
> and its output is a point in a learned latent space, not a table of named predictions. So the
> interface is `encode` (and `reconstruct` / `components` / `assign`), not the `predict(df)` the
> region classifier and spatial encoder expose.

> **Requires an IBL/ONE account for atlas-wide use.** Unlike the other families, the recorded unit
> dataset is **not** shipped here -- it is pulled from IBL S3 via ONE at first use and cached
> locally. Encoding your *own* units needs only the weights below; reproducing the atlas-wide
> latents (`.latents()`) triggers the S3 fetch.

## Quickstart

```python
import numpy as np
from ephysatlas import load_pretrained

model = load_pretrained("{repo_id}", revision="{revision}")
z = model.encode(waveform, acg)          # [n_units, {latent_dim}] latent phenotype
rec = model.reconstruct(waveform, acg)   # waveform + ACG reconstruction
means, log_var = model.components()      # GMM putative cell types
```

`load_pretrained` is the entry point for every ephysatlas model -- it reads
`ephysatlas_model.json` and returns the right wrapper.

## What ships here

- `{ae}` -- the multimodal autoencoder.
- `{gmm}` -- the Point-Transformer GMM.
- the latent `StandardScaler`, and the unconditional GMM baseline.

The recorded per-unit dataset (waveforms, ACGs, positions) is **not** here: it is fetched from S3
via ONE, keeping this repository weights-only.

## Reproducibility

**Pin the revision.** `revision="{revision}"` is an immutable tag. Verify your install reproduces
the shipped output with `model.selftest()` (encode-only -- no S3, no account).

## Citation

Please cite the International Brain Laboratory Ephys Atlas. Model id `{model_id}`, vintage
`{vintage}`.
"""


def build_example(path_model: Path, path_features: Path, n_channels: int, seed: int = 0):
    """Write a small real feature sample plus its golden predictions.

    Args:
        path_model (Path): Model directory to write ``example/`` into.
        path_features (Path): An ``agg_full`` features directory to sample from.
        n_channels (int): Number of channels to include.
        seed (int, optional): Sampling seed, so the sample is reproducible.

    Returns:
        Path: The ``example/`` directory.
    """
    from ephysatlas.regionclassifier import RegionClassifier

    index = json.loads(path_model.joinpath(model_registry.MODEL_MANIFEST_FILE).read_text())
    feature_names = index["inputs"]["features"]
    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    df = ephysatlas.data.read_features_from_disk(
        path_features, brain_atlas=brain_atlas, strict=False
    )
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"features directory lacks columns the model needs: {missing}")

    # Sample whole channels at random but deterministically, spanning many insertions.
    rng = np.random.default_rng(seed)
    take = rng.choice(df.shape[0], size=min(n_channels, df.shape[0]), replace=False)
    sample = df.iloc[np.sort(take)].loc[:, feature_names]

    example = path_model.joinpath("example")
    example.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(example.joinpath("features_sample.parquet"))

    # Golden output, produced by the very model being published.
    predictions = RegionClassifier(path_model).predict(sample)
    predictions.to_parquet(example.joinpath("expected_predictions.parquet"))
    _logger.info(f"wrote example/ with {len(sample)} channels")
    return example


def _read_encoder_source(path_features: Path, feature_names) -> "pd.DataFrame":
    """Read an ``agg_full`` directory into the ``(pid, channel)`` table the encoder needs.

    The encoder is fed positions, so its source must carry both the coordinates (from
    ``channels.pqt``) and the feature columns (from ``raw_ephys_features_denoised.pqt``): the
    bank stores standardised features keyed by position. Rows missing any coordinate or feature
    are dropped, so neither the bank nor the example carries holes. Mirrors the join the training
    pipeline does.

    Args:
        path_features (Path): An ``agg_full`` features directory.
        feature_names (list): The feature columns the model predicts, from ``outputs.columns``.

    Returns:
        pd.DataFrame: Indexed by ``(pid, channel)``, carrying ``x, y, z`` and every feature.
    """
    import pandas as pd

    path_features = Path(path_features)
    feats = pd.read_parquet(path_features.joinpath("raw_ephys_features_denoised.pqt"))
    chans = pd.read_parquet(path_features.joinpath("channels.pqt"))
    df = feats.join(chans.loc[:, ["x", "y", "z"]], how="inner")
    return df.dropna(subset=["x", "y", "z"] + list(feature_names))


def build_encoder_example(
    path_model: Path, index: dict, path_features: Path, n_channels: int, seed: int = 0
):
    """Write a small position sample plus the encoder's golden predictions.

    Unlike the classifier example, the sample holds channel *positions* (``x, y, z``): the
    encoder predicts features from them. It is written to ``example/features_sample.parquet``
    all the same, because ``SpatialEncoder.selftest`` -- shared machinery with the classifier --
    looks for that name.

    Args:
        path_model (Path): Model directory to write ``example/`` into.
        index (dict): The manifest, read for ``outputs.columns``.
        path_features (Path): An ``agg_full`` features directory to sample positions from.
        n_channels (int): Number of channels to include.
        seed (int, optional): Sampling seed, so the sample is reproducible.

    Returns:
        Path: The ``example/`` directory.
    """
    from ephysatlas.models.encoder_inpainting import SpatialEncoder

    features = index["outputs"]["columns"]
    df = _read_encoder_source(path_features, features)
    # Sample whole channels deterministically, spanning many insertions.
    rng = np.random.default_rng(seed)
    take = rng.choice(df.shape[0], size=min(n_channels, df.shape[0]), replace=False)
    sample = df.iloc[np.sort(take)].loc[:, ["x", "y", "z"]]

    example = path_model.joinpath("example")
    example.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(example.joinpath("features_sample.parquet"))

    # Golden output, produced by the very model being published.
    predictions = SpatialEncoder(path_model, index=index).predict(sample)
    predictions.to_parquet(example.joinpath("expected_predictions.parquet"))
    _logger.info(f"wrote example/ with {len(sample)} channels")
    return example


def build_unit_example(path_model: Path, index: dict, arrays_dir: Path, n_units: int, seed: int = 0):
    """Write a small unit sample (waveform + ACG) plus the encoder's golden latents.

    The unit encoder's input is per-unit arrays, not a table, so the sample is an ``.npz`` of
    ``waveform``/``acg`` and the golden is ``expected_latents.npy``. Encode-only, so the golden
    reproduces offline -- no S3 fetch, no ONE -- which is what ``UnitEncoder.selftest`` checks.

    Args:
        path_model (Path): Model directory to write ``example/`` into.
        index (dict): The manifest.
        arrays_dir (Path): A prepared unit-arrays directory (``waveforms.npy``, ``acgs.npy``).
        n_units (int): Number of units to include.
        seed (int, optional): Sampling seed, so the sample is reproducible.

    Returns:
        Path: The ``example/`` directory.
    """
    from ephysatlas.models.unit_encoder import UnitEncoder

    arrays_dir = Path(arrays_dir)
    wav = np.load(arrays_dir.joinpath("waveforms.npy"), mmap_mode="r")
    acg = np.load(arrays_dir.joinpath("acgs.npy"), mmap_mode="r")
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(wav.shape[0], size=min(n_units, wav.shape[0]), replace=False))
    waveform = np.asarray(wav[take], dtype=np.float32)
    acg_sample = np.asarray(acg[take], dtype=np.float32)

    example = path_model.joinpath("example")
    example.mkdir(parents=True, exist_ok=True)
    np.savez(example.joinpath("units_sample.npz"), waveform=waveform, acg=acg_sample)

    # Golden latents, produced by the very model being published.
    latents = UnitEncoder(path_model, index=index).encode(waveform, acg_sample)
    np.save(example.joinpath("expected_latents.npy"), latents)
    _logger.info(f"wrote example/ with {len(take)} units")
    return example


def _render_classifier_card(index: dict, repo_id: str, revision: str) -> str:
    """Fill CARD_TEMPLATE from a region-classifier manifest."""
    training = index.get("training") or {}
    config = index["config"]
    return CARD_TEMPLATE.format(
        vintage=index["vintage"],
        region_map=config["region_map"],
        repo_id=repo_id,
        revision=revision or "main",
        n_folds=len(index["artifacts"]["folds"]) or 1,
        n_features=len(index["inputs"]["features"]),
        n_classes=len(config["classes"]),
        accuracy=config.get("accuracy") or float("nan"),
        training_size=training.get("training_size", "n/a"),
        model_id=index["model_id"],
    )


def _render_encoder_card(index: dict, repo_id: str, revision: str) -> str:
    """Fill CARD_TEMPLATE_ENCODER from a spatial-encoder manifest.

    Reads the ordered feature list from ``outputs`` (not ``inputs``: this family predicts
    features *from* position) and the neighbourhood settings from ``config``.
    """
    config = index.get("config") or {}
    outputs = index.get("outputs") or {}
    artifacts = index.get("artifacts") or {}
    neighbourhood = config.get("neighbourhood") or {}
    return CARD_TEMPLATE_ENCODER.format(
        vintage=index["vintage"],
        repo_id=repo_id,
        revision=revision or "main",
        n_features=len(outputs.get("columns") or []),
        weights=artifacts.get("weights", model_registry.ENCODER_WEIGHTS_FILE),
        radius_um=neighbourhood.get("radius_um", 600.0),
        m_max=neighbourhood.get("m_max", 64),
        model_id=index["model_id"],
    )


def _render_unit_card(index: dict, repo_id: str, revision: str) -> str:
    """Fill CARD_TEMPLATE_UNIT from a unit-encoder manifest."""
    outputs = index.get("outputs") or {}
    artifacts = index.get("artifacts") or {}
    return CARD_TEMPLATE_UNIT.format(
        vintage=index["vintage"],
        repo_id=repo_id,
        revision=revision or "main",
        latent_dim=outputs.get("latent_dim", 32),
        ae=artifacts.get("autoencoder", model_registry.UNIT_AE_FILE),
        gmm=artifacts.get("pt_gmm", model_registry.UNIT_GMM_FILE),
        model_id=index["model_id"],
    )


# The card renderer a task gets. A new family registers here rather than growing an if/elif,
# mirroring model_registry.TASK_BUILDERS.
CARD_RENDERERS = {
    model_registry.TASK_REGION_CLASSIFICATION: _render_classifier_card,
    model_registry.TASK_SPATIAL_ENCODING: _render_encoder_card,
    model_registry.TASK_UNIT_ENCODING: _render_unit_card,
}


def write_card(path_model: Path, index: dict, repo_id: str, revision: str):
    """Render README.md and LICENSE into the model directory, per the manifest's task.

    Raises:
        NotImplementedError: For a task with no registered card renderer. A card makes concrete,
            family-specific claims -- the classifier card quotes region accuracy and a class
            count, the encoder card the neighbour bank and the position->features inversion -- so
            a new family must supply its own template rather than borrow another's and misdescribe
            itself.
    """
    task = index.get("task")
    renderer = CARD_RENDERERS.get(task)
    if renderer is None:
        raise NotImplementedError(
            f"no model card template for task {task!r}; a card states family-specific claims, so "
            f"add a renderer for {task!r} to CARD_RENDERERS before publishing it. "
            f"Known: {sorted(CARD_RENDERERS)}."
        )
    path_model.joinpath("README.md").write_text(renderer(index, repo_id, revision))
    path_model.joinpath("LICENSE").write_text(LICENSE_TEXT)
    _logger.info("wrote README.md and LICENSE")


def _default_repo_id(task: str) -> str:
    """A placeholder repo id for the card when ``--repo-id`` is not given, per family.

    There is no single repo across families, so the card shows a visibly family-appropriate
    placeholder rather than a wrong default.
    """
    names = {
        model_registry.TASK_REGION_CLASSIFICATION: "ea-decoder-channel-xgboost",
        model_registry.TASK_SPATIAL_ENCODING: "ea-encoder-channel",
        model_registry.TASK_UNIT_ENCODING: "ea-encoder-unit",
    }
    return f"{model_registry.DEFAULT_HF_ORG}/{names.get(task, 'ea-model')}"


def _write_example_and_selftest(path_model: Path, index: dict, path_features: Path, n_channels: int):
    """Write the golden ``example/`` and self-test the freshly packaged model, per family.

    Each branch imports its own wrapper lazily so that packaging one family never loads the
    other's heavy dependency (xgboost vs torch); see the module-level import note.

    Raises:
        NotImplementedError: For a task with no packaging path here.
    """
    task = index.get("task")
    if task == model_registry.TASK_REGION_CLASSIFICATION:
        from ephysatlas.regionclassifier import RegionClassifier

        build_example(path_model, path_features, n_channels)
        RegionClassifier(path_model).selftest()
    elif task == model_registry.TASK_SPATIAL_ENCODING:
        from ephysatlas.models.encoder_inpainting import SpatialEncoder

        build_encoder_example(path_model, index, path_features, n_channels)
        SpatialEncoder(path_model, index=index).selftest()
    elif task == model_registry.TASK_UNIT_ENCODING:
        from ephysatlas.models.unit_encoder import UnitEncoder

        build_unit_example(path_model, index, path_features, n_channels)
        UnitEncoder(path_model, index=index).selftest()
    else:
        raise NotImplementedError(
            f"no packaging path for task {task!r}; add one before publishing this family."
        )
    _logger.info("selftest passed against the freshly written golden file")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None, help="local model directory")
    parser.add_argument(
        "--features", type=Path, default=None, help="agg_full dir to build example/ from"
    )
    parser.add_argument("--n-example-channels", type=int, default=500)
    parser.add_argument(
        "--method",
        default=None,
        help="semantic label for the approach (xgboost, transformer, ridge, regionmean, gmm). "
        "Recorded in the manifest; distinct from model_class, which is the implementation.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="target Hugging Face repo, e.g. org/ephys-atlas-region-classifier. "
        "Required with --upload; otherwise only used in the model card's quickstart.",
    )
    parser.add_argument("--revision", default=None, help="branch/tag to publish under")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="actually push to the hub (requires HF_TOKEN with write scope)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="create the repository public immediately. Default is private, so the contents "
        "can be reviewed on the Hub first; use --make-public afterwards to open it up.",
    )
    parser.add_argument(
        "--make-public",
        action="store_true",
        help="flip an existing repository to public and exit. Does not upload anything.",
    )
    args = parser.parse_args(argv)

    # --make-public is a standalone step: no packaging, no upload, no model directory needed.
    if args.make_public:
        token = os.environ.get("HF_TOKEN")
        if not token or not args.repo_id:
            _logger.error("--make-public needs both HF_TOKEN and --repo-id")
            return 1
        model_registry.HFModelSource(
            repo_id=args.repo_id, token=token
        ).set_visibility(private=False)
        _logger.info(f"https://huggingface.co/{args.repo_id}")
        return 0

    if args.model_dir is None:
        parser.error("--model-dir is required")
    path_model = args.model_dir.resolve()
    if not path_model.joinpath("meta.yaml").exists():
        parser.error(f"{path_model} does not look like a model directory (no meta.yaml)")

    index = model_registry.build_model_index(path_model, method=args.method)
    task = index.get("task")

    # The spatial encoder cannot even be validated until its neighbour bank is written: the bank
    # is a runtime input the manifest must list, and build_model_index records only artifacts
    # already on disk. Build it from the same feature source the example uses, then re-scan so
    # artifacts.neighbor_bank is recorded. Skipped without --features, since the bank needs data.
    if task == model_registry.TASK_SPATIAL_ENCODING and args.features is not None:
        from ephysatlas.models.encoder_inpainting import build_neighbor_bank

        df_source = _read_encoder_source(args.features, index["outputs"]["columns"])
        build_neighbor_bank(path_model, df_source, index)
        index = model_registry.build_model_index(path_model, method=args.method)

    # Fail here rather than shipping a repository that looks complete and only breaks when a
    # stranger loads it.
    model_registry.validate_artifacts(path_model, index)
    # Without a repo id the card still renders; the quickstart carries a visible placeholder
    # rather than a wrong default, since there is no single repo across model families.
    write_card(path_model, index, args.repo_id or _default_repo_id(task), args.revision)
    if args.features is not None:
        _write_example_and_selftest(path_model, index, args.features, args.n_example_channels)
    else:
        _logger.warning("--features not given: no example/ or golden file written")

    # Last, so the digests cover the manifest, the card, the licence and the golden example.
    # Anything written after this point would ship unverified.
    model_registry.write_checksums(path_model)
    model_registry.verify_checksums(path_model, missing_ok=False)

    _logger.info(f"packaged model at {path_model}")
    for p in sorted(path_model.rglob("*")):
        if p.is_file():
            _logger.info(f"   {p.relative_to(path_model)} ({p.stat().st_size / 1e6:.2f} MB)")

    if not args.upload:
        _logger.info("packaging only. Re-run with --upload (and HF_TOKEN set) to publish.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        _logger.error("--upload given but HF_TOKEN is not set; refusing to publish")
        return 1
    if not args.repo_id:
        _logger.error("--upload requires --repo-id; refusing to guess a destination")
        return 1
    # Never upload a release that already fails locally: a corrupt publish is far more
    # expensive to withdraw than to prevent, since a pushed tag is meant to be immutable.
    model_registry.verify_checksums(path_model, missing_ok=False)
    source = model_registry.HFModelSource(repo_id=args.repo_id, token=token)
    source.upload(
        path_model,
        revision=args.revision,
        message=f"Publish {path_model.name}",
        private=not args.public,
    )
    _logger.info(f"https://huggingface.co/{args.repo_id}")
    if args.revision:
        _logger.info(f"pinned revision: {args.revision} (tag on main)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
