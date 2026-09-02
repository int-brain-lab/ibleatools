import hashlib
import logging
from typing import List, Tuple
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from one.api import ONE
import iblutil.random
from ephysatlas import features
from ephysatlas import model_registry

logger = logging.getLogger(__name__)


def save_model(path_model, classifier, meta, subfolder="", identifier=None):
    """Save model to disk in ubj format with associated meta-data and a hash.

    The model is a set of files in a folder named after the meta-data 'VINTAGE' and 'REGION_MAP' fields,
    with the hash as suffix e.g. 2023_W41_Cosmos_dfd731f0.

    Only the **root** model carries a ``meta.yaml``; fold subfolders are written weights-only.
    The publication manifest (``ephysatlas_model.json``) lists the folds and supplies their
    ``model_class``, so a per-fold ``meta.yaml`` would only duplicate the root's -- see
    :func:`ephysatlas.model_registry.write_manifest`.

    Args:
        path_model (Path): Base path where the model will be saved.
        classifier: The classifier object to save.
        meta (dict): Metadata dictionary containing model information.
        subfolder (str, optional): Optional level to add to the model path, for example 'FOLD01' will write to
            2023_W41_Cosmos_dfd731f0/FOLD01/. When set, marks this as a fold and suppresses
            the ``meta.yaml`` write. Defaults to "".
        identifier (str, optional): Optional identifier for the model, defaults to a 8 character hexdigest of the meta data.

    Returns:
        Path: Path where the model was saved.
    """
    meta["MODEL_CLASS"] = (
        f"{classifier.__class__.__module__}.{classifier.__class__.__name__}"
    )
    if identifier is None:
        identifier = iblutil.random.name_from_hash(
            hashlib.md5(yaml.dump(meta).encode("utf-8"))
        )
    path_model = path_model.joinpath(
        f"{meta['VINTAGE']}_{meta['REGION_MAP']}_{identifier}", subfolder
    )
    path_model.mkdir(exist_ok=True, parents=True)
    # Folds (subfolder set) hold only model.ubj: the manifest carries their metadata.
    if not subfolder:
        with open(path_model.joinpath("meta.yaml"), "w+") as fid:
            fid.write(yaml.dump(dict(meta)))
    classifier.save_model(path_model.joinpath("model.ubj"))
    return path_model


def download_model(
    local_path: Path,
    model_name: str,
    one: ONE = None,
    overwrite=False,
    revision: str = None,
    source: str = "auto",
) -> Path:
    """Download a trained model, from the Hugging Face Hub or from AWS S3.

    Delegates to :func:`ephysatlas.model_registry.resolve_model`, which tries the public
    Hugging Face Hub first and falls back to the private S3 bucket. ``one`` is therefore
    optional: it is only needed for the S3 route.

    Example:
        >>> download_model(Path('/mnt/s0/ephys-atlas-decoding/models'), '2024_W50_Cosmos_lid-basket-sense', one=one)

    Args:
        local_path (Path): Local directory where the model will be downloaded.
        model_name (str): Model folder name, or a ``org/repo`` Hugging Face id.
        one (ONE, optional): ONE client instance, required only for the S3 fallback.
        overwrite (bool, optional): If True, overwrite existing files. Defaults to False.
        revision (str, optional): Hugging Face branch/tag to pin. Ignored by S3.
        source (str, optional): ``"auto"``, ``"hf"`` or ``"s3"``. Defaults to ``"auto"``.

    Returns:
        Path: Path to the downloaded model directory.
    """
    return model_registry.resolve_model(
        model_name,
        revision=revision,
        source=source,
        cache_dir=Path(local_path),
        one=one,
        overwrite=overwrite,
    )


def _load_xgb(path_model: Path, manifest: dict = None):
    """Load an XGBoost classifier from the weights file its manifest names.

    Args:
        path_model (Path): Model directory.
        manifest (dict, optional): Parsed manifest. For a fold directory the caller passes the
            *parent* manifest (folds ship weights only, with no manifest of their own); its
            ``artifacts.weights`` names the file, defaulting to ``model.ubj``.

    Returns:
        XGBClassifier: The loaded classifier.
    """
    weights = ((manifest or {}).get("artifacts") or {}).get("weights", "model.ubj")
    classifier = XGBClassifier()
    classifier.load_model(path_model.joinpath(weights))
    return classifier


# Dispatch table keyed on the model class named by the manifest (or meta.yaml). Every loader
# takes ``(path_model, manifest)`` and returns an in-memory model: passing the whole manifest
# rather than one hand-picked ``weights`` string is what lets a family whose model is several
# files (the spatial encoder: weights, context volumes, a neighbour bank) register here at all.
# Deliberately xgboost-only. The spatial encoder is NOT registered here even though it has a
# loader, because this module imports xgboost at scope (line 9): any process that reached a torch
# loader through this table would hold both runtimes at once, and on macOS arm64 that segfaults
# at the first torch tensor copy. The encoder is loaded through
# `ephysatlas.models.encoder_inpainting` instead, which is the path `load_pretrained` uses.
# This is the concrete cost the design doc's §9 split (move the registry out of this module)
# would remove.
MODEL_LOADERS = {
    "xgboost.sklearn.XGBClassifier": _load_xgb,
}


def load_model(path_model):
    """Load a trained model from disk, from the manifest or from ``meta.yaml``.

    Either file is sufficient. The publication manifest
    (``ephysatlas_model.json``) is authoritative when present — it names the model class and
    the weights file — so a model published with a manifest and no ``meta.yaml`` loads fine.
    That matters for families that do not go through ``save_model`` at all, such as the torch
    spatial encoder. When only ``meta.yaml`` exists, its ``MODEL_CLASS`` is used instead.

    The returned ``model_info`` is always meta-shaped, so existing callers such as
    ``infer_regions`` (which reads ``model_info["FEATURES"]``) keep working either way.

    Args:
        path_model (Path or str): Directory containing the model files.

    Returns:
        tuple: A tuple containing:
            - classifier: The loaded classifier object (usually an XGBClassifier).
            - model_info (dict): Model metadata, in ``meta.yaml`` key shape.

    Raises:
        FileNotFoundError: If the directory has neither a manifest nor ``meta.yaml``.
        ValueError: If the model class has no registered loader.
    """
    path_model = Path(path_model)
    manifest = model_registry.read_manifest(path_model)
    meta_file = path_model.joinpath("meta.yaml")
    model_info = yaml.safe_load(meta_file.read_text()) if meta_file.exists() else None
    if manifest is None and model_info is None:
        raise FileNotFoundError(
            f"{path_model} has neither {model_registry.MODEL_MANIFEST_FILE} nor meta.yaml"
        )

    model_class = (
        (manifest or {}).get("model_class")
        or (model_info or {}).get("MODEL_CLASS")
        or "xgboost.sklearn.XGBClassifier"
    )
    if model_class not in MODEL_LOADERS:
        raise ValueError(
            f"no loader registered for MODEL_CLASS {model_class!r}; "
            f"known: {sorted(MODEL_LOADERS)}"
        )
    classifier = MODEL_LOADERS[model_class](path_model, manifest or {})
    if model_info is None:
        model_info = model_registry.meta_from_manifest(manifest)
    return classifier, model_info


def _step_viterbi(
    mu_prev: np.ndarray,
    emission_probs: np.ndarray,
    transition_probs: np.ndarray,
    observed_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one step of the Viterbi algorithm.

    Args:
        mu_prev (np.ndarray): Probability distribution with shape (num_hidden), the previous mu.
        emission_probs (np.ndarray): The emission probability matrix (num_hidden, num_observed).
        transition_probs (np.ndarray): The transition probability matrix, with shape (num_hidden, num_hidden).
        observed_state (int): The observed state at the current step.

    Returns:
        tuple: A tuple containing:
            - mu_new (np.ndarray): The mu for the next step.
            - max_prev_states (np.ndarray): The maximizing previous state, before the current state,
              as an int array with shape (num_hidden).
    """

    pre_max = mu_prev * transition_probs.T
    max_prev_states = np.argmax(pre_max, axis=1)
    max_vals = pre_max[np.arange(len(max_prev_states)), max_prev_states]
    mu_new = max_vals * emission_probs[:, observed_state]

    return np.array(mu_new).flatten(), np.array(max_prev_states).flatten()


def viterbi(
    emission_probs: np.ndarray,
    transition_probs: np.ndarray,
    start_probs: np.ndarray,
    observed_states: List[int],
) -> Tuple[List[int], float]:
    """Run the Viterbi algorithm to get the most likely state sequence.

    Args:
        emission_probs (np.ndarray): The emission probability matrix (num_hidden, num_observed).
        transition_probs (np.ndarray): The transition probability matrix, with shape (num_hidden, num_hidden).
        start_probs (np.ndarray): The initial probabilities for each state, with shape (num_hidden).
        observed_states (List[int]): The observed states at each step.

    Returns:
        tuple: A tuple containing:
            - sequence (List[int]): The most likely series of states.
            - sequence_prob (float): The joint probability of that series of states and the observed.

    Note:
        This implementation is based on the article "Coding the Viterbi Algorithm in Numpy" by Benjamin Bolte (2020).
        Available at: https://ben.bolte.cc/viterbi
    """
    num_hidden_states = transition_probs.shape[0]
    num_observed_states = emission_probs.shape[1]
    transition_probs = np.array(
        transition_probs
    )  # if np.matrix, the dimensions are inconsistent
    observed_states = np.array(observed_states).astype(int)

    assert transition_probs.shape == (num_hidden_states, num_hidden_states)
    assert transition_probs.sum(1).mean() == 1
    assert emission_probs.shape == (num_hidden_states, num_observed_states)
    assert emission_probs.sum(1).mean()
    assert start_probs.shape == (num_hidden_states,)

    # Runs the forward pass, storing the most likely previous state.
    mu = start_probs * emission_probs[:, observed_states[0]]
    previous_states = np.zeros((len(observed_states), num_hidden_states)).astype(
        observed_states.dtype
    )
    all_prev_states = []  # tud
    for i, observed_state in enumerate(observed_states[1:]):
        mu, prevs = _step_viterbi(mu, emission_probs, transition_probs, observed_state)
        previous_states[i, :] = prevs
        all_prev_states.append(prevs)  # tud

    # Traces backwards to get the maximum likelihood sequence.
    # Traces backwards
    sequence = np.zeros_like(observed_states)
    sequence[-1] = np.argmax(mu)
    sequence_prob = mu[sequence[-1]]
    for i in np.arange(len(observed_states) - 1, 0, -1):
        sequence[i - 1] = previous_states[i - 1, sequence[i]]
    return sequence, sequence_prob


def infer_regions(df_inference, path_model, n_folds=5, denoise=False):
    """Infer brain regions using a trained classifier model across multiple folds.

    This function loads a trained model for each fold and performs inference on the input data.
    It applies denoising to the input features before prediction and aggregates results across all folds.

    Args:
        df_inference (pd.DataFrame): DataFrame containing features for inference.
        path_model (Path): Path to the directory containing the trained model folds.
        n_folds (int, optional): Number of folds to use for inference. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - predicted_probas (np.ndarray): Array of shape (n_folds, n_samples, n_classes) containing
              prediction probabilities for each fold.
            - predicted_region (np.ndarray): Array of shape (n_folds, n_samples) containing
              predicted region labels for each fold.
    """
    for fold in range(n_folds):
        classifier, model_info = load_model(path_model.joinpath(f"FOLD0{fold}"))

        if denoise:
            df_inference = features.denoise_dataframe(df_inference)

        x_test = df_inference.loc[:, model_info["FEATURES"]].values
        y_pred = classifier.predict(x_test)
        y_probas = classifier.predict_proba(x_test)

        if fold == 0:
            predicted_probas = np.zeros((n_folds, y_probas.shape[0], y_probas.shape[1]))
            predicted_region = np.zeros((n_folds, y_pred.shape[0]))
        predicted_probas[fold] = y_probas
        predicted_region[fold] = y_pred

    return predicted_probas, predicted_region


class RegionClassifier:
    """A trained region classifier, ready to predict on a features DataFrame.

    Wraps the fold models and the ``model_index.json`` manifest so that a caller needs
    neither ``iblatlas`` to read the output nor prior knowledge of the feature list.

    Example:
        >>> clf = RegionClassifier.from_pretrained('international-brain-lab/ephys-atlas-region-classifier')
        >>> out = clf.predict(df_features)
        >>> out.head()

    Attributes:
        path_model (Path): Local model directory.
        index (dict): Contents of ``ephysatlas_model.json`` (or an equivalent derived from
            ``meta.yaml`` when the manifest is absent).
        config (dict): The manifest's task-specific ``config`` block -- class ids, acronyms,
            region map and accuracy.
        inputs (dict): The manifest's ``inputs`` block -- row identity and feature list.
    """

    def __init__(self, path_model, index: dict = None):
        self.path_model = Path(path_model)
        self.index = index if index is not None else self._read_index()
        self.config = self.index["config"]
        self.inputs = self.index["inputs"]

    def _read_index(self) -> dict:
        """Read the manifest, or synthesise one from a legacy ``meta.yaml``."""
        manifest = model_registry.read_manifest(self.path_model)
        if manifest is not None:
            return manifest
        # Models saved before the manifest existed carry the same information in meta.yaml.
        logger.warning(
            f"{model_registry.MODEL_MANIFEST_FILE} missing; deriving it from meta.yaml"
        )
        meta = yaml.safe_load(self.path_model.joinpath("meta.yaml").read_text())
        folds_root = self.path_model.joinpath("folds")
        base = folds_root if folds_root.exists() else self.path_model
        index = {
            "model_id": self.path_model.name,
            "task": model_registry.TASK_REGION_CLASSIFICATION,
            "model_class": meta.get("MODEL_CLASS"),
            "vintage": meta.get("VINTAGE"),
            "artifacts": {
                "weights": "model.ubj",
                "folds": sorted(p.name for p in base.glob("FOLD*")),
            },
        }
        index.update(model_registry._blocks_region_classification(meta, self.path_model))
        return index

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str = None,
        cache_dir: Path = None,
        one=None,
        source: str = "auto",
    ) -> "RegionClassifier":
        """Fetch a model by id and return a ready-to-use classifier.

        Args:
            model_id (str): Hugging Face ``org/repo``, or an S3 model folder name.
            revision (str, optional): Hugging Face branch/tag to pin.
            cache_dir (Path, optional): Download location.
            one (optional): ONE client, needed only for the S3 fallback.
            source (str, optional): ``"auto"``, ``"hf"`` or ``"s3"``.

        Returns:
            RegionClassifier: Classifier backed by the downloaded directory.
        """
        return cls(
            model_registry.resolve_model(
                model_id, revision=revision, source=source, cache_dir=cache_dir, one=one
            )
        )

    def _fold_dirs(self) -> list:
        """Return the fold model directories, or the model root if there are no folds."""
        folds_root = self.path_model.joinpath("folds")
        base = folds_root if folds_root.exists() else self.path_model
        names = (self.index.get("artifacts") or {}).get("folds") or []
        # A fold is loadable when its weights are present. Folds ship weights only -- no
        # meta.yaml -- so keying on the weights file (not meta.yaml) is what keeps every fold
        # in the ensemble instead of silently dropping the meta-less ones.
        weights = (self.index.get("artifacts") or {}).get("weights", "model.ubj")
        dirs = [d for d in (base.joinpath(n) for n in names) if d.joinpath(weights).exists()]
        if names and 0 < len(dirs) < len(names):
            # Losing *some* folds is the dangerous case, and it used to pass in silence: the
            # ensemble quietly averages fewer models than the manifest and the model card
            # advertise, and fold_agreement is then computed over the survivors -- reporting
            # unanimity among two folds while claiming five.
            logger.warning(
                f"manifest declares {len(names)} folds but only {len(dirs)} are loadable "
                f"({sorted(d.name for d in dirs)}); predictions and fold_agreement will be "
                f"computed from those alone"
            )
        if not dirs:
            # No folds published: fall back to the single global model.
            dirs = [self.path_model]
        return dirs

    def _model_dirs(self, estimator: str) -> list:
        """Resolve which model directories an estimator mode should use.

        Args:
            estimator (str): ``"ensemble"`` or ``"global"``.

        Returns:
            list[Path]: Directories to load and average over.

        Raises:
            ValueError: On an unknown mode, or if ``"global"`` is asked of a model that
                publishes only folds.
        """
        if estimator == "global":
            weights = (self.index.get("artifacts") or {}).get("weights", "model.ubj")
            if not self.path_model.joinpath(weights).exists():
                raise ValueError(
                    f"estimator='global' needs {weights} at {self.path_model}, which this "
                    f"model does not publish; use estimator='ensemble'"
                )
            return [self.path_model]
        if estimator == "ensemble":
            dirs = self._fold_dirs()
            if dirs == [self.path_model]:
                logger.warning(
                    "estimator='ensemble' but this model publishes no folds; "
                    "falling back to the single global model and fold_agreement will be NaN"
                )
            return dirs
        raise ValueError(
            f"unknown estimator {estimator!r}; expected 'ensemble' or 'global'"
        )

    def _acronyms(self) -> np.ndarray:
        """Class acronyms, from the manifest if present, otherwise resolved from the atlas."""
        acronyms = self.config.get("class_acronyms")
        if acronyms is None:
            acronyms = model_registry.class_acronyms(
                self.config["classes"], self.config["region_map"]
            )
        if acronyms is None or len(acronyms) != len(self.config["classes"]):
            raise ValueError(
                f"cannot label predictions: {len(self.config['classes'])} classes but "
                f"{0 if acronyms is None else len(acronyms)} acronyms"
            )
        return np.asarray(acronyms)

    def predict(
        self, df, denoise: bool = False, estimator: str = "ensemble"
    ) -> pd.DataFrame:
        """Predict a brain region per channel.

        Args:
            df (pd.DataFrame): Features, indexed by ``(pid, channel)``. Must contain every
                column listed in the manifest's ``features``.
            denoise (bool, optional): Apply :func:`ephysatlas.features.denoise_dataframe`
                first. Leave False if ``df`` already holds denoised features.
            estimator (str, optional): Which weights to use.

                * ``"ensemble"`` (default) -- average the per-fold models. Slightly better
                  calibrated, and the only mode that yields a meaningful ``fold_agreement``.
                * ``"global"`` -- the single model trained on all channels. One fifth of the
                  inference cost; ``fold_agreement`` comes back as NaN because no folds were
                  consulted.

                The two disagree on a small fraction of channels, so do not mix them within
                one analysis.

        Returns:
            pd.DataFrame: Indexed like ``df``, with columns ``predicted_acronym``,
            ``predicted_atlas_id``, ``prediction_probability`` (fold-averaged probability of
            the winning class), ``fold_agreement`` (fraction of folds voting for the winner),
            and one ``p_<acronym>`` column per class holding the fold-averaged probability.

            The prediction columns are deliberately namespaced: both the channel feature
            table and the cluster table already carry histology-derived ``acronym`` and
            ``atlas_id`` columns, so bare names would collide on ``df.join(out)`` and would
            silently overwrite ground truth on assignment.

        Raises:
            KeyError: If any required feature column is absent, naming the missing ones.
            ValueError: If the manifest's ``inputs.features`` no longer matches its recorded
                order digest, i.e. the published feature list has been edited or reordered.
        """
        feature_names = list(self.inputs["features"])
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} feature(s) required by this model are missing from the "
                f"input DataFrame: {missing}. Expected all of: {feature_names}"
            )
        # After the missing-column check on purpose: someone who simply forgot a column should
        # get the KeyError that names it, not an integrity error about the manifest.
        model_registry.validate_feature_order(
            self.inputs["features"], self.inputs.get("feature_order_sha256")
        )
        if denoise:
            df = features.denoise_dataframe(df)

        classes = np.asarray(self.config["classes"])
        x = df.loc[:, feature_names].values.astype(float)

        # Resolve the loader once from the parent manifest's model_class, then apply it to
        # each directory. Folds carry no meta.yaml of their own, so load_model (which insists
        # on a manifest or meta.yaml in the directory) cannot resolve them -- but the family is
        # known from the root manifest, and every fold stores its weights under the same name.
        model_class = self.index["model_class"]
        loader = MODEL_LOADERS.get(model_class)
        if loader is None:
            raise ValueError(
                f"no loader registered for MODEL_CLASS {model_class!r}; "
                f"known: {sorted(MODEL_LOADERS)}"
            )

        model_dirs = self._model_dirs(estimator)
        probas = np.zeros((len(model_dirs), x.shape[0], classes.size))
        for i, model_dir in enumerate(model_dirs):
            classifier = loader(model_dir, self.index)
            p = classifier.predict_proba(x)
            if p.shape[1] != classes.size:
                raise ValueError(
                    f"{model_dir.name}: model emits {p.shape[1]} classes but the manifest "
                    f"declares {classes.size}"
                )
            probas[i] = p

        mean_probas = probas.mean(axis=0)
        winner = np.argmax(mean_probas, axis=1)
        if len(model_dirs) > 1:
            per_fold_winner = np.argmax(probas, axis=2)
            agreement = (per_fold_winner == winner[np.newaxis, :]).mean(axis=0)
        else:
            # A single model has nothing to agree with. NaN rather than 1.0, which would
            # imply unanimity among folds that were never consulted.
            agreement = np.full(x.shape[0], np.nan)

        acronyms = self._acronyms()
        out = pd.DataFrame(index=df.index)
        out["predicted_acronym"] = acronyms[winner]
        out["predicted_atlas_id"] = classes[winner]
        out["prediction_probability"] = mean_probas[np.arange(winner.size), winner]
        out["fold_agreement"] = agreement
        for j, acronym in enumerate(acronyms):
            out[f"p_{acronym}"] = mean_probas[:, j]
        return out

    def selftest(self, rtol: float = 1e-5) -> bool:
        """Reproduce the shipped golden predictions, if the model ships an example.

        Converts silent numerical drift (changed transforms, an incompatible xgboost, a
        reordered class vector) into one explicit failure.

        Args:
            rtol (float, optional): Relative tolerance on the probability comparison.

        Returns:
            bool: True when the recomputed predictions match the shipped ones.

        Raises:
            FileNotFoundError: If the model does not ship ``example/`` files.
            AssertionError: If the predictions differ.
        """
        example = self.path_model.joinpath("example")
        sample_file = example.joinpath("features_sample.parquet")
        expected_file = example.joinpath("expected_predictions.parquet")
        if not (sample_file.exists() and expected_file.exists()):
            raise FileNotFoundError(f"no example/golden files under {example}")
        # Pin the estimator: the golden file was produced by the ensemble, so the comparison
        # must not silently follow a future change to predict()'s default.
        got = self.predict(pd.read_parquet(sample_file), estimator="ensemble")
        expected = pd.read_parquet(expected_file)
        mismatched = (
            got["predicted_acronym"].values != expected["predicted_acronym"].values
        ).sum()
        assert mismatched == 0, f"{mismatched} of {len(got)} predicted acronyms differ"
        np.testing.assert_allclose(
            got["prediction_probability"].values,
            expected["prediction_probability"].values,
            rtol=rtol,
        )
        logger.info(f"selftest passed on {len(got)} channels")
        return True
