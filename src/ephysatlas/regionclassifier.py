import hashlib
import json
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

    Args:
        path_model (Path): Base path where the model will be saved.
        classifier: The classifier object to save.
        meta (dict): Metadata dictionary containing model information.
        subfolder (str, optional): Optional level to add to the model path, for example 'FOLD01' will write to
            2023_W41_Cosmos_dfd731f0/FOLD01/. Defaults to "".
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


def _load_xgb(path_model: Path):
    """Load an XGBoost classifier from ``model.ubj``."""
    classifier = XGBClassifier()
    classifier.load_model(path_model.joinpath("model.ubj"))
    return classifier


# Dispatch table keyed on the 'MODEL_CLASS' field that save_model() records. Adding the
# torch spatial encoder is a new entry here, not a new abstraction.
MODEL_LOADERS = {
    "xgboost.sklearn.XGBClassifier": _load_xgb,
}


def load_model(path_model):
    """Load a trained classifier model from disk.

    This function loads both the model binary and its associated metadata from the
    specified directory. The model is expected to be in UBJ format, and the metadata
    in YAML format. The concrete model class is taken from the ``MODEL_CLASS`` field of
    ``meta.yaml`` and dispatched through :data:`MODEL_LOADERS`; models saved before that
    field existed fall back to XGBoost.

    Args:
        path_model (Path or str): Path to the directory containing the model files.
            The directory should contain 'model.ubj' and 'meta.yaml' files.

    Returns:
        tuple: A tuple containing:
            - classifier: The loaded classifier object (usually an XGBClassifier).
            - model_info (dict): Dictionary containing the model metadata.

    Raises:
        ValueError: If ``MODEL_CLASS`` names a class with no registered loader.
    """
    path_model = Path(path_model)
    # load model
    with open(path_model.joinpath("meta.yaml")) as f:
        model_info = yaml.safe_load(f)
    # The publication manifest is authoritative when present; meta.yaml is the fallback for
    # models saved before it existed.
    index_file = path_model.joinpath(model_registry.MODEL_INDEX_FILE)
    model_class = (
        json.loads(index_file.read_text()).get("model_class")
        if index_file.exists()
        else None
    ) or model_info.get("MODEL_CLASS", "xgboost.sklearn.XGBClassifier")
    if model_class not in MODEL_LOADERS:
        raise ValueError(
            f"no loader registered for MODEL_CLASS {model_class!r}; "
            f"known: {sorted(MODEL_LOADERS)}"
        )
    classifier = MODEL_LOADERS[model_class](path_model)
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
        index (dict): Contents of ``model_index.json`` (or an equivalent derived from
            ``meta.yaml`` when the manifest is absent).
        config (dict): The manifest's task-specific ``config`` block -- for this task the
            feature list, class ids, acronyms and region map.
    """

    def __init__(self, path_model, index: dict = None):
        self.path_model = Path(path_model)
        self.index = index if index is not None else self._read_index()
        self.config = self.index["config"]

    def _read_index(self) -> dict:
        """Read ``model_index.json``, or synthesise one from a legacy ``meta.yaml``."""
        index_file = self.path_model.joinpath(model_registry.MODEL_INDEX_FILE)
        if index_file.exists():
            return json.loads(index_file.read_text())
        # Models saved before the manifest existed carry the same information in meta.yaml.
        logger.warning(
            f"{model_registry.MODEL_INDEX_FILE} missing; deriving it from meta.yaml"
        )
        meta = yaml.safe_load(self.path_model.joinpath("meta.yaml").read_text())
        folds_root = self.path_model.joinpath("folds")
        base = folds_root if folds_root.exists() else self.path_model
        return {
            "model_id": self.path_model.name,
            "task": model_registry.TASK_REGION_CLASSIFICATION,
            "model_class": meta.get("MODEL_CLASS"),
            "vintage": meta.get("VINTAGE"),
            "artifacts": {
                "weights": "model.ubj",
                "folds": sorted(p.name for p in base.glob("FOLD*")),
            },
            "config": model_registry._config_region_classification(meta, self.path_model),
        }

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
        dirs = [d for d in (base.joinpath(n) for n in names) if d.joinpath("meta.yaml").exists()]
        if not dirs:
            # No folds published: fall back to the single global model.
            dirs = [self.path_model]
        return dirs

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

    def predict(self, df, denoise: bool = False) -> pd.DataFrame:
        """Predict a brain region per channel.

        Args:
            df (pd.DataFrame): Features, indexed by ``(pid, channel)``. Must contain every
                column listed in the manifest's ``features``.
            denoise (bool, optional): Apply :func:`ephysatlas.features.denoise_dataframe`
                first. Leave False if ``df`` already holds denoised features.

        Returns:
            pd.DataFrame: Indexed like ``df``, with columns ``acronym`` (predicted region),
            ``atlas_id``, ``probability`` (fold-averaged probability of the winner),
            ``fold_agreement`` (fraction of folds voting for the winner), and one column per
            class holding the fold-averaged probability.

        Raises:
            KeyError: If any required feature column is absent, naming the missing ones.
        """
        feature_names = list(self.config["features"])
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} feature(s) required by this model are missing from the "
                f"input DataFrame: {missing}. Expected all of: {feature_names}"
            )
        if denoise:
            df = features.denoise_dataframe(df)

        classes = np.asarray(self.config["classes"])
        x = df.loc[:, feature_names].values.astype(float)

        fold_dirs = self._fold_dirs()
        probas = np.zeros((len(fold_dirs), x.shape[0], classes.size))
        for i, fold_dir in enumerate(fold_dirs):
            classifier, _ = load_model(fold_dir)
            p = classifier.predict_proba(x)
            if p.shape[1] != classes.size:
                raise ValueError(
                    f"{fold_dir.name}: model emits {p.shape[1]} classes but the manifest "
                    f"declares {classes.size}"
                )
            probas[i] = p

        mean_probas = probas.mean(axis=0)
        winner = np.argmax(mean_probas, axis=1)
        per_fold_winner = np.argmax(probas, axis=2)
        agreement = (per_fold_winner == winner[np.newaxis, :]).mean(axis=0)

        acronyms = self._acronyms()
        out = pd.DataFrame(index=df.index)
        out["acronym"] = acronyms[winner]
        out["atlas_id"] = classes[winner]
        out["probability"] = mean_probas[np.arange(winner.size), winner]
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
        got = self.predict(pd.read_parquet(sample_file))
        expected = pd.read_parquet(expected_file)
        mismatched = (got["acronym"].values != expected["acronym"].values).sum()
        assert mismatched == 0, f"{mismatched} of {len(got)} predicted acronyms differ"
        np.testing.assert_allclose(
            got["probability"].values, expected["probability"].values, rtol=rtol
        )
        logger.info(f"selftest passed on {len(got)} channels")
        return True
