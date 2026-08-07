import hashlib
from typing import List, Tuple
from pathlib import Path
import yaml

import numpy as np
from xgboost import XGBClassifier

from one.remote import aws
from one.api import ONE
import iblutil.random
from ephysatlas import features


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
    local_path: Path, model_name: str, one: ONE, overwrite=False
) -> Path:
    """Download a trained model from AWS S3.

    Example:
        >>> download_model(Path('/mnt/s0/ephys-atlas-decoding/models'), '2024_W50_Cosmos_lid-basket-sense', one=one)

    Args:
        local_path (Path): Local directory where the model will be downloaded.
        model_name (str): Name of the model to download from S3.
        one (ONE): ONE client instance for AWS authentication.
        overwrite (bool, optional): If True, overwrite existing files. Defaults to False.

    Returns:
        Path: Path to the downloaded model directory.
    """
    local_path = Path(local_path)
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_folder(
        f"aggregates/atlas/models/{model_name}",
        local_path.joinpath(model_name),
        s3=s3,
        bucket_name=bucket_name,
        overwrite=overwrite,
    )
    return local_path.joinpath(model_name)


def load_model(path_model, n_jobs=None):
    """Load a trained XGBoost classifier model from disk.

    This function loads both the model binary and its associated metadata from the
    specified directory. The model is expected to be in UBJ format, and the metadata
    in YAML format.

    Args:
        path_model (Path or str): Path to the directory containing the model files.
            The directory should contain 'model.ubj' and 'meta.yaml' files.
        n_jobs (int, optional): Number of threads the classifier may use. Defaults to None,
            leaving XGBoost's own choice (all available cores) untouched. Pass 1 when the
            calling process also loads torch: both ship their own OpenMP runtime, and two
            OpenMP thread pools in one process crash or deadlock on macOS.

    Returns:
        tuple: A tuple containing:
            - classifier: The loaded classifier object (usually an XGBClassifier).
            - model_info (dict): Dictionary containing the model metadata.
    """
    path_model = Path(path_model)
    # load model
    with open(path_model.joinpath("meta.yaml")) as f:
        model_info = yaml.safe_load(f)
    # todo: this should support multiple model classes
    classifier = XGBClassifier(
        model_file=path_model.joinpath("model.ubj"), n_jobs=n_jobs
    )
    classifier.load_model(path_model.joinpath("model.ubj"))
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


def infer_regions(df_inference, path_model, n_folds=5, denoise=False, n_jobs=None):
    """Infer brain regions using a trained classifier model across multiple folds.

    This function loads a trained model for each fold and performs inference on the input data.
    It applies denoising to the input features before prediction and aggregates results across all folds.

    Args:
        df_inference (pd.DataFrame): DataFrame containing features for inference.
        path_model (Path): Path to the directory containing the trained model folds.
        n_folds (int, optional): Number of folds to use for inference. Defaults to 5.
        n_jobs (int, optional): Number of threads each fold's classifier may use, passed to
            :func:`load_model`. Defaults to None, leaving XGBoost's own choice untouched.

    Returns:
        tuple: A tuple containing:
            - predicted_probas (np.ndarray): Array of shape (n_folds, n_samples, n_classes) containing
              prediction probabilities for each fold.
            - predicted_region (np.ndarray): Array of shape (n_folds, n_samples) containing
              predicted region labels for each fold.
    """
    for fold in range(n_folds):
        classifier, model_info = load_model(
            path_model.joinpath(f"FOLD0{fold}"), n_jobs=n_jobs
        )

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
