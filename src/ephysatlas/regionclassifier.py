from typing import List, Tuple
from pathlib import Path
import yaml

import numpy as np
from xgboost import XGBClassifier

from iblutil.util import Bunch
from ephysatlas import features


def load_model(path_model):
    """
    Load a trained XGBoost classifier model from disk.

    This function loads both the model binary and its associated metadata from the
    specified directory. The model is expected to be in UBJ format, and the metadata
    in YAML format.

    Args:
        path_model: Path or str
            Path to the directory containing the model files.
            The directory should contain 'model.ubj' and 'meta.yaml' files.

    Returns:
        dict_model: Bunch
            A dictionary-like object containing:
            - 'classifier': The loaded XGBClassifier model
            - 'meta': Dictionary with model metadata loaded from meta.yaml
    """
    path_model = Path(path_model)
    # load model
    with open(path_model.joinpath("meta.yaml")) as f:
        dict_model = Bunch(
            {
                # TODO: it should be possible to use different model kinds
                "classifier": XGBClassifier(
                    model_file=path_model.joinpath("model.ubj")
                ),
                "meta": yaml.safe_load(f),
            }
        )
    dict_model.classifier.load_model(path_model.joinpath("model.ubj"))
    return dict_model


def _step_viterbi(
    mu_prev: np.ndarray,
    emission_probs: np.ndarray,
    transition_probs: np.ndarray,
    observed_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs one step of the Viterbi algorithm.

    Args:
        mu_prev: probability distribution with shape (num_hidden),
            the previous mu
        emission_probs: the emission probability matrix (num_hidden,
            num_observed)
        transition_probs: the transition probability matrix, with
            shape (num_hidden, num_hidden)
        observed_state: the observed state at the current step

    Returns:
        - the mu for the next step
        - the maximizing previous state, before the current state,
          as an int array with shape (num_hidden)
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
    """Runs the Viterbi algorithm to get the most likely state sequence.

    Args:
        emission_probs: the emission probability matrix (num_hidden,
            num_observed)
        transition_probs: the transition probability matrix, with
            shape (num_hidden, num_hidden)
        start_probs: the initial probabilies for each state, with shape
            (num_hidden)
        observed_states: the observed states at each step

    Returns:
        - the most likely series of states
        - the joint probability of that series of states and the observed

        @article{
            title    = "Coding the Viterbi Algorithm in Numpy",
            journal  = "Ben's Blog",
            author   = "Benjamin Bolte",
            year     = "2020",
            month    = "03",
            url      = "https://ben.bolte.cc/viterbi",
        }
    """
    num_hidden_states = transition_probs.shape[0]
    num_observed_states = emission_probs.shape[1]
    transition_probs = np.array(transition_probs)  # if np.matrix, the dimensions are inconsistent
    observed_states = np.array(observed_states).astype(int)

    assert transition_probs.shape == (num_hidden_states, num_hidden_states)
    assert transition_probs.sum(1).mean() == 1
    assert emission_probs.shape == (num_hidden_states, num_observed_states)
    assert emission_probs.sum(1).mean()
    assert start_probs.shape == (num_hidden_states,)

    # Runs the forward pass, storing the most likely previous state.
    mu = start_probs * emission_probs[:, observed_states[0]]
    previous_states = np.zeros((len(observed_states), num_hidden_states)).astype(observed_states.dtype)
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


def infer_regions(df_inference, path_model, n_folds=5):
    for fold in range(n_folds):
        dict_model = load_model(path_model.joinpath(f"FOLD0{fold}"))
        classifier = dict_model["classifier"]

        df_inference["outside"] = 0
        df_inference_denoised = features.denoise_dataframe(df_inference)

        x_test = df_inference_denoised.loc[:, dict_model["meta"]["FEATURES"]].values
        y_pred = classifier.predict(x_test)
        y_probas = classifier.predict_proba(x_test)

        if fold == 0:
            predicted_probas = np.zeros((n_folds, y_probas.shape[0], y_probas.shape[1]))
            predicted_region = np.zeros((n_folds, y_pred.shape[0]))
        predicted_probas[fold] = y_probas
        predicted_region[fold] = y_pred

    return predicted_probas, predicted_region
