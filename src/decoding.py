import hashlib
from typing import List, Tuple
from pathlib import Path
import yaml

import numpy as np
from xgboost import XGBClassifier

from ephys_atlas.data import save_model  # NO QA
from iblutil.util import Bunch


def load_model(path_model):
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

    return mu_new, max_prev_states


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

    assert transition_probs.shape == (num_hidden_states, num_hidden_states)
    assert transition_probs.sum(1).mean() == 1
    assert emission_probs.shape == (num_hidden_states, num_observed_states)
    assert emission_probs.sum(1).mean()
    assert start_probs.shape == (num_hidden_states,)

    # Runs the forward pass, storing the most likely previous state.
    mu = start_probs * emission_probs[:, observed_states[0]]
    all_prev_states = []
    for observed_state in observed_states[1:]:
        mu, prevs = _step_viterbi(mu, emission_probs, transition_probs, observed_state)
        all_prev_states.append(prevs)

    # Traces backwards to get the maximum likelihood sequence.
    state = np.argmax(mu)
    sequence_prob = mu[state]
    state_sequence = [state]
    for prev_states in all_prev_states[::-1]:
        state = prev_states[state]
        state_sequence.append(state)

    return state_sequence[::-1], sequence_prob
