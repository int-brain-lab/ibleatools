import unittest

import numpy as np
from ephysatlas.regionclassifier import viterbi


class TestViterbi(unittest.TestCase):
    def test_viterbi(self):
        num_hidden_states = 3
        num_observed_states = 2
        num_time_steps = 4
        # Initializes the transition probability matrix (nlatent, nlatent).
        transition_probs = np.array(
            [
                [0.1, 0.2, 0.7],
                [0.1, 0.1, 0.8],
                [0.5, 0.4, 0.1],
            ]
        )
        # Initializes the emission probability matrix. (nlatent, nobs)
        emission_probs = np.array(
            [
                [0.1, 0.9],
                [0.3, 0.7],
                [0.5, 0.5],
            ]
        )
        # Initalizes the initial hidden probabilities (nlatent)
        init_hidden_probs = np.array([0.1, 0.3, 0.6])
        # Defines the sequence of observed states (nsteps)
        observed_states = [1, 1, 0, 1]

        s, p = viterbi(
            emission_probs, transition_probs, init_hidden_probs, observed_states
        )
        np.testing.assert_array_equal(s, [2, 0, 2, 0])
        assert p == 0.0212625
