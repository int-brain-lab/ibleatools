import unittest
import tempfile
from pathlib import Path
import shutil

import numpy as np
import numpy.testing

from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


import ephysatlas.regionclassifier


class TestModelIO(unittest.TestCase):
    def test_read_after_write(self):
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data["data"], data["target"], test_size=0.2
        )
        classifier = XGBClassifier(
            n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic"
        )
        classifier.fit(X_train, y_train)
        # Create a temporary directory that works on both Windows and Linux
        model_info = {"REGION_MAP": "Cosmos", "VINTAGE": "2024_W50"}
        try:
            temp_dir = Path(tempfile.mkdtemp())
            model_path = ephysatlas.regionclassifier.save_model(
                temp_dir, classifier=classifier, meta=model_info
            )
            _classifier, _model_info = ephysatlas.regionclassifier.load_model(
                model_path
            )
            for k, v in model_info.items():
                self.assertEqual(model_info[k], _model_info[k])

            np.testing.assert_equal(
                classifier.predict(X_test), _classifier.predict(X_test)
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestViterbi(unittest.TestCase):
    def test_viterbi(self):
        num_hidden_states = 3
        num_observed_states = 2
        num_time_steps = 4
        print(
            f"Testing viterbi with {num_hidden_states} hidden states, {num_observed_states} observed states, and {num_time_steps} time steps"
        )
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

        s, p = ephysatlas.regionclassifier.viterbi(
            emission_probs, transition_probs, init_hidden_probs, observed_states
        )
        np.testing.assert_array_equal(s, [2, 0, 2, 0])
        assert p == 0.0212625
