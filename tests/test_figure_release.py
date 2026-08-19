"""Tests for the figure loader shim's split-agreement guard.

Pure dict logic over split payloads -- no models, no torch -- so it runs anywhere. Guards
cross-model figures against silently combining models that held out different insertions.
"""

import logging
import sys
import unittest
from pathlib import Path

# examples/ is not a package; the figure scripts import their shim as a sibling by directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].joinpath("examples", "figures")))

from _release import check_split_agreement  # noqa: E402


def _holdout(test_pids, sha=None):
    return {"test_pids": list(test_pids), "split_sha256": sha}


def _kfold(pids, sha):
    return {"pids": list(pids), "folds": [{"fold": 0, "test_pids": list(pids)}], "split_sha256": sha}


class TestSplitAgreement(unittest.TestCase):
    def test_identical_digests_pass(self):
        a = _holdout(["p1", "p2"], sha="deadbeef")
        b = _holdout(["p1", "p2"], sha="deadbeef")
        self.assertTrue(check_split_agreement([a, b]))

    def test_same_test_pids_without_matching_digest_pass(self):
        # Same held-out insertions, digests absent -> compared on the test-pid set.
        a = _holdout(["p2", "p1"])
        b = _holdout(["p1", "p2"])
        self.assertTrue(check_split_agreement([a, b]))

    def test_different_test_pids_raise(self):
        a = _holdout(["p1", "p2"])
        b = _holdout(["p1", "p3"])
        with self.assertRaises(ValueError) as ctx:
            check_split_agreement([a, b], names=["clf", "enc"])
        self.assertIn("DIFFERENT held-out insertions", str(ctx.exception))

    def test_different_test_pids_warn_returns_false(self):
        a = _holdout(["p1", "p2"])
        b = _holdout(["p1", "p3"])
        with self.assertLogs(level=logging.WARNING):
            self.assertFalse(check_split_agreement([a, b], on_mismatch="warn"))

    def test_a_missing_split_cannot_be_verified_and_warns(self):
        a = _holdout(["p1", "p2"], sha="x")
        with self.assertLogs(level=logging.WARNING):
            self.assertTrue(check_split_agreement([a, None]))

    def test_single_model_is_a_noop(self):
        self.assertTrue(check_split_agreement([_holdout(["p1"])]))

    def test_kfold_with_differing_digests_cannot_be_verified(self):
        # No single held-out set to compare and digests differ -> warn, do not falsely fail.
        a = _kfold(["p1", "p2"], sha="aaa")
        b = _kfold(["p1", "p2"], sha="bbb")
        with self.assertLogs(level=logging.WARNING):
            self.assertTrue(check_split_agreement([a, b]))


if __name__ == "__main__":
    unittest.main()
