"""Released preprocessing statistics as accepted by the channel-level data loaders.

``build_channels_plus_emptyvoxels_with_neighbors`` freezes the released ephys preprocessing and may
recompute the context normalisation (for an ablation sampling the context differently); the
inference-time neighbour bank needs the full set.
"""

import unittest

import numpy as np

from ephysatlas.spatial_encoder.utils import (
    CONTEXT_STATS_KEYS,
    EPHYS_STATS_KEYS,
    _coerce_preprocessing_stats,
)


def _stats(keys=EPHYS_STATS_KEYS + CONTEXT_STATS_KEYS):
    return {k: np.full(3, 2.0, np.float32) for k in keys}


class TestCoercePreprocessingStats(unittest.TestCase):
    def test_no_stats_means_first_training(self):
        self.assertIsNone(_coerce_preprocessing_stats(None))

    def test_full_stats_are_converted(self):
        stats = _coerce_preprocessing_stats(_stats())
        self.assertEqual(set(stats), set(EPHYS_STATS_KEYS + CONTEXT_STATS_KEYS))

    def test_context_normalisation_is_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "ctx_mean"):
            _coerce_preprocessing_stats(_stats(EPHYS_STATS_KEYS))

    def test_context_normalisation_may_be_left_out_to_be_recomputed(self):
        stats = _coerce_preprocessing_stats(
            _stats(EPHYS_STATS_KEYS), require_context=False
        )
        self.assertEqual(set(stats), set(EPHYS_STATS_KEYS))

    def test_context_normalisation_is_all_or_nothing(self):
        with self.assertRaisesRegex(ValueError, "ctx_std"):
            _coerce_preprocessing_stats(
                _stats(EPHYS_STATS_KEYS + ("ctx_mean",)), require_context=False
            )

    def test_ephys_preprocessing_is_always_required(self):
        with self.assertRaisesRegex(ValueError, "e_std"):
            _coerce_preprocessing_stats(
                _stats(("rec_ephys_low_pctl", "rec_ephys_high_pctl", "e_mean")),
                require_context=False,
            )


if __name__ == "__main__":
    unittest.main()
