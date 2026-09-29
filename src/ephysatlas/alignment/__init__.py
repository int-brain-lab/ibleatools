"""Probe alignment and localization with the Ephys Atlas models.

Three ways to place a Neuropixels probe's recording channels in the brain:

- :func:`align_histology` -- along the probe's reconstructed histology trace, by warping the
  recorded channel features onto the channel-level model's predictions along the trace;
- :func:`align_histology_with_units` -- the same, adding the spike-sorted units' likelihood
  under the unit-level model to the cost;
- :class:`EphysOnlyLocalizer` (:func:`localize_ephys_only` for a single probe) -- without
  histology: search the brain for the trajectory whose predicted features best explain the
  recording (Student-t likelihood + planned-trajectory prior).

All return an :class:`AlignmentResult`, scored against a reference alignment by
:func:`alignment_metrics` and drawn by :func:`plot_alignment_result`. Long-running calls take a
``progress(fraction, message)`` callback. Models load from the Hugging Face Hub
(:class:`ChannelModel`, :class:`UnitModel`); :mod:`.runner` runs many probes offline.
"""

from .ephys_only import EphysOnlyLocalizer, PlannedTrajectoryUnavailable, localize_ephys_only
from .histology import align_histology
from .metrics import alignment_metrics, result_metrics
from .models import CHANNEL_MODEL_REPO_ID, UNIT_MODEL_REPO_ID, VINTAGE, ChannelModel, UnitModel
from .plotting import plot_alignment_result
from .progress import ProgressCallback, print_progress
from .result import METHOD_LABELS, AlignmentResult
from .unit_cost import align_histology_with_units

__all__ = [
    "AlignmentResult",
    "CHANNEL_MODEL_REPO_ID",
    "ChannelModel",
    "EphysOnlyLocalizer",
    "METHOD_LABELS",
    "PlannedTrajectoryUnavailable",
    "ProgressCallback",
    "UNIT_MODEL_REPO_ID",
    "UnitModel",
    "VINTAGE",
    "align_histology",
    "align_histology_with_units",
    "alignment_metrics",
    "localize_ephys_only",
    "plot_alignment_result",
    "print_progress",
    "result_metrics",
]
