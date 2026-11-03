"""Ephys-only probe localization during acquisition.

The pieces shared by the acquisition-software integrations (SpikeGLX in Python, Open Ephys in
C++):

- :mod:`.features` -- channel features from raw LF and AP snippets, aggregated and denoised as
  the feature release; the reference every online pipeline is checked against;
- :mod:`.plan` -- the planned channel positions the localizer starts from, as a JSON file;
- :mod:`.bundle` -- a fitted localizer exported as plain arrays for software outside Python.

The localizer itself is :class:`ephysatlas.alignment.EphysOnlyLocalizer` with
``feature_set="lf_ap_no_detection"``: the LF and AP features that need no spike detection.
"""

from .features import (
    SNIPPET_S,
    aggregate_snippets,
    feature_matrix,
    read_snippet,
    snippet_features,
)
from .plan import (
    load_plan,
    plan_from_alyx,
    plan_from_release,
    plan_from_trajectory,
    save_plan,
)

# The localizer's feature set for recordings without spike detection.
FEATURE_SET = "lf_ap_no_detection"

__all__ = [
    "FEATURE_SET",
    "SNIPPET_S",
    "aggregate_snippets",
    "feature_matrix",
    "load_plan",
    "plan_from_alyx",
    "plan_from_release",
    "plan_from_trajectory",
    "read_snippet",
    "save_plan",
    "snippet_features",
]
