"""
IBLEA Tools - Tools for IBL data analysis and processing.

This package provides tools for analyzing and processing electrophysiology data from the International Brain Laboratory (IBL).
It includes functionality for feature computation, data aggregation, brain anatomy analysis, and more.

Example:
    >>> import ephysatlas
    >>> print(ephysatlas.__version__)
    >>> from ephysatlas import features, aggregation, anatomy

To run a published model, there is a single entry point for every model family:

    >>> from ephysatlas import load_pretrained
    >>> model = load_pretrained("int-brain-lab/ea-decoder-channel-xgboost", revision="2026_W32")
    >>> out = model.predict(df_features)
"""

__version__ = "0.7.0"

from ephysatlas.models import load_pretrained  # noqa: E402,F401  (public API re-export)
