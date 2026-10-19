"""Features aggregated over several snippets of one recording.

A feature calculator computes one snippet at a time, and a single snippet is noisy. These functions
pick snippet start times, compute every snippet with any ``BaseFeatureCalculator``, then aggregate
them as ``ephysatlas.aggregation.produce_output_dataframes`` does: per-channel median, denoising and
waveform remap.

Functions
---------
get_snippet_start_times
    Spread snippet start times over a recording.
check_snippet_overlap
    Raise or warn when snippet windows overlap.
compute_aggregated_features
    Compute snippets with a calculator and return the aggregated, denoised features.
"""

from __future__ import annotations

import dataclasses
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ephysatlas.aggregation import (
    apply_waveform_remap,
    denoise_raw_features_data,
    get_aggregated_raw_features,
)

from .types import FeatureComputationOptions, SnippetWindow

if TYPE_CHECKING:
    from .base import BaseFeatureCalculator

LOGGER = logging.getLogger(__name__)


def get_snippet_start_times(
    recording_duration: float,
    n_snippets: int,
    snippet_duration: float,
    margin: float = 0.0,
    method: str = "stratified",
    seed: int = 0,
    on_overlap: str = "raise",
) -> np.ndarray:
    """Spread snippet start times over a recording, away from its start and end.

    Args:
        recording_duration (float): Recording duration in seconds.
        n_snippets (int): Number of snippets.
        snippet_duration (float): Duration of each snippet in seconds.
        margin (float): Seconds excluded at the start and at the end of the recording.
        method (str): ``"stratified"`` splits the usable range into ``n_snippets`` equal bins and
            draws one start uniformly in each (random but spread); ``"linspace"`` spaces the
            starts evenly.
        seed (int): Random seed of the ``"stratified"`` draw.
        on_overlap (str): ``"raise"`` or ``"warn"`` when snippets overlap.

    Returns:
        np.ndarray: Sorted start times in seconds, rounded to 0.1 s (the precision of the snippet
        folder names), shape ``(n_snippets,)``.

    Raises:
        ValueError: If the recording is too short for the margins and the snippet duration, or if
            snippets overlap and ``on_overlap="raise"``.
    """
    t_first = margin
    t_last = recording_duration - margin - snippet_duration
    if t_last <= t_first:
        raise ValueError(
            f"Recording of {recording_duration:.1f} s is too short for {margin} s margins "
            f"and {snippet_duration} s snippets"
        )
    if method == "linspace":
        t_starts = np.linspace(t_first, t_last, n_snippets)
    else:
        edges = np.linspace(t_first, t_last, n_snippets + 1)
        t_starts = np.random.default_rng(seed).uniform(edges[:-1], edges[1:])
    t_starts = np.round(t_starts, 1)
    check_snippet_overlap(
        [SnippetWindow(t, snippet_duration, snippet_duration) for t in t_starts],
        on_overlap,
    )
    return t_starts


def check_snippet_overlap(
    windows: Sequence[SnippetWindow], on_overlap: str = "raise"
) -> None:
    """Raise or warn when snippet windows overlap.

    A window covers ``[t_start, t_start + max(duration_ap, duration_lf))``.

    Args:
        windows (Sequence[SnippetWindow]): Snippet windows, in any order.
        on_overlap (str): ``"raise"`` raises a ``ValueError``, ``"warn"`` logs a warning.

    Raises:
        ValueError: If two windows overlap and ``on_overlap="raise"``.
    """
    windows = sorted(windows, key=lambda window: window.t_start)
    for previous, current in zip(windows[:-1], windows[1:]):
        previous_end = previous.t_start + max(
            previous.duration_ap, previous.duration_lf
        )
        if current.t_start < previous_end:
            message = f"Snippets starting at {previous.t_start} s and {current.t_start} s overlap"
            if on_overlap == "raise":
                raise ValueError(message)
            LOGGER.warning(message)


def compute_aggregated_features(
    calculator: BaseFeatureCalculator,
    windows: Sequence[SnippetWindow],
    options: FeatureComputationOptions | None = None,
    on_overlap: str = "raise",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute features on several snippets, then aggregate and denoise them.

    Each snippet is computed with ``calculator.compute_snippet``; its manifest record is one row
    of the snippet table read by ``get_aggregated_raw_features`` (per-channel median), followed by
    ``denoise_raw_features_data`` and ``apply_waveform_remap``.

    Args:
        calculator (BaseFeatureCalculator): Calculator of the recording.
        windows (Sequence[SnippetWindow]): Snippet windows, e.g. from ``get_snippet_start_times``.
        options (FeatureComputationOptions, optional): Options of every snippet computation. The
            aggregation reads the snippet features from disk: without ``output_dir`` they are
            written to a temporary folder, removed afterwards.
        on_overlap (str): ``"raise"`` or ``"warn"`` when windows overlap.
        n_jobs (int): Snippets computed in parallel with joblib (the calculator must pickle).

    Returns:
        pd.DataFrame: Aggregated and denoised features indexed by ``(pid, channel)``, with
        ``pid = calculator.name``.
    """
    options = options or FeatureComputationOptions()
    if options.output_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            options = dataclasses.replace(options, output_dir=Path(tmp))
            return compute_aggregated_features(
                calculator, windows, options, on_overlap, n_jobs
            )

    check_snippet_overlap(windows, on_overlap)
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculator.compute_snippet)(window, options) for window in windows
    )
    snippet_df = pd.DataFrame([result.manifest_record for result in results])
    df = get_aggregated_raw_features(snippet_df)
    df = denoise_raw_features_data(df)
    return apply_waveform_remap(df)
