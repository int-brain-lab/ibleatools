"""Progress reporting shared by the alignment methods, their offline runners and the GUI.

Every long-running function of :mod:`ephysatlas.alignment` accepts an optional ``progress``
callable ``progress(fraction, message)``: ``fraction`` in [0, 1] of the whole call, ``message`` a
short description of the current step (e.g. "Predicting features along the trace"). The GUI binds
it to a progress bar; the offline runners to a log line. ``None`` reports nothing.
"""

from __future__ import annotations

from typing import Callable, Optional

ProgressCallback = Callable[[float, str], None]


def report(progress: Optional[ProgressCallback], fraction: float, message: str) -> None:
    """Call ``progress(fraction, message)`` when a callback was given, with the fraction clipped."""
    if progress is not None:
        progress(min(max(float(fraction), 0.0), 1.0), str(message))


def sub_progress(
    progress: Optional[ProgressCallback], start: float, stop: float
) -> Optional[ProgressCallback]:
    """A callback reporting a sub-step's own [0, 1] progress as [start, stop] of the parent's."""
    if progress is None:
        return None

    def _sub(fraction: float, message: str) -> None:
        report(progress, start + (stop - start) * float(fraction), message)

    return _sub


def print_progress(prefix: str = "") -> ProgressCallback:
    """A callback printing ``[ 42%] message`` lines, each message once (for scripts)."""
    last = {"message": None}

    def _print(fraction: float, message: str) -> None:
        if message != last["message"]:
            print(f"{prefix}[{100 * fraction:3.0f}%] {message}", flush=True)
            last["message"] = message

    return _print
