"""The common result of every alignment / localization method."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Method labels, as used in result files and figure legends.
METHOD_LABELS = {
    "histology": "Histology-based (channel)",
    "histology_unit": "Histology-based (channel + unit)",
    "ephys_only": "Ephys-only",
}


@dataclass
class AlignmentResult:
    """Where one probe's channels are, according to one method.

    Channel arrays are in row order (top of the probe first; see :mod:`.geometry`).

    Attributes:
        method: ``"histology"``, ``"histology_unit"`` or ``"ephys_only"``.
        pid: Insertion id ('' when unknown, e.g. a GUI probe without one).
        channel_xyz: ``[C, 3]`` estimated channel positions (m).
        recorded: ``[C, F]`` recorded features, feature units.
        predicted_std: ``[C, F]`` model prediction at ``channel_xyz``, standardised.
        recorded_std: ``[C, F]`` recorded features, standardised.
        p_good: ``[C]`` probe-confidence p(good) at ``channel_xyz`` (NaN where invalid).
        valid: ``[C]`` channels carrying a signal (used by the method and the metrics).
        trace_xyz: ``[L, 3]`` the trace the channels lie on: the extended histology trace, or
            the inferred trace for the ephys-only method.
        feature_names: The F feature names.
        cost_matrix: Histology methods: ``[C_valid, L]`` alignment cost (valid channels x trace).
        path: Histology methods: ``[P, 2]`` (valid-channel row, trace index) warping path.
        diagnostics: Method-specific values, e.g. ``candidate_scores`` / ``candidate_distance_um``
            for the ephys-only method, the unit-cost summary for the channel + unit method.
        timings: Seconds spent per step.
    """

    method: str
    pid: str
    channel_xyz: np.ndarray
    recorded: np.ndarray
    predicted_std: np.ndarray
    recorded_std: np.ndarray
    p_good: np.ndarray
    valid: np.ndarray
    trace_xyz: np.ndarray
    feature_names: list
    cost_matrix: Optional[np.ndarray] = None
    path: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return METHOD_LABELS.get(self.method, self.method)

    def save(self, path: Path) -> Path:
        """Write the arrays and scalar diagnostics to a compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            k: np.asarray(v)
            for k, v in dict(
                channel_xyz=self.channel_xyz,
                recorded=self.recorded,
                predicted_std=self.predicted_std,
                recorded_std=self.recorded_std,
                p_good=self.p_good,
                valid=self.valid,
                trace_xyz=self.trace_xyz,
                feature_names=np.asarray(self.feature_names, dtype="U"),
            ).items()
        }
        if self.cost_matrix is not None:
            arrays["cost_matrix"] = np.asarray(self.cost_matrix, dtype=np.float32)
        if self.path is not None:
            arrays["path"] = np.asarray(self.path, dtype=np.int32)
        for key, value in self.diagnostics.items():
            if isinstance(value, np.ndarray):
                arrays[f"diag_{key}"] = value
        np.savez_compressed(
            path,
            method=np.asarray(self.method),
            pid=np.asarray(self.pid),
            **arrays,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "AlignmentResult":
        with np.load(Path(path), allow_pickle=False) as f:
            diagnostics = {k[5:]: f[k] for k in f.files if k.startswith("diag_")}
            return cls(
                method=str(f["method"]),
                pid=str(f["pid"]),
                channel_xyz=f["channel_xyz"],
                recorded=f["recorded"],
                predicted_std=f["predicted_std"],
                recorded_std=f["recorded_std"],
                p_good=f["p_good"],
                valid=f["valid"],
                trace_xyz=f["trace_xyz"],
                feature_names=f["feature_names"].astype(str).tolist(),
                cost_matrix=f["cost_matrix"] if "cost_matrix" in f.files else None,
                path=f["path"] if "path" in f.files else None,
                diagnostics=diagnostics,
            )
