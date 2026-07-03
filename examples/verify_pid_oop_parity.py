"""Offline end-to-end parity check: compute_features_from_pid vs *_oop.

Run this manually (it needs a live ONE client and streams a short snippet); it is
NOT part of the CI suite. It computes features for one PID both ways and reports
whether the returned DataFrames match. The deterministic, network-free
orchestration parity is covered by ``tests/test_pid_oop_parity.py``.
"""

# %% Imports and configuration
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE

from ephysatlas.feature_computation import (
    compute_features_from_pid,
    compute_features_from_pid_oop,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Edit these for your own probe / snippet. Small durations keep the check fast.
PID = "4cb60c5c-d15b-4abd-8cfd-776bc5a81dbe"
T_START = 300.0
DURATION_AP = 1.0
DURATION_LF = 1.0
FEATURES = ["lf", "csd", "ap"]  # skip "waveforms" (dartsort is nondeterministic)

# %% Compute features both ways into separate output directories
one = ONE()
tmp = Path(tempfile.mkdtemp(prefix="pid_oop_parity_"))
logger.info("Writing outputs under %s", tmp)

df_proc = compute_features_from_pid(
    pid=PID,
    t_start=T_START,
    duration_ap=DURATION_AP,
    duration_lf=DURATION_LF,
    one=one,
    features_to_compute=FEATURES,
    output_dir=tmp / "procedural",
)
df_oop = compute_features_from_pid_oop(
    pid=PID,
    t_start=T_START,
    duration_ap=DURATION_AP,
    duration_lf=DURATION_LF,
    one=one,
    features_to_compute=FEATURES,
    output_dir=tmp / "oop",
)


# %% Compare the returned DataFrames
def compare(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Log per-column max abs difference over shared columns; return True if equal."""
    a = a.sort_values("channel").reset_index(drop=True)
    b = b.sort_values("channel").reset_index(drop=True)
    only_a = sorted(set(a.columns) - set(b.columns))
    only_b = sorted(set(b.columns) - set(a.columns))
    if only_a or only_b:
        logger.warning(
            "Columns only in procedural: %s | only in oop: %s", only_a, only_b
        )

    ok = True
    for col in sorted(set(a.columns) & set(b.columns)):
        va, vb = a[col], b[col]
        if pd.api.types.is_numeric_dtype(va) and pd.api.types.is_numeric_dtype(vb):
            diff = np.nanmax(np.abs(va.to_numpy(float) - vb.to_numpy(float)))
            if diff > 0:
                logger.warning("Column %-24s max abs diff = %.3e", col, diff)
                ok = False
        elif not va.equals(vb):
            logger.warning("Column %-24s differs (non-numeric)", col)
            ok = False
    return ok


if compare(df_proc, df_oop):
    logger.info(
        "PARITY OK: compute_features_from_pid_oop matches compute_features_from_pid"
    )
else:
    logger.error("PARITY MISMATCH: see column diffs above")
