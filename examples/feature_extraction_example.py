"""Basic feature extraction from a probe insertion ID (PID).

Computes AP / LF / CSD features for a short snippet using the public
``compute_features_from_pid`` entry point, which loads the raw data through ONE,
destripes it, computes features, and writes them under ``output_dir``.

Run as a script or cell-by-cell (``# %%`` cells).
"""

# %% Imports and configuration
import logging
import tempfile
from pathlib import Path

from one.api import ONE

from ephysatlas.feature_calculators import CsdParams, FeatureParams
from ephysatlas.feature_computation import compute_features_from_pid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A valid probe insertion ID and where to write the outputs.
pid = "4cb60c5c-d15b-4abd-8cfd-776bc5a81dbe"
output_dir = Path(tempfile.mkdtemp(prefix="ephysatlas_features_"))
logger.info("Writing features under %s", output_dir)

one = ONE()

# %% Compute features for a 1-second snippet
# duration_ap / duration_lf set the AP and LF snippet lengths independently
# (the older single ``duration`` argument is deprecated).
df = compute_features_from_pid(
    pid=pid,
    one=one,
    t_start=300.0,
    duration_ap=1.0,
    duration_lf=1.0,
    features_to_compute=["lf", "csd", "ap"],
    output_dir=output_dir,
)
logger.info("Computed features: shape=%s, columns=%s", df.shape, sorted(df.columns))

# %% Optional: override per-feature parameters
# ``feature_params`` accepts the typed objects OR a nested dict; here we turn off
# CSD scaling. Only the options you set change; everything else keeps its default.
df_unscaled_csd = compute_features_from_pid(
    pid=pid,
    one=one,
    t_start=300.0,
    duration_ap=1.0,
    duration_lf=1.0,
    features_to_compute=["csd"],
    output_dir=output_dir,
    feature_params=FeatureParams(csd=CsdParams(scale=False)),
    # equivalently: feature_params={"csd": {"scale": False}}
)
logger.info("CSD (scale=False) features: shape=%s", df_unscaled_csd.shape)
