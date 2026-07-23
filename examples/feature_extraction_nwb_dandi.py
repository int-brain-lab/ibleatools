"""Feature extraction from an NWB recording streamed from DANDI.

``NwbFeatureCalculator`` reads NWB ``ElectricalSeries`` through SpikeInterface and
reuses the same OOP compute path as the IBL/SpikeGLX calculators. The acquisition axis
(backend HDF5/Zarr x location local/S3/DANDI) is handled by ``NwbSource``, so the
calculator itself is source-agnostic.

This example streams one probe of the IBL Brain-Wide-Map dandiset ``000409`` (each
session file holds the raw AP + LF bands as separate ``ElectricalSeries``) and computes
the ``lf`` + ``csd`` features from the LF band. Everything streams over HTTP -- nothing
is downloaded whole.

Requires the optional NWB stack: ``pip install ibleatools[full]``. Not part of CI. Run
cell-by-cell (``# %%`` cells).
"""

# %% Imports and configuration
import logging

from ephysatlas.feature_calculators import (
    FeatureComputationOptions,
    NwbFeatureCalculator,
    SnippetWindow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IBL Brain-Wide-Map on DANDI. The sub-CSHL049 session file holds two ElectricalSeries:
# acquisition/ElectricalSeriesProbe00AP and .../ElectricalSeriesProbe00LF.
DANDISET, VERSION = "000409", "0.260309.1324"
DANDI_PATH = (
    "sub-CSHL049/sub-CSHL049_ses-c99d53e6-c317-4c53-99ba-070b26673ac4_desc-raw_ecephys.nwb"
)
AP_ES = "acquisition/ElectricalSeriesProbe00AP"
LF_ES = "acquisition/ElectricalSeriesProbe00LF"
options = FeatureComputationOptions(
    features_to_compute=["lf", "csd"], include_trajectory=False
)

# %% 1. Stream the probe's LF band from DANDI and compute lf + csd features
# from_dandi resolves the asset to its streamable S3 URL; lf_electrical_series selects
# the LF series inside the file (a file with one series needs no selector).
calc = NwbFeatureCalculator.from_dandi(
    DANDISET,
    lf_filepath=DANDI_PATH,
    version=VERSION,
    lf_electrical_series=LF_ES,
    name="CSHL049_probe00",
)
window = SnippetWindow(t_start=100.0, duration_ap=0.0, duration_lf=1.0)
result = calc.compute_snippet(window, options)
logger.info("LF/CSD features: shape=%s", result.features.shape)
print(result.features.sort_values("channel")[["channel", "rms_lf", "psd_delta"]].head())

# %% 2. Both bands from the SAME file -> full feature set (ap + waveforms too)
# One NWB file holds both series, so point ap/lf at the same file with different
# electrical-series selectors. The AP band is large; streaming + dartsort take longer.
full_calc = NwbFeatureCalculator.from_dandi(
    DANDISET,
    ap_filepath=DANDI_PATH,
    lf_filepath=DANDI_PATH,
    version=VERSION,
    ap_electrical_series=AP_ES,
    lf_electrical_series=LF_ES,
    name="CSHL049_probe00_full",
)
# full = full_calc.compute_snippet(
#     SnippetWindow(t_start=100.0, duration_ap=1.0, duration_lf=1.0),
#     FeatureComputationOptions(features_to_compute=["lf", "csd", "ap"], include_trajectory=False),
# )
