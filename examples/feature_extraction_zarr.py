"""Feature extraction from SpikeInterface-native Zarr recordings.

``SpikeInterfaceZarrFeatureCalculator`` reads ``read_zarr`` recordings -- e.g. the
AIND / open-ephys WavPack-compressed ``ecephys_compressed/*.zarr`` stores -- for local
paths and remote object stores alike. It reuses the same OOP compute path as the NWB
calculator; only the reader differs (SI-native Zarr, not NWB).

Section 1 streams a probe of a public AIND session from S3 (anonymous access) and
computes ``lf`` + ``csd`` features. Section 2 shows the local-store form.

Requires: ``pip install ibleatools[full]`` plus ``s3fs`` (S3 access) and
``wavpack_numcodecs`` (to decompress AIND WavPack stores). Not part of CI. Run
cell-by-cell (``# %%`` cells).
"""

# %% Imports and configuration
import logging

# Registers the WavPack codec with numcodecs so read_zarr can decompress AIND stores.
import wavpack_numcodecs  # noqa: F401

from ephysatlas.feature_calculators import (
    FeatureComputationOptions,
    SnippetWindow,
    SpikeInterfaceZarrFeatureCalculator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

options = FeatureComputationOptions(
    features_to_compute=["lf", "csd"], include_trajectory=False
)

# %% 1. Public AIND recording streamed from S3 (anonymous) -- one probe's LF band
# storage_options={"anon": True} => unauthenticated read of the public bucket.
S3_LF = (
    "s3://aind-open-data/ecephys_830794_2026-01-26_12-02-05/ecephys/ecephys_compressed/"
    "experiment1_Record Node 101#Neuropix-PXI-100.ProbeA-LFP.zarr"
)
calc = SpikeInterfaceZarrFeatureCalculator(
    lf_zarr=S3_LF, storage_options={"anon": True}, name="aind_830794_ProbeA"
)
window = SnippetWindow(t_start=100.0, duration_ap=0.0, duration_lf=1.0)
result = calc.compute_snippet(window, options)
logger.info("LF/CSD features: shape=%s", result.features.shape)
print(result.features.sort_values("channel")[["channel", "rms_lf", "psd_delta"]].head())

# %% 2. Local Zarr stores (no storage_options needed) -- AP + LF for the full set
# local = SpikeInterfaceZarrFeatureCalculator(
#     ap_zarr="/path/to/...ProbeD-AP.zarr",
#     lf_zarr="/path/to/...ProbeD-LFP.zarr",
#     name="probeD",
# )
# full = local.compute_snippet(
#     SnippetWindow(t_start=100.0, duration_ap=1.0, duration_lf=1.0),
#     FeatureComputationOptions(features_to_compute=["lf", "csd", "ap"], include_trajectory=False),
# )
