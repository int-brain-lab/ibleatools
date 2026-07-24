"""Integration parity test: IBLPIDFeatureCalculator vs NwbFeatureCalculator.

Streams the *same* IBL recording two independent ways -- the raw ephys served by
ONE, and its mirror in the DANDI "IBL - Brain Wide Map" dandiset (000409) -- and
checks that they agree, both at the raw-voltage level and at the feature level.

Flow exercised (as an end user would):
    PID --one.pid2eid--> (eid, probe) --one.get_details--> subject
        --> DANDI asset  sub-<subject>/..._ses-<eid>_desc-raw_ecephys.nwb
        --> IBLPIDFeatureCalculator (ONE)  vs  NwbFeatureCalculator (DANDI NWB)

What is asserted (the robust facts):
- Raw AP is **bit-identical** channel-for-channel (both decode the same probe).
- Raw LF matches up to a 1-sample shift: the NWB stores LF fs as 2499.9999999...
  (not exactly 2500), so ``int(fs * t_start)`` lands one sample earlier at
  t=100 s. Compared per channel with a correlation floor rather than exact
  equality.
- With ``channel_labels`` forced to zeros on both paths (via the
  FeatureComputationOptions option), the AP and LF features agree tightly
  (r > ~0.98). Left to their own devices they diverge more only because IBL uses
  stored ``channels.labels`` while the NWB path detects bad channels on the
  snippet -- i.e. the divergence is bad-channel handling, not the data.

Deliberately NOT asserted here (documented findings, see the geometry note):
- CSD features agree only moderately, and **waveform features do not** -- the
  DANDI NWB electrode table has a mirrored lateral (x) axis (``x_nwb = 59 - x_ibl``)
  and a +20 um axial offset relative to IBL's spikeglx geometry. Per-channel
  spectral features are immune, but dartsort spike localization uses the geometry,
  so the lateral flip perturbs waveform (and, to a lesser degree, CSD) features.
  Channels are therefore aligned on the positional index, not on coordinates.

Requires network + the optional NWB deps (spikeinterface, pynwb, dandi, remfile),
so it is skipped unless IBLEATOOLS_NETWORK_TESTS is set:

    IBLEATOOLS_NETWORK_TESTS=1 python -m unittest tests.test_ibl_vs_dandi_parity
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

RUN_NETWORK_TESTS = os.environ.get("IBLEATOOLS_NETWORK_TESTS")

# Benchmark recording confirmed present in DANDI:000409/0.260309.1324
# (CSHL049, churchlandlab, 2020-01-09, probe00).
BENCHMARK_PID = "531423f6-d36d-472b-8234-c8f7b8293f79"
DANDISET_ID = "000409"
DANDISET_VERSION = "0.260309.1324"

# Minimum Pearson r between the IBL and DANDI feature maps, with channel_labels
# forced to zeros on both paths (observed values are ~0.97-0.9996).
MIN_FEATURE_CORRELATION = {
    "rms_ap": 0.98,
    "rms_lf": 0.98,
    "psd_theta": 0.95,
    "psd_beta": 0.95,
    "psd_gamma": 0.95,
}


@unittest.skipUnless(
    RUN_NETWORK_TESTS,
    "network + NWB deps required; set IBLEATOOLS_NETWORK_TESTS=1 to run",
)
class TestIblVsDandiParity(unittest.TestCase):
    """Compare raw data and features from ONE vs the DANDI NWB mirror."""

    @classmethod
    def setUpClass(cls):
        from dandi.dandiapi import DandiAPIClient
        from one.api import ONE
        from spikeinterface.extractors import NwbRecordingExtractor

        from ephysatlas.feature_calculators import (
            IBLPIDFeatureCalculator,
            NwbFeatureCalculator,
            NwbSource,
        )

        cls.cache_dir = tempfile.mkdtemp(prefix="one_parity_")
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            password="international",
            silent=True,
            cache_dir=cls.cache_dir,
        )

        # PID -> eid/probe -> subject -> DANDI asset
        eid, pname = one.pid2eid(BENCHMARK_PID)
        eid = str(eid)
        subject = one.get_details(eid)["subject"]
        with DandiAPIClient() as client:
            ds = client.get_dandiset(DANDISET_ID, DANDISET_VERSION)
            hits = list(
                ds.get_assets_by_glob(f"sub-{subject}/*ses-{eid}*desc-raw_ecephys.nwb")
            )
        assert hits, f"no DANDI raw ecephys asset for {subject}/{eid}"
        dandi_path = hits[0].path

        # DANDI AP + LF ElectricalSeries for this probe
        base_src = NwbSource.dandi(DANDISET_ID, dandi_path, version=DANDISET_VERSION)
        es_paths = NwbRecordingExtractor.fetch_available_electrical_series_paths(
            file_path=base_src._resolve_url(), stream_mode="remfile"
        )
        probe_nn = pname.replace("probe", "")

        def es(band):
            return next(
                p for p in es_paths if f"Probe{probe_nn}" in p and p.endswith(band)
            )

        cls.ibl = IBLPIDFeatureCalculator(pid=BENCHMARK_PID, one=one)
        cls.nwb = NwbFeatureCalculator(
            ap_source=NwbSource.dandi(
                DANDISET_ID,
                dandi_path,
                version=DANDISET_VERSION,
                electrical_series=es("AP"),
            ),
            lf_source=NwbSource.dandi(
                DANDISET_ID,
                dandi_path,
                version=DANDISET_VERSION,
                electrical_series=es("LF"),
            ),
            name=f"{subject}_{pname}",
        )

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(getattr(cls, "cache_dir", ""), ignore_errors=True)

    def test_raw_ap_identical_lf_close(self):
        from ephysatlas.feature_calculators import SnippetWindow

        window = SnippetWindow(t_start=100.0, duration_ap=0.5, duration_lf=0.5)
        raw_ibl = self.ibl.load_raw_snippet(window)
        raw_nwb = self.nwb.load_raw_snippet(window)

        # AP: same probe decoded two ways -> identical to float precision.
        self.assertEqual(raw_ibl.raw_ap.shape, raw_nwb.raw_ap.shape)
        self.assertTrue(
            np.allclose(raw_ibl.raw_ap, raw_nwb.raw_ap, atol=1e-6),
            f"AP raw differs: max|Δ|={np.max(np.abs(raw_ibl.raw_ap - raw_nwb.raw_ap)):.2e}",
        )

        # LF: identical up to the 1-sample fs-precision offset -> high per-channel r.
        n = min(raw_ibl.raw_lf.shape[1], raw_nwb.raw_lf.shape[1])
        corr = np.array(
            [
                np.corrcoef(raw_ibl.raw_lf[i, :n], raw_nwb.raw_lf[i, :n])[0, 1]
                for i in range(raw_ibl.raw_lf.shape[0])
            ]
        )
        self.assertGreater(np.nanmedian(corr), 0.95, "LF raw per-channel corr too low")

    def test_ap_lf_features_agree_with_zero_labels(self):
        from ephysatlas.feature_calculators import (
            FeatureComputationOptions,
            SnippetWindow,
        )

        window = SnippetWindow(t_start=100.0, duration_ap=1.0, duration_lf=1.0)
        n_channels = len(self.ibl.load_geometry()["x"])
        # Force identical (zero) bad-channel labels so destriping matches -> the
        # comparison isolates the data path, not bad-channel detection.
        options = FeatureComputationOptions(
            features_to_compute=["ap", "lf"],
            channel_labels=np.zeros(n_channels, dtype=int),
            include_trajectory=False,
        )
        df_ibl = self.ibl.compute_snippet(window, options).features
        df_nwb = self.nwb.compute_snippet(window, options).features

        merged = df_ibl.merge(df_nwb, on="channel", suffixes=("_ibl", "_nwb"))
        self.assertEqual(len(merged), len(df_ibl))
        # Same physical channels by depth (axial coordinates track 1:1).
        axial_corr = np.corrcoef(merged["axial_um_ibl"], merged["axial_um_nwb"])[0, 1]
        self.assertGreater(axial_corr, 0.999, f"axial mismatch (r={axial_corr:.4f})")

        for feature, min_r in MIN_FEATURE_CORRELATION.items():
            x = merged[f"{feature}_ibl"].to_numpy(dtype=float)
            y = merged[f"{feature}_nwb"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            self.assertGreater(finite.sum(), 100, f"{feature}: too few finite values")
            r = np.corrcoef(x[finite], y[finite])[0, 1]
            self.assertGreater(
                r, min_r, f"{feature}: IBL/DANDI correlation r={r:.3f} < {min_r}"
            )


if __name__ == "__main__":
    unittest.main()
