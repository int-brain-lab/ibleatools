import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pathlib import Path

import numpy as np
import pandas as pd

import ephysatlas.data
import ephysatlas.anatomy
import ephysatlas.plots

FIXTURE_PATH = Path(ephysatlas.data.__file__).parents[2].joinpath("tests", "fixtures")


def _fixture_creation_code_for_reference():
    """
    Creates test fixtures by extracting a subset of ephys-atlas feature data.

    This function reads parquet files from a source directory containing ephys-atlas features
    and creates smaller test fixtures by filtering the data to include only benchmark probe IDs.
    """
    import ephysatlas.fixtures
    import pandas as pd

    pids = ephysatlas.fixtures.benchmark_pids
    output_test_fixture_path = (
        Path(ephysatlas.fixtures.__file__).parents[3].joinpath("tests", "fixtures")
    )
    for VINTAGE in ["2024_W50", "2025_W28"]:
        path_features = Path(
            f"/mnt/s0/ephys-atlas-decoding/features/{VINTAGE}"
        )  # parede
        for pqt_file_name in [
            "channels.pqt",
            "channels_labels.pqt",
            "raw_ephys_features.pqt",
            "raw_ephys_features_denoised.pqt",
        ]:
            pqt_file = path_features.joinpath(pqt_file_name)
            if pqt_file.exists():
                output_file = output_test_fixture_path.joinpath(
                    "features", VINTAGE, pqt_file.name
                )
                output_file.parent.mkdir(
                    parents=True, exist_ok=True
                )  # create directory if not exists
                df = pd.read_parquet(pqt_file)
                df.loc[pids, :].to_parquet(output_file)


class TestFeaturesDataframeIO(unittest.TestCase):
    def setUp(self):
        self.mock_brain_atlas = MagicMock()
        self.mock_brain_atlas.get_labels.return_value = 0
        self.mock_regions = MagicMock()
        self.mock_regions.remap.return_value = 0
        self.mock_brain_atlas.regions = self.mock_regions

    def test_load_features_dataframe(self):
        VINTAGES = ["2024_W50", "2025_W28"]
        for VINTAGE in VINTAGES:
            with self.subTest(vintage=VINTAGE):
                path_features = FIXTURE_PATH.joinpath("features", VINTAGE)
                # once features and anatomy are downloaded, this will load the features Dataframe
                df_features = ephysatlas.data.read_features_from_disk(
                    path_features,
                    brain_atlas=self.mock_brain_atlas,
                    strict=True,
                    mappings=[],
                )
                ephysatlas.plots.plot_features_distributions(
                    df_features, title=f"Features distributions for {VINTAGE}"
                )


def _schema_example(model_cls, n=5):
    """Build a minimal valid DataFrame from a pandera DataFrameModel schema.

    Generates deterministic values based on each column's dtype so that
    the result passes schema validation without requiring hypothesis.
    """
    schema = model_cls.to_schema()
    data = {}
    for col_name, col in schema.columns.items():
        type_str = str(col.dtype)
        if "float" in type_str.lower():
            data[col_name] = np.ones(n, dtype=float)
        elif "int" in type_str.lower():
            data[col_name] = np.zeros(n, dtype=int)
        elif "bool" in type_str:
            data[col_name] = [True] * n
        else:  # str / object
            data[col_name] = [f"{col_name}_{i}" for i in range(n)]
    return pd.DataFrame(data)


def _make_probe_details(n=3):
    """Synthetic df_probe_details fixture generated from ModelProbeDetails schema."""
    import ephysatlas.features

    return _schema_example(ephysatlas.features.ModelProbeDetails, n=n)


def _make_cluster_aggregates(path, n=10):
    """Write synthetic cell-aggregate files to path/cells_aggregates/."""
    import ephysatlas.cells

    agg_path = path / "cells_aggregates"
    agg_path.mkdir(parents=True, exist_ok=True)
    df = _schema_example(ephysatlas.cells.ModelClusters, n=n)
    df.to_parquet(agg_path / "clusters.table.pqt")
    df.to_parquet(agg_path / "clusters_good.table.pqt")
    np.save(agg_path / "clusters.acgs_log.npy", np.zeros((n, 128), dtype=np.float16))
    np.save(agg_path / "acgs_log.times.npy", np.linspace(1e-3, 2.0, 128))
    np.save(
        agg_path / "clusters.waveforms_peak.npy", np.zeros((n, 128), dtype=np.float16)
    )
    np.save(agg_path / "clusters_good.stpc.npy", np.zeros((n, 1000), dtype=np.float16))
    np.save(agg_path / "clusters_good.stlfp.npy", np.zeros((n, 250), dtype=np.float16))
    return agg_path


class TestProjectDataIO(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.project = "test_project"
        self.project_path = self.tmp / self.project
        self.project_path.mkdir(parents=True, exist_ok=True)
        _make_probe_details().to_parquet(self.project_path / "df_probe_details.pqt")
        _make_cluster_aggregates(self.project_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_read_probe_details(self):
        df = ephysatlas.data.read_probe_details(self.project_path, strict=True)
        self.assertEqual(df.shape[0], 3)
        self.assertIn("pid", df.columns)
        self.assertIn("bwm", df.columns)

    def test_read_cells_features(self):
        r = ephysatlas.data.read_cells_features(self.project_path)
        self.assertEqual(r["df_clusters"].shape[0], 10)
        self.assertEqual(r["stpc"].shape, (10, 1000))
        self.assertEqual(r["stlfp"].shape, (10, 250))
        # waveforms not present unless large_files=True was passed to download
        self.assertNotIn("waveforms", r)
        self.assertNotIn("df_waveforms", r)

    def test_read_cells_features_with_waveforms(self):
        n_traces = 50
        agg_path = self.project_path / "cells_aggregates"
        np.save(
            agg_path / "waveforms.voltage.npy",
            np.zeros((n_traces, 128), dtype=np.float16),
        )
        pd.DataFrame(
            {
                "pid": ["x"] * n_traces,
                "cluster_id": range(n_traces),
                "abs_channel": range(n_traces),
            }
        ).to_parquet(agg_path / "waveforms.table.pqt")
        r = ephysatlas.data.read_cells_features(self.project_path)
        self.assertIn("waveforms", r)
        self.assertEqual(r["waveforms"].shape, (n_traces, 128))
        self.assertIn("df_waveforms", r)

    def test_download_probe_details(self):
        mock_s3 = MagicMock()
        mock_one = MagicMock()
        mock_one.alyx = MagicMock()
        with patch("ephysatlas.data.aws") as mock_aws:
            mock_aws.get_s3_from_alyx.return_value = (mock_s3, "test-bucket")
            ephysatlas.data.download_probe_details(
                self.tmp / "dl", project=self.project, one=mock_one
            )
        mock_s3.Bucket.assert_called_once_with("test-bucket")
        mock_s3.Bucket.return_value.download_file.assert_called_once()
        args = mock_s3.Bucket.return_value.download_file.call_args[0]
        self.assertIn("df_probe_details.pqt", args[0])  # S3 key
        self.assertIn("df_probe_details.pqt", args[1])  # local path

    def test_download_cells_features(self):
        mock_one = MagicMock()
        mock_one.alyx = MagicMock()
        with patch("ephysatlas.data.aws") as mock_aws:
            mock_aws.get_s3_from_alyx.return_value = (MagicMock(), "test-bucket")
            mock_aws.s3_download_file.return_value = Path("file1")
            ephysatlas.data.download_cells_features(
                self.tmp / "dl", project=self.project, one=mock_one
            )
        # standard files only — no waveform files by default
        calls = mock_aws.s3_download_file.call_args_list
        s3_keys = [c[0][0] for c in calls]
        self.assertEqual(
            mock_aws.s3_download_file.call_count,
            len(ephysatlas.data._CELLS_AGGREGATES_FILES),
        )
        self.assertFalse(any("waveforms.voltage" in k for k in s3_keys))
        self.assertTrue(all("cells_aggregates" in k for k in s3_keys))

    def test_download_cells_features_large_files(self):
        mock_one = MagicMock()
        mock_one.alyx = MagicMock()
        with patch("ephysatlas.data.aws") as mock_aws:
            mock_aws.get_s3_from_alyx.return_value = (MagicMock(), "test-bucket")
            mock_aws.s3_download_file.return_value = Path("file1")
            ephysatlas.data.download_cells_features(
                self.tmp / "dl", project=self.project, one=mock_one, large_files=True
            )
        s3_keys = [c[0][0] for c in mock_aws.s3_download_file.call_args_list]
        n_expected = len(ephysatlas.data._CELLS_AGGREGATES_FILES) + len(
            ephysatlas.data._WAVEFORMS_FILES
        )
        self.assertEqual(mock_aws.s3_download_file.call_count, n_expected)
        self.assertTrue(any("waveforms.voltage" in k for k in s3_keys))

    def test_download_project_data(self):
        mock_one = MagicMock()
        mock_one.alyx = MagicMock()
        with (
            patch("ephysatlas.data.download_probe_details") as mock_pd,
            patch("ephysatlas.data.download_cells_features") as mock_cf,
        ):
            result = ephysatlas.data.download_project_data(
                self.tmp / "dl", project=self.project, one=mock_one
            )
        mock_pd.assert_called_once_with(
            self.tmp / "dl", project=self.project, one=mock_one, overwrite=False
        )
        mock_cf.assert_called_once_with(
            self.tmp / "dl",
            project=self.project,
            one=mock_one,
            overwrite=False,
            large_files=False,
        )
        self.assertEqual(result, self.tmp / "dl" / self.project)
