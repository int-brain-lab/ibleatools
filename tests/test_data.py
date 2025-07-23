import unittest
from unittest.mock import MagicMock

from pathlib import Path

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
