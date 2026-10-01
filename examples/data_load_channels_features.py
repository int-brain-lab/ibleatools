# %%
from pathlib import Path
from one.api import ONE

import ephysatlas.data
import ephysatlas.anatomy
import ephysatlas.plots

VINTAGE = None
# this will download the Allen brain templates
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()

if VINTAGE is None:
    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    VINTAGE = ephysatlas.data.get_latest_label(one=one, project='ea_active')

path_features = Path(f'/Users/olivier/Documents/datadisk/ephys-atlas-decoding/features')  # mac

if not path_features.joinpath(VINTAGE).exists():
    # an ONE account is required to access the private IBL datasets
    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    download_path = ephysatlas.data.download_tables(path_features, label=VINTAGE, one=one)
    print(download_path)  # PosixPath('/home/olivier/scratch/2025_W27')

# once features and anatomy are downloaded, this will load the features Dataframe
df_features = ephysatlas.data.read_features_from_disk(download_path, brain_atlas=brain_atlas, strict=False)
ephysatlas.plots.plot_features_distributions(df_features, title=f"Features distributions for {VINTAGE}")
