from pathlib import Path
from one.api import ONE

import ephysatlas.data
import ephysatlas.anatomy

VINTAGE = '2025_W27'
# this will download the Allen brain templates
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()

path_features = Path(f'/home/olivier/scratch/{VINTAGE}')  # put ht
if not path_features.exists():
    # an ONE account is required to access the private IBL datasets
    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    download_path = ephysatlas.data.download_tables(path_features.parent, label=VINTAGE, one=one)
    print(download_path)  # PosixPath('/home/olivier/scratch/2025_W27')

# once features and anatomy are downloaded, this will load the features Dataframe
df_features = ephysatlas.data.read_features_from_disk(path_features, brain_atlas=brain_atlas)
