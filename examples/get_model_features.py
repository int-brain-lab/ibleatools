from pathlib import Path
from ephysatlas.decoding import load_model

# Example on how to retrieve the features a specific model uses.

model_path = Path('/Users/gaellechapuis/Desktop/Reports/EphysAtlas/atlas-features/models/2024_W50_Cosmos_lid-basket-sense')
fold = 0
dict_model = load_model(model_path.joinpath(f'FOLD0{fold}'))
features_model = dict_model['meta']['FEATURES']
