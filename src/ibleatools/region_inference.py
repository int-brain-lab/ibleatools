
from pathlib import Path
from . import decoding
from . import features
import numpy as np


model_paths = [
    Path('/mnt/s0/ephys-atlas-decoding/models/2024_W50_Cosmos_voter-snap-pudding/'),  # without waveforms
    Path('/mnt/s0/ephys-atlas-decoding/models/2024_W50_Cosmos_lid-basket-sense/') #with waveforms
]


n_folds = 5

def infer_regions(df_inference, path_model):
    
    for fold in range(n_folds):
        
        dict_model = decoding.load_model(path_model.joinpath(f'FOLD0{fold}'))
        classifier = dict_model['classifier']
        
        df_inference['outside'] = 0
        df_inference_denoised = features.denoise_dataframe(df_inference)
        
        x_test = df_inference_denoised.loc[:, dict_model['meta']['FEATURES']].values
        y_pred = classifier.predict(x_test)
        y_probas = classifier.predict_proba(x_test)
        
        if fold == 0:
            predicted_probas = np.zeros((n_folds, y_probas.shape[0], y_probas.shape[1]))
            predicted_region = np.zeros((n_folds, y_pred.shape[0]))
        predicted_probas[fold] = y_probas
        predicted_region[fold] = y_pred

    return predicted_probas, predicted_region

