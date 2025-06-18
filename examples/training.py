from pathlib import Path

import pandas as pd
import numpy as np
import sklearn.metrics
from xgboost import XGBClassifier  # pip install xgboost  # https://xgboost.readthedocs.io/en/stable/prediction.html

from iblutil.numerical import ismember
import ephysatlas.anatomy
import ephysatlas.data
import ephysatlas.regionclassifier
# we are shooting for around 55% accuracy


# import ephys_atlas.data
# from one.api import ONE
# one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
# df_voltage, _, df_channels, df_probes = ephys_atlas.data.download_tables(local_path='/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding/features', label='2024_W50', one=one)



brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
# brain_atlas = ephys_atlas.anatomy.AllenAtlas()  # Accuracy: 0.5536619920744102

path_features = Path('/Users/olivier/Documents/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding/features/2024_W50')  # mac
path_features = Path('/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding/features/2024_W50')  # mac
path_features = Path('/mnt/s0/ephys-atlas-decoding/features/2024_W50')  # parede
path_models = path_features.parents[1].joinpath('models')
path_models.mkdir(exist_ok=True)
df_features = pd.read_parquet(path_features / 'raw_ephys_features_denoised.pqt')
df_features = df_features.merge(pd.read_parquet(path_features / 'channels.pqt'), how='inner', right_index=True, left_index=True)
df_features = df_features.merge(pd.read_parquet(path_features / 'channels_labels.pqt').fillna(0), how='inner', right_index=True, left_index=True)
ephysatlas.data.load_tables(local_path=path_features)


FEATURE_SET = ['raw_lf', 'raw_lf_csd', 'raw_ap', 'localisation', 'waveforms', 'micro-manipulator']
FEATURE_SET = ['raw_lf', 'raw_lf_csd', 'raw_ap', 'micro-manipulator']
x_list = ephysatlas.features.voltage_features_set(FEATURE_SET)

df_features['outside'] = df_features['labels'] == 3
x_list.append('outside')


aids = brain_atlas.get_labels(df_features.loc[:, ['x', 'y', 'z']].values, mode='clip')
df_features['Allen_id'] = aids
df_features['Cosmos_id'] = brain_atlas.regions.remap(aids, 'Allen', 'Cosmos')
df_features['Beryl_id'] = brain_atlas.regions.remap(aids, 'Allen', 'Beryl')

TRAIN_LABEL = 'Cosmos_id'  # ['Beryl_id', 'Cosmos_id']



test_sets = {
    'benchmark': ephysatlas.data.BENCHMARK_PIDS,
    'nemo': ephysatlas.data.NEMO_TEST_PIDS,
}
all_classes = np.unique(df_features.loc[:, TRAIN_LABEL])

def train(test_idx, fold_label):

    train_idx = ~test_idx
    print(f"{fold_label}: {df_features.shape[0]} channels", f'training set {np.sum(test_idx) / test_idx.size}')
    df_features.loc[train_idx, :].groupby(TRAIN_LABEL).count()
    x_train = df_features.loc[train_idx, x_list].values
    x_test = df_features.loc[test_idx, x_list].values
    y_train = df_features.loc[train_idx, TRAIN_LABEL].values
    y_test = df_features.loc[test_idx, TRAIN_LABEL].values
    df_benchmarks = df_features.loc[ismember(df_features.index.get_level_values(0), ephys_atlas.data.BENCHMARK_PIDS)[0], :].copy()
    df_test = df_features.loc[test_idx, :].copy()
    classes = np.unique(df_features.loc[train_idx, TRAIN_LABEL])

    _, iy_train = ismember(y_train, classes)
    _, iy_test = ismember(y_test, classes)
    # 0.5376271321378102
    #  create model instance
    classifier = XGBClassifier(device='gpu', verbosity=2)

    # fit model
    classifier.fit(x_train, iy_train)
    # make predictions
    y_pred = classes[classifier.predict(x_test)]
    df_test[f'cosmos_prediction'] = classes[classifier.predict(df_test.loc[:, x_list].values)]
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='true')  # row: true, col: predicted
    print(f"{fold_label} Accuracy: {accuracy}")

    np.testing.assert_array_equal(classes, all_classes)
    return classifier.predict_proba(x_test), classifier, accuracy

# %%
IDENTIFIER = 'lid-basket-sense'
IDENTIFIER = 'voter-snap-pudding'
n_folds = 5
all_pids = np.array(df_features.index.get_level_values(0).unique())
np.random.seed(12345)
np.random.shuffle(all_pids)
ifold = np.floor(np.arange(len(all_pids)) / len(all_pids) * n_folds)

df_predictions = pd.DataFrame(index=df_features.index, columns=list(all_classes))
for i in range(n_folds):
    test_pids = all_pids[ifold == i]
    test_idx = np.isin(df_features.index.get_level_values(0), test_pids)
    probas, classifier, accuracy = train(test_idx=test_idx, fold_label=f'fold {i}')
    df_predictions.loc[test_idx, all_classes] = probas
    meta = dict(
        RANDOM_SEED=713705,
        VINTAGE="2024_W50",
        REGION_MAP="Cosmos",
        FEATURES=x_list,
        CLASSES=[int(c) for c in all_classes],
        ACCURACY=accuracy,
        )
    path_model = ephysatlas.regionclassifier.save_model(path_models, classifier, meta, subfolder=f'FOLD{i :02d}', identifier=IDENTIFIER)

df_predictions.to_parquet(path_models / 'predictions_Cosmos.pqt')

# lid_basket_sense
# fold 0: 384215 channels training set 0.2008120453391981
# fold 0 Accuracy: 0.6679541183332254
# fold 1: 384215 channels training set 0.19975013989563134
# fold 1 Accuracy: 0.6525857688248401
# fold 2: 384215 channels training set 0.20078601824499304
# fold 2 Accuracy: 0.6892734461079785
# fold 3: 384215 channels training set 0.19982041304998505
# fold 3 Accuracy: 0.6961862088727955
# fold 4: 384215 channels training set 0.19883138347019247
# fold 4 Accuracy: 0.6831033850825982
