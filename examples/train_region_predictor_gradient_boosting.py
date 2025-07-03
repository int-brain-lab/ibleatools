from pathlib import Path
import tqdm

import pandas as pd
import numpy as np

import sklearn.metrics
from xgboost import XGBClassifier  # pip install xgboost  # https://xgboost.readthedocs.io/en/stable/prediction.html

import iblutil.numerical
import ephysatlas.anatomy
import ephysatlas.data
import ephysatlas.regionclassifier
import ephysatlas.fixtures

LOWQ = ephysatlas.fixtures.misaligned_pids
path_features = Path('/Users/olivier/Documents/datadisk/ephys-atlas-decoding/features/2024_W50')  # mac
path_features = Path('/mnt/s0/ephys-atlas-decoding/features/2024_W50')  # parede

brain_atlas = ephysatlas.anatomy.ClassifierAtlas()

path_models = path_features.parents[1].joinpath('models')
path_models.mkdir(exist_ok=True)
df_features = pd.read_parquet(path_features / "raw_ephys_features_denoised.pqt")
df_features = df_features.merge(
    pd.read_parquet(path_features / "channels.pqt"),
    how="inner",
    right_index=True,
    left_index=True,
)
df_features = df_features.merge(
    pd.read_parquet(path_features / "channels_labels.pqt").fillna(0),
    how="inner",
    right_index=True,
    left_index=True,
)
ephysatlas.data.load_tables(local_path=path_features)

FEATURE_SET = ["raw_lf", "raw_lf_csd", "raw_ap", "micro-manipulator"]
FEATURE_SET = [
    "raw_lf",
    "raw_lf_csd",
    "raw_ap",
    "localisation",
    "waveforms",
    "micro-manipulator",
]
x_list = ephysatlas.features.voltage_features_set(FEATURE_SET)

df_features["outside"] = df_features["labels"] == 3
x_list.append("outside")

aids = brain_atlas.get_labels(df_features.loc[:, ["x", "y", "z"]].values, mode="clip")
df_features["Allen_id"] = aids
df_features["Cosmos_id"] = brain_atlas.regions.remap(aids, "Allen", "Cosmos")
df_features["Beryl_id"] = brain_atlas.regions.remap(aids, "Allen", "Beryl")

TRAIN_LABEL = "Cosmos_id"  # ['Beryl_id', 'Cosmos_id']

test_sets = {
    "benchmark": ephysatlas.data.BENCHMARK_PIDS,
    "nemo": ephysatlas.data.NEMO_TEST_PIDS,
}
rids = np.unique(df_features.loc[:, TRAIN_LABEL])

def train(test_idx, fold_label, train_idx=None):
    train_idx = ~test_idx if train_idx is None else train_idx
    print(f"{fold_label}: {df_features.shape[0]} channels", f'training set {np.sum(test_idx) / test_idx.size}')
    df_features.loc[train_idx, :].groupby(TRAIN_LABEL).count()
    x_train = df_features.loc[train_idx, x_list].values
    x_test = df_features.loc[test_idx, x_list].values
    y_train = df_features.loc[train_idx, TRAIN_LABEL].values
    y_test = df_features.loc[test_idx, TRAIN_LABEL].values
    df_benchmarks = df_features.loc[iblutil.numerical.ismember(df_features.index.get_level_values(0), ephysatlas.data.BENCHMARK_PIDS)[0], :].copy()
    df_test = df_features.loc[test_idx, :].copy()
    classes = np.unique(df_features.loc[train_idx, TRAIN_LABEL])

    _, iy_train = iblutil.numerical.ismember(y_train, classes)
    _, iy_test = iblutil.numerical.ismember(y_test, classes)
    # 0.5376271321378102
    #  create model instance
    classifier = XGBClassifier(device="gpu", verbosity=2)

    # fit model
    classifier.fit(x_train, iy_train)
    # make predictions
    y_pred = classes[classifier.predict(x_test)]
    df_test[f"cosmos_prediction"] = classes[
        classifier.predict(df_test.loc[:, x_list].values)
    ]
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_test, y_pred, normalize="true"
    )  # row: true, col: predicted
    print(f"{fold_label} Accuracy: {accuracy}")

    np.testing.assert_array_equal(classes, rids)
    return classifier.predict_proba(x_test), classifier, accuracy, confusion_matrix


# %%
n_folds = 5
df_features = df_features[~df_features.index.get_level_values(0).isin(LOWQ)]
all_pids = np.array(df_features.index.get_level_values(0).unique())
np.random.seed(12345)
np.random.shuffle(all_pids)
ifold = np.floor(np.arange(len(all_pids)) / len(all_pids) * n_folds)

df_predictions = pd.DataFrame(index=df_features.index, columns=list(rids))
for i in range(n_folds):
    test_pids = all_pids[ifold == i]
    train_pids = all_pids[ifold!= i]
    test_idx = np.isin(df_features.index.get_level_values(0), test_pids)
    probas, classifier, accuracy, confusion_matrix = train(
        test_idx=test_idx, fold_label=f"fold {i}"
    )

    df_predictions.loc[test_idx, rids] = probas
    meta = dict(
        RANDOM_SEED=12345,
        VINTAGE="2024_W50",
        REGION_MAP="Cosmos",
        FEATURES=x_list,
        CLASSES=[int(c) for c in rids],
        ACCURACY=accuracy,
        TRAINING=dict(
            training_size=len(train_pids),
            testing_size=len(test_pids),
            hash_training=iblutil.numerical.hash_uuids(train_pids),
            hash_testing=iblutil.numerical.hash_uuids(test_pids)
        ),
    )
    # here we will use the confusion matrix as an emission matrix P(observation|state) = P(prediction|class)
    path_model = ephysatlas.regionclassifier.save_model(path_models, classifier, meta, subfolder=f"FOLD{i:02}", identifier='medical-rosewood-kestrel')
print(f"Model saved to {path_model}")

# df_predictions.to_parquet(path_models / 'predictions_Cosmos.pqt')
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

# %%
import scipy.stats
import pandas as pd


def hmm_post_process(probas, rids, emission=None, transmission=None, priors=None, n_runs=1000, method="two_ways"):
    up2down = transmission / np.sum(transmission, axis=1)[:, np.newaxis]
    down2up = up2down.T / np.sum(up2down.T, axis=1)[:, np.newaxis]
    priors = priors / np.sum(priors)
    nd, nr = probas.shape[0], hmm_rids.size
    if nd <= 2:
        return np.argmax(probas, axis=1)
    all_obs = np.zeros((nd, n_runs), dtype=int)
    all_vit = np.zeros((nd, n_runs))
    vprobs = np.zeros(n_runs)
    vit, _ = ephysatlas.regionclassifier.viterbi(emission, up2down, priors, observed_states=np.argmax(probas, axis=1))
    for i in range(n_runs):
        # generates a random set of observed states according to the classifier output probabilities
        all_obs[:, i] = np.flipud(
            np.mod(
                np.searchsorted(
                    probas.flatten().cumsum(), np.random.rand(nd) + np.arange(nd)
                ),
                nr,
            )
        )
        # run the viterbi algorithm once on the predicted labels for reference
        match method:
            case "one_way":
                init_hidden_probs = probas[0, :]
                cl, p = ephysatlas.regionclassifier.viterbi(
                    emission, up2down, init_hidden_probs, all_obs[:, i]
                )
                vprobs[i] = p
                all_vit[:, i] = cl
            case "two_ways":
                # we take a random depth and start from this point, upwards, and downwards
                dstart = np.random.randint(1, nd - 1)
                init_hidden_probs = probas[dstart, :]
                # we take a random set of observed states according to the classifier output probabilities
                # this is the downward part
                all_vit[dstart:nd, i], pdown = ephysatlas.regionclassifier.viterbi(
                    emission, up2down, init_hidden_probs, all_obs[dstart:nd, i]
                )
                # this is the upward part, note that we transpose the transition matrix to reflect the shift in direction
                _vit, pup = ephysatlas.regionclassifier.viterbi(
                    confusion_matrix,
                    down2up,
                    init_hidden_probs,
                    np.flipud(all_obs[0:dstart, i]),
                )
                all_vit[0:dstart, i] = np.flipud(_vit)
                vprobs[i] = np.sqrt(pup * pdown)

    vcprobs = scipy.sparse.coo_array(
        (
            all_vit.flatten() * 0 + 1,
            (np.tile(np.arange(nd), (n_runs,)), all_vit.flatten()),
        ),
        shape=(nd, nr),
    ).todense()
    vcprobs = np.cumsum(vcprobs / vcprobs.sum(axis=1)[:, np.newaxis], axis=1)

    all_vit = np.flipud(all_vit.astype(int))
    vit_pred, _ = scipy.stats.mode(all_vit, axis=-1)
    return vit_pred

def process_pid(pid, df_features, **kwargs):
    df_predictions_depths = pd.merge(df_features.loc[pid]['axial_um'], df_predictions.loc[pid], left_index=True, right_index=True).groupby('axial_um').mean()
    probas_pid = df_predictions_depths[rids].to_numpy().astype(float)  # this is the (ndepths, nregions) matrix of probabilities
    hpred = hmm_post_process(probas_pid, **kwargs)
    _, idpth = iblutil.numerical.ismember(df_features.loc[pid]['axial_um'].to_numpy(), df_predictions_depths.index)
    return hpred[idpth]


transmission, hmm_priors, hmm_rids = ephysatlas.anatomy.regions_transition_matrix(ba=brain_atlas, mapping='Cosmos')
hmm_kwargs = {
    'transmission': transmission,
    'emission': confusion_matrix,
    'priors': hmm_priors,
    'rids': hmm_rids,
    'n_runs': 1000,
}
# import joblib
# n_jobs = min(joblib.cpu_count(), 8)  # Adjust based on your system
# jobs = (joblib.delayed(process_pid)(pid, df_features, **hmm_kwargs) for pid in test_pids)
# results = list(tqdm.tqdm(joblib.Parallel(return_as='generator', n_jobs=n_jobs)(jobs), total=len(test_pids)))

results = []
for pid in tqdm.tqdm(test_pids):
    results.append(process_pid(pid, df_features, **hmm_kwargs))
# %%

sklearn.metrics.accuracy_score(
    df_features.loc[test_pids, 'Cosmos_id'].values,
    rids[np.concatenate(results)])

