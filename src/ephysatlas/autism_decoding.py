
# Get autism data

import numpy as np
import pandas as pd
from pathlib import Path
from one.api import ONE
import ephysatlas.data
import ephysatlas.aggregation
from ephysatlas.features import voltage_features_set
from ephysatlas.features import EphysTransformer
from ephysatlas.plots import plot_histogram, select_series
from ephysatlas.anatomy import ClassifierRegions, NEW_VOID
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
import shap
from collections import Counter
import matplotlib.pyplot as plt

BASE_PATH = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas')

def remove_void_root(df_voltage, mapping, br, list_acronyms=None):
    if list_acronyms is None:
        list_acronyms = ['void', 'root', 'void_fluid'] # Todo remove void liquid
    vect_void_root = br.acronym2id(list_acronyms)
    df_voltage = df_voltage[~df_voltage[mapping + "_id"].isin(vect_void_root)]
    return df_voltage


def set_common_region(df_voltage, df_autism, id_region_voltage, id_region_autism, mapping):
    set_common = set(id_region_voltage).intersection(set(id_region_autism))
    df_voltage = df_voltage[df_voltage[mapping + "_id"].isin(set_common)]
    df_autism = df_autism[df_autism[mapping + "_id"].isin(set_common)]
    return df_voltage, df_autism

def get_br_id_min_pid(df, mapping, min_pid):
    df_pid = df[[mapping+'_id', 'pid']].copy()
    df_pid = df_pid.drop_duplicates()
    gr_reg = df_pid.groupby([mapping+'_id']).aggregate('count')
    id_region = gr_reg[gr_reg['pid']>=min_pid].index
    # Keeps rows with brain region col values
    filtered_df = df[df[mapping+'_id'].isin(id_region)]
    return filtered_df, id_region


def get_br_id_min_channel(df, mapping, min_channel):
    df_channel = df[[mapping+'_id', 'pid', 'channel']].copy()
    df_channel = df_channel.drop_duplicates()  # This should not change the df
    gr_reg = df_channel.groupby([mapping+'_id']).aggregate('count')
    id_region = gr_reg[gr_reg['channel']>=min_channel].index
    # Keeps rows with brain region col values
    filtered_df = df[df[mapping+'_id'].isin(id_region)]
    return filtered_df, id_region

def download_baseline_data(local_data_path=BASE_PATH, label='2025_W28', one=None):
    if one is None: one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    download_path = ephysatlas.data.download_tables(local_data_path, label=label, one=one)
    return download_path

def clean_df(df_autism, df_voltage, remove_voidroot, mapping, br, min_pid, min_channel, features):
    df_voltage = df_voltage.reset_index().dropna()
    df_autism = df_autism.reset_index().dropna()

    # Remove bad channels ; keep only good channels with label 0
    df_voltage = df_voltage[df_voltage["channel_labels"] == 0]
    df_autism = df_autism[df_autism["channel_labels"] == 0]

    # Remove void and root
    if remove_voidroot:
        df_voltage = remove_void_root(df_voltage, mapping, br)
        df_autism = remove_void_root(df_autism, mapping, br)

    # Check per region that there is min number of PID
    df_voltage, id_region_voltage = get_br_id_min_pid(df_voltage, mapping, min_pid)
    df_autism, id_region_autism = get_br_id_min_pid(df_autism, mapping, min_pid)
    # Take union of region remaining in Autism / EA
    df_voltage, df_autism = set_common_region(df_voltage, df_autism, id_region_voltage, id_region_autism, mapping)

    # Check per region that there is at least number channels
    df_voltage, id_region_voltage = get_br_id_min_channel(df_voltage, mapping, min_channel)
    df_autism, id_region_autism = get_br_id_min_channel(df_autism, mapping, min_channel)
    # Take union of region remaining in Autism / EA
    df_voltage, df_autism = set_common_region(df_voltage, df_autism, id_region_voltage, id_region_autism, mapping)

    # -- Check that features are in df columns
    setfeature = set(df_voltage.columns).intersection(set(features))
    setfeature.discard("channel_labels")  # Remove channel labels as a feature
    features = sorted(list(setfeature))

    return df_autism, df_voltage, features


def get_autism_baseline_data(path_autism, path_baseline,
                             APPLY_DENOISING = False, APPLY_TRANSFORMER = True,
                             strict = True, load_denoised = False,
                             features=None, br=None,
                             min_pid=5, min_channel=200, mapping='Beryl',
                             remove_voidroot=True):

    # This function assumes the data is already downloaded

    if br is None:
        br = ClassifierRegions()
        br.add_new_region(NEW_VOID)

    if features is None: features = voltage_features_set()
    if not load_denoised:  strict = False

    # ---- BASELINE DATA ----
    df_voltage = ephysatlas.data.read_features_from_disk(path_baseline,
                                                         strict=strict,
                                                         load_denoised=load_denoised)

    # ---- AUTISM DATA ----
    df_autism = ephysatlas.data.read_features_from_disk(path_autism,
                                                         strict=strict,
                                                         load_denoised=load_denoised)
    # Apply denoising or transform
    if not load_denoised:
        if APPLY_DENOISING:  # This applies the transformer automatically
            # Apply denoising without bad channel interpolation
            df_voltage['channel_labels'] = 0
            df_voltage = ephysatlas.aggregation.denoise_raw_features_data(df_voltage)

            df_autism['channel_labels'] = 0
            df_autism = ephysatlas.aggregation.denoise_raw_features_data(df_autism)

        elif APPLY_TRANSFORMER:  # Applies transformer without denoising first
            df_voltage = EphysTransformer().fit_transform(df_voltage)
            df_autism = EphysTransformer().fit_transform(df_autism)

    df_autism, df_voltage, features = clean_df(df_autism, df_voltage, remove_voidroot, mapping, br, min_pid, min_channel, features)

    return df_autism, df_voltage, features


# ====
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def pid_cv_region_summary(df_baseline, df_autism, feature_cols, model_type="log_reg"):
    """
    PID-wise cross-validation for a single brain region, with region-level aggregation.

    Parameters
    ----------
    df_baseline : pd.DataFrame
        Baseline data for this region
    df_autism : pd.DataFrame
        Autism data for this region
    feature_cols : list
        Features to use
    model_type : str
        "log_reg" (default) or "random_forest"

    Returns
    -------
    pid_metrics : pd.DataFrame
        Per-PID metrics: accuracy, precision, recall, f1
    region_summary : pd.Series
        Region-level mean metrics aggregated over autism PIDs
    final_model : fitted sklearn Pipeline on all data
    """
    # Label datasets
    df_baseline = df_baseline.copy()
    df_autism = df_autism.copy()
    df_baseline["label"] = 0
    df_autism["label"] = 1

    # Combine data for this region
    df_region = pd.concat([df_baseline, df_autism], ignore_index=True)
    X, y = get_set_classifier_xy(df_region, feature_cols)
    # Select classifier
    if model_type == "log_reg":
        clf = LogisticRegression(max_iter=500)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Do cross val
    autism_pids = df_autism["pid"].unique()
    pid_metrics = cross_val(df_region, feature_cols, clf, cv_pids=autism_pids)
    weighted_recall = aggregate_pid_scores(pid_metrics, min_channels=20)

    # Train final model on all data
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])
    final_pipe.fit(X, y)

    feature_importances = pd.DataFrame({
        "feature": feature_cols,
        "coef": clf.coef_.flatten(),
        "abs_coef": np.abs(clf.coef_.flatten())
    }).sort_values("abs_coef", ascending=False)

    return pid_metrics, weighted_recall, feature_importances



def get_set_classifier_xy(df_region, feature_cols, label_col="label"):
    X = df_region[feature_cols].to_numpy()
    y = df_region[label_col].to_numpy()
    return X, y



##
def cross_val(df_region, feature_cols, clf, cv_pids=None):
    X, y = get_set_classifier_xy(df_region, feature_cols)

    if cv_pids is None:
        cv_pids = df_region["pid"]

    logo = LeaveOneGroupOut()
    pid_rows = []

    # Cross-val each PID
    groups = df_region["pid"].to_numpy()  # PID grouping
    for train_idx, test_idx in logo.split(df_region, y, groups=groups):
        test_pid = groups[test_idx][0]
        # Only leave out the PIDs to do cross-validation on
        if test_pid not in cv_pids:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Compute per-PID metrics

        pid_rows.append({
            "pid": test_pid,
            "n_channel" : len(test_idx),
            "accuracy": accuracy_score(y_test, y_pred),
            # These will be set to 0 if only autism data cross-val
            "precision_0": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
            "recall_0": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
            "f1_0": f1_score(y_test, y_pred, pos_label=0, zero_division=0),
            # These will represent autism data as label =1
            "precision_1": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "recall_1": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "f1_1": f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        })

    pid_metrics = pd.DataFrame(pid_rows)
    return pid_metrics


def aggregate_pid_scores(pid_metrics, min_channels=20, min_pid=3):
    """
    Aggregate per-PID recall scores to region-level score.
    """
    # Keep only PIDs with enough channels
    pid_metrics_ch = pid_metrics[pid_metrics["n_channel"] >= min_channels]

    if pid_metrics_ch.empty or len(pid_metrics_ch)<min_pid:
        return np.nan

    # Weighted mean by number of channels
    weights = pid_metrics["n_channel"]
    weighted_recall = np.average(pid_metrics["recall_1"], weights=weights)
    return weighted_recall


## ----- BINARY DECODER ------

N_SPLIT = 3
RANDOM_STATE = 42


def get_data_per_fold(df_classes, features, label_col="label",
                      n_split=N_SPLIT, shuffle=True, random_state=RANDOM_STATE):
    # Get data sets
    X, y = get_set_classifier_xy(df_classes, features, label_col=label_col)

    # Stratify groups to remove entire PIDs
    sgkf = StratifiedGroupKFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
    # Instantiate dict to save statistics of fold
    fold_data = dict()

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=df_classes["pid"]), start=1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        y_test_random = np.random.permutation(y_test)

        # Get the train/test PIDs from the split
        train_pids = df_classes.iloc[train_idx]["pid"].unique()
        test_pids = df_classes.iloc[test_idx]["pid"].unique()

        # Count how many unique PIDs of each label are in train/test
        train_counts = df_classes[df_classes["pid"].isin(train_pids)].drop_duplicates("pid")["label"].value_counts()
        test_counts = df_classes[df_classes["pid"].isin(test_pids)].drop_duplicates("pid")["label"].value_counts()

        # Compute class weight within the fold
        # Compute in 2 ways, as this depends on the type of classifier used: class_weight and scale_weight
        # y_train and y_test come from your split
        n_train = len(y_train)
        classes_train = np.unique(y_train)
        K = len(classes_train)  # This will always be 2 in our case
        # count samples per class in y_train
        counts = Counter(y_train)
        # compute weights
        class_weights = {cls: n_train / (K * counts[cls]) for cls in classes_train}
        # Scale weight
        n_pos = sum(y_train == 1)
        n_neg = sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos

        # Check there are the right N classes in test set
        classes_test = np.unique(y_test)
        is_test_balanced = len(classes_test) == K

        fold_data[fold_idx] = {"fold": fold_idx,
                               "features": features,
                               "X_train": X_train,
                               "X_test": X_test,
                               "y_train": y_train,
                               "y_test": y_test,
                               "y_test_random": y_test_random,
                               "n_train_channels": np.bincount(y_train),
                               "n_test_channels": np.bincount(y_test),
                               "n_train_pids": train_counts,
                               "n_test_pids": test_counts,
                               "list_train_pids": train_pids,
                               "list_test_pids": test_pids,
                               "class_weights": class_weights,
                               "scale_pos_weight": scale_pos_weight,
                               "random_state": random_state,
                               "is_test_balanced": is_test_balanced}

    return fold_data


def get_weight_classes(df_baseline, df_compare):
    scale_pos_weight = len(df_baseline) / len(df_compare)
    return scale_pos_weight

def train_classifer_one_fold(fold, eval_metric="logloss"):
    X_train = fold["X_train"]
    X_test = fold["X_test"]
    y_train = fold["y_train"]
    y_test = fold["y_test"]
    y_test_random = fold['y_test_random']
    features = fold['features']
    scale_pos_weight = fold['scale_pos_weight']

    report_perf = dict()

    # Train classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric=eval_metric, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    # Check on performance
    y_pred = model.predict(X_test)

    report_perf['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    report_perf['classification_report_random'] = classification_report(y_test, y_test_random, output_dict=True)

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # Take mean absolute SHAP value across samples
    importance = np.abs(shap_values).mean(axis=0)

    report_perf['shap_values'] = shap_values
    report_perf['importance'] = importance
    report_perf['features'] = features

    return report_perf


def print_classification_report(report):
    # Convert to DataFrame
    df_report = pd.DataFrame(report).transpose()
    print(df_report.head())


def get_sorted_list_feature_importance(features, importance):
    # Put into a DataFrame for clarity
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": importance
    })
    # Sort by importance
    feature_importance = feature_importance.sort_values("importance", ascending=False).reset_index(drop=True)
    list_imp = feature_importance['feature'].to_list()
    return list_imp

def plot_one_fold(fold, report_perf, label='label'):
    importance = report_perf['importance']
    shap_values = report_perf['shap_values']
    features = report_perf['features']

    report = report_perf['classification_report']
    report_random = report_perf['classification_report_random']
    print_classification_report(report)
    print_classification_report(report_random)
    print(fold['is_test_balanced'])


    X_test = fold['X_test']
    X_test_df = pd.DataFrame(X_test, columns=features)
    plot_df = X_test_df.copy()
    plot_df[label] = fold['y_test']

    # 1. Global feature importance (summary plot)
    # This shows which features matter most overall.
    shap.summary_plot(shap_values, X_test, feature_names=features)

    # 2. Cross plot of first 3 important features
    list_feat = get_sorted_list_feature_importance(features, importance)
    # Select first 3 for plotting
    features_plot = list_feat[0:3]
    # print(features_plot)
    sns.pairplot(data=plot_df[features_plot + [label]], hue=label)

    # 3. Feature-level effect (dependence plot)
    # To see how a single feature influences predictions:
    shap.dependence_plot(features_plot[0], shap_values, X_test_df)


def plot_one_fold_region(fold, report_perf, mapping='Beryl'):
    # Plot specific to training with region as information column
    X_test = fold['X_test']
    features = report_perf['features']
    shap_values = report_perf['shap_values']
    X_test_df = pd.DataFrame(X_test, columns=features)

    # Place absolute SHAP values into dataframe
    df_shap = pd.DataFrame(np.abs(shap_values), columns=features)
    # Overwrite region column otherwise it contain the SHAP value for region ID
    df_shap[mapping + "_id"] = X_test_df[mapping + "_id"].astype('int')
    # Mean absolute SHAP per feature, stratified by region
    grouped = df_shap.groupby(mapping + "_id").mean()

    plt.figure(figsize=(12,6))
    sns.heatmap(grouped, cmap="viridis", annot=False)
    plt.title("Mean |SHAP| values per feature and region")
    plt.ylabel(mapping + "_id (region)")
    plt.xlabel("Feature")
    plt.show()

