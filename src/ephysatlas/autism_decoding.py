
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

BASE_PATH = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas')

def download_baseline_data(local_data_path=BASE_PATH, label='2025_W28', one=None):
    if one is None: one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    download_path = ephysatlas.data.download_tables(local_data_path, label=label, one=one)
    return download_path

def get_autism_baseline_data(path_autism, path_baseline,
                             APPLY_DENOISING = False, APPLY_TRANSFORMER = True,
                             strict = True, load_denoised = False,
                             features=None, br=None):

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

    df_voltage = df_voltage.reset_index().dropna()
    df_autism = df_autism.reset_index().dropna()

    # -- Check that features are in df columns
    setfeature = set(df_voltage.columns).intersection(set(features))
    setfeature.discard("channel_labels")  # Remove channel labels as a feature
    features = sorted(list(setfeature))

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



def get_set_classifier_xy(df_region, feature_cols):
    X = df_region[feature_cols].to_numpy()
    y = df_region["label"].to_numpy()
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