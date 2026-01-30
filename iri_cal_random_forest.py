"""
IRI Random Forest Training with Group Cross-Validation

This script trains a Random Forest regression model to predict IRI
using Z-axis vibration features and vehicle speed.

Key design choices:
- GroupKFold is used to avoid leakage between frames of the same video
- IRI is modeled in log-space to stabilize variance
- Results are saved per fold for later analysis and comparison
"""

import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.dummy import DummyRegressor



# CONFIGURATION


# Output directory where all fold results will be stored
BASE_RESULTS_DIR = "results_rf_3"

# Input features used for the model
FEATURES = ["z_std", "z_rms", "z_peak_to_peak", "speed"]

# Target variable
TARGET = "iri_est"

# Grouping column to ensure video-level separation during CV
GROUP = "sensor_video_id"

# Number of cross-validation folds
N_SPLITS = 4

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}

# UTILITY FUNCTIONS


def ensure_dir(path):
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)
    return path


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for regression.

    Metrics:
    - RMSE
    - Relative RMSE (normalized by mean target)
    - Pearson correlation

    Parameters:
        y_true (array): Ground truth IRI values
        y_pred (array): Predicted IRI values

    Returns:
        tuple: (rmse, rrmse, correlation)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rrmse = rmse / (np.mean(y_true) + 1e-8)
    corr = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    return rmse, rrmse, corr


def compute_dummy_rmse(y_true):
    """
    Compute RMSE of a dummy baseline model that always predicts
    the mean IRI value.

    Used as a reference to assess whether the RF model
    performs better than a naive baseline.

    Parameters:
        y_true (array): Ground truth IRI values

    Returns:
        float: Dummy model RMSE
    """
    dum = DummyRegressor(strategy="mean")
    dum.fit(np.zeros((len(y_true), 1)), y_true)
    y_pred = dum.predict(np.zeros((len(y_true), 1)))
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_rf():
    """
    Build and return a RandomForestRegressor
    using predefined hyperparameters.
    """
    return RandomForestRegressor(**RF_PARAMS)


def plot_per_video(df, out_dir, title_prefix):
    """
    Plot true vs predicted IRI per video and save the plots.

    Each video is plotted separately to visually inspect
    how predictions follow the ground truth.

    Parameters:
        df (DataFrame): Data containing predictions
        out_dir (str): Directory to save plots
        title_prefix (str): 'train' or 'test'
    """
    ensure_dir(out_dir)

    for vid in df[GROUP].unique():
        vdf = df[df[GROUP] == vid]

        plt.figure(figsize=(10, 4))
        plt.plot(vdf["mt"], vdf[TARGET], label="True IRI", linewidth=2)
        plt.plot(vdf["mt"], vdf["iri_pred"], "--", label="Predicted IRI")
        plt.xlabel("Frame (MT)")
        plt.ylabel("IRI")
        plt.title(f"{title_prefix} | Video {vid}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{title_prefix}_video_{vid}.png"))
        plt.close()


def predict_per_frame(model, df):
    """
    Generate per-frame IRI predictions using the trained model.

    Predictions are made in log-space and converted back
    using expm1.

    Parameters:
        model: Trained RandomForestRegressor
        df (DataFrame): Input features

    Returns:
        DataFrame: Copy of df with iri_pred column added
    """
    preds = np.expm1(model.predict(df[FEATURES]))
    df = df.copy()
    df["iri_pred"] = preds
    return df



# MAIN TRAINING PIPELINE


def run_iri_rf_cv(df):
    """
    Run group-based cross-validation for IRI prediction.

    - Splits data by sensor_video_id using GroupKFold
    - Trains Random Forest on log(IRI)
    - Evaluates against dummy baseline
    - Saves metrics, predictions, and plots per fold

    Parameters:
        df (DataFrame): Training dataset
    """
    ensure_dir(BASE_RESULTS_DIR)

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold = 1

    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df[GROUP].values

    for train_idx, test_idx in gkf.split(X, y, groups):
        print(f"\n========== FOLD {fold} ==========")

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # Log-transform target for stability
        train_df["iri_log"] = np.log1p(train_df[TARGET])
        test_df["iri_log"] = np.log1p(test_df[TARGET])

        fold_dir = ensure_dir(os.path.join(BASE_RESULTS_DIR, f"fold_{fold}"))
        train_dir = ensure_dir(os.path.join(fold_dir, "train_results"))
        test_dir = ensure_dir(os.path.join(fold_dir, "test_results"))

        # -------- TRAIN MODEL --------
        model = build_rf()
        model.fit(train_df[FEATURES], train_df["iri_log"])

        joblib.dump(model, os.path.join(train_dir, "models_rf.pkl"))

        # -------- EVALUATION --------
        train_pred = np.expm1(model.predict(train_df[FEATURES]))
        test_pred = np.expm1(model.predict(test_df[FEATURES]))

        train_rmse = np.sqrt(mean_squared_error(train_df[TARGET], train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_df[TARGET], test_pred))

        train_corr = pearsonr(train_df[TARGET], train_pred)[0]
        test_corr = pearsonr(test_df[TARGET], test_pred)[0]

        train_dummy_rmse = compute_dummy_rmse(train_df[TARGET].values)
        test_dummy_rmse = compute_dummy_rmse(test_df[TARGET].values)

        train_rrmse = train_rmse / (train_dummy_rmse + 1e-8)
        test_rrmse = test_rmse / (test_dummy_rmse + 1e-8)

        pd.DataFrame([{
            "rmse": train_rmse,
            "dummy_rmse": train_dummy_rmse,
            "rrmse": train_rrmse,
            "correlation": train_corr
        }]).to_csv(os.path.join(train_dir, "train_metrics.csv"), index=False)

        pd.DataFrame([{
            "rmse": test_rmse,
            "dummy_rmse": test_dummy_rmse,
            "rrmse": test_rrmse,
            "correlation": test_corr
        }]).to_csv(os.path.join(test_dir, "test_metrics.csv"), index=False)

        # -------- PER-FRAME PREDICTIONS --------
        train_out = predict_per_frame(model, train_df)
        test_out = predict_per_frame(model, test_df)

        ensure_dir(os.path.join(train_dir, "perframe"))
        ensure_dir(os.path.join(test_dir, "perframe"))

        train_out.to_csv(
            os.path.join(train_dir, "perframe", "train_predictions.csv"),
            index=False
        )
        test_out.to_csv(
            os.path.join(test_dir, "perframe", "test_predictions.csv"),
            index=False
        )

        # -------- PLOTS --------
        plot_per_video(train_out, os.path.join(train_dir, "plots"), "train")
        plot_per_video(test_out, os.path.join(test_dir, "plots"), "test")

        print(f"Fold {fold} | Test RMSE: {test_rmse:.3f} | Corr: {test_corr:.3f}")
        fold += 1

    print("\nAll RF folds completed.")



# ENTRY POINT


if __name__ == "__main__":
    df = pd.read_csv("iri_training_data_1.csv")
    run_iri_rf_cv(df)
