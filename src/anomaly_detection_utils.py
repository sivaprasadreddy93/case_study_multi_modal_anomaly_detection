"""
anomaly_detection.py  (v2.0)
───────────────────────────────────────────────
Optimised, modular anomaly‑detection toolkit.
• Isolation Forest, One‑Class SVM, LSTM Autoencoder
• Shared utility helpers (scaling, sequencing, plotting)
• Each detector adds a *named* anomaly flag:
    ─ anomaly_iforest
    ─ anomaly_ocsvm
    ─ anomaly_lstm
• CPU‑only TensorFlow fallback handled.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Optional ‑ only if TF available
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, RepeatVector, TimeDistributed, Input
    )
    from tensorflow.keras.callbacks import EarlyStopping
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ─────────────────────────────────────────────
# File Loading helper
# ─────────────────────────────────────────────

def load_processed_files(folder: str = "../data/processed_oilrig_data") -> Dict[str, pd.DataFrame]:
    """Return dict of {filename: DataFrame} for all CSVs in *folder*."""
    return {
        f.name: pd.read_csv(f, parse_dates=["timestamp"])
        for f in Path(folder).glob("*.csv")
    }


# ─────────────────────────────────────────────
# Isolation Forest Anomaly Detection
# ─────────────────────────────────────────────

def detect_iforest(
    df: pd.DataFrame,
    numeric_cols: List[str] | None = None,
    contamination: float = 0.01,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, IsolationForest]:
    """Detect anomalies using Isolation Forest and return dataframe with added column 'anomaly_iforest'."""
    numeric_cols = numeric_cols or df.select_dtypes(include="number").columns.tolist()

    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state
    )
    preds = clf.fit_predict(df[numeric_cols])
    df["anomaly_iforest"] = (preds == -1).astype(int)

    return df, clf

# ─────────────────────────────────────────────
# Contamination Grid Search Runner
# ─────────────────────────────────────────────

def run_iforest_grid(
    datasets: Dict[str, pd.DataFrame],
    contaminations: List[float]
) -> pd.DataFrame:
    """
    Run Isolation Forest across all datasets for a grid of contamination values.
    Returns a DataFrame summarizing anomaly counts.
    """
    results = []

    for c in contaminations:
        for fname, df in datasets.items():
            try:
                df_out, _ = detect_iforest(df.copy(), contamination=c)
                n_anom = int(df_out["anomaly_iforest"].sum())
            except Exception as e:
                print(f"[!] {fname} | contamination={c:.3f} failed: {e}")
                n_anom = np.nan

            results.append({
                "file": fname,
                "contamination": c,
                "anomalies": n_anom
            })

    return pd.DataFrame(results).dropna(subset=["anomalies"])


# ─────────────────────────────────────────────
# One‑Class SVM Anomaly Detection
# ─────────────────────────────────────────────
def detect_ocsvm(
    df: pd.DataFrame,
    numeric_cols: List[str] | None = None,
    nu: float = 0.05,
    kernel: str = "rbf",
) -> Tuple[pd.DataFrame, OneClassSVM]:
    """Detect anomalies with One‑Class SVM and add column 'anomaly_ocsvm'."""
    numeric_cols = numeric_cols or df.select_dtypes(include="number").columns.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[numeric_cols])

    clf = OneClassSVM(nu=nu, kernel=kernel, gamma="auto")
    preds = clf.fit_predict(X)
    df["anomaly_ocsvm"] = (preds == -1).astype(int)

    return df, clf

# ─────────────────────────────────────────────
# ν‑Grid Search Runner for OCSVM
# ─────────────────────────────────────────────
def run_ocsvm_grid(
    datasets: Dict[str, pd.DataFrame],
    nu_grid: List[float],
    kernel: str = "rbf"
) -> pd.DataFrame:
    """
    Run One‑Class SVM across all datasets for a grid of ν values.
    Returns a DataFrame summarizing anomaly counts.
    """
    results = []

    for nu in nu_grid:
        for fname, df in datasets.items():
            try:
                df_out, _ = detect_ocsvm(df.copy(), nu=nu, kernel=kernel)
                n_anom = int(df_out["anomaly_ocsvm"].sum())
            except Exception as e:
                print(f"[!] {fname} | nu={nu:.3f} failed: {e}")
                n_anom = np.nan

            results.append({
                "file": fname,
                "nu": nu,
                "anomalies": n_anom
            })

    return pd.DataFrame(results).dropna(subset=["anomalies"])


def _to_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Convert 2‑D array into overlapping sequences."""
    return np.stack([arr[i : i + seq_len] for i in range(len(arr) - seq_len + 1)])

def _build_autoencoder(input_shape: tuple[int, int]) -> Sequential:
    """
    Builds an LSTM autoencoder model for anomaly detection.

    Parameters:
    - input_shape: Tuple (timesteps, num_features)

    Returns:
    - Compiled Keras Sequential model
    """
    timesteps, num_features = input_shape

    model = Sequential([
        LSTM(128, activation='relu', input_shape=(timesteps, num_features), return_sequences=True),
        LSTM(64, activation='relu', return_sequences=False),
        RepeatVector(timesteps),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(num_features))
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def detect_lstm(
    df: pd.DataFrame,
    numeric_cols: List[str] | None = None,
    seq_len: int = 30,
    epochs: int = 40,
    contamination: float = 0.02,
    batch_size: int = 128,
    verbose: int = 0,
):
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras not installed – LSTM unavailable.")

    numeric_cols = numeric_cols or df.select_dtypes("number").columns.tolist()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numeric_cols])

    seqs = _to_sequences(scaled, seq_len)
    n_samples, _, n_feat = seqs.shape

    model = _build_autoencoder(input_shape = (seq_len, n_feat))
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(seqs, seqs, epochs=epochs, verbose=verbose, batch_size=batch_size, shuffle=True, callbacks=[es])

    recon = model.predict(seqs, verbose=0)
    mse = np.mean(np.square(seqs - recon), axis=(1, 2))
    padded_mse = np.concatenate([np.zeros(seq_len - 1), mse])
    df["recon_error"] = padded_mse

    thresh = np.quantile(mse, 1 - contamination)
    df["anomaly_lstm"] = (df["recon_error"] > thresh).astype(int)
    return df, model, thresh

# ─────────────────────────────────────────────
# Generic plot helper
# ─────────────────────────────────────────────

def plot_anomalies(df: pd.DataFrame, sensor: str, anomaly_col: str):
    df = df.sort_values("timestamp")
    plt.figure(figsize=(14, 4))
    sns.lineplot(data=df, x="timestamp", y=sensor, label=sensor)
    sns.scatterplot(
        data=df[df[anomaly_col] == 1], x="timestamp", y=sensor, color="red", s=40, label="Anomaly"
    )
    plt.title(f"{sensor} – {anomaly_col}")
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# Batch runner utility (re‑usable in notebook)
# ─────────────────────────────────────────────

def batch_detect(
    folder: str = "processed_data",
    models: List[str] = ["iforest", "ocsvm", "lstm"],
    out_dir: str = "anomaly_outputs",
    **kwargs,
):
    Path(out_dir).mkdir(exist_ok=True)
    data = load_processed_files(folder)
    for fname, df in data.items():
        if "iforest" in models:
            df, _ = detect_iforest(df, contamination=kwargs.get("if_contam", 0.01))
        if "ocsvm" in models:
            df, _ = detect_ocsvm(df, nu=kwargs.get("ocsvm_nu", 0.05))
        if "lstm" in models:
            df, _, _ = detect_lstm(
                df,
                seq_len=kwargs.get("seq_len", 30),
                epochs=kwargs.get("epochs", 30),
                contamination=kwargs.get("lstm_contam", 0.02),
                verbose=kwargs.get("verbose", 0),
            )
        df.to_csv(Path(out_dir) / fname, index=False)
        print(f"[✓] {fname} saved to {out_dir}")

# If run as CLI ----------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run anomaly detection on processed CSVs")
    p.add_argument("--folder", default="processed_data")
    p.add_argument("--models", default="iforest,ocsvm", help="comma‑sep list")
    p.add_argument("--out", default="anomaly_outputs")
    p.add_argument("--if_contam", type=float, default=0.01)
    p.add_argument("--ocsvm_nu", type=float, default=0.05)
    p.add_argument("--lstm", action="store_true", help="include LSTM (requires TF)")
    args = p.parse_args()

    model_list = args.models.split(",")
    if args.lstm and "lstm" not in model_list:
        model_list.append("lstm")
    batch_detect(
        folder=args.folder,
        models=model_list,
        out_dir=args.out,
        if_contam=args.if_contam,
        ocsvm_nu=args.ocsvm_nu,
        lstm_contam=0.02,
        seq_len=30,
        epochs=30,
        verbose=1,
    )