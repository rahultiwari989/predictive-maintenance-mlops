import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import COLS, DROP_COLS, RUL_CAP, WINDOW_SIZE


def load_cmapss(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=COLS,
    )
    return df.drop(columns=DROP_COLS)


def add_rul(df, cap=RUL_CAP):
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].max()
    df["RUL"] = df["engine_id"].map(max_cycle) - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def get_feature_cols(df):
    return [
        c for c in df.columns
        if c.startswith("op_setting") or c.startswith("sensor_")
    ]


def split_by_engine(df, train_fraction=0.8):
    engine_ids = df["engine_id"].unique()
    split = int(train_fraction * len(engine_ids))
    train_ids = engine_ids[:split]
    val_ids = engine_ids[split:]
    return (
        df[df["engine_id"].isin(train_ids)].copy(),
        df[df["engine_id"].isin(val_ids)].copy(),
    )


def fit_scaler(train_df, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df, scaler, feature_cols):
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def create_sequences(df, feature_cols, window_size=WINDOW_SIZE):
    X, y = [], []
    for engine_id in df["engine_id"].unique():
        engine_data = df[df["engine_id"] == engine_id].sort_values("cycle")
        features = engine_data[feature_cols].to_numpy()
        rul = engine_data["RUL"].to_numpy()
        for i in range(len(engine_data) - window_size):
            X.append(features[i:i + window_size])
            y.append(rul[i + window_size])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def create_test_sequences(df, feature_cols, window_size=WINDOW_SIZE):
    X = []
    for engine_id in df["engine_id"].unique():
        engine_data = df[df["engine_id"] == engine_id].sort_values("cycle")
        features = engine_data[feature_cols].to_numpy()
        if len(features) >= window_size:
            X.append(features[-window_size:])
        else:
            pad = np.zeros((window_size - len(features), features.shape[1]))
            X.append(np.vstack([pad, features]))
    return np.asarray(X, dtype=np.float32)
