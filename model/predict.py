import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_lstm_model(path: str):
    """Load trained LSTM model from disk."""
    from tensorflow.keras.models import load_model

    return load_model(path)


def load_series(path: str) -> np.ndarray:
    """Load a time series from a CSV file and return it as a (N, 1) array."""
    df = pd.read_csv(path)
    return df["value"].values.reshape(-1, 1)


def load_scaler(path: str) -> MinMaxScaler:
    """Load a previously fitted MinMaxScaler using joblib."""
    return joblib.load(path)


def prepare_input(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Extract the last `window` points and format into (1, window, 1)."""
    if len(values) < window:
        raise ValueError(f"Not enough data: at least {window} values are required.")
    return values[-window:].reshape(1, window, 1)


def predict_next(
    model, scaler: MinMaxScaler, raw_values: np.ndarray, window: int = 50
) -> float:
    """Scale raw series, prepare sliding window, predict next step, and inverse scale."""
    scaled = scaler.transform(raw_values)
    inp = prepare_input(scaled, window=window)
    scaled_pred = model.predict(inp)
    return float(np.asarray(scaler.inverse_transform(scaled_pred)).ravel()[0])
