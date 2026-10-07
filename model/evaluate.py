import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.dataloader import create_dataset


def evaluate_model(
    model_path: str = "models/lstm_model.keras",
    scaler_path: str = "models/scaler.pkl",
    data_path: str = "data/preprocess-QDL-OPEC.csv",
    window: int = 60,
    test_size: int = 100,
    plot: bool = False,
):
    """Evaluate trained LSTM model against test split."""
    if not os.path.exists(scaler_path):
        if os.path.exists("checkpoint/scaler.pkl"):
            scaler_path = "checkpoint/scaler.pkl"
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    if not os.path.exists(model_path):
        if os.path.exists("checkpoint/best_model.h5"):
            model_path = "checkpoint/best_model.h5"
        elif os.path.exists("checkpoint/fold_3.h5"):
            model_path = "checkpoint/fold_3.h5"
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    scaler = joblib.load(scaler_path)

    if not os.path.exists(data_path):
        if os.path.exists("data/QDL-OPEC.csv"):
            data_path = "data/QDL-OPEC.csv"
        else:
            raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    series = df["value"].dropna().values.reshape(-1, 1)

    train_series = series[:-test_size]
    test_series = series[-test_size:]

    # Scale with fitted scaler
    train_scaled = scaler.transform(train_series)

    x_train, y_train = create_dataset(train_scaled, window=window)

    model = load_model(model_path)
    y_pred_scaled = model.predict(x_train, verbose=0)

    y_train_inv = scaler.inverse_transform(y_train)
    y_pred_inv = scaler.inverse_transform(y_pred_scaled)

    mae = mean_absolute_error(y_train_inv, y_pred_inv)
    rmse = float(np.sqrt(mean_squared_error(y_train_inv, y_pred_inv)))

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(y_train_inv, label="Actual", color="blue")
        plt.plot(y_pred_inv, label="Predicted", color="red")
        plt.title("Actual vs Predicted Oil Prices")
        plt.xlabel("Time Step")
        plt.ylabel("Value (USD)")
        plt.legend()
        plt.show()

    return {"mae": float(mae), "rmse": float(rmse)}


if __name__ == "__main__":
    evaluate_model(window=60, plot=False)
