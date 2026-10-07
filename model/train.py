import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import ModelCheckpoint

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.dataloader import create_dataset, scale_series
from model.lstm_model import build_advanced_lstm_model


def train_pipeline(
    data_path: str = "./data/preprocess-QDL-OPEC.csv",
    window_size: int = 60,
    epochs: int = 25,
):
    """Run TimeSeriesSplit cross-validation training on the dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    values = df["value"].values.reshape(-1, 1)

    tscv = TimeSeriesSplit(n_splits=3)
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    last_model = None
    last_scaler = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(values)):
        print(f"\n--- Fold {fold + 1} ---")

        train_raw, test_raw = values[train_idx], values[test_idx]

        # Scale train/test
        train_scaled, test_scaled, scaler = scale_series(train_raw, test_raw)
        last_scaler = scaler

        # Make dataset
        X_train, y_train = create_dataset(train_scaled, window=window_size)
        X_test, y_test = create_dataset(test_scaled, window=window_size)

        # Build fresh model for each fold
        model = build_advanced_lstm_model(input_shape=(window_size, 1))

        checkpoint_path = f"checkpoint/fold_{fold + 1}.h5"
        checkpoint = ModelCheckpoint(
            checkpoint_path, monitor="loss", save_best_only=True, verbose=1
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test) if len(X_test) > 0 else None,
            epochs=epochs,
            batch_size=32,
            callbacks=[checkpoint],
            verbose=1,
        )
        last_model = model

    # Save final model & scaler to models/ for API consumption
    if last_model is not None and last_scaler is not None:
        last_model.save("models/lstm_model.keras")
        joblib.dump(last_scaler, "models/scaler.pkl")
        joblib.dump(last_scaler, "checkpoint/scaler.pkl")
        print(
            "\n[INFO] Final model saved to models/lstm_model.keras and scaler saved to models/scaler.pkl"
        )


if __name__ == "__main__":
    train_pipeline(window_size=60, epochs=10)
