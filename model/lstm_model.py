import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


def build_base_lstm_model(input_shape=(None, 1)):
    """Build a standard 2-layer LSTM model accepting variable sequence lengths."""
    keras_input_shape = (
        (None, input_shape[-1]) if len(input_shape) == 2 else input_shape
    )
    model = Sequential(
        [
            Input(shape=keras_input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mae")
    return model


def build_advanced_lstm_model(input_shape=(None, 1)):
    """Build an advanced CNN + BiLSTM + LSTM stacked architecture accepting variable sequence lengths."""
    keras_input_shape = (
        (None, input_shape[-1]) if len(input_shape) == 2 else input_shape
    )
    model = Sequential(
        [
            Input(shape=keras_input_shape),
            # 1. CNN Feature Extractor
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            # 2. BiLSTM layer
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.15),
            # 3. Stacked LSTM layers
            LSTM(64, return_sequences=True),
            Dropout(0.1),
            LSTM(32, return_sequences=False),
            Dropout(0.1),
            # 4. Dense Head
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_absolute_error")
    return model


class LSTMModel:
    """Wrapper class providing clean lifecycle methods for training, inference, and serialization."""

    def __init__(self, input_shape=(None, 1), model=None, model_type="advanced"):
        if model is not None:
            self.model = model
        else:
            if model_type == "base":
                self.model = build_base_lstm_model(input_shape)
            else:
                self.model = build_advanced_lstm_model(input_shape)

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=20,
        batch_size=32,
        callbacks=None,
    ):
        validation_data = (
            (X_val, y_val) if (X_val is not None and len(X_val) > 0) else None
        )
        return self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            verbose=1,
        )

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        keras_model = load_model(path)
        return cls(model=keras_model)
