import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import ModelCheckpoint
from lstm_model import build_advanced_lstm_model
from dataloader import create_dataset, scale_series


# Load data (after preprocess)
df = pd.read_csv('./data/preprocess-QDL-OPEC.csv')
values = df['value'].values.reshape(-1, 1)

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

window_size = 5

for fold, (train_idx, test_idx) in enumerate(tscv.split(values)):
    print(f"\n Fold {fold + 1}")

    train_raw, test_raw = values[train_idx], values[test_idx]

    # Scale train/test
    train_scaled, test_scaled, scaler = scale_series(train_raw, test_raw)

    # Make dataset
    X_train, y_train = create_dataset(train_scaled, window=window_size)

    # Build fresh model for each fold
    model = build_advanced_lstm_model(input_shape=(window_size, 1))

    # Save the best model
    checkpoint = ModelCheckpoint(
        f"checkpoint/fold_{fold+1}.h5",
        monitor='loss',
        save_best_only=True,
        verbose=1
    )

    # Train
    model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=32,
        callbacks=[checkpoint],
        verbose=1
    )
