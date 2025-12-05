import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_dataset(series, window=5):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i-window:i])
        y.append(series[i])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y.reshape(-1, 1)

def scale_series(train, test=None):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)

    if test is not None:
        test_scaled = scaler.transform(test)
        return train_scaled, test_scaled, scaler
    return train_scaled, scaler


# -----------------------------
# USE CASE TIME SERIES SPLIT
# -----------------------------
test_size = 50
train_series = series[:-test_size]
test_series = series[-test_size:]

train_scaled, test_scaled, scaler = scale_series(
    train_series.reshape(-1, 1),
    test_series.reshape(-1, 1)
)

window = 5
X_train, y_train = create_dataset(train_scaled, window)
X_test, y_test = create_dataset(test_scaled, window)
