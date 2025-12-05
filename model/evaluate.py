import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load scaler
scaler = joblib.load("checkpoint/scaler.pkl")

# Load raw data
df = pd.read_csv("data/raw_data.csv")
series = df['value'].values.reshape(-1, 1)

# Split train/test
test_size = 50
train_series = series[:-test_size]

# Scale with fitted scaler
train_scaled = scaler.transform(train_series)

# Create dataset for evaluation
def create_dataset(series, window=5):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i-window:i])
        y.append(series[i])
    return np.array(X), np.array(y)

window = 5
x_train, y_train = create_dataset(train_scaled, window)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Load trained LSTM model
model = load_model("checkpoint/best_model.h5")

# Predict
y_pred_scaled = model.predict(x_train)

# Inverse transform
y_train_inv = scaler.inverse_transform(y_train)
y_pred_inv = scaler.inverse_transform(y_pred_scaled)

# Metrics
mae = mean_absolute_error(y_train_inv, y_pred_inv)
rmse = mean_squared_error(y_train_inv, y_pred_inv, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_train_inv, label="Actual", color="blue")
plt.plot(y_pred_inv, label="Predicted", color="red")
plt.title("Training Data: Actual vs Predicted")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
