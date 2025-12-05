import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load original data
df = pd.read_csv("./data/preprocess-QDL-OPEC.csv")
values = df['value'].values.reshape(-1, 1)

# Apply the same scaling used during training
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# Use the last 50 values as input
last_50 = scaled_values[-50:]
input_data = np.array(last_50).reshape(1, 50, 1)

# Load trained model
model = load_model("model_lstm.h5")

# Predict
scaled_pred = model.predict(input_data)
prediction = scaler.inverse_transform(scaled_pred)[0][0]

print("Predicted value:", prediction)
