import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator

try:
    from model.dataloader import create_dataset, scale_series
except ImportError:
    from dataloader import create_dataset, scale_series


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV file."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def create_continuous_series(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to continuous daily time series."""
    df = df.copy()
    df = df.set_index("date")
    daily_df = df.resample("D").asfreq()
    return daily_df


def fill_missing_with_rolling(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Fill missing values using centered rolling average (with forward/backward fallback)."""
    df = df.copy()

    # Tính rolling mean
    rolling = (
        df["value"]
        .rolling(window=window_size * 2 + 1, min_periods=1, center=True)
        .mean()
    )

    # Điền missing bằng rolling mean, sau đó bfill/ffill cho các cạnh nếu còn sót
    df["value"] = df["value"].fillna(rolling)
    df["value"] = df["value"].bfill().ffill()

    return df


def preprocess_data(csv_path: str, window: int | None = None, test_ratio: float = 0.2):
    """
    Full preprocessing pipeline.
    - If window is None: returns cleaned DataFrame with ['date', 'value', 'Year'].
    - If window is an int: returns (X_train, X_test, y_train, y_test, scaler).
    """
    df = load_raw_data(csv_path)
    df = create_continuous_series(df)
    df = fill_missing_with_rolling(df, window_size=5)

    # Reset index + thêm Year
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df["Year"] = df["date"].dt.year
    df = df.sort_values("date")

    if window is None:
        return df

    # Chuyển thành supervised dataset cho model training
    values = df["value"].values.reshape(-1, 1)
    test_count = max(int(len(values) * test_ratio), window + 1)
    train_raw = values[:-test_count]
    test_raw = values[-test_count:]

    train_scaled, test_scaled, scaler = scale_series(train_raw, test_raw)

    X_train, y_train = create_dataset(train_scaled, window=window)
    X_test, y_test = create_dataset(test_scaled, window=window)

    return X_train, X_test, y_train, y_test, scaler


def plot_series(df: pd.DataFrame):
    """Plot series (optional visualization)."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["value"], label="Value")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title("Average Oil Price of OPEC Member Countries")
    plt.legend()

    years = YearLocator()
    years_fmt = DateFormatter("%Y")
    months = MonthLocator()

    ax = plt.gca()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    plt.tight_layout()
    plt.show()


def save_preprocessed(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    input_path = "./data/QDL-OPEC.csv"
    output_path = "./data/preprocess-QDL-OPEC.csv"

    df = preprocess_data(input_path)
    print(df.head())

    plot_series(df)
    save_preprocessed(df, output_path)
