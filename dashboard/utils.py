import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from model.lstm_model import LSTMModel


def load_artifacts(
    model_path: str = "models/lstm_model.keras",
    scaler_path: str = "models/scaler.pkl",
) -> Tuple[Optional[LSTMModel], Optional[Any]]:
    """Load the trained LSTM model and MinMaxScaler scaler."""
    model = None
    scaler = None
    if os.path.exists(model_path):
        try:
            model = LSTMModel.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading scaler: {e}")
    return model, scaler


def load_opec_dataset(path: str = "data/preprocess-QDL-OPEC.csv") -> pd.DataFrame:
    """Load historical OPEC spot price series with cleaned datetime index."""
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "..", path)

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Keep only relevant columns
    cols_to_keep = ["date", "value"]
    if "Year" in df.columns:
        cols_to_keep.append("Year")
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df


def fetch_live_market_data(ticker: str = "CL=F", period: str = "2y") -> pd.DataFrame:
    """
    Fetch real-time crude oil futures data from Yahoo Finance.
    Tickers:
        - 'CL=F': WTI Crude Oil
        - 'BZ=F': Brent Crude Oil
    """
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    if data.empty:
        raise ValueError(f"Could not retrieve data for ticker '{ticker}'")

    # Handle multi-level columns in newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    df = data.reset_index()
    date_col = "Date" if "Date" in df.columns else "date"
    close_col = "Close" if "Close" in df.columns else "value"

    df["date"] = pd.to_datetime(df[date_col])
    df["value"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)

    # Keep OHLCV if available for candlestick charts
    output_cols = ["date", "value"]
    for c in ["Open", "High", "Low", "Volume"]:
        if c in df.columns:
            output_cols.append(c)

    return df[output_cols].copy()


def compute_technical_indicators(
    df: pd.DataFrame, price_col: str = "value"
) -> pd.DataFrame:
    """Calculate moving averages, Bollinger Bands, and RSI for market context."""
    res = df.copy()
    prices = res[price_col]

    # Simple Moving Averages
    res["SMA_20"] = prices.rolling(window=20, min_periods=1).mean()
    res["SMA_50"] = prices.rolling(window=50, min_periods=1).mean()
    res["EMA_20"] = prices.ewm(span=20, adjust=False).mean()

    # Bollinger Bands (20-day, 2 std)
    rolling_std = prices.rolling(window=20, min_periods=1).std().fillna(0)
    res["BB_Upper"] = res["SMA_20"] + (rolling_std * 2)
    res["BB_Lower"] = (res["SMA_20"] - (rolling_std * 2)).clip(lower=0)

    # RSI (14 periods)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss.replace(0, 1e-9))
    res["RSI_14"] = 100 - (100 / (1 + rs))

    # Rolling Volatility (annualized %)
    log_ret = np.log(prices / prices.shift(1))
    res["Volatility_30D"] = (
        log_ret.rolling(window=30, min_periods=5).std() * np.sqrt(252) * 100
    )

    return res


def predict_multistep(
    model: LSTMModel,
    scaler: Any,
    raw_series: np.ndarray,
    start_date: pd.Timestamp,
    window: int = 60,
    horizon: int = 30,
    confidence_level: float = 0.90,
) -> pd.DataFrame:
    """
    Perform autoregressive multi-step recursive forecasting with confidence intervals.
    """
    if len(raw_series) < window:
        raise ValueError(
            f"Need at least {window} historical data points, got {len(raw_series)}"
        )

    # Extract historical window
    current_window = raw_series[-window:].reshape(-1, 1)
    scaled_window = scaler.transform(current_window).flatten()

    # Base volatility for uncertainty bounds
    recent_returns = np.diff(raw_series[-window:].ravel())
    std_residual = float(np.std(recent_returns)) if len(recent_returns) > 1 else 1.5

    # Z-multiplier for confidence intervals
    z_table = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z_score = z_table.get(confidence_level, 1.645)

    predictions = []
    lower_bounds = []
    upper_bounds = []
    forecast_dates = []

    curr_date = start_date
    rolling_scaled = list(scaled_window)

    for h in range(1, horizon + 1):
        # Step forward by 1 calendar day
        curr_date = curr_date + pd.Timedelta(days=1)
        forecast_dates.append(curr_date)

        # Prepare input tensor: [1, window, 1]
        inp = np.array(rolling_scaled[-window:]).reshape(1, window, 1)
        scaled_next = model.predict(inp)
        next_scaled_val = float(scaled_next[0, 0])

        # Append to rolling buffer for autoregression
        rolling_scaled.append(next_scaled_val)

        # Inverse transform to get actual price ($/bbl)
        inv_val = float(
            np.asarray(scaler.inverse_transform(np.array([[next_scaled_val]]))).ravel()[
                0
            ]
        )
        predictions.append(inv_val)

        # Expanding uncertainty cone: sigma * sqrt(h)
        uncertainty = z_score * std_residual * np.sqrt(h)
        lower_bounds.append(max(0.0, inv_val - uncertainty))
        upper_bounds.append(inv_val + uncertainty)

    return pd.DataFrame(
        {
            "date": forecast_dates,
            "forecast": predictions,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
            "horizon_day": list(range(1, horizon + 1)),
        }
    )


def run_backtest_simulation(
    model: LSTMModel,
    scaler: Any,
    full_df: pd.DataFrame,
    cutoff_idx: int,
    window: int = 60,
    horizon: int = 30,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Backtest the forecasting model at a specific historical point.
    Compares the predicted trajectory against real ground truth.
    """
    if cutoff_idx < window:
        raise ValueError(f"Cutoff index must be >= window size ({window})")
    if cutoff_idx + horizon > len(full_df):
        horizon = len(full_df) - cutoff_idx

    # Split history and future ground truth
    train_history = full_df.iloc[:cutoff_idx]
    actual_future = (
        full_df.iloc[cutoff_idx : cutoff_idx + horizon].copy().reset_index(drop=True)
    )

    history_series = train_history["value"].values
    start_date = pd.to_datetime(train_history["date"].iloc[-1])

    # Run multi-step forecast starting from cutoff point
    forecast_df = predict_multistep(
        model=model,
        scaler=scaler,
        raw_series=history_series,
        start_date=start_date,
        window=window,
        horizon=horizon,
    )

    actuals = actual_future["value"].values
    preds = forecast_df["forecast"].values[: len(actuals)]

    # Metrics computation
    mae = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mape = float(np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1e-5))) * 100)

    # Directional Accuracy (comparing step-by-step movement direction)
    if len(actuals) > 1:
        prev_actual = np.insert(actuals[:-1], 0, history_series[-1])
        actual_diff = np.sign(actuals - prev_actual)
        pred_diff = np.sign(preds - prev_actual)
        dir_acc = float(np.mean(actual_diff == pred_diff) * 100)
    else:
        dir_acc = (
            100.0
            if np.sign(preds[0] - history_series[-1])
            == np.sign(actuals[0] - history_series[-1])
            else 0.0
        )

    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "directional_accuracy": round(dir_acc, 1),
        "test_points": len(actuals),
    }

    comparison_df = pd.DataFrame(
        {
            "date": actual_future["date"].values[: len(actuals)],
            "actual": actuals,
            "forecast": preds,
            "lower_bound": forecast_df["lower_bound"].values[: len(actuals)],
            "upper_bound": forecast_df["upper_bound"].values[: len(actuals)],
            "error": np.abs(actuals - preds),
        }
    )

    return metrics, comparison_df
