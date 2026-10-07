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


def compute_portfolio_risk_metrics(
    price_series: pd.Series,
    dates: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04,
    notional: float = 100000.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive quantitative risk metrics for energy portfolio assets.
    - Value at Risk (VaR 95%, 99% for 1-day & 1-week, historical & parametric)
    - Conditional Value at Risk (CVaR / Expected Shortfall 95%, 99%)
    - Sharpe Ratio, Sortino Ratio, and Calmar Ratio
    - Maximum Drawdown (MDD) and Drawdown series
    - Stress Testing Scenarios with Dollar PnL impact
    """
    clean_series = price_series.dropna()
    if len(clean_series) < 10:
        return {
            "error": "Insufficient data points for quantitative risk modeling (need >= 10)",
            "ann_return": 0.0,
            "ann_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "mdd_pct": 0.0,
            "mdd_dollar": 0.0,
            "var_95_1d_pct": 0.0,
            "var_95_1d_dollar": 0.0,
            "var_99_1d_pct": 0.0,
            "var_99_1d_dollar": 0.0,
            "var_95_1w_pct": 0.0,
            "var_95_1w_dollar": 0.0,
            "var_99_1w_pct": 0.0,
            "var_99_1w_dollar": 0.0,
            "var_95_param_pct": 0.0,
            "var_99_param_pct": 0.0,
            "cvar_95_1d_pct": 0.0,
            "cvar_95_1d_dollar": 0.0,
            "cvar_99_1d_pct": 0.0,
            "cvar_99_1d_dollar": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "returns": pd.Series(dtype=float),
            "drawdown_series": pd.Series(dtype=float),
            "rolling_df": pd.DataFrame(),
            "stress_scenarios": [],
        }

    returns = clean_series.pct_change().dropna()
    ann_factor = 252

    mean_daily = float(returns.mean())
    std_daily = float(returns.std())
    ann_return = mean_daily * ann_factor * 100
    ann_volatility = std_daily * np.sqrt(ann_factor) * 100

    skewness = float(returns.skew())
    kurtosis = float(returns.kurtosis())

    # Value at Risk (Historical)
    var_95_1d_pct = float(-np.percentile(returns, 5) * 100)
    var_99_1d_pct = float(-np.percentile(returns, 1) * 100)

    # Parametric Gaussian VaR
    var_95_param_pct = float(-(mean_daily - 1.644853 * std_daily) * 100)
    var_99_param_pct = float(-(mean_daily - 2.326348 * std_daily) * 100)

    # 1-Week (5 trading days) VaR scaling: sqrt(5)
    var_95_1w_pct = float(var_95_1d_pct * np.sqrt(5))
    var_99_1w_pct = float(var_99_1d_pct * np.sqrt(5))

    var_95_1d_dollar = float(notional * (var_95_1d_pct / 100.0))
    var_99_1d_dollar = float(notional * (var_99_1d_pct / 100.0))
    var_95_1w_dollar = float(notional * (var_95_1w_pct / 100.0))
    var_99_1w_dollar = float(notional * (var_99_1w_pct / 100.0))

    # Conditional Value at Risk (CVaR / Expected Shortfall)
    tail_95 = returns[returns <= np.percentile(returns, 5)]
    cvar_95_1d_pct = float(-tail_95.mean() * 100) if len(tail_95) > 0 else var_95_1d_pct
    cvar_95_1d_dollar = float(notional * (cvar_95_1d_pct / 100.0))

    tail_99 = returns[returns <= np.percentile(returns, 1)]
    cvar_99_1d_pct = float(-tail_99.mean() * 100) if len(tail_99) > 0 else var_99_1d_pct
    cvar_99_1d_dollar = float(notional * (cvar_99_1d_pct / 100.0))

    # Maximum Drawdown (MDD)
    cum_max = clean_series.cummax()
    drawdown_series = ((clean_series - cum_max) / cum_max) * 100
    mdd_pct = float(drawdown_series.min())
    mdd_dollar = float(notional * (abs(mdd_pct) / 100.0))

    # Risk-Adjusted Ratios
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / ann_factor) - 1.0
    sharpe = (
        float((mean_daily - rf_daily) / std_daily * np.sqrt(ann_factor))
        if std_daily > 0
        else 0.0
    )

    downside = np.minimum(0.0, returns - rf_daily)
    downside_std = float(np.sqrt(np.mean(downside**2)) * np.sqrt(ann_factor))
    sortino = (
        float((ann_return / 100.0 - risk_free_rate) / downside_std)
        if downside_std > 0
        else 0.0
    )

    calmar = (
        float((ann_return / 100.0) / (abs(mdd_pct) / 100.0))
        if abs(mdd_pct) > 0
        else 0.0
    )

    # Rolling Metrics DataFrame (60-day window)
    rolling_vol = (
        returns.rolling(window=60, min_periods=20).std() * np.sqrt(ann_factor) * 100
    )
    rolling_mean = returns.rolling(window=60, min_periods=20).mean()
    rolling_sharpe = (
        (rolling_mean - rf_daily)
        / (returns.rolling(window=60, min_periods=20).std().replace(0, 1e-9))
        * np.sqrt(ann_factor)
    )

    date_vals = (
        dates.iloc[1:].values
        if dates is not None and len(dates) == len(clean_series)
        else (clean_series.index[1:] if len(clean_series) > 1 else clean_series.index)
    )

    rolling_df = pd.DataFrame(
        {
            "date": date_vals,
            "returns": returns.values,
            "drawdown": drawdown_series.iloc[1:].values,
            "rolling_vol_60d": rolling_vol.values,
            "rolling_sharpe_60d": rolling_sharpe.values,
        }
    )

    # Stress Testing Scenarios
    stress_scenarios = [
        {
            "event": "2008 Global Financial Crisis",
            "benchmark_shock": -54.2,
            "impact_dollar": notional * (-0.542),
            "description": "Subprime liquidity crisis and global demand contraction",
        },
        {
            "event": "2014 OPEC Price War",
            "benchmark_shock": -48.6,
            "impact_dollar": notional * (-0.486),
            "description": "US shale boom and market share defense by OPEC",
        },
        {
            "event": "2020 COVID-19 Demand Shock",
            "benchmark_shock": -68.4,
            "impact_dollar": notional * (-0.684),
            "description": "Global aviation groundings and storage capacity exhaustion",
        },
        {
            "event": "2022 Geopolitical Conflict Surge",
            "benchmark_shock": 42.1,
            "impact_dollar": notional * 0.421,
            "description": "Eastern European conflict and immediate supply risk premium",
        },
        {
            "event": "3-Sigma Extreme Daily Shock",
            "benchmark_shock": round(float(-3.0 * std_daily * 100), 2),
            "impact_dollar": round(float(notional * (-3.0 * std_daily)), 2),
            "description": "Statistical tail event (99.73% normal distribution bounds)",
        },
    ]

    return {
        "ann_return": round(ann_return, 2),
        "ann_volatility": round(ann_volatility, 2),
        "mean_daily_return": round(mean_daily * 100, 3),
        "std_daily": round(std_daily * 100, 3),
        "skewness": round(skewness, 2),
        "kurtosis": round(kurtosis, 2),
        "var_95_1d_pct": round(var_95_1d_pct, 2),
        "var_95_1d_dollar": round(var_95_1d_dollar, 2),
        "var_99_1d_pct": round(var_99_1d_pct, 2),
        "var_99_1d_dollar": round(var_99_1d_dollar, 2),
        "var_95_1w_pct": round(var_95_1w_pct, 2),
        "var_95_1w_dollar": round(var_95_1w_dollar, 2),
        "var_99_1w_pct": round(var_99_1w_pct, 2),
        "var_99_1w_dollar": round(var_99_1w_dollar, 2),
        "var_95_param_pct": round(var_95_param_pct, 2),
        "var_99_param_pct": round(var_99_param_pct, 2),
        "cvar_95_1d_pct": round(cvar_95_1d_pct, 2),
        "cvar_95_1d_dollar": round(cvar_95_1d_dollar, 2),
        "cvar_99_1d_pct": round(cvar_99_1d_pct, 2),
        "cvar_99_1d_dollar": round(cvar_99_1d_dollar, 2),
        "mdd_pct": round(mdd_pct, 2),
        "mdd_dollar": round(mdd_dollar, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "returns": returns,
        "drawdown_series": drawdown_series,
        "rolling_df": rolling_df,
        "stress_scenarios": stress_scenarios,
    }
