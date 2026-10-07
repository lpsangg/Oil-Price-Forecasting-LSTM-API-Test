import numpy as np
import pandas as pd
import pytest

from dashboard.utils import (
    compute_technical_indicators,
    load_artifacts,
    load_opec_dataset,
    predict_multistep,
    run_backtest_simulation,
)


class DummyModel:
    def predict(self, x):
        # Predict slightly higher than input mean
        return np.array([[float(np.mean(x)) + 0.01]])


class DummyScaler:
    def transform(self, x):
        return np.asarray(x) / 100.0

    def inverse_transform(self, x):
        return np.asarray(x) * 100.0


def test_load_artifacts():
    model, scaler = load_artifacts("nonexistent_model.keras", "nonexistent_scaler.pkl")
    assert model is None
    assert scaler is None


def test_load_opec_dataset():
    df = load_opec_dataset("data/preprocess-QDL-OPEC.csv")
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert "value" in df.columns
    assert len(df) > 100
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_compute_technical_indicators():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=60),
            "value": np.linspace(70, 90, 60),
        }
    )
    res = compute_technical_indicators(df)
    assert "SMA_20" in res.columns
    assert "SMA_50" in res.columns
    assert "EMA_20" in res.columns
    assert "BB_Upper" in res.columns
    assert "BB_Lower" in res.columns
    assert "RSI_14" in res.columns
    assert "Volatility_30D" in res.columns
    assert not res["SMA_20"].isna().all()


def test_predict_multistep_shape_and_bounds():
    model = DummyModel()
    scaler = DummyScaler()
    raw_series = np.linspace(70, 80, 60)
    start_date = pd.to_datetime("2023-01-01")

    res = predict_multistep(
        model=model,
        scaler=scaler,
        raw_series=raw_series,
        start_date=start_date,
        window=30,
        horizon=7,
        confidence_level=0.90,
    )

    assert len(res) == 7
    assert "forecast" in res.columns
    assert "lower_bound" in res.columns
    assert "upper_bound" in res.columns
    # Check that upper bound is >= forecast >= lower bound
    assert (res["upper_bound"] >= res["forecast"]).all()
    assert (res["forecast"] >= res["lower_bound"]).all()
    assert res["date"].iloc[0] == pd.to_datetime("2023-01-02")


def test_predict_multistep_insufficient_data():
    model = DummyModel()
    scaler = DummyScaler()
    raw_series = np.array([70.0, 71.0])
    with pytest.raises(ValueError, match="Need at least"):
        predict_multistep(
            model=model,
            scaler=scaler,
            raw_series=raw_series,
            start_date=pd.to_datetime("2023-01-01"),
            window=10,
            horizon=5,
        )


def test_run_backtest_simulation():
    model = DummyModel()
    scaler = DummyScaler()
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=100),
            "value": np.linspace(70, 90, 100),
        }
    )

    metrics, comp_df = run_backtest_simulation(
        model=model,
        scaler=scaler,
        full_df=df,
        cutoff_idx=70,
        window=30,
        horizon=10,
    )

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "directional_accuracy" in metrics
    assert len(comp_df) == 10
    assert "actual" in comp_df.columns
    assert "forecast" in comp_df.columns
    assert "error" in comp_df.columns


def test_compute_portfolio_risk_metrics_comprehensive():
    """Verify VaR, CVaR, Sharpe, Sortino, MDD, and stress testing logic."""
    from dashboard.utils import compute_portfolio_risk_metrics

    # Simulated price series with known fluctuations
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    prices = pd.Series(70.0 + np.cumsum(np.random.normal(0.05, 1.2, 100)), index=dates)

    res = compute_portfolio_risk_metrics(
        price_series=prices,
        dates=pd.Series(dates),
        risk_free_rate=0.04,
        notional=100000.0,
    )

    assert "var_95_1d_pct" in res
    assert "var_99_1d_pct" in res
    assert "cvar_95_1d_pct" in res
    assert "mdd_pct" in res
    assert "sharpe_ratio" in res
    assert "sortino_ratio" in res
    assert "calmar_ratio" in res

    # Monotonicity of tail risk: 99% VaR >= 95% VaR
    assert res["var_99_1d_pct"] >= res["var_95_1d_pct"]

    # Expected shortfall CVaR >= VaR
    assert res["cvar_95_1d_pct"] >= res["var_95_1d_pct"]

    # 1-Week VaR should be greater than 1-Day VaR (time-scaling)
    assert res["var_95_1w_pct"] > res["var_95_1d_pct"]

    # Maximum Drawdown must be non-positive
    assert res["mdd_pct"] <= 0.0

    # Dollar amounts align with notional
    assert res["var_95_1d_dollar"] == pytest.approx(
        100000.0 * (res["var_95_1d_pct"] / 100.0), rel=1e-3
    )

    # Stress testing scenarios
    assert len(res["stress_scenarios"]) == 5
    for sc in res["stress_scenarios"]:
        assert "event" in sc
        assert "benchmark_shock" in sc
        assert "impact_dollar" in sc

    # Rolling metrics DataFrame
    assert len(res["rolling_df"]) == 99
    assert "drawdown" in res["rolling_df"].columns
    assert "rolling_vol_60d" in res["rolling_df"].columns
    assert "rolling_sharpe_60d" in res["rolling_df"].columns


def test_compute_portfolio_risk_metrics_insufficient_data():
    """Verify safe fallback when series has fewer than 10 points."""
    from dashboard.utils import compute_portfolio_risk_metrics

    short_prices = pd.Series([70.0, 71.0, 72.0])
    res = compute_portfolio_risk_metrics(short_prices)
    assert "error" in res
    assert res["var_95_1d_pct"] == 0.0
    assert res["mdd_pct"] == 0.0
