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
