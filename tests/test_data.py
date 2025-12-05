import pandas as pd
from model.preprocess import load_raw_data

def test_load_raw_data_returns_dataframe():
    df = load_raw_data("data/QDL-OPEC.csv")
    assert isinstance(df, pd.DataFrame)
    assert "value" in df.columns
    assert "date" in df.columns
