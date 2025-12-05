import pandas as pd
from model.preprocess import fill_missing_with_rolling

def test_preprocess_fills_nan():
    df = pd.DataFrame({"value": [1, None, 3]})
    result = fill_missing_with_rolling(df, window_size=1)

    assert result["value"].isna().sum() == 0
