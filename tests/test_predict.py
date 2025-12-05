import numpy as np
import pytest
from model.predict import prepare_input, predict_next


class DummyModel:
    def predict(self, x):
        return np.array([[0.5]])   # giả sử mô hình dự đoán 0.5 scaled value


class DummyScaler:
    def transform(self, x):
        return x * 0.1
    def inverse_transform(self, x):
        return x * 10


def test_prepare_input_shape():
    arr = np.arange(60).reshape(-1, 1)
    result = prepare_input(arr, window=50)

    assert result.shape == (1, 50, 1)


def test_prepare_input_not_enough_data():
    arr = np.arange(10).reshape(-1, 1)

    with pytest.raises(ValueError):
        prepare_input(arr, window=50)


def test_predict_next():
    raw = np.ones((60, 1))  # 60 giá trị
    model = DummyModel()
    scaler = DummyScaler()

    output = predict_next(model, scaler, raw)

    # raw -> raw*0.1 -> predict -> inverse_predict => 0.5 * 10 = 5.0
    assert output == 5.0
