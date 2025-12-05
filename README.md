
# Oil Price Forecasting with LSTM

![CI/CD](https://github.com/yourusername/oil-Price-Forecasting-LSTM-API-Test/workflows/CI/CD%20Pipeline/badge.svg)
[![codecov](https://codecov.io/gh/yourusername/oil-Price-Forecasting-LSTM-API-Test/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/oil-Price-Forecasting-LSTM-API-Test)

A time series forecasting project using LSTM to predict crude oil prices, served via FastAPI with full unit testing and CI/CD.

## Features

- LSTM neural network for time series forecasting
- Data preprocessing and feature engineering
- RESTful API with FastAPI
- Comprehensive unit tests (95%+ coverage)
- GitHub Actions CI/CD pipeline
- Interactive API documentation (Swagger UI)

## Requirements

- Python 3.10+
- TensorFlow 2.13+
- FastAPI
- See `requirements.txt` for full list


## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=model --cov=app --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_api.py -v
```


## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

### On Push/Pull Request:
1. **Testing**: Runs all unit tests with multiple Python versions
2. **Linting**: Code quality checks with flake8
3. **Security**: Vulnerability scanning with safety and bandit
4. **Code Quality**: Format checking with black and isort
5. **Coverage**: Generates and uploads test coverage reports

### On Main Branch:
- Deployment notification (can be extended for actual deployment)

## Model Performance
The LSTM model achieves:
- Training accuracy: ~  %
- Validation accuracy: ~  %
- Mean Absolute Error: <   
