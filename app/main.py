import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.lstm_model import LSTMModel
from model.preprocess import preprocess_data

# Rate limiter initialization (Keyed by client IP address)
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

# Asynchronous lock to guarantee only one model training session runs at a time
train_lock = asyncio.Lock()


def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for RateLimitExceeded returning standard 429 response"""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Too Many Requests",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": "60 seconds",
        },
        headers={"Retry-After": "60"},
    )


# Load model and scaler at startup
model = None
scaler = None


def load_saved_model():
    global model, scaler
    try:
        if os.path.exists("models/lstm_model.keras"):
            model = LSTMModel.load("models/lstm_model.keras")
            print("[INFO] Model loaded successfully")
        if os.path.exists("models/scaler.pkl"):
            scaler = joblib.load("models/scaler.pkl")
            print("[INFO] Scaler loaded successfully")
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_saved_model()
    yield


app = FastAPI(
    title="Oil Price Forecasting API",
    description="Quantitative LSTM-based crude oil price forecasting API with rate limiting and concurrency protection",
    version="1.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load on import for direct TestClient usage
load_saved_model()


# Request/Response schemas
class PredictionRequest(BaseModel):
    data: List[float] = Field(
        default_factory=list,
        max_length=1000,
        description="Historical oil price sequence (at least 'window' items, maximum 1000 items)",
    )
    window: int = Field(
        60,
        ge=3,
        le=200,
        description="Lookback window size (between 3 and 200)",
    )


class PredictionResponse(BaseModel):
    prediction: float
    timestamp: str


class TrainRequest(BaseModel):
    csv_path: str = Field(
        "data/preprocess-QDL-OPEC.csv",
        description="Path to CSV file with oil price data",
    )
    window: int = Field(
        60,
        ge=3,
        le=200,
        description="Lookback window size (between 3 and 200)",
    )
    epochs: int = Field(
        50,
        ge=1,
        le=100,
        description="Number of training epochs (between 1 and 100)",
    )
    batch_size: int = Field(
        32,
        ge=1,
        le=512,
        description="Batch size for training (between 1 and 512)",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# API endpoints
@app.get("/", response_model=dict)
@limiter.limit("120/minute")
async def root(request: Request):
    """Root endpoint with API information"""
    return {
        "message": "Oil Price Forecasting API",
        "version": "1.1.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Make prediction (POST)",
            "/train": "Train model (POST)",
            "/model/info": "Get model information",
            "/docs": "Interactive API documentation",
        },
    }


@app.get("/health", response_model=HealthResponse)
@limiter.limit("120/minute")
async def health(request: Request):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute")
async def predict(request: Request, body: PredictionRequest):
    """
    Make oil price prediction with rate limiting and payload validation.

    - **data**: List of historical prices (at least 'window' data points, max 1000)
    - **window**: Lookback window size (default: 60, range: 3-200)
    """

    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using /train endpoint",
        )

    if len(body.data) < body.window:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {body.window} data points, got {len(body.data)}",
        )

    try:
        # Take last window values
        recent_data = np.array(body.data[-body.window :])

        # Scale data
        scaled_data = scaler.transform(recent_data.reshape(-1, 1))

        # Reshape for LSTM [samples, timesteps, features]
        X = scaled_data.reshape(1, body.window, 1)

        # Predict
        scaled_prediction = model.predict(X)

        # Inverse transform
        pred_val = float(
            np.asarray(scaler.inverse_transform(scaled_prediction)).ravel()[0]
        )

        return {"prediction": pred_val, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/train")
@limiter.limit("10/minute")
async def train(request: Request, body: TrainRequest):
    """
    Train the LSTM model with concurrency protection, rate limiting, and parameter validation.

    - **csv_path**: Path to CSV file with oil price data
    - **window**: Lookback window size (3 - 200)
    - **epochs**: Number of training epochs (1 - 100)
    - **batch_size**: Batch size for training (1 - 512)
    """
    global model, scaler

    # 1. Concurrency Check: If a training session is active, reject immediately with 409 Conflict
    if train_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A model training job is already in progress. Please wait for completion before submitting a new job.",
        )

    if not os.path.exists(body.csv_path):
        raise HTTPException(
            status_code=404, detail=f"Data file not found: {body.csv_path}"
        )

    async with train_lock:
        try:
            # Execute CPU-intensive training in worker thread to prevent blocking event loop
            def _run_training():
                global model, scaler
                X_train, X_test, y_train, y_test, new_scaler = preprocess_data(
                    body.csv_path, window=body.window
                )

                input_shape = (body.window, 1)
                new_model = LSTMModel(input_shape)

                history = new_model.train(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    epochs=body.epochs,
                    batch_size=body.batch_size,
                )

                os.makedirs("models", exist_ok=True)
                new_model.save("models/lstm_model.keras")
                joblib.dump(new_scaler, "models/scaler.pkl")

                model = new_model
                scaler = new_scaler

                return {
                    "message": "Training completed successfully",
                    "final_loss": float(history.history["loss"][-1]),
                    "final_val_loss": float(history.history["val_loss"][-1]),
                    "epochs_completed": len(history.history["loss"]),
                }

            result = await asyncio.to_thread(_run_training)
            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.get("/model/info")
@limiter.limit("120/minute")
async def model_info(request: Request):
    """Get information about the loaded model"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    trainable_params = (
        int(sum([np.prod(v.shape) for v in model.model.trainable_weights]))
        if model.model.trainable_weights
        else 0
    )
    return {
        "input_shape": str(model.model.input_shape),
        "output_shape": str(model.model.output_shape),
        "total_params": int(model.model.count_params()),
        "trainable_params": trainable_params,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
