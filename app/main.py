from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lstm_model import LSTMModel
from model.preprocess import preprocess_data

app = FastAPI(
    title="Oil Price Forecasting API",
    description="LSTM-based crude oil price forecasting API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler at startup
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        if os.path.exists("models/lstm_model.keras"):
            model = LSTMModel.load("models/lstm_model.keras")
            print("✓ Model loaded successfully")
        if os.path.exists("models/scaler.pkl"):
            scaler = joblib.load("models/scaler.pkl")
            print("✓ Scaler loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load model: {e}")

# Request/Response schemas
class PredictionRequest(BaseModel):
    data: List[float]
    window: int = 60

class PredictionResponse(BaseModel):
    prediction: float
    timestamp: str

class TrainRequest(BaseModel):
    csv_path: str = "data/oil_price.csv"
    window: int = 60
    epochs: int = 50
    batch_size: int = 32

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# API endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Oil Price Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Make prediction (POST)",
            "/train": "Train model (POST)",
            "/model/info": "Get model information",
            "/docs": "Interactive API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make oil price prediction
    
    - **data**: List of historical prices (at least 'window' data points)
    - **window**: Lookback window size (default: 60)
    """
    global model, scaler
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first using /train endpoint"
        )
    
    if len(request.data) < request.window:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {request.window} data points, got {len(request.data)}"
        )
    
    try:
        # Take last window values
        recent_data = np.array(request.data[-request.window:])
        
        # Scale data
        scaled_data = scaler.transform(recent_data.reshape(-1, 1))
        
        # Reshape for LSTM [samples, timesteps, features]
        X = scaled_data.reshape(1, request.window, 1)
        
        # Predict
        scaled_prediction = model.predict(X)
        
        # Inverse transform
        prediction = scaler.inverse_transform(scaled_prediction)[0][0]
        
        return {
            "prediction": float(prediction),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/train")
async def train(request: TrainRequest):
    """
    Train the LSTM model
    
    - **csv_path**: Path to CSV file with oil price data
    - **window**: Lookback window size
    - **epochs**: Number of training epochs
    - **batch_size**: Batch size for training
    """
    global model, scaler
    
    if not os.path.exists(request.csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {request.csv_path}"
        )
    
    try:
        # Preprocess data
        X_train, X_test, y_train, y_test, new_scaler = preprocess_data(
            request.csv_path,
            window=request.window
        )
        
        # Create and train model
        input_shape = (request.window, 1)
        new_model = LSTMModel(input_shape)
        
        history = new_model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=request.epochs,
            batch_size=request.batch_size
        )
        
        # Create models directory if not exists
        os.makedirs("models", exist_ok=True)
        
        # Save model and scaler
        new_model.save("models/lstm_model.keras")
        joblib.save(new_scaler, "models/scaler.pkl")
        
        # Update global variables
        model = new_model
        scaler = new_scaler
        
        return {
            "message": "Training completed successfully",
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "epochs_completed": len(history.history['loss'])
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Training error: {str(e)}"
        )

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    return {
        "input_shape": str(model.model.input_shape),
        "output_shape": str(model.model.output_shape),
        "total_params": int(model.model.count_params()),
        "trainable_params": int(sum([np.prod(v.shape) for v in model.model.trainable_weights]))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)