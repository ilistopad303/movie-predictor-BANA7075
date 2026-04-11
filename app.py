"""
FastAPI backend for Movie Predictor application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = FastAPI(
    title="Movie Predictor API",
    description="API for predicting movie performance",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class MovieInput(BaseModel):
    """Movie input features for prediction"""
    budget: float
    runtime: int
    year: int
    num_reviews: int = 0
    avg_rating: float = 0.0


class PredictionResponse(BaseModel):
    """Prediction response"""
    input_data: MovieInput
    prediction: float
    confidence: Optional[float] = None
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


# Global model placeholder (will be loaded/created as needed)
model = None
scaler = None


def load_or_create_model():
    """Load model from file or create a simple default model"""
    global model, scaler

    if model is None:
        # For now, we'll use a simple mock model
        # In production, you'd load a trained model here
        model = "mock_model"
        scaler = StandardScaler()

        # Fit scaler with some reasonable default ranges
        sample_data = np.array([
            [1e7, 120, 2020, 100, 7.0],
            [1e8, 150, 2022, 500, 8.0],
            [1e6, 90, 2024, 50, 6.0],
        ])
        scaler.fit(sample_data)

    return model, scaler


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(movie: MovieInput):
    """
    Make a prediction for a movie's performance

    Parameters:
    - budget: Movie budget in USD
    - runtime: Movie runtime in minutes
    - year: Release year
    - num_reviews: Number of reviews
    - avg_rating: Average rating (0-10)
    """
    try:
        load_or_create_model()

        # Prepare features for prediction
        features = np.array([[
            movie.budget,
            movie.runtime,
            movie.year,
            movie.num_reviews,
            movie.avg_rating
        ]])

        # Scale features
        scaled_features = scaler.transform(features)

        # For demo purposes, generate a simple prediction based on features
        # In production, use your trained model
        base_prediction = (
            (movie.budget / 1e8) * 0.3 +
            (movie.avg_rating / 10) * 0.5 +
            (movie.num_reviews / 1000) * 0.2
        )

        # Add some noise for realism
        prediction = np.clip(base_prediction * 100, 0, 1000)

        return PredictionResponse(
            input_data=movie,
            prediction=float(prediction),
            confidence=0.75,
            message="Prediction successful"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Movie Predictor API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

