"""
FastAPI backend for Movie Predictor application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from src.models.predictor import predict_movie

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
    quarter: int
    language: str
    director: int
    genres: List[str]


class PredictionResponse(BaseModel):
    """Prediction response"""
    input_data: MovieInput
    pred_roi: float
    pred_revenue: float
    pred_profit: float
    lower_roi: float
    upper_roi: float
    lower_revenue: float
    upper_revenue: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


# Global model placeholder (will be loaded/created as needed)
model = None
scaler = None


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
        pred_roi, pred_revenue, pred_profit, lower_roi, upper_roi, lower_revenue, upper_revenue = predict_movie(movie.budget,movie.runtime,movie.quarter,movie.language,movie.director,movie.genres)
        return PredictionResponse(
            input_data = movie,
            pred_roi = pred_roi,
            pred_revenue = pred_revenue,
            pred_profit = pred_profit,
            lower_roi = lower_roi,
            upper_roi = upper_roi,
            lower_revenue = lower_revenue,
            upper_revenue = upper_revenue
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
