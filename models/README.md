# Models Directory

This directory is intended to store trained machine learning models and scalers.

## File Organization

```
models/
├── model.pkl          # Your trained prediction model
├── scaler.pkl         # Feature scaler (StandardScaler, etc.)
└── metadata.json      # Model metadata and version info
```

## Loading Models

To integrate your trained model, update the `load_or_create_model()` function in `app.py`:

```python
import joblib

def load_or_create_model():
    global model, scaler
    
    if model is None:
        # Load from file
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    
    return model, scaler
```

Then update the prediction endpoint to use your model:

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(movie: MovieInput):
    load_or_create_model()
    
    features = np.array([[
        movie.budget,
        movie.runtime,
        movie.year,
        movie.num_reviews,
        movie.avg_rating
    ]])
    
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    return PredictionResponse(
        input_data=movie,
        prediction=float(prediction),
        confidence=0.95,
        message="Prediction successful"
    )
```

## Saving Models

Use joblib to save your trained models:

```python
import joblib
from sklearn.preprocessing import StandardScaler

# After training your model
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
```

## Supported Model Types

- scikit-learn models (RandomForest, XGBoost, LightGBM, etc.)
- Neural networks (with serialization)
- Custom models (with proper serialization)

**Note:** This directory is in .gitignore to prevent uploading large model files.

