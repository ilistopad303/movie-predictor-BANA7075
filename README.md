                          mjbvn # Movie Predictor

A FastAPI backend with Streamlit frontend for predicting movie performance.

## Project Structure

```
movie-predictor-BANA7075/
├── app.py                  # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .streamlit/            # Streamlit configuration (optional)
    └── config.toml
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Backend

```bash
python app.py
```

Or use uvicorn directly:

```bash
uvicorn app:app --reload
```

The API will be available at: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 3. Run the Streamlit Frontend (in a new terminal)

```bash
streamlit run streamlit_app.py
```

The frontend will open in your browser at: `http://localhost:8501`

## Features

### FastAPI Backend
- **Health Check**: `/health` - Check API status
- **Prediction Endpoint**: `/POST /predict` - Get movie performance predictions
- **Interactive Documentation**: Automatic API docs via Swagger UI
- **CORS Support**: Enabled for frontend communication

### Streamlit Frontend
- Interactive input form for movie parameters
- Real-time predictions
- Visual result display with metrics
- API connection status indicator
- Responsive design

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "budget": 50000000,
    "runtime": 120,
    "year": 2024,
    "num_reviews": 500,
    "avg_rating": 7.5
  }'
```

## Input Parameters

- **budget**: Movie budget in USD (float)
- **runtime**: Movie runtime in minutes (integer, 1-500)
- **year**: Release year (integer, 1990-2050)
- **num_reviews**: Expected number of reviews (integer, 0+)
- **avg_rating**: Expected average rating 0-10 (float, 0.0-10.0)

## Output

```json
{
  "input_data": {
    "budget": 50000000,
    "runtime": 120,
    "year": 2024,
    "num_reviews": 500,
    "avg_rating": 7.5
  },
  "prediction": 234.56,
  "confidence": 0.75,
  "message": "Prediction successful"
}
```

## Development

### Adding a Real Model

Replace the mock prediction logic in `app.py` with your trained model:

```python
# In app.py, replace the prediction logic
model = joblib.load('path/to/your/model.pkl')
prediction = model.predict(scaled_features)[0]
```

### Customizing the Frontend

Edit `streamlit_app.py` to:
- Change layout and styling
- Add new input parameters
- Modify result visualization
- Add charts and analytics

## Requirements

- Python 3.8+
- FastAPI 0.104.1+
- Streamlit 1.29.0+
- scikit-learn 1.3.2+
- pandas 2.1.1+
- numpy 1.24.3+

## Troubleshooting

### Frontend can't connect to API
- Make sure both `app.py` and `streamlit_app.py` are running
- Verify the API URL in `streamlit_app.py` matches your API endpoint
- Check firewall settings

### Port conflicts
- Change API port: `uvicorn app:app --port 8001`
- Change Streamlit port: `streamlit run streamlit_app.py --server.port 8502`

## Next Steps

1. Integrate your trained machine learning model
2. Add database integration for storing predictions
3. Implement authentication if needed
4. Deploy to cloud (AWS, Heroku, etc.)
5. Add advanced features (batch predictions, model versioning, etc.)

## License

MIT License

