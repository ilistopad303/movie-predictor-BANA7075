# Quick Start Guide

## 🚀 Getting Started with Movie Predictor

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

#### 1. Install Dependencies
```bash
cd /Users/ian/PycharmProjects/movie-predictor-BANA7075
pip install -r requirements.txt
```

#### 2. Start the FastAPI Backend (Terminal 1)
```bash
python app.py
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### 3. Start the Streamlit Frontend (Terminal 2)
```bash
streamlit run streamlit_app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

### How to Use

1. **Enter Movie Parameters:**
   - Budget: Production budget in USD
   - Runtime: Movie duration in minutes
   - Release Year: Expected/actual release year
   - Expected Reviews: Anticipated review count
   - Expected Average Rating: Predicted rating (0-10)

2. **Click "🔮 Make Prediction"**

3. **View Results:**
   - Predicted Revenue
   - Confidence Score
   - Input Summary Table
   - Raw JSON Response

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Make Prediction
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

### Project Structure
```
movie-predictor-BANA7075/
├── app.py                  # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Dependencies
├── README.md              # Full documentation
├── QUICKSTART.md          # This file
├── run.sh                 # Startup script
├── .env.example           # Example environment variables
└── .streamlit/            # Streamlit config
    └── config.toml
```

### Troubleshooting

**Issue: "Cannot connect to API"**
- Make sure the FastAPI backend is running
- Verify it's running on http://localhost:8000
- Check firewall settings

**Issue: "Port 8000 already in use"**
- Start API on different port: `uvicorn app:app --port 8001`
- Update API_URL in streamlit_app.py

**Issue: Missing dependencies**
- Run: `pip install -r requirements.txt --upgrade`

### Next Steps

1. **Integrate Your Model:** Replace the mock prediction logic in `app.py` with your trained ML model
2. **Add Database:** Store predictions for analytics
3. **Deploy:** Deploy to Heroku, AWS, or other platforms
4. **Extend Features:** Add more input parameters, charts, etc.

### Tips

- The API uses CORS, so it's accessible from any frontend
- Streamlit automatically hot-reloads on code changes
- FastAPI provides automatic API documentation at /docs
- Both services can run on different machines if needed

### Contact & Support
For questions or issues, check the README.md file for more detailed information.

