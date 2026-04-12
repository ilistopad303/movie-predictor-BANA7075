"""
Streamlit frontend for Movie Predictor
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Movie Predictor",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Predictor")
st.markdown("Predict movie performance based on key metrics")

# API Configuration
API_URL = "http://localhost:8000"

@st.cache_data
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_prediction(budget, runtime, year, num_reviews, avg_rating):
    """Make a prediction via the API"""
    try:
        payload = {
            "budget": float(budget),
            "runtime": int(runtime),
            "year": int(year),
            "num_reviews": int(num_reviews),
            "avg_rating": float(avg_rating)
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API at " + API_URL)
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# API Status
api_healthy = check_api_health()
if api_healthy:
    st.success("✅ API Connected")
else:
    st.warning("⚠️ API Not Connected")

# Input section
st.subheader("📊 Movie Information")
col1, col2 = st.columns(2)

with col1:
    budget = st.number_input(
        "Budget (USD)",
        min_value=0.0,
        value=50000000.0,
        step=1000000.0
    )
    runtime = st.number_input(
        "Runtime (minutes)",
        min_value=1,
        max_value=500,
        value=120
    )

with col2:
    year = st.number_input(
        "Release Year",
        min_value=1990,
        max_value=2050,
        value=2024
    )
    num_reviews = st.number_input(
        "Expected Reviews",
        min_value=0,
        value=500,
        step=100
    )

avg_rating = st.slider(
    "Expected Average Rating",
    min_value=0.0,
    max_value=10.0,
    value=7.5,
    step=0.1
)

# Prediction button
if st.button("🔮 Make Prediction", use_container_width=True, type="primary"):
    with st.spinner("Making prediction..."):
        result = make_prediction(budget, runtime, year, num_reviews, avg_rating)
    
    if result:
        st.subheader("📈 Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Revenue",
                f"${result['prediction']:.2f}M"
            )
        with col2:
            confidence = result.get('confidence', 0)
            st.metric(
                "Confidence",
                f"{confidence*100:.1f}%" if confidence else "N/A"
            )
        with col3:
            st.metric("Year", int(result['input_data']['year']))
        
        st.markdown("**Input Summary:**")
        input_df = pd.DataFrame({
            "Metric": ["Budget", "Runtime", "Rating", "Reviews"],
            "Value": [
                f"${result['input_data']['budget']:,.0f}",
                f"{result['input_data']['runtime']} min",
                f"{result['input_data']['avg_rating']}/10",
                f"{result['input_data']['num_reviews']}"
            ]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)
        
        with st.expander("Raw Response"):
            st.json(result)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("Predict movie box office performance")
    st.markdown(f"API: {API_URL}")
