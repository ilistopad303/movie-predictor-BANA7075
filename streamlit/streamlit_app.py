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

def make_prediction(budget, runtime, quarter, language, top_5_director, genre):
    """Make a prediction via the API"""
    if top_5_director == "True":
        top_5_director = 1 
    else:
        top_5_director = 0
    try:
        payload = {
            "budget": float(budget),
            "runtime": int(runtime),
            "quarter": int(quarter),
            "language": str(language),
            "top_5_director": int(top_5_director),
            "genres": list(genre)
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
    quarter = st.number_input(
        "Release quarter",
        min_value=1,
        max_value=4,
        value=2
    )


with col2:
    language = st.text_input(
        "Language (ISO code)",
        value="en"
    )
    top_5_director = st.selectbox(
    "Is this director in the top 5?",
    ["True", "False"]
    )
    genre = st.multiselect(
        "What Genres",
        ["Drama", "Comedy", "Romance", "Horror", "Thriller", "Crime",
          "Action", "Adventure", "Science Fiction", "Fantasy",
          "Animation", "Documentary"]
    )

# Prediction button
if st.button("🔮 Make Prediction", use_container_width=True, type="primary"):
    with st.spinner("Making prediction..."):
        result = make_prediction(budget, runtime, quarter, language, top_5_director, genre)
    
    if result:
        st.subheader("📈 Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted profit",
                f"${result['pred_profit']:.2f}M"
            )
            st.metric(
                "Predicted ROI",
                f"{result['pred_roi']*100:.1f}%" if result['pred_roi'] is not None else "N/A"
            )
        with col2:
            st.metric(
                "Lower_ROI",
                f"{result['lower_roi']*100:.1f}%"
            )
            st.metric(
                "Upper_ROI",
                f"{result['upper_roi']*100:.1f}%"
            )
        # with col3:
        #     st.metric("Year", int(result['input_data']['year']))
        
        st.markdown("**Input Summary:**")
        input_df = pd.DataFrame({
            "Metric": ["Budget", "Runtime", "Quarter", "Language", "Directors", "Genres"],
            "Value": [
                f"${budget:,.2f}",
                f"{runtime} min",
                f"Q{quarter}",
                language,
                directors,
                ", ".join(genre) if genre else "None"
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
