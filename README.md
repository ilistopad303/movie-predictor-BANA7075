# Predicting Movie ROI with Machine Learning   

---

## Team Members  
- Jay Weil  
- Drew Cobble  
- M. Zafrul Hossain  
- Will Morrison  
- Ian Listopad 

## Overview  
The film industry operates under a high level of uncertainty when deciding which projects to fund. Studios often invest millions of dollars into production and marketing without a clear understanding of expected financial performance.  

This project develops a **machine learning system to predict movie return on investment (ROI)** using pre-release features such as budget, runtime, genre, release timing, and director history. The goal is to provide a **data-driven tool** that helps stakeholders evaluate potential projects and reduce financial risk.

## Problem Statement  

Movie studios and investors are required to make high-cost decisions with limited predictive insight. Movies often require significant upfront investment, yet there is no reliable way to estimate financial performance before release. As a result, decisions around budgeting, casting, and release strategy are often based on intuition or limited analysis.  

Movie success is influenced by many factors such as budget, genre, runtime, and release timing. However, these factors interact in complex and non-linear ways, making them difficult to model using traditional statistical methods.  

This project addresses that gap by using machine learning to:
- Predict **return on investment (ROI)** using pre-release features  
- Estimate **revenue and profit** outcomes  
- Provide a **range of possible outcomes** to reflect uncertainty  

The goal is to support more structured, data-driven decision-making and reduce financial risk in the film investment process.

## Why Machine Learning?  

Movie performance is influenced by many interacting variables such as budget, genre, timing, and production characteristics. These relationships are often non-linear and difficult to capture using traditional modeling approaches.  

Machine learning is well-suited for this problem because it can:
- Capture **non-linear relationships** between variables  
- Incorporate **multiple features simultaneously**  
- Identify **hidden patterns** in historical data  
- Improve performance as more data becomes available  

By leveraging machine learning, this project is able to move beyond simple assumptions and provide more accurate and flexible predictions for movie ROI and financial outcomes.

## How to Run the Project  

### 1. Clone the Repository  
```bash
git clone https://github.com/ilistopad303/movie-predictor-BANA7075.git
cd movie-predictor-BANA7075
```

---

### 2. Install Dependencies  
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

---

### 3. Run the FastAPI Backend  

Start the API server:

```bash
code idk IAN help
```



### 4. Run the Streamlit Frontend  

In a separate terminal:

```bash
streamlit run streamlit_app.py
```

This will launch the user interface in your browser.

---

### 5. Using the Application  

- Enter movie features such as:
  - Budget  
  - Runtime  
  - Year  
  - Reviews / Rating  
- Submit the form to receive:
  - Predicted ROI  
  - Estimated performance  
  - Confidence output  

---




## Repository Structure  

```
├── .dvc/                     # Data version control configuration  
├── .streamlit/              # Streamlit app configuration  
├── data/                    # Raw and processed datasets  
├── models/                  # Saved machine learning models  
├── src/                     # Data processing / pipeline scripts  
│   └── data/                # Data transformation logic  
├── app.py                   # FastAPI backend (API endpoints)  
├── streamlit_app.py         # Streamlit frontend (user interface)  
├── Movie_ROI_Prediction_Model_Version_2_0.ipynb  # Main modeling notebook  
├── requirements.txt         # Python dependencies  
├── run.sh                   # Script to run application components  
├── QUICKSTART.md            # Quick start instructions  
├── SETUP_COMPLETE.md        # Setup documentation  
├── README.md                # Project overview and documentation  
├── .env.example             # Environment variable template  
├── .gitignore               # Git ignore rules  
```

---

## Project Components  

- **FastAPI Backend (`app.py`)**  
  Handles prediction requests and serves model outputs  

- **Streamlit Frontend (`streamlit_app.py`)**  
  Provides an interactive interface for users  

- **Model (`models/`)**  
  Stores trained machine learning models  

- **Data (`data/`)**  
  Contains datasets used for training and evaluation  

- **Pipeline (`src/`)**  
  Includes data processing and transformation logic  


## System Architecture  

The system is designed as a modular machine learning pipeline using Python. It consists of the following key components:

### 1. Data Layer  
- Input dataset: `movie_dataset_cleaned.csv`  
- Contains film-level features such as budget, revenue, genres, runtime, and release date  

### 2. Feature Engineering Layer  
- ROI calculation:
- Log transformation of budget (`log_budget`) to reduce skew  
- Extraction of release quarter from release date  
- Binary encoding of genres  
- Frequency encoding of directors (`director_count`)  

### 3. Preprocessing Pipeline  
- Numerical features:
- Median imputation for missing values  
- Standard scaling  
- Categorical features:
- One-hot encoding for variables like release quarter and language  

### 4. Modeling Layer  
- Primary model: **Random Forest Regressor**  
- Benchmark model: **LASSO Regression**  

### 5. Prediction Layer  
- Generates predicted ROI  
- Calculates estimated revenue and profit  
- Produces a prediction range using model residuals

## ML Pipeline  

### Data Ingestion  
- Data loaded from `movie_dataset_cleaned.csv`  
- Initial checks performed:
  - Missing values  
  - Duplicate records  
  - Summary statistics for numeric features  

---

### Data Processing & Feature Engineering  
Key transformations include:
- Filtering out invalid rows (budget > 0)  
- Creating ROI target variable  
- Log transformation of budget to reduce skew  
- Extracting release quarter from release date  
- Encoding genres as binary features  
- Frequency encoding directors (`director_count`)  

---

### Data Splitting & Versioning  
- Train/Test split: **80/20**  
- Processed datasets saved for reproducibility:
  - `train_processed_v2.csv`  
  - `test_processed_v2.csv`  

---

### Preprocessing Pipeline  
Implemented using **ColumnTransformer + Pipeline**:
- Numerical features:
  - Median imputation  
  - Standard scaling  
- Categorical features:
  - One-hot encoding  

This ensures consistency between training and prediction environments.

---

## Model Performance  

### Models Used  
- **Random Forest Regressor (Primary Model)**
  - n_estimators = 200  
  - max_depth = 10  

- **LASSO Regression (Benchmark Model)**
  - Alpha selected using cross-validation  

---

### Results  

| Model           | MAE  | RMSE  | R²    |
|----------------|------|------|------|
| Random Forest  | 4.51 | 16.76 | 0.271 |
| LASSO          | 6.92 | 19.04 | 0.060 |

The Random Forest model outperformed LASSO across all evaluation metrics, achieving lower error and higher explanatory power.

---

### Feature Importance  
- Log-transformed budget is the strongest predictor  
- Genre variables contribute meaningful signal  
- Director frequency provides additional predictive value  

---

##  Experiment Tracking  

Experiments are logged in:
- `experiment_log_v2.csv`  

Tracked information includes:
- Model type  
- Hyperparameters  
- Train/test split  
- Performance metrics (MAE, RMSE, R²)

## ROI Calculator (Prediction Tool)  

The project includes a custom ROI calculator that allows users to input key movie features and receive predictions.

### Inputs:
- Budget  
- Runtime  
- Release quarter  
- Original language  
- Director experience (frequency count)  
- Genre selection  

### Outputs:
- Predicted ROI  
- Estimated revenue  
- Estimated profit  
- Revenue range (based on model residuals)  
- Profitability classification (Profitable / Not Profitable)  

This tool provides a simple way to interact with the model and simulate real-world decision scenarios.

---

## User Interface  

An interactive interface was built using **ipywidgets**:
- Input fields for movie attributes  
- Multi-select genre input  
- Button-triggered predictions  
- Real-time output display  

This serves as a prototype for a production-level application.

---

## Deployment Strategy  

### Current (MVP)
- Model is deployed locally within a Python environment  
- Saved using `joblib`:
  - `random_forest_roi_v2.pkl`  
- Predictions generated through scripts and interactive widgets  

---

### Future Deployment  
In a real-world setting, this system could be deployed using:
- **FastAPI + Docker** for serving predictions as an API  
- **Cloud platforms** such as AWS, Azure, or GCP  
- **CI/CD pipelines** for automated updates and retraining  
- **Monitoring systems** to track model performance and drift  

---

## Challenges  

- Movie success is inherently unpredictable  
- Limited availability of key predictive features (e.g., marketing spend, competition)  
- Data integration across multiple sources  
- Difficulty deploying a full web application in the current environment  

---

## Future Improvements  

- Incorporate additional features:
  - Marketing budget  
  - Social media sentiment  
  - Competitive releases  
- Improve model performance:
  - Gradient Boosting / XGBoost  
  - Ensemble methods  
- Enhance deployment:
  - Fully functional web application (Streamlit or FastAPI)  
- Implement:
  - Model monitoring  
  - Automated retraining pipelines  

---


