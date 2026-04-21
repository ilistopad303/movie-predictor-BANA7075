from IPython.display import clear_output
import pandas as pd
import numpy as np
# from notebooks.Movie_ROI_Prediction.Movie_ROI_Prediction_Model_Version_2_0 import rf_model
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path

# Error bounds from your model
lower_error = -1.5
upper_error = 1.8

def train_data():
    base_dir = Path(__file__).resolve().parent
    cleaned_data_file = "../../data/processed/movie_dataset_cleaned.parquet"
    # load in dataset
    movie_dataset = pd.read_parquet(base_dir / cleaned_data_file)

    # show column names
    print("\nCOLUMNS:")
    print(movie_dataset.columns.tolist())

    # summary stats for numeric columns
    print("\nDESCRIBE:")
    print(movie_dataset.describe())

    # check missing values count
    print("\nMISSING VALUES:")
    print(movie_dataset.isnull().sum())

    # check duplicate rows
    print("\nDUPLICATE ROWS:", movie_dataset.duplicated().sum())

    # copy dataset and create target
    df = movie_dataset.copy()

    # keep only rows with positive budget
    df = df[df['budget'] > 0]

    # create ROI target
    df['ROI'] = (df['revenue'] - df['budget']) / df['budget']

    # -> log (budget was dominating before)
    df['log_budget'] = np.log(df['budget'])

    # create release quarter
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_quarter'] = df['release_date'].dt.quarter.astype(str)

    # define core genre features
    core_genres = [
        'Drama',
        'Comedy',
        'Romance',
        'Horror',
        'Thriller',
        'Crime',
        'Action',
        'Adventure',
        'Science Fiction',
        'Fantasy',
        'Animation',
        'Documentary'
    ]

    # create binary genre columns
    for genre in core_genres:
        df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)

    # fill in missing director values
    df['director'] = df['director'].fillna('Unknown')

    # frequency encode director
    df['director_count'] = df['director'].map(df['director'].value_counts())

    # select features and target
    features = [
                   'log_budget',
                   'runtime',
                   'release_quarter',
                   'original_language',
                   'director_count'
               ] + core_genres

    target = 'ROI'

    X = df[features]
    y = df[target]

    # train/split test
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # save processed train/test data
    train_processed = X_train.copy()
    train_processed[target] = y_train

    test_processed = X_test.copy()
    test_processed[target] = y_test

    train_processed.to_csv("train_processed_v2.csv", index=False)
    test_processed.to_csv("test_processed_v2.csv", index=False)

    print("Processed train/test files saved.")
    return core_genres, target, X_train, y_train, X_test

def preproccessing_pipeline(core_genres):
    categorical_features = ['release_quarter', 'original_language']
    numeric_features = ['log_budget', 'runtime', 'director_count'] + core_genres

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor

def rf_model(X_train, y_train, preprocessor):
    # Train the model and return fitted model
    from sklearn.ensemble import RandomForestRegressor

    rf_model_obj = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=123
        ))
    ])

    rf_model_obj.fit(X_train, y_train)
    return rf_model_obj

def predict_movie(budget,runtime,quarter,language,directors,genres):
    core_genres, target, X_train, y_train, X_test = train_data()
    preprocessor = preproccessing_pipeline(core_genres)

    budget_input = int(budget)
    runtime_input = runtime
    release_quarter_input = str(quarter)
    language_input = language.lower()
    director_count_input = directors
    selected_genres = list(genres)

    errors = []

    if budget_input <= 0:
        errors.append("Budget must be greater than 0.")
    if runtime_input <= 0:
        errors.append("Runtime must be greater than 0.")
    if director_count_input < 0:
        errors.append("Director count must be 0 or greater.")
    if language_input == "":
        errors.append("Language cannot be blank.")
    if len(selected_genres) == 0:
        errors.append("Select at least one genre.")

    if errors:
        for err in errors:
            print(err)
        return

    # Create genre data for ALL core genres, not just selected ones
    genre_data = {genre: [1 if genre in selected_genres else 0] for genre in core_genres}

    new_movie = pd.DataFrame({
        "log_budget": [np.log(budget_input)],
        "runtime": [runtime_input],
        "release_quarter": [release_quarter_input],
        "original_language": [language_input],
        "director_count": [director_count_input],
        **genre_data
    })

    pred_roi = rf_model(X_train, y_train, X_test, preprocessor).predict(new_movie)[0]
    pred_revenue = budget_input * (1 + pred_roi)
    pred_profit = pred_revenue - budget_input

    lower_roi = max(-1, pred_roi + lower_error)
    upper_roi = pred_roi + upper_error

    lower_revenue = max(0, budget_input * (1 + lower_roi))
    upper_revenue = max(0, budget_input * (1 + upper_roi))
    return pred_roi, pred_revenue, pred_profit, lower_roi, upper_roi, lower_revenue, upper_revenue
