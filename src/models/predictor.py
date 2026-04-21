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

    # create top 5 director binary feature
    top_5_directors = ['Guy Hamilton', 'George Lucas', 'James Wan', 'James Cameron', 'Mel Brooks']

    df['top_5_director_binary'] = df['director'].apply(
        lambda x: 1 if x in top_5_directors else 0
    )

    # select features and target
    features = [
                   'log_budget',
                   'runtime',
                   'release_quarter',
                   'original_language',
                   'top_5_director_binary'
               ] + core_genres

    target = 'ROI'

    X = df[features]
    y = df[target]

            # --- Filter directors before ranking ---
    # change 5 to another cutoff if you want
    director_summary = (
        df.groupby('director')
        .agg(
            movie_count=('director', 'count'),
            total_budget=('budget', 'sum')
        )
        .query('movie_count >= 5')
        .reset_index()
    )

    # keep only directors who meet the minimum movie threshold
    df_filtered = df.merge(
        director_summary[['director', 'movie_count']],
        on='director',
        how='inner'
    )

    # --- Weighted ROI ranking ---
    # weighted ROI = sum(ROI * budget) / sum(budget)
    top_weighted_roi = (
        df_filtered.groupby('director')
        .apply(lambda x: pd.Series({
            'weighted_ROI': (x['ROI'] * x['budget']).sum() / x['budget'].sum(),
            'avg_ROI': x['ROI'].mean(),
            'movie_count': x['director'].count(),
            'total_budget': x['budget'].sum()
        }))
        .reset_index()
        .sort_values('weighted_ROI', ascending=False)
        .head(5)
    )

    # --- Create Top 5 Director group dynamically ---
    top_5_directors = top_weighted_roi['director'].tolist()

    df['top_5_director_group'] = df['director'].apply(
        lambda x: 'Top 5 Director' if x in top_5_directors else 'Other'
    )

    # optional binary version if needed for modeling
    df['top_5_director_binary'] = df['director'].apply(
        lambda x: 1 if x in top_5_directors else 0
    )

    # --- Top 5 directors by number of movies directed ---
    top_count = (
        df.groupby('director')
        .size()
        .reset_index(name='movie_count')
        .sort_values('movie_count', ascending=False)
        .head(5)
    )

    print("Top 5 Directors by Weighted ROI (minimum 5 movies)")
    print(top_weighted_roi)

    print("\nTop 5 Directors by Number of Movies Directed")
    print(top_count)

    print("\nTop 5 directors used for grouping:")
    print(top_5_directors)

    print("\nPreview of director grouping:")
    print(df[['director', 'top_5_director_group', 'top_5_director_binary']].head())

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
    numeric_features = ['log_budget', 'runtime', 'top_5_director_binary'] + core_genres

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

def rf_model(X_train, y_train, X_test, preprocessor):
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

def predict_movie(budget,runtime,quarter,language,top_5_director,genres):
    core_genres, target, X_train, y_train, X_test = train_data()
    preprocessor = preproccessing_pipeline(core_genres)

    budget_input = int(budget)
    runtime_input = runtime
    release_quarter_input = str(quarter)
    language_input = language.lower()
    top_5_director_binary = top_5_director
    selected_genres = list(genres)

    errors = []

    if budget_input <= 0:
        errors.append("Budget must be greater than 0.")
    if runtime_input <= 0:
        errors.append("Runtime must be greater than 0.")
    if top_5_director > 1:
        errors.append("Top 5 Director is not a binary.")
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
        "top_5_director_binary": [top_5_director_binary],
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
