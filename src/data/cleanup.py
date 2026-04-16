"""
Data cleaning script for movie dataset
Removes bad rows and handles data quality issues
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath):
    """Load the CSV file"""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def analyze_data_quality(df):
    """Analyze data quality issues"""
    logger.info("\n=== Data Quality Analysis ===")
    
    # Check missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    print("\nMissing Values:")
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
    
    # Check data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    logger.info(f"Duplicate rows: {duplicates}")
    
    # Check numeric column statistics
    print("\nNumeric Columns Summary:")
    print(df.describe())
    
    return missing, missing_pct


def clean_data(df):
    """
    Clean the dataset by removing bad rows
    """
    initial_rows = len(df)
    logger.info(f"\n=== Starting Data Cleaning ===")
    logger.info(f"Initial rows: {initial_rows}")

    # 1. Remove unnecessary columns (if any)
    df = df.drop(["crew", "homepage", "overview", "production_countries", "keywords", "spoken_languages", "tagline",
                  "vote_count", ], axis=1)
    # 2. Remove complete duplicates
    df = df.drop_duplicates()
    logger.info(f"After removing duplicates: {len(df)} rows (removed {initial_rows - len(df)})")

    # 3. Remove rows with all NaN values
    df = df.dropna(how='all')
    logger.info(f"After removing all-NaN rows: {len(df)} rows")

    # 4. Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 5. Handle missing values in numeric columns
    if numeric_cols:
        # Remove rows where key numeric columns are missing
        key_numeric = [col for col in numeric_cols if col in ['budget', 'revenue', 'runtime', 'year']]
        if key_numeric:
            df = df.dropna(subset=key_numeric, how='any')
            logger.info(f"After removing rows with missing key numeric values: {len(df)} rows")

    # 6. Remove rows with negative values in numeric columns (where not applicable)
    for col in numeric_cols:
        negative_count = (df[col] <= 0).sum()
        if negative_count > 0:
            # Some columns can be negative (e.g., profit), so be selective
            if col in ['budget', 'revenue', 'runtime', 'year', 'votes', 'num_reviews', 'imdb_rating']:
                df = df[df[col] >= 0]
                logger.info(f"Removed {negative_count} rows with negative {col}")

    # 7. Remove rows with invalid year values
    if 'year' in numeric_cols:
        current_year = 2026
        invalid_years = (df['year'] < 1800) | (df['year'] > current_year)
        invalid_count = invalid_years.sum()
        if invalid_count > 0:
            df = df[~invalid_years]
            logger.info(f"Removed {invalid_count} rows with invalid year values")

    # 8. Remove rows with invalid runtime
    if 'runtime' in numeric_cols:
        invalid_runtime = (df['runtime'] <= 0) | (df['runtime'] > 1000)
        invalid_count = invalid_runtime.sum()
        if invalid_count > 0:
            df = df[~invalid_runtime]
            logger.info(f"Removed {invalid_count} rows with invalid runtime (<=0 or >1000)")

    # 9. Remove rows with invalid ratings
    rating_cols = [col for col in df.columns if 'rating' in col.lower()]
    for col in rating_cols:
        if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            invalid_rating = (df[col] < 0) | (df[col] > 10)
            invalid_count = invalid_rating.sum()
            if invalid_count > 0:
                df = df[~invalid_rating]
                logger.info(f"Removed {invalid_count} rows with invalid {col} (not 0-10)")

    # 10. Remove rows with extremely low budget or revenue (likely data entry errors)
    if 'budget' in numeric_cols:
        invalid_budget = df['budget'] < 1000  # Less than $1000 is likely error
        invalid_count = invalid_budget.sum()
        if invalid_count > 0:
            df = df[~invalid_budget]
            logger.info(f"Removed {invalid_count} rows with budget < $1000")

    # 11. Handle missing values in text columns (fill with 'Unknown' if critical)
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        missing_text = df[col].isnull().sum()
        if missing_text > 0:
            if col in ['title', 'genres', 'director', 'actors']:
                # Don't remove rows for these, fill with placeholder
                df[col] = df[col].fillna('Unknown')
                logger.info(f"Filled {missing_text} missing values in '{col}' with 'Unknown'")
            else:
                # For non-critical columns, just log
                logger.info(f"Column '{col}' has {missing_text} missing values")

    # 12. Remove rows with invalid budget/revenue relationships
    if 'budget' in df.columns and 'revenue' in df.columns:
        # Remove rows where revenue is way too high compared to budget (likely errors)
        # Allow reasonable multipliers (e.g., up to 100x for blockbusters)
        valid_ratio = (df['revenue'] <= df['budget'] * 1000) | (df['budget'] == 0)
        invalid_count = (~valid_ratio).sum()
        if invalid_count > 0:
            df = df[valid_ratio]
            logger.info(f"Removed {invalid_count} rows with unrealistic revenue/budget ratio")

    # 13. Remove rows with empty titles
    if 'title' in df.columns:
        empty_titles = (df['title'] == '') | (df['title'] == 'Unknown')
        empty_count = empty_titles.sum()
        if empty_count > 0:
            df = df[~empty_titles]
            logger.info(f"Removed {empty_count} rows with empty/unknown titles")

    final_rows = len(df)
    removed_rows = initial_rows - final_rows
    logger.info(f"\n=== Cleaning Complete ===")
    logger.info(f"Final rows: {final_rows}")
    logger.info(f"Total rows removed: {removed_rows} ({(removed_rows/initial_rows)*100:.2f}%)")

    return df


def save_cleaned_data(df, output_filepath):
    """Save cleaned data to new CSV"""
    try:
        df.to_parquet(output_filepath)
        logger.info(f"✅ Cleaned data saved to: {output_filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving cleaned data: {e}")
        return False


def main():
    """Main execution"""
    # File paths - use absolute paths based on script location
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / 'data' / 'raw' / 'movie_dataset.csv'
    output_file = project_root / 'data' / 'processed' / 'movie_dataset_cleaned.parquet'
    
    # Load data
    logger.info("Starting data cleaning process...")
    df = load_data(str(input_file))
    
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Analyze original data
    analyze_data_quality(df)
    original_df = df.copy()
    
    # Clean data
    df = clean_data(df)
    
    # Save cleaned data
    if save_cleaned_data(df, str(output_file)):
        logger.info(f"✅ Data cleaning successful!")
    else:
        logger.error("❌ Failed to save cleaned data")
        return

if __name__ == "__main__":
    main()
