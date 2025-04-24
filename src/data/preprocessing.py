"""
Data preprocessing module for retail analytics
"""
import os
import logging
from typing import Tuple

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV or JSON file

    Args:
        file_path: Path to the data file

    Returns:
        DataFrame containing the loaded data
    """
    logger.info(f"Loading data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        # Try reading with explicit comma separator
        try:
            # Attempt standard comma separation first
            return pd.read_csv(file_path, sep=',')
        except Exception as e_comma:
            logger.warning(f"Failed to read {file_path} with comma separator: {e_comma}. Attempting auto-detection.")
            # Fallback to pandas auto-detection if explicit comma fails
            try:
                return pd.read_csv(file_path)
            except Exception as e_auto:
                 logger.error(f"Failed to read CSV {file_path} with auto-detection: {e_auto}")
                 raise e_auto # Re-raise the exception if auto-detect also fails
    elif file_extension == '.json' or file_extension == '.jsonl':
        return pd.read_json(file_path, lines=file_extension == '.jsonl')
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def clean_retail_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean retail sales data

    Args:
        df: Raw retail sales DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning retail sales data")

    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()

    # Convert date to datetime
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

    # Handle missing values
    # For numeric columns, fill with median
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

    # For categorical columns, fill with mode
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'date' and col != 'store_id':  # Skip date and ID columns
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

    # Convert boolean columns
    if 'special_event' in cleaned_df.columns:
        cleaned_df['special_event'] = cleaned_df['special_event'].astype(bool)

    # Handle outliers in sales data
    for col in ['total_sales', 'online_sales', 'in_store_sales', 'avg_transaction']:
        if col in cleaned_df.columns:
            # Cap outliers at 3 standard deviations
            mean, std = cleaned_df[col].mean(), cleaned_df[col].std()
            lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
            cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)

    logger.info(f"Cleaned retail sales data: {cleaned_df.shape}")
    return cleaned_df


def clean_product_reviews_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean product reviews data

    Args:
        df: Raw product reviews DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning product reviews data")

    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()

    # Convert date to datetime if present
    if 'date' in cleaned_df.columns:
        cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

    # Ensure rating is numeric
    if 'rating' in cleaned_df.columns:
        cleaned_df['rating'] = pd.to_numeric(cleaned_df['rating'], errors='coerce')
        # Fill missing ratings with median
        cleaned_df['rating'] = cleaned_df['rating'].fillna(cleaned_df['rating'].median())

    # Clean review text
    if 'review_text' in cleaned_df.columns:
        # Remove rows with empty reviews
        cleaned_df = cleaned_df[cleaned_df['review_text'].notna()]
        cleaned_df = cleaned_df[cleaned_df['review_text'].str.strip() != '']

        # Basic text cleaning
        cleaned_df['review_text'] = cleaned_df['review_text'].str.strip()

    # Ensure required columns exist
    required_cols = ['product', 'category', 'rating', 'review_text']
    for col in required_cols:
        if col not in cleaned_df.columns:
            logger.warning(f"Required column '{col}' not found in product reviews data")

    # Generate product_id based on product name
    if 'product' in cleaned_df.columns and 'product_id' not in cleaned_df.columns:
        logger.info("Generating product_id from product name")
        product_names = cleaned_df['product'].unique()
        product_id_map = {name: f"P{i+1:03d}" for i, name in enumerate(product_names)}
        cleaned_df['product_id'] = cleaned_df['product'].map(product_id_map)
        logger.info("Added 'product_id' column")

    logger.info(f"Cleaned product reviews data: {cleaned_df.shape}")
    return cleaned_df


def split_time_series_data(
    df: pd.DataFrame,
    date_column: str = 'date',
    test_size: float = 0.2,
    validation_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets

    Args:
        df: DataFrame with time series data
        date_column: Name of the date column
        test_size: Proportion of data to use for testing
        validation_size: Proportion of data to use for validation

    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    logger.info("Splitting time series data")

    # Sort by date
    df = df.sort_values(by=date_column)

    # Calculate split points
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - validation_size))

    # Split data
    train = df.iloc[:val_idx].copy()
    validation = df.iloc[val_idx:test_idx].copy()
    test = df.iloc[test_idx:].copy()

    logger.info(f"Split data: train={train.shape}, validation={validation.shape}, test={test.shape}")
    return train, validation, test


def preprocess_text_data(
    df: pd.DataFrame,
    text_column: str = 'review_text',
    min_words: int = 3
) -> pd.DataFrame:
    """
    Preprocess text data for NLP tasks

    Args:
        df: DataFrame with text data
        text_column: Name of the text column
        min_words: Minimum number of words required in text

    Returns:
        DataFrame with preprocessed text
    """
    logger.info(f"Preprocessing text data in column '{text_column}'")

    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Basic text cleaning
    processed_df[text_column] = processed_df[text_column].str.lower()

    # Remove very short reviews
    processed_df['word_count'] = processed_df[text_column].str.split().str.len()
    processed_df = processed_df[processed_df['word_count'] >= min_words]

    # Remove word_count column
    processed_df = processed_df.drop(columns=['word_count'])

    logger.info(f"Preprocessed text data: {processed_df.shape}")
    return processed_df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed data to file

    Args:
        df: Processed DataFrame
        output_path: Path to save the processed data
    """
    logger.info(f"Saving processed data to {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save data
    file_extension = os.path.splitext(output_path)[1].lower()

    if file_extension == '.csv':
        df.to_csv(output_path, index=False, sep=',') # Explicitly set separator
    elif file_extension == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    logger.info(f"Saved processed data: {df.shape}")


def preprocess_pipeline(
    input_path: str,
    output_path: str,
    data_type: str = 'retail_sales'
) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline

    Args:
        input_path: Path to the input data file
        output_path: Path to save the processed data
        data_type: Type of data ('retail_sales' or 'product_reviews')

    Returns:
        Processed DataFrame
    """
    logger.info(f"Running preprocessing pipeline for {data_type} data")

    # Load data
    df = load_data(input_path)

    # Clean data based on type
    if data_type == 'retail_sales':
        cleaned_df = clean_retail_sales_data(df)
    elif data_type == 'product_reviews':
        cleaned_df = clean_product_reviews_data(df)
        cleaned_df = preprocess_text_data(cleaned_df)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Save processed data
    save_processed_data(cleaned_df, output_path)

    return cleaned_df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with handled missing values
    """
    logger.info("Handling missing values")
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # For numeric columns, fill with median
    numeric_cols = processed_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'date' and col != 'store_id':  # Skip date and ID columns
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    
    return processed_df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sales data (alias for clean_retail_sales_data)
    
    Args:
        df: Raw sales DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    return clean_retail_sales_data(df)


def clean_review_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean review data (alias for clean_product_reviews_data)
    
    Args:
        df: Raw review DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    return clean_product_reviews_data(df)


def encode_categorical_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding.
    Args:
        df: Input DataFrame
        columns: List of column names to encode
    Returns:
        DataFrame with encoded features
    """
    logger.info(f"Encoding categorical features: {columns}")
    result_df = df.copy()
    for col in columns:
        if col in result_df.columns:
            dummies = pd.get_dummies(result_df[col], prefix=col)
            result_df = pd.concat([result_df, dummies], axis=1)
            result_df.drop(columns=[col], inplace=True)
    return result_df


def normalize_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Normalize specified features to [0, 1] range."""
    df = df.copy()
    for col in feature_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val == max_val:
            df[f"{col}_normalized"] = 0.0
        else:
            df[f"{col}_normalized"] = (df[col] - min_val) / (max_val - min_val)
    return df


def extract_date_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Extract date-related features from a datetime column
    
    Args:
        df: Input DataFrame
        date_column: Name of the datetime column
        
    Returns:
        DataFrame with additional date features
    """
    logger.info(f"Extracting date features from {date_column}")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract basic date components with date_ prefix
    df['date_year'] = df[date_column].dt.year
    df['date_month'] = df[date_column].dt.month
    df['date_day'] = df[date_column].dt.day
    df['date_day_of_week'] = df[date_column].dt.dayofweek
    df['date_quarter'] = df[date_column].dt.quarter
    
    # Add derived features with date_ prefix
    df['date_is_weekend'] = df['date_day_of_week'].isin([5, 6]).astype(int)
    df['date_is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['date_is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['date_is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['date_is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    
    # Add seasonal features
    df['date_season'] = pd.cut(df['date_month'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'],
                         include_lowest=True)
    
    logger.info(f"Added date features with 'date_' prefix")
    return df


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess retail data')
    parser.add_argument('--input', required=True, help='Path to input data file')
    parser.add_argument('--output', required=True, help='Path to save processed data')
    parser.add_argument('--type', choices=['retail_sales', 'product_reviews'],
                        default='retail_sales', help='Type of data to process')

    args = parser.parse_args()

    preprocess_pipeline(args.input, args.output, args.type)
