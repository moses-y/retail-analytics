"""
Data preprocessing module for retail analytics
"""
import os
import logging
from typing import Dict, Tuple, Optional, Union, List

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
        return pd.read_csv(file_path)
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
        df.to_csv(output_path, index=False)
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