"""
Feature engineering module for retail analytics
"""
import os
import logging
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import holidays

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_time_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Create time-based features from date column

    Args:
        df: DataFrame with date column
        date_column: Name of the date column

    Returns:
        DataFrame with additional time features
    """
    logger.info("Creating time features")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure date column is datetime
    result_df[date_column] = pd.to_datetime(result_df[date_column])

    # Extract date components
    result_df['year'] = result_df[date_column].dt.year
    result_df['month'] = result_df[date_column].dt.month
    result_df['day'] = result_df[date_column].dt.day
    result_df['day_of_week'] = result_df[date_column].dt.dayofweek
    result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
    result_df['quarter'] = result_df[date_column].dt.quarter
    result_df['week_of_year'] = result_df[date_column].dt.isocalendar().week

    # Create cyclical features for month, day of week, etc.
    result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
    result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
    result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)

    logger.info(f"Created time features: {result_df.shape}")
    return result_df


def add_holiday_features(df: pd.DataFrame, date_column: str = 'date', country: str = 'US') -> pd.DataFrame:
    """
    Add holiday indicators to the DataFrame

    Args:
        df: DataFrame with date column
        date_column: Name of the date column
        country: Country code for holidays

    Returns:
        DataFrame with holiday features
    """
    logger.info(f"Adding holiday features for {country}")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Ensure date column is datetime
    result_df[date_column] = pd.to_datetime(result_df[date_column])

    # Get holidays for the relevant years
    years = result_df[date_column].dt.year.unique()
    country_holidays = holidays.country_holidays(country, years=years)

    # Create holiday indicator
    result_df['is_holiday'] = result_df[date_column].dt.date.isin(country_holidays).astype(int)

    # Create days before/after holiday features
    result_df['is_day_before_holiday'] = result_df[date_column].dt.date.apply(
        lambda x: (x + pd.Timedelta(days=1)).isin(country_holidays)
    ).astype(int)

    result_df['is_day_after_holiday'] = result_df[date_column].dt.date.apply(
        lambda x: (x - pd.Timedelta(days=1)).isin(country_holidays)
    ).astype(int)

    logger.info(f"Added holiday features: {result_df.shape}")
    return result_df


def create_lag_features(
    df: pd.DataFrame,
    target_column: str,
    group_columns: List[str],
    lag_periods: List[int]
) -> pd.DataFrame:
    """
    Create lag features for time series data

    Args:
        df: DataFrame with time series data
        target_column: Column to create lags for
        group_columns: Columns to group by (e.g., store_id, category)
        lag_periods: List of lag periods to create

    Returns:
        DataFrame with lag features
    """
    logger.info(f"Creating lag features for {target_column}")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Sort by date within groups
    result_df = result_df.sort_values(by=group_columns + ['date'])

    # Create lag features
    for lag in lag_periods:
        lag_name = f"{target_column}_lag_{lag}"
        result_df[lag_name] = result_df.groupby(group_columns)[target_column].shift(lag)

    logger.info(f"Created lag features: {result_df.shape}")
    return result_df


def create_rolling_features(
    df: pd.DataFrame,
    target_column: str,
    group_columns: List[str],
    windows: List[int],
    functions: Dict[str, callable] = {'mean': np.mean, 'std': np.std, 'max': np.max, 'min': np.min}
) -> pd.DataFrame:
    """
    Create rolling window features for time series data

    Args:
        df: DataFrame with time series data
        target_column: Column to create rolling features for
        group_columns: Columns to group by (e.g., store_id, category)
        windows: List of window sizes
        functions: Dictionary of functions to apply to rolling windows

    Returns:
        DataFrame with rolling features
    """
    logger.info(f"Creating rolling features for {target_column}")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Sort by date within groups
    result_df = result_df.sort_values(by=group_columns + ['date'])

    # Create rolling features
    for window in windows:
        for func_name, func in functions.items():
            feature_name = f"{target_column}_roll_{window}_{func_name}"
            result_df[feature_name] = result_df.groupby(group_columns)[target_column].transform(
                lambda x: x.rolling(window, min_periods=1).apply(func)
            )

    logger.info(f"Created rolling features: {result_df.shape}")
    return result_df


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Create interaction features between pairs of columns

    Args:
        df: DataFrame
        feature_pairs: List of column pairs to create interactions for

    Returns:
        DataFrame with interaction features
    """
    logger.info("Creating interaction features")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Create interaction features
    for col1, col2 in feature_pairs:
        if col1 in result_df.columns and col2 in result_df.columns:
            # For numeric columns, create product
            if pd.api.types.is_numeric_dtype(result_df[col1]) and pd.api.types.is_numeric_dtype(result_df[col2]):
                result_df[f"{col1}_{col2}_product"] = result_df[col1] * result_df[col2]

            # For categorical columns, create combined feature
            elif pd.api.types.is_object_dtype(result_df[col1]) or pd.api.types.is_object_dtype(result_df[col2]):
                result_df[f"{col1}_{col2}_combined"] = result_df[col1].astype(str) + "_" + result_df[col2].astype(str)

    logger.info(f"Created interaction features: {result_df.shape}")
    return result_df


def create_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create retail-specific sales features

    Args:
        df: DataFrame with sales data

    Returns:
        DataFrame with additional sales features
    """
    logger.info("Creating sales-specific features")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Calculate online sales percentage
    if 'online_sales' in result_df.columns and 'total_sales' in result_df.columns:
        result_df['online_sales_pct'] = result_df['online_sales'] / result_df['total_sales']

    # Calculate in-store sales percentage
    if 'in_store_sales' in result_df.columns and 'total_sales' in result_df.columns:
        result_df['in_store_sales_pct'] = result_df['in_store_sales'] / result_df['total_sales']

    # Calculate sales per customer
    if 'total_sales' in result_df.columns and 'num_customers' in result_df.columns:
        result_df['sales_per_customer'] = result_df['total_sales'] / result_df['num_customers']

    # Calculate return rate impact
    if 'return_rate' in result_df.columns and 'total_sales' in result_df.columns:
        result_df['return_value'] = result_df['total_sales'] * result_df['return_rate']

    # Calculate net sales (after returns)
    if 'total_sales' in result_df.columns and 'return_value' in result_df.columns:
        result_df['net_sales'] = result_df['total_sales'] - result_df['return_value']

    logger.info(f"Created sales-specific features: {result_df.shape}")
    return result_df


def create_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from product review data

    Args:
        df: DataFrame with review data

    Returns:
        DataFrame with additional review features
    """
    logger.info("Creating review-specific features")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Calculate review length
    if 'review_text' in result_df.columns:
        result_df['review_length'] = result_df['review_text'].str.len()
        result_df['word_count'] = result_df['review_text'].str.split().str.len()

    # Create rating categories
    if 'rating' in result_df.columns:
        result_df['rating_category'] = pd.cut(
            result_df['rating'],
            bins=[0, 2, 3, 5],
            labels=['negative', 'neutral', 'positive']
        )

    # Extract mentioned features if available
    if 'feature_mentioned' in result_df.columns:
        result_df['has_feature_mention'] = result_df['feature_mentioned'].notna().astype(int)

    # Extract mentioned attributes if available
    if 'attribute_mentioned' in result_df.columns:
        result_df['has_attribute_mention'] = result_df['attribute_mentioned'].notna().astype(int)

    logger.info(f"Created review-specific features: {result_df.shape}")
    return result_df


def create_feature_pipeline(
    df: pd.DataFrame,
    data_type: str = 'retail_sales',
    date_column: str = 'date',
    target_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline

    Args:
        df: Input DataFrame
        data_type: Type of data ('retail_sales' or 'product_reviews')
        date_column: Name of the date column
        target_column: Name of the target column (if applicable)

    Returns:
        DataFrame with engineered features
    """
    logger.info(f"Running feature engineering pipeline for {data_type} data")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    if data_type == 'retail_sales':
        # Time features
        result_df = create_time_features(result_df, date_column)

        # Holiday features
        result_df = add_holiday_features(result_df, date_column)

        # Sales-specific features
        result_df = create_sales_features(result_df)

        # Lag and rolling features if target is specified
        if target_column:
            group_columns = ['store_id', 'category']
            result_df = create_lag_features(
                result_df,
                target_column,
                group_columns,
                lag_periods=[1, 7, 14, 28]
            )

            result_df = create_rolling_features(
                result_df,
                target_column,
                group_columns,
                windows=[7, 14, 30],
                functions={'mean': np.mean, 'std': np.std, 'max': np.max}
            )

        # Interaction features
        feature_pairs = [
            ('is_weekend', 'is_holiday'),
            ('month', 'is_holiday'),
            ('weather', 'is_weekend')
        ]
        result_df = create_interaction_features(result_df, feature_pairs)

    elif data_type == 'product_reviews':
        # Review-specific features
        result_df = create_review_features(result_df)

        # Time features if date column exists
        if date_column in result_df.columns:
            result_df = create_time_features(result_df, date_column)

    logger.info(f"Completed feature engineering: {result_df.shape}")
    return result_df


def save_feature_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save feature-engineered data to file

    Args:
        df: Feature-engineered DataFrame
        output_path: Path to save the data
    """
    logger.info(f"Saving feature-engineered data to {output_path}")

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

    logger.info(f"Saved feature-engineered data: {df.shape}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Feature engineering for retail data')
    parser.add_argument('--input', required=True, help='Path to input data file')
    parser.add_argument('--output', required=True, help='Path to save feature-engineered data')
    parser.add_argument('--type', choices=['retail_sales', 'product_reviews'],
                        default='retail_sales', help='Type of data to process')
    parser.add_argument('--target', help='Target column for lag/rolling features')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Run feature engineering
    result_df = create_feature_pipeline(df, args.type, target_column=args.target)

    # Save results
    save_feature_data(result_df, args.output)