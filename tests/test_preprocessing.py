"""
Unit tests for data preprocessing functions
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import (
    clean_sales_data,
    clean_review_data,
    handle_missing_values,
    encode_categorical_features,
    normalize_features,
    extract_date_features
)


@pytest.fixture
def sample_sales_data():
    """Create sample sales data for testing"""
    dates = [datetime.now() - timedelta(days=i) for i in range(10)]
    return pd.DataFrame({
        'date': dates,
        'store_id': ['store_1'] * 5 + ['store_2'] * 5,
        'category': ['Electronics', 'Clothing', 'Groceries', 'Home Goods', 'Beauty'] * 2,
        'weather': ['Sunny', 'Rainy', 'Sunny', 'Sunny', 'Rainy'] * 2,
        'promotion': ['Discount', None, 'Seasonal', 'Discount', None] * 2,
        'special_event': [False, False, True, False, True] * 2,
        'dominant_age_group': ['25-34', '55+', '18-24', '55+', '35-44'] * 2,
        'num_customers': [137, 116, 120, 132, 120] * 2,
        'total_sales': [1409.76, 1612.79, 1307.37, 1756.65, 1764.91] * 2,
        'online_sales': [430.08, 1238.71, 168.46, 220.98, 686.24] * 2,
        'in_store_sales': [979.68, 374.07, 1138.92, 1535.66, 1078.67] * 2,
        'avg_transaction': [10.29, 13.9, 10.89, 13.31, 14.71] * 2,
        'return_rate': [0.0453, 0.0407, 0.0793, 0.0442, 0.0512] * 2
    })


@pytest.fixture
def sample_review_data():
    """Create sample review data for testing"""
    return pd.DataFrame({
        'review_id': [f'REV{i}' for i in range(1, 6)],
        'product': ['TechPro X20', 'SmartWatch Pro', 'HomeConnect Hub', 'StudioQuality Earbuds', 'PixelView 7'],
        'category': ['Smartphones', 'Wearables', 'Smart Home', 'Audio', 'Smartphones'],
        'rating': [4, 4, 4, 3, 3],
        'review_text': [
            'The TechPro X20 is amazing! facial recognition works perfectly and the design is outstanding.',
            'The SmartWatch Pro is amazing! app integration works perfectly and the durability is outstanding.',
            'Very impressed with the HomeConnect Hub. Great connectivity and the app interface is exactly what I needed.',
            'The StudioQuality Earbuds is decent. microphone works as expected but the build quality could be better.',
            'An average Smartphones option. The PixelView 7 has good battery life but the camera quality is disappointing.'
        ],
        'feature_mentioned': ['facial recognition', 'app integration', 'app interface', 'microphone', 'battery life'],
        'attribute_mentioned': ['design', 'durability', 'connectivity', 'build quality', 'camera quality'],
        'date': [
            '2023-03-09', '2023-03-09', '2022-12-19', '2022-03-23', '2022-05-17'
        ],
        'sentiment': ['positive', 'positive', 'positive', 'neutral', 'neutral']
    })


def test_clean_sales_data(sample_sales_data):
    """Test cleaning sales data"""
    cleaned_data = clean_sales_data(sample_sales_data)

    # Check that all expected columns are present
    expected_columns = sample_sales_data.columns.tolist()
    assert all(col in cleaned_data.columns for col in expected_columns)

    # Check that date column is datetime
    assert pd.api.types.is_datetime64_dtype(cleaned_data['date'])

    # Check that numeric columns are numeric
    numeric_cols = ['num_customers', 'total_sales', 'online_sales', 'in_store_sales', 'avg_transaction', 'return_rate']
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(cleaned_data[col])

    # Check that no NaN values in required columns
    required_cols = ['date', 'store_id', 'category', 'total_sales']
    for col in required_cols:
        assert cleaned_data[col].isna().sum() == 0


def test_clean_review_data(sample_review_data):
    """Test cleaning review data"""
    cleaned_data = clean_review_data(sample_review_data)

    # Check that all expected columns are present
    expected_columns = sample_review_data.columns.tolist()
    assert all(col in cleaned_data.columns for col in expected_columns)

    # Check that date column is datetime
    assert pd.api.types.is_datetime64_dtype(cleaned_data['date'])

    # Check that rating is numeric
    assert pd.api.types.is_numeric_dtype(cleaned_data['rating'])

    # Check that no NaN values in required columns
    required_cols = ['review_id', 'product', 'rating', 'review_text']
    for col in required_cols:
        assert cleaned_data[col].isna().sum() == 0


def test_handle_missing_values(sample_sales_data):
    """Test handling missing values"""
    # Introduce missing values
    data_with_missing = sample_sales_data.copy()
    data_with_missing.loc[0, 'total_sales'] = np.nan
    data_with_missing.loc[1, 'online_sales'] = np.nan
    data_with_missing.loc[2, 'promotion'] = np.nan

    # Handle missing values
    filled_data = handle_missing_values(data_with_missing)

    # Check that numeric columns have no NaN values
    numeric_cols = ['num_customers', 'total_sales', 'online_sales', 'in_store_sales', 'avg_transaction', 'return_rate']
    for col in numeric_cols:
        assert filled_data[col].isna().sum() == 0

    # Check that categorical columns have no NaN values
    cat_cols = ['store_id', 'category', 'weather', 'promotion', 'dominant_age_group']
    for col in cat_cols:
        assert filled_data[col].isna().sum() == 0


def test_encode_categorical_features(sample_sales_data):
    """Test encoding categorical features"""
    # Encode categorical features
    encoded_data = encode_categorical_features(sample_sales_data, ['store_id', 'category', 'weather'])

    # Check that encoded columns are present
    for col in ['store_id', 'category', 'weather']:
        encoded_cols = [c for c in encoded_data.columns if c.startswith(f"{col}_")]
        assert len(encoded_cols) > 0

    # Check that encoded columns are binary
    for col in encoded_data.columns:
        if col.startswith(('store_id_', 'category_', 'weather_')):
            assert set(encoded_data[col].unique()).issubset({0, 1})


def test_normalize_features(sample_sales_data):
    """Test normalizing features"""
    # Normalize features
    features = ['num_customers', 'total_sales', 'online_sales', 'in_store_sales']
    normalized_data = normalize_features(sample_sales_data, features)

    # Check that normalized columns are present
    for col in features:
        assert f"{col}_normalized" in normalized_data.columns

    # Check that normalized values are between 0 and 1
    for col in features:
        normalized_col = f"{col}_normalized"
        assert normalized_data[normalized_col].min() >= 0
        assert normalized_data[normalized_col].max() <= 1


def test_extract_date_features(sample_sales_data):
    """Test extracting date features"""
    # Extract date features
    date_features = extract_date_features(sample_sales_data, 'date')

    # Check that date features are present
    expected_features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'quarter']
    for feature in expected_features:
        assert f"date_{feature}" in date_features.columns

    # Check that is_weekend is binary
    assert set(date_features['date_is_weekend'].unique()).issubset({0, 1})

    # Check that month is between 1 and 12
    assert date_features['date_month'].min() >= 1
    assert date_features['date_month'].max() <= 12