"""
Integration tests for data processing and model pipeline
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import os
import tempfile

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import clean_sales_data, clean_review_data
from src.data.feature_engineering import (
    create_sales_features,
    create_customer_features,
    create_review_features
)
from src.models.forecasting import train_forecasting_model, predict_sales
from src.models.segmentation import train_segmentation_model, predict_segment
from src.models.sentiment import train_sentiment_model, predict_sentiment


@pytest.fixture
def sample_sales_data():
    """Create sample sales data for testing"""
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    return pd.DataFrame({
        'date': dates,
        'store_id': ['store_1'] * 15 + ['store_2'] * 15,
        'category': ['Electronics', 'Clothing', 'Groceries', 'Home Goods', 'Beauty'] * 6,
        'weather': ['Sunny', 'Rainy', 'Sunny', 'Sunny', 'Rainy'] * 6,
        'promotion': ['Discount', None, 'Seasonal', 'Discount', None] * 6,
        'special_event': [False, False, True, False, True] * 6,
        'dominant_age_group': ['25-34', '55+', '18-24', '55+', '35-44'] * 6,
        'num_customers': np.random.randint(100, 200, 30),
        'total_sales': np.random.uniform(1000, 2000, 30),
        'online_sales': np.random.uniform(100, 500, 30),
        'in_store_sales': np.random.uniform(500, 1500, 30),
        'avg_transaction': np.random.uniform(10, 20, 30),
        'return_rate': np.random.uniform(0.01, 0.1, 30)
    })


@pytest.fixture
def sample_review_data():
    """Create sample review data for testing"""
    sentiments = ['positive', 'neutral', 'negative']
    return pd.DataFrame({
        'review_id': [f'REV{i:04d}' for i in range(100)],
        'product': np.random.choice(['TechPro X20', 'SmartWatch Pro', 'HomeConnect Hub', 'StudioQuality Earbuds', 'PixelView 7'], 100),
        'category': np.random.choice(['Smartphones', 'Wearables', 'Smart Home', 'Audio'], 100),
        'rating': np.random.randint(1, 6, 100),
        'review_text': [
            f"This product is {'great' if i % 3 == 0 else 'okay' if i % 3 == 1 else 'terrible'}. " +
            f"The {'battery' if i % 5 == 0 else 'screen' if i % 5 == 1 else 'camera' if i % 5 == 2 else 'design' if i % 5 == 3 else 'performance'} " +
            f"is {'excellent' if i % 3 == 0 else 'decent' if i % 3 == 1 else 'poor'}."
            for i in range(100)
        ],
        'feature_mentioned': np.random.choice(['battery', 'screen', 'camera', 'design', 'performance'], 100),
        'attribute_mentioned': np.random.choice(['quality', 'durability', 'speed', 'ease of use', 'value'], 100),
        'date': [datetime.now() - timedelta(days=i) for i in range(100)],
        'sentiment': np.random.choice(sentiments, 100)
    })


def test_sales_forecasting_pipeline(sample_sales_data):
    """Test the complete sales forecasting pipeline"""
    # Step 1: Clean data
    cleaned_data = clean_sales_data(sample_sales_data)

    # Step 2: Create features
    features_data = create_sales_features(cleaned_data)

    # Step 3: Split data
    train_data = features_data.iloc[:20]
    test_data = features_data.iloc[20:]

    # Step 4: Define features and target
    feature_cols = [col for col in features_data.columns if col.startswith(('date_', 'store_id_', 'category_', 'weather_'))]
    feature_cols += ['num_customers', 'promotion_encoded', 'special_event']
    target = 'total_sales'

    # Step 5: Train model
    model, feature_importance = train_forecasting_model(train_data, feature_cols, target)

    # Step 6: Make predictions
    predictions = predict_sales(model, test_data, feature_cols)

    # Check that predictions are returned
    assert len(predictions) == len(test_data)
    assert all(isinstance(p, (int, float)) for p in predictions)

    # Step 7: Save and load model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'forecasting_model.pkl')

        # Save model
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Check that file exists
        assert os.path.exists(model_path)

        # Load model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        # Make predictions with loaded model
        loaded_predictions = predict_sales(loaded_model, test_data, feature_cols)

        # Check that predictions are the same
        assert np.allclose(predictions, loaded_predictions)


def test_customer_segmentation_pipeline(sample_sales_data):
    """Test the complete customer segmentation pipeline"""
    # Step 1: Clean data
    cleaned_data = clean_sales_data(sample_sales_data)

    # Step 2: Create customer features
    customer_data = create_customer_features(cleaned_data)

    # Step 3: Define features
    feature_cols = ['total_spend', 'avg_transaction', 'purchase_frequency', 'days_since_last_purchase', 'online_ratio']

    # Step 4: Train model
    model, cluster_centers = train_segmentation_model(customer_data, feature_cols, n_clusters=3)

    # Step 5: Make predictions
    segments = predict_segment(model, customer_data, feature_cols)

    # Check that segments are returned
    assert len(segments) == len(customer_data)
    assert set(segments).issubset({0, 1, 2})

    # Step 6: Add segments to data
    customer_data['segment'] = segments

    # Check that segments are added
    assert 'segment' in customer_data.columns
    assert customer_data['segment'].nunique() <= 3


def test_sentiment_analysis_pipeline(sample_review_data):
    """Test the complete sentiment analysis pipeline"""
    # Step 1: Clean data
    cleaned_data = clean_review_data(sample_review_data)

    # Step 2: Create review features
    review_features = create_review_features(cleaned_data)

    # Step 3: Split data
    train_data = review_features.iloc[:70]
    test_data = review_features.iloc[70:]

    # Step 4: Train model
    model, vectorizer = train_sentiment_model(train_data, text_column='review_text', target_column='sentiment')

    # Step 5: Make predictions
    test_texts = test_data['review_text'].tolist()
    predictions = predict_sentiment(model, vectorizer, test_texts)

    # Check that predictions are returned
    assert len(predictions) == len(test_data)
    assert set(predictions).issubset({'positive', 'neutral', 'negative'})

    # Step 6: Evaluate predictions
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_data['sentiment'], predictions)

    # Check that accuracy is reasonable
    assert accuracy > 0.5  # This is a very basic check