"""
Unit tests for model functions
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

from src.models.forecasting import (
    train_forecasting_model,
    evaluate_forecasting_model,
    predict_sales
)
from src.models.segmentation import (
    train_segmentation_model,
    evaluate_segmentation_model,
    predict_segment
)
from src.models.sentiment import (
    train_sentiment_model,
    evaluate_sentiment_model,
    predict_sentiment,
    extract_features
)


@pytest.fixture
def sample_sales_data():
    """Create sample sales data for testing"""
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    return pd.DataFrame({
        'date': dates,
        'store_id': ['store_1'] * 15 + ['store_2'] * 15,
        'category': ['Electronics', 'Clothing', 'Groceries', 'Home Goods', 'Beauty'] * 6,
        'weather': ['Sunny', 'Rainy', 'Sunny', 'Sunny', 'Rainy'] * 6,
        'promotion': ['Discount', 'None', 'Seasonal', 'Discount', 'None'] * 6,
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
def sample_customer_data():
    """Create sample customer data for testing"""
    return pd.DataFrame({
        'customer_id': [f'C{i:04d}' for i in range(100)],
        'total_spend': np.random.uniform(100, 5000, 100),
        'avg_transaction': np.random.uniform(10, 100, 100),
        'purchase_frequency': np.random.uniform(1, 20, 100),
        'days_since_last_purchase': np.random.randint(1, 100, 100),
        'online_ratio': np.random.uniform(0, 1, 100),
        'return_rate': np.random.uniform(0, 0.2, 100)
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


def test_train_forecasting_model(sample_sales_data):
    """Test training a forecasting model"""
    # Prepare features and target
    features = ['num_customers', 'promotion', 'special_event', 'weather', 'day_of_week']
    target = 'total_sales'

    # Add date features
    sample_sales_data['day_of_week'] = sample_sales_data['date'].dt.dayofweek
    sample_sales_data['promotion'] = sample_sales_data['promotion'].fillna('None')

    # Train model
    model, feature_importance = train_forecasting_model(sample_sales_data, features, target)

    # Check that model is trained
    assert model is not None

    # Check that feature importance is returned
    assert feature_importance is not None
    assert len(feature_importance) == len(features)


def test_evaluate_forecasting_model(sample_sales_data):
    """Test evaluating a forecasting model"""
    # Prepare features and target
    features = ['num_customers', 'day_of_week']
    target = 'total_sales'

    # Add date features
    sample_sales_data['day_of_week'] = sample_sales_data['date'].dt.dayofweek

    # Split data
    train_data = sample_sales_data.iloc[:20]
    test_data = sample_sales_data.iloc[20:]

    # Train model
    model, _ = train_forecasting_model(train_data, features, target)

    # Evaluate model
    metrics = evaluate_forecasting_model(model, test_data, features, target)

    # Check that metrics are returned
    assert 'r2' in metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'mape' in metrics


def test_predict_sales(sample_sales_data):
    """Test predicting sales"""
    # Prepare features and target
    features = ['num_customers', 'day_of_week']
    target = 'total_sales'

    # Add date features
    sample_sales_data['day_of_week'] = sample_sales_data['date'].dt.dayofweek

    # Train model
    model, _ = train_forecasting_model(sample_sales_data, features, target)

    # Create future data
    future_dates = [datetime.now() + timedelta(days=i) for i in range(7)]
    future_data = pd.DataFrame({
        'date': future_dates,
        'num_customers': np.random.randint(100, 200, 7),
        'day_of_week': [d.dayofweek for d in future_dates]
    })

    # Predict sales
    predictions = predict_sales(model, future_data, features)

    # Check that predictions are returned
    assert len(predictions) == len(future_data)
    assert all(isinstance(p, (int, float)) for p in predictions)


def test_train_segmentation_model(sample_customer_data):
    """Test training a segmentation model"""
    # Prepare features
    features = ['total_spend', 'avg_transaction', 'purchase_frequency', 'days_since_last_purchase', 'online_ratio']

    # Train model
    model, cluster_centers = train_segmentation_model(sample_customer_data, features, n_clusters=3)

    # Check that model is trained
    assert model is not None

    # Check that cluster centers are returned
    assert cluster_centers is not None
    assert cluster_centers.shape == (3, len(features))


def test_evaluate_segmentation_model(sample_customer_data):
    """Test evaluating a segmentation model"""
    # Prepare features
    features = ['total_spend', 'avg_transaction', 'purchase_frequency', 'days_since_last_purchase', 'online_ratio']

    # Train model
    model, _ = train_segmentation_model(sample_customer_data, features, n_clusters=3)

    # Evaluate model
    metrics = evaluate_segmentation_model(model, sample_customer_data, features)

    # Check that metrics are returned
    assert 'silhouette_score' in metrics
    assert 'davies_bouldin_score' in metrics
    assert 'calinski_harabasz_score' in metrics


def test_predict_segment(sample_customer_data):
    """Test predicting customer segment"""
    # Prepare features
    features = ['total_spend', 'avg_transaction', 'purchase_frequency', 'days_since_last_purchase', 'online_ratio']

    # Train model
    model, _ = train_segmentation_model(sample_customer_data, features, n_clusters=3)

    # Predict segment
    segments = predict_segment(model, sample_customer_data, features)

    # Check that segments are returned
    assert len(segments) == len(sample_customer_data)
    assert set(segments).issubset({0, 1, 2})


def test_train_sentiment_model(sample_review_data):
    """Test training a sentiment model"""
    # Train model
    model, vectorizer = train_sentiment_model(sample_review_data, text_column='review_text', target_column='sentiment')

    # Check that model is trained
    assert model is not None
    assert vectorizer is not None

    # Save and load model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'sentiment_model.pkl')
        vectorizer_path = os.path.join(tmpdir, 'vectorizer.pkl')

        # Save model
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

        # Check that files exist
        assert os.path.exists(model_path)
        assert os.path.exists(vectorizer_path)


def test_predict_sentiment(sample_review_data):
    """Test predicting sentiment"""
    # Train model
    model, vectorizer = train_sentiment_model(sample_review_data, text_column='review_text', target_column='sentiment')

    # Predict sentiment
    texts = [
        "This product is amazing! I love it.",
        "It's okay, but could be better.",
        "Terrible product, don't buy it."
    ]

    sentiments = predict_sentiment(model, vectorizer, texts)

    # Check that sentiments are returned
    assert len(sentiments) == len(texts)
    assert set(sentiments).issubset({'positive', 'neutral', 'negative'})