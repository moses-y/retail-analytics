# tests/test_models.py

"""
Unit tests for forecasting, segmentation, and sentiment model functions
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import os
import tempfile
import pickle
import xgboost as xgb
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Add the project root to the path if tests are run from the root
# This helps Python find the 'src' module
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Import Functions from Modules ---
from src.models.forecasting import (
    prepare_forecasting_data as prepare_forecast_data, # Alias to avoid name clash
    train_xgboost_model,
    train_lightgbm_model,
    train_prophet_model,
    evaluate_forecast_model,
    save_model as save_forecast_model, # Alias
    load_model as load_forecast_model, # Alias
    generate_forecast,
    plot_feature_importance as plot_forecast_importance, # Alias
    plot_forecast_vs_actual
)
from src.models.segmentation import (
    prepare_segmentation_data,
    find_optimal_clusters,
    train_kmeans_model,
    train_dbscan_model,
    train_hierarchical_model,
    evaluate_clustering,
    analyze_clusters,
    plot_clusters_2d,
    plot_cluster_profiles,
    save_segmentation_model,
    load_segmentation_model,
    assign_segments,
    generate_segment_descriptions
)
from src.models.sentiment import (
    preprocess_text,
    prepare_sentiment_data,
    extract_features as extract_sentiment_features, # Alias
    train_sentiment_model,
    evaluate_sentiment_model,
    plot_confusion_matrix,
    plot_feature_importance as plot_sentiment_importance, # Alias
    save_sentiment_model,
    load_sentiment_model,
    predict_sentiment,
    analyze_feature_sentiment,
    plot_feature_sentiment
)

# --- Fixtures ---

@pytest.fixture(scope="module") # Scope module to create data once per test module run
def sample_sales_data():
    """Create sample sales data for testing forecasting"""
    dates = pd.to_datetime([datetime(2023, 1, 1) + timedelta(days=i) for i in range(60)])
    data = {
        'date': dates,
        'store_id': ['store_1'] * 30 + ['store_2'] * 30,
        'category': (['Electronics', 'Clothing'] * 15) + (['Groceries', 'Home Goods'] * 15),
        'num_customers': np.random.randint(50, 150, 60),
        'promotion': np.random.choice(['None', 'Discount', 'Seasonal'], 60),
        'weather': np.random.choice(['Sunny', 'Rainy'], 60),
        'total_sales': np.random.uniform(500, 2500, 60),
        'day_of_week': dates.dayofweek
    }
    df = pd.DataFrame(data)
    df.loc[5, 'total_sales'] = np.nan
    df.loc[10, 'num_customers'] = np.nan
    return df

@pytest.fixture(scope="module")
def sample_customer_data():
    """Create sample customer data for testing segmentation"""
    # Use sales data aggregated by store_id as proxy for customer data
    # In a real scenario, this would be actual customer-level data
    dates = pd.to_datetime([datetime(2023, 1, 1) + timedelta(days=i) for i in range(60)])
    df = pd.DataFrame({
        'date': dates,
        'store_id': [f'store_{i%10}' for i in range(60)], # 10 stores
        'num_customers': np.random.randint(50, 150, 60),
        'total_sales': np.random.uniform(500, 2500, 60),
        'avg_transaction': np.random.uniform(10, 50, 60),
        'online_ratio': np.random.uniform(0.1, 0.9, 60)
    })
    # Aggregate to store level
    agg_df = df.groupby('store_id').agg(
        total_spend=('total_sales', 'sum'),
        avg_transaction=('avg_transaction', 'mean'),
        purchase_frequency=('store_id', 'size'), # Number of days with sales
        days_since_last_purchase=('date', lambda x: (datetime(2023,3,2) - x.max()).days), # Example end date
        online_ratio=('online_ratio', 'mean')
    ).reset_index()
    return agg_df


@pytest.fixture(scope="module")
def sample_review_data():
    """Create sample review data for testing sentiment"""
    sentiments = ['positive', 'neutral', 'negative']
    data = {
        'review_id': [f'REV{i:04d}' for i in range(100)],
        'product': np.random.choice(['TechPro X20', 'SmartWatch Pro', 'HomeConnect Hub'], 100),
        'rating': np.random.randint(1, 6, 100),
        'review_text': [
            f"This product is {'great' if i % 3 == 0 else 'okay' if i % 3 == 1 else 'terrible'}. " +
            f"The {'battery' if i % 5 == 0 else 'screen'} is {'excellent' if i % 3 == 0 else 'poor'}."
            for i in range(100)
        ],
        'feature_mentioned': np.random.choice(['battery', 'screen', 'camera', 'design'], 100),
        'sentiment': np.random.choice(sentiments, 100) # Explicit sentiment for testing
    }
    return pd.DataFrame(data)


# --- Forecasting Tests ---

@pytest.fixture(scope="module")
def prepared_forecast_data(sample_sales_data):
    """Fixture to provide prepared data for forecasting tests"""
    X_dict, y_dict, feature_cols = prepare_forecast_data(
        sample_sales_data.copy(),
        target_column='total_sales',
        date_column='date',
        group_columns=['store_id', 'category'],
        test_size=0.25
    )
    X_train_combined = pd.concat(X_dict['train'].values())
    y_train_combined = pd.concat(y_dict['train'].values())
    X_test_combined = pd.concat(X_dict['test'].values())
    y_test_combined = pd.concat(y_dict['test'].values())
    # Get dates for plotting/prophet eval
    test_dates = sample_sales_data[sample_sales_data['date'] >= X_test_combined['date'].min()]['date']

    return {
        'X_train': X_train_combined, 'y_train': y_train_combined,
        'X_test': X_test_combined, 'y_test': y_test_combined,
        'features': feature_cols,
        'original_df': sample_sales_data.dropna(subset=['total_sales']),
        'test_dates': test_dates
    }

def test_prepare_forecasting_data(sample_sales_data):
    """Test the forecasting data preparation function"""
    X_dict, y_dict, feature_cols = prepare_forecast_data(
        sample_sales_data.copy(), test_size=0.25
    )
    assert isinstance(X_dict, dict)
    assert isinstance(y_dict, dict)
    assert isinstance(feature_cols, list)
    assert 'train' in X_dict and 'test' in X_dict
    assert 'num_customers' in feature_cols

def test_train_xgboost_model(prepared_forecast_data):
    model = train_xgboost_model(prepared_forecast_data['X_train'], prepared_forecast_data['y_train'])
    assert isinstance(model, xgb.XGBRegressor)
    assert hasattr(model, 'feature_importances_')

def test_evaluate_forecast_model_xgboost(prepared_forecast_data):
    model = train_xgboost_model(prepared_forecast_data['X_train'], prepared_forecast_data['y_train'])
    metrics = evaluate_forecast_model(model, prepared_forecast_data['X_test'], prepared_forecast_data['y_test'], model_type='xgboost')
    assert isinstance(metrics, dict)
    assert 'rmse' in metrics

def test_save_and_load_forecast_xgboost(prepared_forecast_data, tmp_path):
    model = train_xgboost_model(prepared_forecast_data['X_train'], prepared_forecast_data['y_train'])
    model_path = tmp_path / "xgboost_model.json"
    save_forecast_model(model, str(model_path), model_type='xgboost')
    assert model_path.exists()
    loaded_model = load_forecast_model(str(model_path), model_type='xgboost')
    assert isinstance(loaded_model, xgb.XGBRegressor)

def test_generate_forecast_xgboost(prepared_forecast_data):
    model = train_xgboost_model(prepared_forecast_data['X_train'], prepared_forecast_data['y_train'])
    X_future = prepared_forecast_data['X_test'].head(5)
    forecast_result = generate_forecast(model, X_future, model_type='xgboost')
    assert 'forecast' in forecast_result
    assert len(forecast_result['forecast']) == 5

# --- Segmentation Tests ---

@pytest.fixture(scope="module")
def prepared_segment_data(sample_customer_data):
    """Fixture for prepared segmentation data"""
    features_df, ids_df, scaler = prepare_segmentation_data(
        sample_customer_data.copy(),
        id_column='store_id', # Using store_id from fixture
        scale_data=True
    )
    # Define explicit features used in fixture creation
    feature_cols = ['total_spend', 'avg_transaction', 'purchase_frequency', 'days_since_last_purchase', 'online_ratio']
    return {'features': features_df[feature_cols], 'ids': ids_df, 'scaler': scaler, 'original': sample_customer_data}


def test_prepare_segmentation_data(sample_customer_data):
    features_df, ids_df, scaler = prepare_segmentation_data(sample_customer_data.copy(), id_column='store_id')
    assert isinstance(features_df, pd.DataFrame)
    assert isinstance(ids_df, pd.DataFrame)
    assert isinstance(scaler, StandardScaler)
    assert not features_df.isnull().sum().any() # Check NAs handled
    assert 'store_id' not in features_df.columns

def test_train_kmeans_model(prepared_segment_data):
    model = train_kmeans_model(prepared_segment_data['features'], n_clusters=3) # Explicit n_clusters
    assert isinstance(model, KMeans)
    assert model.n_clusters == 3
    assert hasattr(model, 'labels_')

def test_evaluate_clustering(prepared_segment_data):
    model = train_kmeans_model(prepared_segment_data['features'], n_clusters=3)
    metrics = evaluate_clustering(prepared_segment_data['features'], model.labels_)
    assert isinstance(metrics, dict)
    # Silhouette might be low/negative with random data, just check presence
    assert 'silhouette_score' in metrics
    assert 'n_clusters' in metrics

def test_save_and_load_segmentation_model(prepared_segment_data, tmp_path):
    model = train_kmeans_model(prepared_segment_data['features'], n_clusters=3)
    scaler = prepared_segment_data['scaler']
    model_path = tmp_path / "kmeans_model.pkl"
    save_segmentation_model(model, scaler, str(model_path), model_type='kmeans')
    assert model_path.exists()
    loaded_model, loaded_scaler, model_type = load_segmentation_model(str(model_path))
    assert isinstance(loaded_model, KMeans)
    assert isinstance(loaded_scaler, StandardScaler)
    assert model_type == 'kmeans'

def test_assign_segments(prepared_segment_data):
    model = train_kmeans_model(prepared_segment_data['features'], n_clusters=3)
    scaler = prepared_segment_data['scaler']
    feature_cols = prepared_segment_data['features'].columns.tolist()
    # Use original data to test assignment
    segmented_df = assign_segments(prepared_segment_data['original'], model, scaler, feature_cols)
    assert 'segment' in segmented_df.columns
    assert len(segmented_df) == len(prepared_segment_data['original'])
    assert segmented_df['segment'].nunique() <= 3


# --- Sentiment Tests ---

@pytest.fixture(scope="module")
def prepared_sentiment_data(sample_review_data):
    """Fixture for prepared sentiment data"""
    X_train, X_test, y_train, y_test = prepare_sentiment_data(
        sample_review_data.copy(),
        text_column='review_text',
        sentiment_column='sentiment', # Use explicit sentiment
        test_size=0.25
    )
    X_train_feat, X_test_feat, vectorizer = extract_sentiment_features(X_train, X_test)
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_feat': X_train_feat, 'X_test_feat': X_test_feat,
        'vectorizer': vectorizer,
        'original': sample_review_data
    }

def test_prepare_sentiment_data(sample_review_data):
    X_train, X_test, y_train, y_test = prepare_sentiment_data(
        sample_review_data.copy(), text_column='review_text', rating_column='rating' # Test rating conversion
    )
    assert isinstance(X_train, pd.Series)
    assert isinstance(y_train, pd.Series)
    assert len(X_train) > 0
    assert len(y_train) > 0
    assert set(y_train.unique()).issubset({'positive', 'neutral', 'negative'})

def test_extract_sentiment_features(prepared_sentiment_data):
    X_train_feat = prepared_sentiment_data['X_train_feat']
    X_test_feat = prepared_sentiment_data['X_test_feat']
    vectorizer = prepared_sentiment_data['vectorizer']
    assert X_train_feat.shape[0] == len(prepared_sentiment_data['y_train'])
    assert X_test_feat.shape[0] == len(prepared_sentiment_data['y_test'])
    assert isinstance(vectorizer, TfidfVectorizer)

def test_train_sentiment_model(prepared_sentiment_data):
    model = train_sentiment_model(
        prepared_sentiment_data['X_train_feat'],
        prepared_sentiment_data['y_train'],
        model_type='logistic'
    )
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, 'coef_') # Check if fitted

def test_evaluate_sentiment_model(prepared_sentiment_data):
    model = train_sentiment_model(
        prepared_sentiment_data['X_train_feat'],
        prepared_sentiment_data['y_train']
    )
    metrics = evaluate_sentiment_model(
        model,
        prepared_sentiment_data['X_test_feat'],
        prepared_sentiment_data['y_test']
    )
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'classification_report' in metrics
    assert 'confusion_matrix' in metrics

def test_save_and_load_sentiment_model(prepared_sentiment_data, tmp_path):
    model = train_sentiment_model(
        prepared_sentiment_data['X_train_feat'],
        prepared_sentiment_data['y_train']
    )
    vectorizer = prepared_sentiment_data['vectorizer']
    model_path = tmp_path / "sentiment_model.pkl"
    save_sentiment_model(model, vectorizer, str(model_path), model_type='logistic')
    assert model_path.exists()
    loaded_model, loaded_vectorizer, model_type = load_sentiment_model(str(model_path))
    assert isinstance(loaded_model, LogisticRegression)
    assert isinstance(loaded_vectorizer, TfidfVectorizer)
    assert model_type == 'logistic'

def test_predict_sentiment(prepared_sentiment_data):
    model = train_sentiment_model(
        prepared_sentiment_data['X_train_feat'],
        prepared_sentiment_data['y_train']
    )
    vectorizer = prepared_sentiment_data['vectorizer']
    texts_to_predict = ["This is fantastic!", "This is okay.", "This is awful."]
    predictions = predict_sentiment(texts_to_predict, model, vectorizer)
    assert isinstance(predictions, list)
    assert len(predictions) == len(texts_to_predict)
    assert set(predictions).issubset({'positive', 'neutral', 'negative'})

def test_analyze_feature_sentiment(prepared_sentiment_data):
    # Use original data which has feature_mentioned and sentiment
    feature_sentiment_df = analyze_feature_sentiment(
        prepared_sentiment_data['original'],
        feature_column='feature_mentioned',
        sentiment_column='sentiment'
    )
    assert isinstance(feature_sentiment_df, pd.DataFrame)
    assert 'total_mentions' in feature_sentiment_df.columns
    assert 'sentiment_score' in feature_sentiment_df.columns
    assert not feature_sentiment_df.empty

# --- Plotting Tests (Check Execution) ---

# Switch backend for all plotting tests in this module before defining tests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_plot_forecast_importance_runs(prepared_forecast_data, tmp_path):
    # Backend is set above
    model = train_xgboost_model(prepared_forecast_data['X_train'], prepared_forecast_data['y_train'])
    plot_path = tmp_path / "forecast_importance.png"
    try:
        plot_forecast_importance(model, prepared_forecast_data['features'], model_type='xgboost', output_path=str(plot_path))
        assert plot_path.exists()
    except Exception as e:
        pytest.fail(f"plot_forecast_importance raised: {e}")

def test_plot_clusters_2d_runs(prepared_segment_data, tmp_path):
    model = train_kmeans_model(prepared_segment_data['features'], n_clusters=3)
    plot_path = tmp_path / "clusters_2d.png"
    try:
        plot_clusters_2d(prepared_segment_data['features'], model.labels_, output_path=str(plot_path))
        assert plot_path.exists()
    except Exception as e:
        pytest.fail(f"plot_clusters_2d raised: {e}")

def test_plot_sentiment_importance_runs(prepared_sentiment_data, tmp_path):
    model = train_sentiment_model(prepared_sentiment_data['X_train_feat'], prepared_sentiment_data['y_train'])
    vectorizer = prepared_sentiment_data['vectorizer']
    plot_path = tmp_path / "sentiment_importance.png"
    try:
        plot_sentiment_importance(model, vectorizer, output_path=str(plot_path))
        assert plot_path.exists()
    except Exception as e:
        pytest.fail(f"plot_sentiment_importance raised: {e}")
