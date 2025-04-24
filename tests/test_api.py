"""
Unit tests for API functions
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.main import app
from api.models import (
    SalesRequest,
    SalesResponse,
    ReviewRequest,
    ReviewResponse,
    ProductRequest,
    ProductResponse,
    RAGRequest,
    RAGResponse
)


@pytest.fixture
def client():
    """Create a test client for the API"""
    return TestClient(app)


@pytest.fixture
def sample_sales_request():
    """Create a sample sales request"""
    return {
        "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "store_id": "store_1",
        "category": "Electronics",
        "forecast_days": 7
    }


@pytest.fixture
def sample_review_request():
    """Create a sample review request"""
    return {
        "product": "TechPro X20",
        "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "min_rating": 1,
        "sentiment": "all"
    }


@pytest.fixture
def sample_product_request():
    """Create a sample product request"""
    return {
        "product_id": "P001",
        "include_reviews": True,
        "include_sales": True
    }


@pytest.fixture
def sample_rag_request():
    """Create a sample RAG request"""
    return {
        "query": "What are the best features of the TechPro X20?",
        "product_id": "P001"
    }


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sales_forecast_endpoint(client, sample_sales_request):
    """Test sales forecast endpoint"""
    response = client.post("/api/forecast/sales", json=sample_sales_request)

    # Check response status
    assert response.status_code == 200

    # Check response structure
    data = response.json()
    assert "forecast" in data
    assert "metrics" in data
    assert "feature_importance" in data

    # Check forecast data
    assert len(data["forecast"]) == sample_sales_request["forecast_days"]
    assert all("date" in item and "value" in item for item in data["forecast"])

    # Check metrics
    assert "r2" in data["metrics"]
    assert "rmse" in data["metrics"]
    assert "mae" in data["metrics"]

    # Check feature importance
    assert len(data["feature_importance"]) > 0
    assert all("feature" in item and "importance" in item for item in data["feature_importance"])


def test_sales_analysis_endpoint(client, sample_sales_request):
    """Test sales analysis endpoint"""
    response = client.post("/api/analysis/sales", json=sample_sales_request)

    # Check response status
    assert response.status_code == 200

    # Check response structure
    data = response.json()
    assert "total_sales" in data
    assert "sales_by_category" in data
    assert "sales_by_channel" in data
    assert "sales_trend" in data

    # Check sales by category
    assert len(data["sales_by_category"]) > 0
    assert all("category" in item and "value" in item for item in data["sales_by_category"])

    # Check sales by channel
    assert "online" in data["sales_by_channel"]
    assert "in_store" in data["sales_by_channel"]

    # Check sales trend
    assert len(data["sales_trend"]) > 0
    assert all("date" in item and "value" in item for item in data["sales_trend"])


def test_review_analysis_endpoint(client, sample_review_request):
    """Test review analysis endpoint"""
    response = client.post("/api/analysis/reviews", json=sample_review_request)

    # Check response status
    assert response.status_code == 200

    # Check response structure
    data = response.json()
    assert "sentiment_distribution" in data
    assert "feature_sentiment" in data
    assert "top_reviews" in data

    # Check sentiment distribution
    assert "positive" in data["sentiment_distribution"]
    assert "neutral" in data["sentiment_distribution"]
    assert "negative" in data["sentiment_distribution"]

    # Check feature sentiment
    assert len(data["feature_sentiment"]) > 0
    assert all("feature" in item and "sentiment" in item for item in data["feature_sentiment"])

    # Check top reviews
    assert len(data["top_reviews"]) > 0
    assert all("review_id" in item and "text" in item and "sentiment" in item for item in data["top_reviews"])


def test_product_info_endpoint(client, sample_product_request):
    """Test product info endpoint"""
    response = client.post("/api/products/info", json=sample_product_request)

    # Check response status
    assert response.status_code == 200

    # Check response structure
    data = response.json()
    assert "product_id" in data
    assert "name" in data
    assert "category" in data
    assert "average_rating" in data

    # Check reviews if included
    if sample_product_request["include_reviews"]:
        assert "reviews" in data
        assert len(data["reviews"]) > 0

    # Check sales if included
    if sample_product_request["include_sales"]:
        assert "sales" in data
        assert "total" in data["sales"]
        assert "trend" in data["sales"]


def test_rag_query_endpoint(client, sample_rag_request):
    """Test RAG query endpoint"""
    response = client.post("/api/rag/query", json=sample_rag_request)

    # Check response status
    assert response.status_code == 200

    # Check response structure
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "related_products" in data

    # Check answer
    assert len(data["answer"]) > 0

    # Check sources
    assert len(data["sources"]) > 0
    assert all("id" in item and "text" in item for item in data["sources"])

    # Check related products
    assert len(data["related_products"]) > 0


def test_invalid_date_range(client, sample_sales_request):
    """Test invalid date range"""
    # Set end date before start date
    invalid_request = sample_sales_request.copy()
    invalid_request["end_date"] = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    response = client.post("/api/forecast/sales", json=invalid_request)

    # Check response status
    assert response.status_code == 400

    # Check error message
    data = response.json()
    assert "detail" in data


def test_invalid_product_id(client, sample_product_request):
    """Test invalid product ID"""
    # Set invalid product ID
    invalid_request = sample_product_request.copy()
    invalid_request["product_id"] = "nonexistent_product"

    response = client.post("/api/products/info", json=invalid_request)

    # Check response status
    assert response.status_code == 404

    # Check error message
    data = response.json()
    assert "detail" in data