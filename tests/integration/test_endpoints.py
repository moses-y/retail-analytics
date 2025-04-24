"""
Integration tests for API endpoints
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


@pytest.fixture
def client():
    """Create a test client for the API"""
    return TestClient(app)


def test_complete_sales_workflow(client):
    """Test a complete sales analysis and forecasting workflow"""
    # Step 1: Get sales data
    sales_request = {
        "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "store_id": "all",
        "category": "all"
    }

    response = client.post("/api/analysis/sales", json=sales_request)
    assert response.status_code == 200
    sales_data = response.json()

    # Step 2: Get sales forecast
    forecast_request = {
        "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "store_id": "all",
        "category": "all",
        "forecast_days": 7
    }

    response = client.post("/api/forecast/sales", json=forecast_request)
    assert response.status_code == 200
    forecast_data = response.json()

    # Step 3: Get sales by store
    for store_id in ["store_1", "store_2"]:
        store_request = sales_request.copy()
        store_request["store_id"] = store_id

        response = client.post("/api/analysis/sales", json=store_request)
        assert response.status_code == 200
        store_data = response.json()

        # Check that store data is returned
        assert store_data["total_sales"] > 0

    # Step 4: Get sales by category
    for category in ["Electronics", "Clothing", "Groceries"]:
        category_request = sales_request.copy()
        category_request["category"] = category

        response = client.post("/api/analysis/sales", json=category_request)
        assert response.status_code == 200
        category_data = response.json()

        # Check that category data is returned
        assert category_data["total_sales"] > 0


def test_complete_review_workflow(client):
    """Test a complete review analysis workflow"""
    # Step 1: Get review data
    review_request = {
        "product": "all",
        "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "min_rating": 1,
        "sentiment": "all"
    }

    response = client.post("/api/analysis/reviews", json=review_request)
    assert response.status_code == 200
    review_data = response.json()

    # Step 2: Get reviews by product
    for product in ["TechPro X20", "SmartWatch Pro"]:
        product_request = review_request.copy()
        product_request["product"] = product

        response = client.post("/api/analysis/reviews", json=product_request)
        assert response.status_code == 200
        product_data = response.json()

        # Check that product data is returned
        assert sum(product_data["sentiment_distribution"].values()) > 0

    # Step 3: Get reviews by sentiment
    for sentiment in ["positive", "neutral", "negative"]:
        sentiment_request = review_request.copy()
        sentiment_request["sentiment"] = sentiment

        response = client.post("/api/analysis/reviews", json=sentiment_request)
        assert response.status_code == 200
        sentiment_data = response.json()

        # Check that sentiment data is returned
        assert sentiment_data["sentiment_distribution"][sentiment] > 0

    # Step 4: Get feature sentiment
    response = client.get("/api/reviews/feature-sentiment")
    assert response.status_code == 200
    feature_data = response.json()

    # Check that feature data is returned
    assert len(feature_data) > 0


def test_complete_product_workflow(client):
    """Test a complete product info workflow"""
    # Step 1: Get all products
    response = client.get("/api/products")
    assert response.status_code == 200
    products = response.json()

    # Check that products are returned
    assert len(products) > 0

    # Step 2: Get product info
    for product in products:
        product_id = product["id"]

        product_request = {
            "product_id": product_id,
            "include_reviews": True,
            "include_sales": True
        }

        response = client.post("/api/products/info", json=product_request)
        assert response.status_code == 200
        product_data = response.json()

        # Check that product data is returned
        assert product_data["product_id"] == product_id
        assert "reviews" in product_data
        assert "sales" in product_data

    # Step 3: Compare products
    if len(products) >= 2:
        product_ids = [p["id"] for p in products[:2]]

        compare_request = {
            "product_ids": product_ids
        }

        response = client.post("/api/products/compare", json=compare_request)
        assert response.status_code == 200
        compare_data = response.json()

        # Check that comparison data is returned
        assert "features" in compare_data
        assert "comparison" in compare_data
        assert len(compare_data["comparison"]) == len(product_ids)


def test_complete_rag_workflow(client):
    """Test a complete RAG workflow"""
    # Step 1: Get all products
    response = client.get("/api/products")
    assert response.status_code == 200
    products = response.json()

    # Check that products are returned
    assert len(products) > 0

    # Step 2: Query about a product
    product = products[0]
    product_id = product["id"]

    rag_request = {
        "query": f"What are the best features of the {product['name']}?",
        "product_id": product_id
    }

    response = client.post("/api/rag/query", json=rag_request)
    assert response.status_code == 200
    rag_data = response.json()

    # Check that RAG data is returned
    assert "answer" in rag_data
    assert "sources" in rag_data
    assert "related_products" in rag_data

    # Step 3: Query without specifying a product
    general_request = {
        "query": "Which product has the best battery life?"
    }

    response = client.post("/api/rag/query", json=general_request)
    assert response.status_code == 200
    general_data = response.json()

    # Check that general data is returned
    assert "answer" in general_data
    assert "sources" in general_data
    assert "related_products" in general_data