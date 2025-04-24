"""
Dependencies for FastAPI application
"""
import os
from functools import lru_cache
from typing import Dict, Optional, List

import yaml
from fastapi import Depends, HTTPException, Header, status
from pydantic import BaseSettings, Field

import mlflow
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.cluster import KMeans


class Settings(BaseSettings):
    """Application settings"""
    environment: str = Field("development", env="ENVIRONMENT")
    api_key: str = Field("", env="API_KEY")
    google_api_key: str = Field("", env="GOOGLE_API_KEY")
    huggingface_token: str = Field("", env="HUGGINGFACE_TOKEN")

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    """Get application settings"""
    return Settings()


def get_api_config():
    """Get API configuration from YAML file"""
    config_path = os.path.join("config", "api_config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_config():
    """Get model configuration from YAML file"""
    config_path = os.path.join("config", "model_config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def verify_api_key(api_key: str = Header(None), settings: Settings = Depends(get_settings)):
    """Verify API key if authentication is enabled"""
    api_config = get_api_config()

    # Skip authentication if disabled
    if not api_config["auth"]["enabled"]:
        return True

    if api_key is None or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "API key"},
        )
    return True


@lru_cache(maxsize=5)
def get_forecasting_model(model_type: str = "xgboost"):
    """Get forecasting model from MLflow or local file"""
    model_config = get_model_config()

    try:
        # Try to load from MLflow if available
        mlflow_model_uri = f"models:/forecasting-{model_type}/Production"
        model = mlflow.pyfunc.load_model(mlflow_model_uri)
        return model
    except Exception:
        # Fallback to loading from local file
        if model_type == "xgboost":
            # Create a simple XGBoost model for demonstration
            model = xgb.XGBRegressor(**model_config["forecasting"]["xgboost"]["params"])

            # Check if model file exists
            model_path = os.path.join("models", "forecasting", "xgboost_model.json")
            if os.path.exists(model_path):
                model.load_model(model_path)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                # Train a dummy model on random data
                X = np.random.rand(100, 10)
                y = np.random.rand(100)
                model.fit(X, y)

                # Save the model
                model.save_model(model_path)

            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


@lru_cache(maxsize=5)
def get_segmentation_model():
    """Get customer segmentation model"""
    model_config = get_model_config()

    try:
        # Try to load from MLflow if available
        mlflow_model_uri = "models:/segmentation-kmeans/Production"
        model = mlflow.sklearn.load_model(mlflow_model_uri)
        return model
    except Exception:
        # Fallback to loading from local file or creating a new one
        model_path = os.path.join("models", "segmentation", "kmeans_model.pkl")

        if os.path.exists(model_path):
            import joblib
            model = joblib.load(model_path)
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Create a simple KMeans model
            model = KMeans(**model_config["segmentation"]["kmeans"]["params"])

            # Train on dummy data
            X = np.random.rand(100, 5)
            model.fit(X)

            # Save the model
            import joblib
            joblib.dump(model, model_path)

        return model


@lru_cache(maxsize=2)
def get_sentiment_model():
    """Get sentiment analysis model"""
    from transformers import pipeline

    try:
        # Try to load from Hugging Face
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
        )
        return sentiment_analyzer
    except Exception as e:
        # Log the error and return a simple function for fallback
        print(f"Error loading sentiment model: {e}")

        def simple_sentiment(text):
            """Simple sentiment analysis fallback"""
            positive_words = ["good", "great", "excellent", "amazing", "love", "best", "perfect"]
            negative_words = ["bad", "poor", "terrible", "worst", "hate", "disappointing", "awful"]

            text = text.lower()
            pos_count = sum(word in text for word in positive_words)
            neg_count = sum(word in text for word in negative_words)

            if pos_count > neg_count:
                return [{"label": "POSITIVE", "score": 0.8}]
            elif neg_count > pos_count:
                return [{"label": "NEGATIVE", "score": 0.8}]
            else:
                return [{"label": "NEUTRAL", "score": 0.5}]

        return simple_sentiment


@lru_cache(maxsize=1)
def get_embedding_model():
    """Get text embedding model"""
    model_config = get_model_config()
    model_name = model_config["embeddings"]["sentence_transformer"]["model_name"]

    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        # Log the error and raise
        print(f"Error loading embedding model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load embedding model"
        )


def get_vector_db():
    """Get vector database for RAG"""
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        model_config = get_model_config()
        persist_directory = model_config["embeddings"]["vector_db"]["params"]["persist_directory"]
        collection_name = model_config["embeddings"]["vector_db"]["params"]["collection_name"]

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
        except ValueError:
            # Collection doesn't exist, create it
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        return collection
    except Exception as e:
        # Log the error and raise
        print(f"Error connecting to vector database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to connect to vector database"
        )


def get_rag_model(settings: Settings = Depends(get_settings)):
    """Get RAG model"""
    try:
        import google.generativeai as genai

        # Configure the Gemini API
        genai.configure(api_key=settings.google_api_key)

        # Get the model
        model = genai.GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        # Log the error and raise
        print(f"Error loading RAG model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load RAG model"
        )


def get_retail_data():
    """Get retail sales data"""
    try:
        # Try to load processed data first
        processed_path = os.path.join("data", "processed", "retail_sales_processed.csv")
        if os.path.exists(processed_path):
            return pd.read_csv(processed_path)

        # Fall back to raw data
        raw_path = os.path.join("data", "raw", "retail_sales_data.csv")
        if os.path.exists(raw_path):
            return pd.read_csv(raw_path)

        # If no data is available, raise an exception
        raise FileNotFoundError("Retail sales data not found")
    except Exception as e:
        # Log the error and raise
        print(f"Error loading retail data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load retail data"
        )


def get_review_data():
    """Get product review data"""
    try:
        # Try to load processed data first
        processed_path = os.path.join("data", "processed", "product_reviews_processed.csv")
        if os.path.exists(processed_path):
            return pd.read_csv(processed_path)

        # Fall back to raw data
        raw_path = os.path.join("data", "raw", "product_reviews.csv")
        if os.path.exists(raw_path):
            return pd.read_csv(raw_path)

        # If no data is available, raise an exception
        raise FileNotFoundError("Product review data not found")
    except Exception as e:
        # Log the error and raise
        print(f"Error loading review data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load review data"
        )