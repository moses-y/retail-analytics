"""
Pydantic models for API request and response schemas
"""
from datetime import date as _date, datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict


class WeatherCondition(str, Enum):
    """Enum for weather conditions"""
    SUNNY = "Sunny"
    CLOUDY = "Cloudy"
    RAINY = "Rainy"
    SNOWY = "Snowy"
    WINDY = "Windy"


class PromotionType(str, Enum):
    """Enum for promotion types"""
    NONE = "None"
    DISCOUNT = "Discount"
    BOGO = "BOGO"
    CLEARANCE = "Clearance"
    HOLIDAY = "Holiday"
    SEASONAL = "Seasonal"


class AgeGroup(str, Enum):
    """Enum for age groups"""
    GROUP_18_24 = "18-24"
    GROUP_25_34 = "25-34"
    GROUP_35_44 = "35-44"
    GROUP_45_54 = "45-54"
    GROUP_55_PLUS = "55+"


class Category(str, Enum):
    """Enum for product categories"""
    ELECTRONICS = "Electronics"
    CLOTHING = "Clothing"
    GROCERIES = "Groceries"
    HOME_GOODS = "Home Goods"
    BEAUTY = "Beauty"


class Sentiment(str, Enum):
    """Enum for sentiment values"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class SalesDataInput(BaseModel):
    """Input model for sales data"""
    date: _date = Field(..., description="Date of the sales data")
    store_id: str = Field(..., description="Store identifier")
    category: Category = Field(..., description="Product category")
    weather: Optional[WeatherCondition] = Field(None, description="Weather condition")
    promotion: Optional[PromotionType] = Field(None, description="Promotion type")
    special_event: bool = Field(False, description="Whether there was a special event")
    dominant_age_group: Optional[AgeGroup] = Field(None, description="Dominant customer age group")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "date": "2023-06-15",
                "store_id": "store_1",
                "category": "Electronics",
                "weather": "Sunny",
                "promotion": "Discount",
                "special_event": False,
                "dominant_age_group": "25-34"
            }
        }
    )


class SalesForecastRequest(BaseModel):
    """Request model for sales forecasting"""
    data: List[SalesDataInput] = Field(..., description="Sales data for forecasting")
    horizon: int = Field(7, description="Forecast horizon in days", ge=1, le=90)
    store_ids: Optional[List[str]] = Field(None, description="Filter by store IDs")
    categories: Optional[List[Category]] = Field(None, description="Filter by categories")

    @field_validator('horizon')
    @classmethod
    def validate_horizon(cls, v):
        if v < 1 or v > 90:
            raise ValueError("Forecast horizon must be between 1 and 90 days")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": [
                    {
                        "date": "2023-06-15",
                        "store_id": "store_1",
                        "category": "Electronics",
                        "weather": "Sunny",
                        "promotion": "Discount",
                        "special_event": False,
                        "dominant_age_group": "25-34"
                    }
                ],
                "horizon": 7,
                "store_ids": ["store_1", "store_2"],
                "categories": ["Electronics", "Clothing"]
            }
        }
    )


class ForecastPoint(BaseModel):
    """Model for a single forecast point"""
    date: _date = Field(..., description="Forecast date")
    store_id: str = Field(..., description="Store identifier")
    category: Category = Field(..., description="Product category")
    predicted_sales: float = Field(..., description="Predicted sales value")
    lower_bound: Optional[float] = Field(None, description="Lower bound of prediction interval")
    upper_bound: Optional[float] = Field(None, description="Upper bound of prediction interval")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "date": "2023-06-16",
                "store_id": "store_1",
                "category": "Electronics",
                "predicted_sales": 1250.75,
                "lower_bound": 1150.25,
                "upper_bound": 1350.50
            }
        }
    )


class SalesForecastResponse(BaseModel):
    """Response model for sales forecasting"""
    forecasts: List[ForecastPoint] = Field(..., description="List of forecast points")
    model_version: str = Field(..., description="Version of the model used")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of forecast creation")
    metrics: Optional[Dict[str, float]] = Field(None, description="Forecast evaluation metrics")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "forecasts": [
                    {
                        "date": "2023-06-16",
                        "store_id": "store_1",
                        "category": "Electronics",
                        "predicted_sales": 1250.75,
                        "lower_bound": 1150.25,
                        "upper_bound": 1350.50
                    }
                ],
                "model_version": "xgboost-v1.2.3",
                "created_at": "2023-06-15T14:30:45.123Z",
                "metrics": {
                    "rmse": 125.5,
                    "mae": 98.3,
                    "r2": 0.87
                }
            }
        }
    )


class SalesRequest(BaseModel):
    """Request model for sales data retrieval and forecasting"""
    start_date: str = Field(..., description="Start date for sales data (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for sales data (YYYY-MM-DD)")
    store_id: str = Field(..., description="Store identifier")
    category: str = Field(..., description="Product category")
    forecast_days: int = Field(7, description="Number of days to forecast", ge=1, le=90)

    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v, values):
        start = datetime.strptime(values.data.get('start_date', ''), '%Y-%m-%d')
        end = datetime.strptime(v, '%Y-%m-%d')
        if end < start:
            raise ValueError("end_date must be after start_date")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_date": "2023-05-01",
                "end_date": "2023-05-31",
                "store_id": "store_1",
                "category": "Electronics",
                "forecast_days": 7
            }
        }
    )


class SalesResponse(BaseModel):
    """Response model for sales data and forecasts"""
    forecast: List[Dict[str, Any]] = Field(..., description="Forecast data points")
    metrics: Dict[str, float] = Field(..., description="Forecast evaluation metrics")
    feature_importance: List[Dict[str, Any]] = Field(..., description="Feature importance scores")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of response creation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "forecast": [
                    {
                        "date": "2023-06-01",
                        "value": 1500.75
                    }
                ],
                "metrics": {
                    "r2": 0.85,
                    "rmse": 125.5,
                    "mae": 98.3
                },
                "feature_importance": [
                    {
                        "feature": "previous_sales",
                        "importance": 0.4
                    }
                ],
                "created_at": "2023-05-31T14:30:00.123Z"
            }
        }
    )


class ReviewInput(BaseModel):
    """Input model for a product review"""
    review_id: Optional[str] = Field(None, description="Review identifier")
    product: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    rating: int = Field(..., description="Rating (1-5)", ge=1, le=5)
    review_text: str = Field(..., description="Review text content")
    date: Optional[_date] = Field(None, description="Review date")

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError("Rating must be between 1 and 5")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "review_id": "REV12345",
                "product": "TechPro X20",
                "category": "Smartphones",
                "rating": 4,
                "review_text": "The TechPro X20 is amazing! The battery life is excellent and the camera quality is outstanding.",
                "date": "2023-05-20"
            }
        }
    )


class ReviewAnalysisRequest(BaseModel):
    """Request model for review analysis"""
    reviews: List[ReviewInput] = Field(..., description="List of reviews to analyze")
    include_sentiment: bool = Field(True, description="Include sentiment analysis")
    include_features: bool = Field(True, description="Extract mentioned features")
    include_summary: bool = Field(True, description="Generate review summary")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reviews": [
                    {
                        "review_id": "REV12345",
                        "product": "TechPro X20",
                        "category": "Smartphones",
                        "rating": 4,
                        "review_text": "The TechPro X20 is amazing! The battery life is excellent and the camera quality is outstanding.",
                        "date": "2023-05-20"
                    }
                ],
                "include_sentiment": True,
                "include_features": True,
                "include_summary": True
            }
        }
    )


class FeatureSentiment(BaseModel):
    """Model for feature-level sentiment"""
    feature: str = Field(..., description="Product feature")
    sentiment_score: float = Field(..., description="Sentiment score (-1 to 1)")
    sentiment: Sentiment = Field(..., description="Sentiment category")
    mentions: int = Field(1, description="Number of mentions")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature": "battery life",
                "sentiment_score": 0.85,
                "sentiment": "positive",
                "mentions": 3
            }
        }
    )


class ReviewAnalysisResult(BaseModel):
    """Model for a single review analysis result"""
    review_id: str = Field(..., description="Review identifier")
    product: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    rating: int = Field(..., description="Rating (1-5)")
    sentiment_score: Optional[float] = Field(None, description="Overall sentiment score (-1 to 1)")
    sentiment: Optional[Sentiment] = Field(None, description="Overall sentiment category")
    features: Optional[List[FeatureSentiment]] = Field(None, description="Feature-level sentiment")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "review_id": "REV12345",
                "product": "TechPro X20",
                "category": "Smartphones",
                "rating": 4,
                "sentiment_score": 0.75,
                "sentiment": "positive",
                "features": [
                    {
                        "feature": "battery life",
                        "sentiment_score": 0.85,
                        "sentiment": "positive",
                        "mentions": 1
                    },
                    {
                        "feature": "camera quality",
                        "sentiment_score": 0.9,
                        "sentiment": "positive",
                        "mentions": 1
                    }
                ]
            }
        }
    )


class ProductSummary(BaseModel):
    """Model for product review summary"""
    product: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    average_rating: float = Field(..., description="Average rating")
    review_count: int = Field(..., description="Number of reviews")
    sentiment_distribution: Dict[Sentiment, float] = Field(..., description="Distribution of sentiment")
    top_positive_features: List[FeatureSentiment] = Field(..., description="Top positive features")
    top_negative_features: List[FeatureSentiment] = Field(..., description="Top negative features")
    summary: str = Field(..., description="Generated summary of reviews")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product": "TechPro X20",
                "category": "Smartphones",
                "average_rating": 4.2,
                "review_count": 120,
                "sentiment_distribution": {
                    "positive": 0.75,
                    "neutral": 0.15,
                    "negative": 0.1
                },
                "top_positive_features": [
                    {
                        "feature": "battery life",
                        "sentiment_score": 0.85,
                        "sentiment": "positive",
                        "mentions": 45
                    }
                ],
                "top_negative_features": [
                    {
                        "feature": "price",
                        "sentiment_score": -0.6,
                        "sentiment": "negative",
                        "mentions": 30
                    }
                ],
                "summary": "The TechPro X20 is highly regarded for its excellent battery life and camera quality. However, some users find the price to be too high."
            }
        }
    )


class ReviewAnalysisResponse(BaseModel):
    """Response model for review analysis"""
    results: List[ReviewAnalysisResult] = Field(..., description="Analysis results for individual reviews")
    product_summaries: Optional[List[ProductSummary]] = Field(None, description="Product-level summaries")
    model_version: str = Field(..., description="Version of the model used")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of analysis creation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "review_id": "REV12345",
                        "product": "TechPro X20",
                        "category": "Smartphones",
                        "rating": 4,
                        "sentiment_score": 0.75,
                        "sentiment": "positive",
                        "features": [
                            {
                                "feature": "battery life",
                                "sentiment_score": 0.85,
                                "sentiment": "positive",
                                "mentions": 1
                            }
                        ]
                    }
                ],
                "product_summaries": [
                    {
                        "product": "TechPro X20",
                        "category": "Smartphones",
                        "average_rating": 4.2,
                        "review_count": 120,
                        "sentiment_distribution": {
                            "positive": 0.75,
                            "neutral": 0.15,
                            "negative": 0.1
                        },
                        "top_positive_features": [
                            {
                                "feature": "battery life",
                                "sentiment_score": 0.85,
                                "sentiment": "positive",
                                "mentions": 45
                            }
                        ],
                        "top_negative_features": [
                            {
                                "feature": "price",
                                "sentiment_score": -0.6,
                                "sentiment": "negative",
                                "mentions": 30
                            }
                        ],
                        "summary": "The TechPro X20 is highly regarded for its excellent battery life and camera quality. However, some users find the price to be too high."
                    }
                ],
                "model_version": "distilbert-v1.0.0",
                "created_at": "2023-06-15T14:35:12.456Z"
            }
        }
    )


class RAGQuery(BaseModel):
    """Request model for RAG query"""
    query: str = Field(..., description="User query about products")
    products: Optional[List[str]] = Field(None, description="Filter by specific products")
    categories: Optional[List[str]] = Field(None, description="Filter by product categories")
    max_results: int = Field(5, description="Maximum number of results to return", ge=1, le=20)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are the best smartphones with good battery life?",
                "products": ["TechPro X20", "PixelView 7"],
                "categories": ["Smartphones"],
                "max_results": 5
            }
        }
    )


class RAGResponse(BaseModel):
    """Response model for RAG query"""
    answer: str = Field(..., description="Generated answer to the query")
    sources: List[Dict[str, Any]] = Field(..., description="Source reviews used for the answer")
    products_mentioned: List[str] = Field(..., description="Products mentioned in the answer")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of response creation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Based on customer reviews, the TechPro X20 is highly regarded for its excellent battery life, with many users reporting that it lasts a full day with heavy use. The PixelView 7 also has good battery performance but is not as long-lasting as the TechPro X20.",
                "sources": [
                    {
                        "review_id": "REV12345",
                        "product": "TechPro X20",
                        "rating": 4,
                        "review_text": "The TechPro X20 is amazing! The battery life is excellent and lasts all day.",
                        "relevance_score": 0.92
                    }
                ],
                "products_mentioned": ["TechPro X20", "PixelView 7"],
                "created_at": "2023-06-15T14:40:22.789Z"
            }
        }
    )


class ReviewRequest(BaseModel):
    """Request model for retrieving product reviews"""
    product: str = Field(..., description="Product name to get reviews for")
    start_date: str = Field(..., description="Start date for review period (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for review period (YYYY-MM-DD)")
    min_rating: int = Field(1, description="Minimum rating to include", ge=1, le=5)
    sentiment: str = Field("all", description="Filter by sentiment (all/positive/negative/neutral)")

    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v, values):
        start = datetime.strptime(values.data.get('start_date', ''), '%Y-%m-%d')
        end = datetime.strptime(v, '%Y-%m-%d')
        if end < start:
            raise ValueError("end_date must be after start_date")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product": "TechPro X20",
                "start_date": "2023-05-01",
                "end_date": "2023-05-31",
                "min_rating": 1,
                "sentiment": "all"
            }
        }
    )


class ReviewResponse(BaseModel):
    """Response model for product review analysis"""
    sentiment_distribution: Dict[str, float] = Field(..., description="Distribution of sentiments")
    feature_sentiment: List[Dict[str, Any]] = Field(..., description="Sentiment analysis by feature")
    top_reviews: List[Dict[str, Any]] = Field(..., description="Most relevant reviews")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of response creation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sentiment_distribution": {
                    "positive": 0.65,
                    "neutral": 0.25,
                    "negative": 0.10
                },
                "feature_sentiment": [
                    {
                        "feature": "battery life",
                        "sentiment": "positive",
                        "score": 0.85,
                        "mentions": 45
                    }
                ],
                "top_reviews": [
                    {
                        "review_id": "REV12345",
                        "text": "Great battery life!",
                        "sentiment": "positive"
                    }
                ],
                "created_at": "2023-05-31T14:30:00.123Z"
            }
        }
    )


class ProductRequest(BaseModel):
    """Request model for product information"""
    product_id: str = Field(..., description="Product identifier")
    include_reviews: bool = Field(True, description="Include review data")
    include_sales: bool = Field(True, description="Include sales data")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_id": "P001",
                "include_reviews": True,
                "include_sales": True
            }
        }
    )


class ProductResponse(BaseModel):
    """Response model for product information"""
    product_id: str = Field(..., description="Product identifier")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    average_rating: float = Field(..., description="Average product rating")
    reviews: Optional[List[Dict[str, Any]]] = Field(None, description="Product reviews")
    sales: Optional[Dict[str, Any]] = Field(None, description="Sales information")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp of response creation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_id": "P001",
                "name": "TechPro X20",
                "category": "Smartphones",
                "average_rating": 4.2,
                "reviews": [
                    {
                        "review_id": "REV12345",
                        "text": "Great battery life!",
                        "rating": 5
                    }
                ],
                "sales": {
                    "total": 15000,
                    "trend": [
                        {
                            "date": "2023-05-01",
                            "value": 500
                        }
                    ]
                },
                "created_at": "2023-05-31T14:30:00.123Z"
            }
        }
    )


class RAGRequest(BaseModel):
    """Request model for RAG queries"""
    query: str = Field(..., description="User query about products")
    product_id: Optional[str] = Field(None, description="Optional product identifier to focus query on") # Made optional

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are the best features of the TechPro X20?",
                "product_id": "P001"
            }
        }
    )


class ErrorResponse(BaseModel):
    """Model for API error responses"""
    detail: str = Field(..., description="Error detail message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Invalid input: Rating must be between 1 and 5"
            }
        }
    )
