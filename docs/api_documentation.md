### docs/api_documentation.md

# API Documentation

This document provides detailed information about the Retail Analytics API endpoints, request/response formats, and usage examples.

## Base URL

```
http://localhost:8000
```

## Authentication

The API uses API key authentication. Include your API key in the request header:

```
X-API-Key: your_api_key_here
```

## Endpoints

### Health Check

#### GET /api/health

Check if the API is running.

**Response:**

```json
{
  "status": "ok"
}
```

### Sales Forecasting

#### POST /api/forecast/sales

Generate sales forecasts for a specified time period.

**Request:**

```json
{
  "start_date": "2023-01-01",
  "end_date": "2023-03-31",
  "store_id": "store_1",  // "all" for all stores
  "category": "Electronics",  // "all" for all categories
  "forecast_days": 14
}
```

**Response:**

```json
{
  "forecast": [
    {
      "date": "2023-04-01",
      "value": 1245.67
    },
    {
      "date": "2023-04-02",
      "value": 1356.89
    },
    // ...more dates
  ],
  "metrics": {
    "r2": 0.87,
    "rmse": 123.45,
    "mae": 98.76,
    "mape": 5.43
  },
  "feature_importance": [
    {
      "feature": "num_customers",
      "importance": 0.35
    },
    {
      "feature": "day_of_week",
      "importance": 0.25
    },
    // ...more features
  ]
}
```

### Sales Analysis

#### POST /api/analysis/sales

Analyze historical sales data.

**Request:**

```json
{
  "start_date": "2023-01-01",
  "end_date": "2023-03-31",
  "store_id": "store_1",  // "all" for all stores
  "category": "Electronics"  // "all" for all categories
}
```

**Response:**

```json
{
  "total_sales": 125678.90,
  "sales_by_category": [
    {
      "category": "Electronics",
      "value": 45678.90
    },
    {
      "category": "Clothing",
      "value": 35678.90
    },
    // ...more categories
  ],
  "sales_by_channel": {
    "online": 75678.90,
    "in_store": 50000.00
  },
  "sales_trend": [
    {
      "date": "2023-01-01",
      "value": 1234.56
    },
    {
      "date": "2023-01-02",
      "value": 2345.67
    },
    // ...more dates
  ]
}
```

### Customer Segmentation

#### GET /api/segmentation/data

Get customer segmentation data.

**Response:**

```json
[
  {
    "customer_id": "C001",
    "total_spend": 1234.56,
    "avg_transaction": 45.67,
    "purchase_frequency": 27,
    "days_since_last_purchase": 5,
    "online_ratio": 0.75,
    "cluster": 0
  },
  // ...more customers
]
```

#### GET /api/segmentation/profiles

Get segment profiles.

**Response:**

```json
[
  {
    "segment_id": 0,
    "name": "High-Value Customers",
    "description": "Frequent shoppers with high average transaction values",
    "size": 1234,
    "percentage": 25.6,
    "profile": {
      "total_spend": 0.85,
      "avg_transaction": 0.92,
      "purchase_frequency": 0.78,
      "days_since_last_purchase": 0.15,
      "online_ratio": 0.65
    },
    "recommendations": [
      "Offer loyalty rewards",
      "Provide early access to new products",
      "Create personalized promotions"
    ]
  },
  // ...more segments
]
```

### Review Analysis

#### POST /api/analysis/reviews

Analyze product reviews.

**Request:**

```json
{
  "product": "TechPro X20",  // "all" for all products
  "start_date": "2023-01-01",
  "end_date": "2023-03-31",
  "min_rating": 1,
  "sentiment": "all"  // "positive", "neutral", "negative"
}
```

**Response:**

```json
{
  "sentiment_distribution": {
    "positive": 75,
    "neutral": 15,
    "negative": 10
  },
  "feature_sentiment": [
    {
      "feature": "battery life",
      "sentiment": 0.85,
      "mention_count": 45
    },
    {
      "feature": "camera quality",
      "sentiment": 0.92,
      "mention_count": 38
    },
    // ...more features
  ],
  "top_reviews": [
    {
      "review_id": "REV001",
      "text": "The battery life is amazing!",
      "rating": 5,
      "sentiment": "positive",
      "date": "2023-03-15"
    },
    // ...more reviews
  ]
}
```

#### GET /api/reviews/feature-sentiment

Get feature-level sentiment across products.

**Query Parameters:**

- `product` (optional): Filter by product

**Response:**

```json
[
  {
    "product": "TechPro X20",
    "feature": "battery life",
    "sentiment_score": 0.85,
    "mention_count": 45
  },
  {
    "product": "TechPro X20",
    "feature": "camera quality",
    "sentiment_score": 0.92,
    "mention_count": 38
  },
  // ...more product-feature pairs
]
```

#### GET /api/reviews/summary

Get product summaries based on reviews.

**Query Parameters:**

- `product` (optional): Filter by product

**Response:**

```json
[
  {
    "product": "TechPro X20",
    "average_rating": 4.2,
    "review_count": 156,
    "sentiment_counts": {
      "positive": 117,
      "neutral": 23,
      "negative": 16
    },
    "summary": "The TechPro X20 is highly regarded for its exceptional battery life and camera quality. Users particularly appreciate the facial recognition feature and premium design. Some concerns were raised about the price and occasional software glitches.",
    "strengths": [
      "Excellent battery life lasting over 24 hours",
      "High-quality camera performance in various lighting conditions",
      "Fast and accurate facial recognition"
    ],
    "weaknesses": [
      "Higher price point compared to competitors",
      "Occasional software glitches reported by some users",
      "Limited customization options"
    ],
    "feature_sentiment": {
      "battery": 0.92,
      "camera": 0.85,
      "design": 0.78,
      "performance": 0.65,
      "price": -0.25
    }
  },
  // ...more products
]
```

### Product Information

#### GET /api/products

Get a list of all products.

**Response:**

```json
[
  {
    "id": "P001",
    "name": "TechPro X20",
    "category": "Smartphones",
    "average_rating": 4.2,
    "price": 799.99,
    "features": ["5G", "6.7\" AMOLED Display", "Triple Camera", "128GB Storage"],
    "image_url": "https://example.com/images/techpro-x20.jpg"
  },
  // ...more products
]
```

#### POST /api/products/info

Get detailed product information.

**Request:**

```json
{
  "product_id": "P001",
  "include_reviews": true,
  "include_sales": true
}
```

**Response:**

```json
{
  "product_id": "P001",
  "name": "TechPro X20",
  "category": "Smartphones",
  "description": "The latest flagship smartphone with cutting-edge features.",
  "price": 799.99,
  "average_rating": 4.2,
  "features": ["5G", "6.7\" AMOLED Display", "Triple Camera", "128GB Storage"],
  "image_url": "https://example.com/images/techpro-x20.jpg",
  "reviews": [
    {
      "review_id": "REV001",
      "rating": 5,
      "text": "The battery life is amazing!",
      "sentiment": "positive",
      "date": "2023-03-15"
    },
    // ...more reviews
  ],
  "sales": {
    "total": 12345,
    "trend": [
      {
        "date": "2023-01-01",
        "value": 123
      },
      // ...more dates
    ]
  }
}
```

#### POST /api/products/compare

Compare multiple products.

**Request:**

```json
{
  "product_ids": ["P001", "P002", "P003"]
}
```

**Response:**

```json
{
  "features": ["price", "battery life", "camera quality", "screen size", "storage"],
  "comparison": {
    "P001": {
      "price": "799.99",
      "battery life": "24 hours",
      "camera quality": "Excellent",
      "screen size": "6.7\"",
      "storage": "128GB"
    },
    "P002": {
      "price": "699.99",
      "battery life": "20 hours",
      "camera quality": "Good",
      "screen size": "6.4\"",
      "storage": "64GB"
    },
    "P003": {
      "price": "899.99",
      "battery life": "30 hours",
      "camera quality": "Excellent",
      "screen size": "6.9\"",
      "storage": "256GB"
    }
  }
}
```

### RAG (Retrieval Augmented Generation)

#### POST /api/rag/query

Answer product-related questions.

**Request:**

```json
{
  "query": "What are the best features of the TechPro X20?",
  "product_id": "P001"  // Optional
}
```

**Response:**

```json
{
  "answer": "Based on customer reviews, the best features of the TechPro X20 are its exceptional battery life, high-quality camera system, and facial recognition technology. Many users specifically praise the all-day battery performance, the camera's ability to take clear photos in low light, and the fast, reliable facial recognition for unlocking the device.",
  "sources": [
    {
      "id": "REV001",
      "text": "The battery life is amazing! I can go a full day of heavy use without charging.",
      "product": "TechPro X20"
    },
    {
      "id": "REV002",
      "text": "The camera quality is outstanding, especially in low light conditions.",
      "product": "TechPro X20"
    },
    {
      "id": "REV003",
      "text": "Facial recognition works perfectly every time, even in the dark.",
      "product": "TechPro X20"
    }
  ],
  "related_products": ["P002", "P005"]
}
```

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Request succeeded
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

Error responses include a detail message:

```json
{
  "detail": "Invalid date range. End date must be after start date."
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- 100 requests per minute per API key
- 1000 requests per day per API key

When rate limits are exceeded, the API returns a `429 Too Many Requests` status code.

## Versioning

The API version is included in the response headers:

```
X-API-Version: 1.0
```

Future versions will be available at `/v2/api/...`, `/v3/api/...`, etc.


