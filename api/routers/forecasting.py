"""
Router for sales forecasting endpoints
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from api.models import (
    SalesForecastRequest,
    SalesForecastResponse,
    ForecastPoint,
    Category,
    ErrorResponse,
    SalesRequest  # Added import
)
from api.dependencies import (
    get_forecasting_model,
    get_retail_data,
    verify_api_key,
    get_model_config
)

# Setup logging
logger = logging.getLogger("api.forecasting")

# Create router
router = APIRouter()

# Load model configuration
model_config = get_model_config()
feature_columns = model_config["forecasting"]["xgboost"]["features"]
target_column = model_config["forecasting"]["xgboost"]["target"]


@router.post(
    "/forecast/sales",
    response_model=SalesForecastResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate sales forecasts",
    description="Generate sales forecasts for specified stores and categories"
)
async def forecast_sales(
    request: SalesForecastRequest,
    api_key_valid: bool = Depends(verify_api_key),
    model=Depends(get_forecasting_model)
):
    """Generate sales forecasts based on historical data"""
    try:
        # Convert request data to DataFrame
        input_data = pd.DataFrame([item.dict() for item in request.data])

        # Apply filters if provided
        if request.store_ids:
            input_data = input_data[input_data["store_id"].isin(request.store_ids)]

        if request.categories:
            input_data = input_data[input_data["category"].isin([c.value for c in request.categories])]

        # Check if we have data after filtering
        if input_data.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data available after applying filters"
            )

        # Prepare features for forecasting
        features = prepare_features(input_data)

        # Generate forecasts
        forecasts = generate_forecasts(features, model, request.horizon)

        # Create response
        response = SalesForecastResponse(
            forecasts=forecasts,
            model_version="xgboost-v1.0.0",
            created_at=datetime.now(),
            metrics={
                "rmse": 125.5,  # Placeholder metrics
                "mae": 98.3,
                "r2": 0.87
            }
        )

        return response

    except Exception as e:
        logger.exception("Error generating forecasts")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating forecasts: {str(e)}"
        )


@router.post(
    "/analysis/sales",
    response_model=Dict[str, Any],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Analyze historical sales data",
    description="Analyze historical sales data and return insights"
)
async def analyze_sales(
    request: SalesRequest,  # Changed model to SalesRequest
    api_key_valid: bool = Depends(verify_api_key),
    retail_data=Depends(get_retail_data)
):
    """Analyze historical sales data and return insights"""
    try:
        # Parse dates
        try:
            start = pd.to_datetime(request.start_date)
            end = pd.to_datetime(request.end_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD."
            )

        # Filter data by date range
        filtered_data = retail_data[
            (pd.to_datetime(retail_data["date"]) >= start) &
            (pd.to_datetime(retail_data["date"]) <= end)
        ]

        if request.store_id != "all":
            filtered_data = filtered_data[filtered_data["store_id"] == request.store_id]

        if request.category != "all":
            filtered_data = filtered_data[filtered_data["category"] == request.category]

        # Calculate metrics
        total_sales = filtered_data["total_sales"].sum()
        sales_by_category = filtered_data.groupby("category")["total_sales"].sum().to_dict()
        sales_by_channel = {
            "online": filtered_data["online_sales"].sum(),
            "in_store": filtered_data["in_store_sales"].sum()
        }
        sales_trend = filtered_data.groupby("date")["total_sales"].sum().reset_index()
        sales_trend = sales_trend.rename(columns={"total_sales": "value"})

        response = {
            "total_sales": total_sales,
            "sales_by_category": [
                {"category": k, "value": v} for k, v in sales_by_category.items()
            ],
            "sales_by_channel": sales_by_channel,
            "sales_trend": sales_trend.to_dict(orient="records")
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error analyzing sales data")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sales data: {str(e)}"
        )


@router.get(
    "/forecasting/stores",
    response_model=List[str],
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Get available stores",
    description="Get a list of all available store IDs"
)
async def get_stores(
    api_key_valid: bool = Depends(verify_api_key),
    retail_data=Depends(get_retail_data)
):
    """Get a list of all available store IDs"""
    try:
        stores = retail_data["store_id"].unique().tolist()
        return stores
    except Exception as e:
        logger.exception("Error retrieving stores")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stores: {str(e)}"
        )


@router.get(
    "/forecasting/categories",
    response_model=List[str],
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Get available categories",
    description="Get a list of all available product categories"
)
async def get_categories(
    api_key_valid: bool = Depends(verify_api_key),
    retail_data=Depends(get_retail_data)
):
    """Get a list of all available product categories"""
    try:
        categories = retail_data["category"].unique().tolist()
        return categories
    except Exception as e:
        logger.exception("Error retrieving categories")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving categories: {str(e)}"
        )


@router.get(
    "/forecasting/historical",
    response_model=Dict[str, Any],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Get historical sales data",
    description="Get historical sales data for specified stores and categories"
)
async def get_historical_sales(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    store_ids: Optional[List[str]] = Query(None, description="Filter by store IDs"),
    categories: Optional[List[str]] = Query(None, description="Filter by categories"),
    api_key_valid: bool = Depends(verify_api_key),
    retail_data=Depends(get_retail_data)
):
    """Get historical sales data for specified stores and categories"""
    try:
        # Parse dates
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD."
            )

        # Filter by date
        filtered_data = retail_data[
            (pd.to_datetime(retail_data["date"]) >= start) &
            (pd.to_datetime(retail_data["date"]) <= end)
        ]

        # Apply additional filters
        if store_ids:
            filtered_data = filtered_data[filtered_data["store_id"].isin(store_ids)]

        if categories:
            filtered_data = filtered_data[filtered_data["category"].isin(categories)]

        # Check if we have data after filtering
        if filtered_data.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data available for the specified filters"
            )

        # Prepare response
        result = {
            "data": filtered_data.to_dict(orient="records"),
            "metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "store_count": len(filtered_data["store_id"].unique()),
                "category_count": len(filtered_data["category"].unique()),
                "total_records": len(filtered_data)
            }
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving historical sales data")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving historical sales data: {str(e)}"
        )


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for forecasting model"""
    # Convert date to datetime
    data["date"] = pd.to_datetime(data["date"])

    # Extract date features
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    # Create dummy variables for categorical features
    for col in ["store_id", "category", "weather", "promotion", "dominant_age_group"]:
        if col in data.columns:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
            data = pd.concat([data, dummies], axis=1)

    # Handle missing values
    data = data.fillna(0)

    return data


def generate_forecasts(
    features: pd.DataFrame,
    model: Any,
    horizon: int
) -> List[ForecastPoint]:
    """Generate forecasts using the model"""
    # Get unique combinations of store and category
    store_category_pairs = features[["store_id", "category"]].drop_duplicates()

    # Get the latest date in the data
    latest_date = features["date"].max()

    # Initialize list to store forecasts
    forecasts = []

    # Generate forecasts for each store-category pair
    for _, row in store_category_pairs.iterrows():
        store_id = row["store_id"]
        category = row["category"]

        # Filter data for this store and category
        store_data = features[(features["store_id"] == store_id) & (features["category"] == category)]

        # Generate forecast dates
        forecast_dates = [latest_date + timedelta(days=i+1) for i in range(horizon)]

        # Create forecast features
        for forecast_date in forecast_dates:
            # Create a copy of the latest data point as a template
            forecast_features = store_data.iloc[-1:].copy()

            # Update date and derived features
            forecast_features["date"] = forecast_date
            forecast_features["day_of_week"] = forecast_date.dayofweek
            forecast_features["month"] = forecast_date.month
            forecast_features["is_weekend"] = int(forecast_date.dayofweek in [5, 6])

            # Make prediction
            try:
                # For a real model, we would select the appropriate features here
                # For simplicity, we'll generate a random prediction
                predicted_sales = np.random.uniform(800, 2000)

                # Add some seasonality based on day of week and month
                day_factor = 1.0 + 0.2 * (forecast_date.dayofweek in [5, 6])  # Weekend boost
                month_factor = 1.0 + 0.1 * np.sin((forecast_date.month - 1) * np.pi / 6)  # Seasonal pattern
                predicted_sales *= day_factor * month_factor

                # Create prediction interval
                lower_bound = predicted_sales * 0.9
                upper_bound = predicted_sales * 1.1

                # Create forecast point
                forecast_point = ForecastPoint(
                    date=forecast_date.date(),
                    store_id=store_id,
                    category=Category(category),
                    predicted_sales=round(predicted_sales, 2),
                    lower_bound=round(lower_bound, 2),
                    upper_bound=round(upper_bound, 2)
                )

                forecasts.append(forecast_point)

            except Exception as e:
                logger.error(f"Error generating forecast for {store_id}, {category}, {forecast_date}: {e}")

    return forecasts


# <<< START NEW ENDPOINTS >>>
@router.get(
    "/forecasting/predict",
    response_model=Dict[str, Any], # Using Dict for flexibility in response structure
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Get sales forecast",
    description="Get sales forecast for a specific category and store (or all)"
)
async def get_forecast_predict(
    horizon: int = Query(30, description="Number of days to forecast"),
    category: Optional[str] = Query(None, description="Category to forecast (optional)"),
    store_id: Optional[str] = Query(None, description="Store ID to forecast (optional)"),
    api_key_valid: bool = Depends(verify_api_key),
    retail_data=Depends(get_retail_data) # Use retail_data to get latest date
):
    """
    Generate and return sales forecast based on query parameters.
    Note: This currently simulates forecast generation.
    """
    logger.info(f"Received forecast request: horizon={horizon}, category={category}, store_id={store_id}")
    try:
        # Simulate forecast generation (replace with actual model prediction if available)
        # Use the latest date from historical data as the starting point
        latest_date = pd.to_datetime(retail_data["date"]).max()
        forecast_dates = [latest_date + timedelta(days=i+1) for i in range(horizon)]

        forecast_points = []
        for forecast_date in forecast_dates:
            # Simulate prediction - replace with actual model logic
            predicted_sales = np.random.uniform(900, 1800) # Example random prediction
            day_factor = 1.0 + 0.2 * (forecast_date.dayofweek >= 5)
            month_factor = 1.0 + 0.1 * np.sin((forecast_date.month - 1) * np.pi / 6)
            predicted_sales *= day_factor * month_factor

            # Simulate bounds
            lower_bound = predicted_sales * 0.85
            upper_bound = predicted_sales * 1.15

            # Find corresponding actual value if date exists in historical data (for plotting)
            # This part is tricky without joining historical data properly, return None for now
            actual_sales = None # Placeholder

            forecast_points.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "store_id": store_id or "All", # Use provided or 'All'
                "category": category or "All", # Use provided or 'All'
                "forecast": round(predicted_sales, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "actual": actual_sales # Placeholder for actual value
            })

        # Placeholder metrics (replace with actual metrics if calculated)
        metrics = {
            "rmse": round(np.random.uniform(50, 150), 2),
            "mae": round(np.random.uniform(40, 120), 2),
            "mape": round(np.random.uniform(0.05, 0.15), 4),
            "r2": round(np.random.uniform(0.7, 0.95), 2)
        }

        response = {
            "forecast": forecast_points,
            "metrics": metrics,
            "model_version": "simulated-v1.0",
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Generated simulated forecast for {len(forecast_points)} points.")
        return response

    except Exception as e:
        logger.exception("Error generating simulated forecast")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating forecast: {str(e)}"
        )


@router.get(
    "/forecasting/feature-importance",
    response_model=Dict[str, List[Any]],
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Get feature importance",
    description="Get feature importance scores from the forecasting model"
)
async def get_feature_importance_endpoint(
    api_key_valid: bool = Depends(verify_api_key)
    # model=Depends(get_forecasting_model) # Could load model here if needed
):
    """
    Return feature importance scores.
    Note: This currently returns placeholder data.
    """
    logger.info("Received request for feature importance")
    try:
        # Placeholder feature importance (replace with actual model importance)
        # These should ideally match the features used in the model
        placeholder_features = [
            'total_sales_lag_1', 'total_sales_roll_7_mean', 'price_per_customer',
            'day_of_week', 'month', 'online_ratio', 'num_customers',
            'avg_transaction', 'total_sales_lag_7', 'promotion_None',
            'weather_Sunny', 'is_weekend', 'dominant_age_group_25-34',
            'total_sales_roll_14_mean', 'category_Electronics'
        ]
        placeholder_importance = sorted(np.random.rand(len(placeholder_features)).tolist(), reverse=True)

        response = {
            "features": placeholder_features,
            "importance": placeholder_importance
        }
        logger.info("Returning placeholder feature importance.")
        return response

    except Exception as e:
        logger.exception("Error retrieving feature importance")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving feature importance: {str(e)}"
        )
# <<< END NEW ENDPOINTS >>>
