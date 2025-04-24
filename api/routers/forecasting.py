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
    ErrorResponse
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
    "/forecasting/predict",
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