"""
API Router for Sales Data
"""
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
import os

logger = logging.getLogger("api.routers.sales")

router = APIRouter()

PROCESSED_DATA_PATH = "data/processed/retail_sales_data.csv"

@router.get("/sales/data", tags=["sales"])
async def get_sales_data():
    """
    Retrieve processed sales data.
    """
    logger.info("Received request for /sales/data")
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error(f"Processed sales data file not found: {PROCESSED_DATA_PATH}")
        raise HTTPException(status_code=404, detail="Processed sales data not found.")

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        # Convert to dictionary records for JSON response
        data = df.to_dict(orient="records")
        logger.info(f"Successfully loaded and returning {len(data)} sales records.")
        return data
    except Exception as e:
        logger.exception(f"Error loading or processing sales data from {PROCESSED_DATA_PATH}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
