"""
API Router for Customer Segmentation Data
"""
import logging
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler
import os

logger = logging.getLogger("api.routers.segmentation")

router = APIRouter()

SEGMENTS_DATA_PATH = "data/processed/customer_segments.csv"

@router.get("/segmentation/data", tags=["segmentation"])
async def get_segmentation_data():
    """
    Retrieve customer segmentation data (including cluster assignments).
    """
    logger.info("Received request for /segmentation/data")
    if not os.path.exists(SEGMENTS_DATA_PATH):
        logger.error(f"Customer segments data file not found: {SEGMENTS_DATA_PATH}")
        raise HTTPException(status_code=404, detail="Customer segments data not found.")

    try:
        df = pd.read_csv(SEGMENTS_DATA_PATH)
        data = df.to_dict(orient="records")
        logger.info(f"Successfully loaded and returning {len(data)} customer segment records.")
        return data
    except Exception as e:
        logger.exception(f"Error loading or processing segmentation data from {SEGMENTS_DATA_PATH}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/segmentation/profiles", tags=["segmentation"])
async def get_segmentation_profiles():
    """
    Retrieve aggregated profiles for each customer segment.
    """
    logger.info("Received request for /segmentation/profiles")
    if not os.path.exists(SEGMENTS_DATA_PATH):
        logger.error(f"Customer segments data file not found: {SEGMENTS_DATA_PATH}")
        raise HTTPException(status_code=404, detail="Customer segments data not found.")

    try:
        df = pd.read_csv(SEGMENTS_DATA_PATH)

        # Define features for profiling (adjust if needed based on generate_segments.py)
        profile_features = [
            'total_sales', 'online_sales', 'in_store_sales', 'num_customers',
            'avg_transaction', 'return_rate', 'online_ratio', 'price_per_customer',
            'is_weekend', 'online_to_instore_ratio'
        ]
        # Filter for features that actually exist in the dataframe
        available_features = [f for f in profile_features if f in df.columns]

        if 'cluster' not in df.columns:
             logger.error("Column 'cluster' not found in segmentation data.")
             raise HTTPException(status_code=500, detail="Segmentation data is missing cluster assignments.")

        # Group by cluster and calculate mean for available features
        profiles_raw = df.groupby('cluster')[available_features].mean()

        # Normalize the profile features (0-1 scaling for radar chart)
        scaler = MinMaxScaler()
        profiles_normalized = pd.DataFrame(scaler.fit_transform(profiles_raw), index=profiles_raw.index, columns=profiles_raw.columns)

        # Calculate segment sizes and percentages
        segment_counts = df['cluster'].value_counts()
        total_count = len(df)

        # --- Placeholder Segment Names/Descriptions/Recommendations ---
        # In a real application, these might come from a config file, database, or be generated
        segment_details = {
            0: {"name": "High Value", "description": "High spending, frequent online shoppers.", "recommendations": ["Loyalty program", "Premium offers"]},
            1: {"name": "Regular In-Store", "description": "Consistent shoppers, prefer in-store.", "recommendations": ["In-store promotions", "Cross-selling"]},
            2: {"name": "Occasional / At Risk", "description": "Low frequency, lower spend.", "recommendations": ["Re-engagement campaigns", "Special discounts"]},
            # Add more placeholders if more clusters are possible
        }
        default_details = {"name": "General Segment", "description": "Standard customer segment.", "recommendations": ["Standard marketing"]}
        # --- End Placeholders ---


        # Structure the final response
        structured_profiles = []
        for cluster_id, profile_norm in profiles_normalized.iterrows():
            details = segment_details.get(cluster_id, default_details) # Get details or default
            count = segment_counts.get(cluster_id, 0)
            percentage = round((count / total_count) * 100, 1) if total_count > 0 else 0

            structured_profiles.append({
                "segment_id": cluster_id,
                "name": details["name"],
                "description": details["description"],
                "size": count,
                "percentage": percentage,
                "profile": profile_norm.to_dict(), # Nested dictionary of normalized features
                "recommendations": details["recommendations"]
            })

        logger.info(f"Successfully calculated and returning structured profiles for {len(structured_profiles)} segments.")
        return structured_profiles
    except Exception as e:
        logger.exception(f"Error calculating or structuring segment profiles from {SEGMENTS_DATA_PATH}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
