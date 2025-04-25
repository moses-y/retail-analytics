"""
API Router for Product Data (Placeholder)
"""
import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger("api.routers.products")

router = APIRouter()

@router.get("/products", tags=["products"])
async def get_products():
    """
    Retrieve product list (Placeholder).
    """
    logger.warning("Endpoint /products called, but is currently a placeholder.")
    # Returning an empty list for now to avoid 404
    # In a real implementation, load product data here
    return []
