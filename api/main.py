"""
Main FastAPI application for Retail Analytics AI
"""
import logging
import os
import time
import yaml
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from prometheus_client import Counter, Histogram, start_http_server

from api.routers import forecasting, reviews, sales, segmentation, products, rag # Added new routers
from api.dependencies import get_settings, Settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", mode="a")
    ]
)
logger = logging.getLogger("api")

# Load API configuration
config_path = os.path.join("config", "api_config.yml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Create FastAPI app
app = FastAPI(
    title="Retail Analytics AI API",
    description="API for retail sales forecasting, customer segmentation, and product review analysis",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["cors"]["allow_origins"],
    allow_credentials=config["cors"]["allow_credentials"],
    allow_methods=config["cors"]["allow_methods"],
    allow_headers=config["cors"]["allow_headers"],
)

# Include routers
app.include_router(forecasting.router, prefix="/api", tags=["forecasting"])
app.include_router(reviews.router, prefix="/api", tags=["reviews"])
app.include_router(sales.router, prefix="/api", tags=["sales"]) # Added sales router
app.include_router(segmentation.router, prefix="/api", tags=["segmentation"]) # Added segmentation router
app.include_router(products.router, prefix="/api", tags=["products"]) # Added products router
app.include_router(rag.router, prefix="/api", tags=["rag"]) # Added rag router

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_request_count", "Number of requests received", ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"]
)

# Start Prometheus metrics server if monitoring is enabled
if config["monitoring"]["enabled"]:
    start_http_server(9090)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics for each request"""
    start_time = time.time()

    # Process the request
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        logger.exception("Request failed")
        status_code = 500
        response = JSONResponse(
            status_code=status_code,
            content={"detail": "Internal server error"}
        )

    # Record metrics
    endpoint = request.url.path
    method = request.method
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)

    return response

@app.get("/", tags=["root"])
async def root(settings: Settings = Depends(get_settings)):
    """Root endpoint returning API information"""
    return {
        "name": "Retail Analytics AI API",
        "version": "1.0.0",
        "description": "API for retail sales forecasting, customer segmentation, and product review analysis",
        "environment": settings.environment,
        "docs_url": "/docs",
        "endpoints": {
            "forecasting": "/api/forecasting", # Corrected path prefix
            "reviews": "/api/reviews", # Corrected path prefix
            "segmentation": "/api/segmentation", # Corrected path prefix
            "sales": "/api/sales", # Added sales endpoint info
            "products": "/api/products", # Added products endpoint info
            "rag": "/api/rag" # Corrected path prefix
        }
    }

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Retail Analytics AI API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """OpenAPI schema endpoint"""
    return get_openapi(
        title="Retail Analytics AI API",
        version="1.0.0",
        description="API for retail sales forecasting, customer segmentation, and product review analysis",
        routes=app.routes,
    )

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Run the API server
    uvicorn.run(
        "api.main:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=config["server"]["reload"],
        workers=config["server"]["workers"],
    )
