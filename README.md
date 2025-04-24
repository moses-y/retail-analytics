### README.md


# Retail Analytics Platform

![Retail Analytics Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![ML Models](https://img.shields.io/badge/ML-XGBoost%20|%20KMeans%20|%20BERT-yellow)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive retail analytics platform that combines sales forecasting, customer segmentation, product review analysis, and AI-powered product insights. This production-ready application provides actionable business intelligence for retail operations.

## Features

- **Sales Analytics & Forecasting**: Analyze sales trends and predict future sales with ML models
- **Customer Segmentation**: Identify customer segments based on purchasing behavior
- **Product Review Analysis**: Extract insights from product reviews using NLP
- **AI-Powered Q&A**: Answer product questions using RAG (Retrieval Augmented Generation)
- **Interactive Dashboard**: Visualize data and insights with Streamlit
- **RESTful API**: Access all functionality programmatically via FastAPI

## Architecture

![Architecture](docs/images/architecture.png)

The platform consists of:
- **Data Processing Pipeline**: Clean and transform retail data
- **ML Models**: Train and deploy forecasting, segmentation, and NLP models
- **API Layer**: Expose functionality through RESTful endpoints
- **Dashboard**: Visualize insights through interactive UI
- **MLOps Components**: Monitor model performance and manage deployments

## Installation

### Prerequisites

- Python 3.9+
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/moses-y/retail-analytics.git
   cd retail-analytics
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

### Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

The API will be available at http://localhost:8000. API documentation is available at http://localhost:8000/docs.

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at http://localhost:8501.

### Running from Docker

```bash
docker-compose up -d
```

## Development

### Project Structure

```
retail-analytics/
├── api/                # FastAPI application
├── config/             # Configuration files
├── dashboard/          # Streamlit dashboard
├── data/               # Data files
├── docs/               # Documentation
├── models/             # Saved models
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
│   ├── data/           # Data processing
│   ├── models/         # ML models
│   ├── utils/          # Utilities
│   └── visualization/  # Visualization
├── tests/              # Tests
└── .env                # Environment variables
```

### Running Tests

```bash
pytest tests/
```

## Documentation

- [Implementation Plan](docs/implementation_plan.md)
- [API Documentation](docs/api_documentation.md)
- [Dashboard Guide](docs/dashboard_guide.md)
- [MLOps Design](docs/mlops_design.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

[moses-y](https://github.com/moses-y)
