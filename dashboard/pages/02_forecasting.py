"""
Sales Forecasting Dashboard Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.charts import (
    forecast_chart,
    feature_importance_chart,
    create_kpi_row
)
from dashboard.components.filters import (
    date_range_filter,
    category_filter,
    create_filter_sidebar
)

# Import data utilities
from src.utils.config import get_api_config
from src.utils.metrics import format_metrics
import requests
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Sales Forecasting | Retail Analytics",
    page_icon="🔮",
    layout="wide"
)

# Load configuration
api_config = get_api_config()
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_sales_data():
    """Load sales data from API or file"""
    try:
        # Try to load from API
        response = requests.get(f"{API_URL}/api/sales/data")
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not load data from API: {e}")
    
    # Fallback to file
    try:
        data = pd.read_csv("data/processed/retail_sales_data.csv")
        return data
    except Exception as e:
        st.error(f"Could not load data from file: {e}")
        # Return empty dataframe with expected columns
        return pd.DataFrame({
            'date': [],
            'store_id': [],
            'category': [],
            'weather': [],
            'promotion': [],
            'special_event': [],
            'dominant_age_group': [],
            'num_customers': [],
            'total_sales': [],
            'online_sales': [],
            'in_store_sales': [],
            'avg_transaction': [],
            'return_rate': []
        })


@st.cache_data(ttl=3600)
def get_forecast(category=None, store_id=None, horizon=30):
    """Get forecast from API"""
    try:
        # Prepare request parameters
        params = {"horizon": horizon}
        if category:
            params["category"] = category
        if store_id:
            params["store_id"] = store_id
            
        # Make API request
        response = requests.get(
            f"{API_URL}/api/forecasting/predict",
            params=params
        )
        
        if response.status_code == 200:
            data = response.json()
            forecast_df = pd.DataFrame(data["forecast"])
            metrics = data["metrics"]
            return forecast_df, metrics
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None, None
    except Exception as e:
        st.error(f"Error getting forecast: {e}")
        return None, None


@st.cache_data(ttl=3600)
def get_feature_importance():
    """Get feature importance from API"""
    try:
        response = requests.get(f"{API_URL}/api/forecasting/feature-importance")
        if response.status_code == 200:
            data = response.json()
            return data["features"], data["importance"]
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None, None
    except Exception as e:
        st.error(f"Error getting feature importance: {e}")
        return None, None


# Load data
sales_data = load_sales_data()

# Convert date column to datetime
if 'date' in sales_data.columns:
    sales_data['date'] = pd.to_datetime(sales_data['date'])

# Main content
st.markdown('<div class="main-header">Sales Forecasting</div>', unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.markdown("## Forecast Settings")
    
    # Category filter
    if 'category' in sales_data.columns:
        category = st.selectbox(
            "Category",
            options=["All"] + sorted(sales_data['category'].unique().tolist()),
            index=0
        )
    else:
        category = "All"
    
    # Store filter
    if 'store_id' in sales_data.columns:
        store = st.selectbox(
            "Store",
            options=["All"] + sorted(sales_data['store_id'].unique().tolist()),
            index=0
        )
    else:
        store = "All"
    
    # Forecast horizon
    horizon = st.slider(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    # Generate forecast button
    generate_forecast = st.button("Generate Forecast")

# Initialize session state
if 'forecast_generated' not in st.session_state:
    st.session_state.forecast_generated = False
    st.session_state.forecast_data = None
    st.session_state.forecast_metrics = None

# Generate forecast if button is clicked
if generate_forecast:
    with st.spinner("Generating forecast..."):
        # Convert "All" to None for API
        api_category = None if category == "All" else category
        api_store = None if store == "All" else store
        
        # Get forecast
        forecast_data, forecast_metrics = get_forecast(
            category=api_category,
            store_id=api_store,
            horizon=horizon
        )
        
        if forecast_data is not None:
            st.session_state.forecast_generated = True
            st.session_state.forecast_data = forecast_data
            st.session_state.forecast_metrics = forecast_metrics
        else:
            st.error("Failed to generate forecast. Please try again.")

# Display forecast if available
if st.session_state.forecast_generated and st.session_state.forecast_data is not None:
    # Get forecast data
    forecast_df = st.session_state.forecast_data
    metrics = st.session_state.forecast_metrics
    
    # Convert date column to datetime
    if 'date' in forecast_df.columns:
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    
    # Display metrics
    st.markdown('<div class="sub-header">Forecast Metrics</div>', unsafe_allow_html=True)
    
    # Create KPI row
    kpi_metrics = [
        {
            'label': 'RMSE',
            'value': f"{metrics.get('rmse', 0):,.2f}",
            'help_text': 'Root Mean Squared Error'
        },
        {
            'label': 'MAE',
            'value': f"{metrics.get('mae', 0):,.2f}",
            'help_text': 'Mean Absolute Error'
        },
        {
            'label': 'MAPE',
            'value': f"{metrics.get('mape', 0) * 100:,.2f}",
            'suffix': '%',
            'help_text': 'Mean Absolute Percentage Error'
        },
        {
            'label': 'R²',
            'value': f"{metrics.get('r2', 0):,.2f}",
            'help_text': 'Coefficient of Determination'
        }
    ]
    
    create_kpi_row(kpi_metrics)
    
    # Display forecast chart
    st.markdown('<div class="sub-header">Sales Forecast</div>', unsafe_allow_html=True)
    
    # Split data into historical and forecast
    historical_mask = forecast_df['actual'].notna()
    historical_data = forecast_df[historical_mask].copy()
    forecast_data = forecast_df.copy()
    
    # Create forecast chart
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['actual'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        )
    )
    
    # Add forecast data
    fig.add_trace(
        go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Add confidence intervals if available
    if 'lower_bound' in forecast_data.columns and 'upper_bound' in forecast_data.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['lower_bound'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        legend_title="Legend",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown('<div class="sub-header">Feature Importance</div>', unsafe_allow_html=True)
    
    # Get feature importance
    features, importance = get_feature_importance()
    
    if features is not None and importance is not None:
        # Create feature importance chart
        fig = feature_importance_chart(features, importance)
        st.plotly_chart(fig, use_container_width=True)
    
    # Forecast details
    st.markdown('<div class="sub-header">Forecast Details</div>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Chart View", "Table View"])
    
    with tab1:
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Forecast by Day of Week")
            # Add day of week
            forecast_df['day_of_week'] = pd.to_datetime(forecast_df['date']).dt.day_name()
            # Group by day of week
            day_forecast = forecast_df.groupby('day_of_week')['forecast'].mean().reset_index()
            # Order days
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_forecast['day_of_week'] = pd.Categorical(day_forecast['day_of_week'], categories=days_order, ordered=True)
            day_forecast = day_forecast.sort_values('day_of_week')
            
            fig = px.bar(
                day_forecast, 
                x='day_of_week', 
                y='forecast',
                title="Forecast by Day of Week",
                color='day_of_week',
                template="plotly_white"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Forecast Distribution")
            
            fig = px.histogram(
                forecast_df, 
                x='forecast',
                title="Forecast Distribution",
                template="plotly_white",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Forecast Data Table")
        
        # Format date column
        forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
        
        # Display table
        st.dataframe(forecast_df, use_container_width=True)
        
        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name="sales_forecast.csv",
            mime="text/csv"
        )

else:
    # Display instructions
    st.info("Use the sidebar to configure and generate a sales forecast.")
    
    # Display sample forecast
    st.markdown('<div class="sub-header">Sample Forecast</div>', unsafe_allow_html=True)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=60)
    historical_data = pd.DataFrame({
        'date': dates[:30],
        'actual': np.random.normal(1000, 100, 30) + np.linspace(0, 200, 30)
    })
    
    forecast_data = pd.DataFrame({
        'date': dates[30:],
        'forecast': np.random.normal(1200, 120, 30) + np.linspace(200, 400, 30)
    })
    
    # Create sample forecast chart
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['actual'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        )
    )
    
    # Add forecast data
    fig.add_trace(
        go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Sample Forecast (Demo Only)",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        legend_title="Legend",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    This is a sample forecast for demonstration purposes. Use the sidebar to generate a real forecast.
    
    The forecast will show:
    - Historical sales data
    - Predicted future sales
    - Confidence intervals
    - Feature importance
    - Forecast metrics (RMSE, MAE, MAPE, R²)
    """)