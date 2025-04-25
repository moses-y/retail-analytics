"""
Sales Analysis Dashboard Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.charts import (
    sales_trend_chart,
    sales_by_category_chart,
    feature_importance_chart,
    create_kpi_row
)
from dashboard.components.filters import (
    date_range_filter,
    category_filter,
    create_filter_sidebar
)

# Import data utilities
from src.data.preprocessing import clean_sales_data # Renamed from preprocess_sales_data
from src.utils.config import get_api_config
import requests
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Sales Analysis | Retail Analytics",
    page_icon="📈",
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


# Load data
sales_data = load_sales_data()

# Convert date column to datetime
if 'date' in sales_data.columns:
    sales_data['date'] = pd.to_datetime(sales_data['date'])

# Main content
st.markdown('<div class="main-header">Sales Analysis</div>', unsafe_allow_html=True)

# Apply filters
filtered_data = create_filter_sidebar(
    sales_data,
    date_column='date',
    category_columns=['category', 'store_id', 'weather', 'promotion'],
    numeric_columns=['total_sales', 'num_customers']
)

# Calculate KPIs
total_sales = filtered_data['total_sales'].sum()
avg_transaction = filtered_data['avg_transaction'].mean()
num_customers = filtered_data['num_customers'].sum()
online_ratio = (filtered_data['online_sales'].sum() / total_sales) * 100

# Create KPI row
kpi_metrics = [
    {
        'label': 'Total Sales',
        'value': f"{total_sales:,.2f}",
        'prefix': '$',
        'help_text': 'Sum of all sales in the selected period'
    },
    {
        'label': 'Customers',
        'value': f"{num_customers:,.0f}",
        'help_text': 'Total number of customers in the selected period'
    },
    {
        'label': 'Avg. Transaction',
        'value': f"{avg_transaction:,.2f}",
        'prefix': '$',
        'help_text': 'Average transaction value in the selected period'
    },
    {
        'label': 'Online Sales',
        'value': f"{online_ratio:,.1f}",
        'suffix': '%',
        'help_text': 'Percentage of sales made online'
    }
]

create_kpi_row(kpi_metrics)

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Sales Trends", "Category Analysis", "Channel Analysis"])

with tab1:
    st.markdown('<div class="sub-header">Sales Trends</div>', unsafe_allow_html=True)

    # Group data by date
    daily_sales = filtered_data.groupby('date')['total_sales'].sum().reset_index()

    # Create sales trend chart
    fig = sales_trend_chart(daily_sales, date_col='date', value_col='total_sales')
    st.plotly_chart(fig, use_container_width=True)

    # Additional analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Sales by Day of Week")
        # Add day of week
        filtered_data['day_of_week'] = filtered_data['date'].dt.day_name()
        # Group by day of week
        day_sales = filtered_data.groupby('day_of_week')['total_sales'].sum().reset_index()
        # Order days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_sales['day_of_week'] = pd.Categorical(day_sales['day_of_week'], categories=days_order, ordered=True)
        day_sales = day_sales.sort_values('day_of_week')

        fig = px.bar(
            day_sales,
            x='day_of_week',
            y='total_sales',
            title="Sales by Day of Week",
            color='day_of_week',
            template="plotly_white"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Sales by Month")
        # Add month
        filtered_data['month'] = filtered_data['date'].dt.month_name()
        # Group by month
        month_sales = filtered_data.groupby('month')['total_sales'].sum().reset_index()
        # Order months
        months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']
        month_sales['month'] = pd.Categorical(month_sales['month'], categories=months_order, ordered=True)
        month_sales = month_sales.sort_values('month')

        fig = px.bar(
            month_sales,
            x='month',
            y='total_sales',
            title="Sales by Month",
            color='month',
            template="plotly_white"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="sub-header">Category Analysis</div>', unsafe_allow_html=True)

    # Group data by category
    category_sales = filtered_data.groupby('category')['total_sales'].sum().reset_index()

    # Create category sales chart
    fig = sales_by_category_chart(category_sales, category_col='category', value_col='total_sales')
    st.plotly_chart(fig, use_container_width=True)

    # Additional analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Category Performance by Weather")
        # Group by category and weather
        weather_cat_sales = filtered_data.groupby(['category', 'weather'])['total_sales'].sum().reset_index()

        fig = px.bar(
            weather_cat_sales,
            x='category',
            y='total_sales',
            color='weather',
            title="Category Sales by Weather",
            barmode='group',
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Category Performance by Promotion")
        # Group by category and promotion
        promo_cat_sales = filtered_data.groupby(['category', 'promotion'])['total_sales'].sum().reset_index()

        fig = px.bar(
            promo_cat_sales,
            x='category',
            y='total_sales',
            color='promotion',
            title="Category Sales by Promotion",
            barmode='group',
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="sub-header">Channel Analysis</div>', unsafe_allow_html=True)

    # Calculate channel metrics
    channel_data = pd.DataFrame({
        'Channel': ['In-Store', 'Online'],
        'Sales': [filtered_data['in_store_sales'].sum(), filtered_data['online_sales'].sum()]
    })

    # Create channel sales chart
    fig = px.pie(
        channel_data,
        values='Sales',
        names='Channel',
        title="Sales by Channel",
        hole=0.4,
        color_discrete_sequence=['#1E88E5', '#FFC107'],
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Additional analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Channel Performance by Category")
        # Melt data for channel comparison
        channel_cat_data = filtered_data.groupby('category')[['in_store_sales', 'online_sales']].sum().reset_index()
        channel_cat_data_melted = pd.melt(
            channel_cat_data,
            id_vars=['category'],
            value_vars=['in_store_sales', 'online_sales'],
            var_name='Channel',
            value_name='Sales'
        )
        # Rename channels
        channel_cat_data_melted['Channel'] = channel_cat_data_melted['Channel'].replace({
            'in_store_sales': 'In-Store',
            'online_sales': 'Online'
        })

        fig = px.bar(
            channel_cat_data_melted,
            x='category',
            y='Sales',
            color='Channel',
            title="Channel Sales by Category",
            barmode='group',
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Channel Performance Over Time")
        # Group by date
        channel_time_data = filtered_data.groupby('date')[['in_store_sales', 'online_sales']].sum().reset_index()
        channel_time_data_melted = pd.melt(
            channel_time_data,
            id_vars=['date'],
            value_vars=['in_store_sales', 'online_sales'],
            var_name='Channel',
            value_name='Sales'
        )
        # Rename channels
        channel_time_data_melted['Channel'] = channel_time_data_melted['Channel'].replace({
            'in_store_sales': 'In-Store',
            'online_sales': 'Online'
        })

        fig = px.line(
            channel_time_data_melted,
            x='date',
            y='Sales',
            color='Channel',
            title="Channel Sales Over Time",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# Insights section
st.markdown('<div class="sub-header">Key Insights</div>', unsafe_allow_html=True)

# Generate insights based on the data
insights = []

# Sales trend insights
if 'date' in filtered_data.columns:
    # Find peak sales day
    peak_day = filtered_data.loc[filtered_data['total_sales'].idxmax()]
    insights.append(f"Peak sales occurred on {peak_day['date'].strftime('%Y-%m-%d')} with ${peak_day['total_sales']:,.2f} in sales.")

    # Compare online vs in-store
    if online_ratio > 50:
        insights.append(f"Online sales dominate with {online_ratio:.1f}% of total sales.")
    else:
        insights.append(f"In-store sales dominate with {100-online_ratio:.1f}% of total sales.")

# Category insights
if 'category' in filtered_data.columns:
    # Top category
    top_category = category_sales.loc[category_sales['total_sales'].idxmax()]
    insights.append(f"The top-performing category is {top_category['category']} with ${top_category['total_sales']:,.2f} in sales.")

    # Category with highest average transaction
    cat_avg_trans = filtered_data.groupby('category')['avg_transaction'].mean().reset_index()
    top_avg_cat = cat_avg_trans.loc[cat_avg_trans['avg_transaction'].idxmax()]
    insights.append(f"{top_avg_cat['category']} has the highest average transaction value at ${top_avg_cat['avg_transaction']:,.2f}.")

# Weather insights
if 'weather' in filtered_data.columns:
    # Sales by weather
    weather_sales = filtered_data.groupby('weather')['total_sales'].sum().reset_index()
    top_weather = weather_sales.loc[weather_sales['total_sales'].idxmax()]
    insights.append(f"Sales are highest during {top_weather['weather']} weather with ${top_weather['total_sales']:,.2f} in total sales.")

# Promotion insights
if 'promotion' in filtered_data.columns and 'promotion' in filtered_data.columns:
    # Effect of promotions
    promo_effect = filtered_data.groupby('promotion')['total_sales'].sum().reset_index()
    if len(promo_effect) > 1:
        promo_impact = promo_effect.set_index('promotion').loc['Discount', 'total_sales'] - promo_effect.set_index('promotion').loc['None', 'total_sales']
        if promo_impact > 0:
            insights.append(f"Promotions have a positive impact, increasing sales by ${promo_impact:,.2f}.")
        else:
            insights.append(f"Promotions do not appear to increase overall sales.")

# Display insights
for i, insight in enumerate(insights):
    st.markdown(f"**Insight {i+1}:** {insight}")

# Download data button
st.markdown("### Download Filtered Data")
csv = filtered_data.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="sales_analysis.csv",
    mime="text/csv"
)
