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


@st.cache_data(ttl=3600) # Keep caching for performance
def load_sales_data(file_path="data/raw/retail_sales_data.csv"):
    """Load sales data directly from the raw CSV file."""
    try:
        data = pd.read_csv(file_path)
        st.success(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        st.error(f"Error: Raw data file not found at {file_path}")
        # Return empty dataframe with expected columns if file not found
        return pd.DataFrame({
            'date': [], 'store_id': [], 'category': [], 'weather': [],
            'promotion': [], 'special_event': [], 'dominant_age_group': [],
            'num_customers': [], 'total_sales': [], 'online_sales': [],
            'in_store_sales': [], 'avg_transaction': [], 'return_rate': []
        })
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        # Return empty dataframe on other errors
        return pd.DataFrame({
            'date': [], 'store_id': [], 'category': [], 'weather': [],
            'promotion': [], 'special_event': [], 'dominant_age_group': [],
            'num_customers': [], 'total_sales': [], 'online_sales': [],
            'in_store_sales': [], 'avg_transaction': [], 'return_rate': []
        })


# Load data
raw_sales_data = load_sales_data()

# Clean and preprocess data using the standardized function
# Ensure the function exists and handles potential errors
if not raw_sales_data.empty:
    st.info("Applying data cleaning and preprocessing...")
    sales_data = clean_sales_data(raw_sales_data)
    st.success("Data cleaning and preprocessing applied.")
else:
    sales_data = raw_sales_data # Use the empty dataframe if loading failed

# Convert date column to datetime (redundant if done in clean_sales_data, but safe to keep)
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
tab1, tab2, tab3, tab4 = st.tabs([
    "Sales Trends",
    "Category Analysis",
    "Channel Analysis",
    "Additional Analysis" # New Tab
])

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
if 'promotion' in filtered_data.columns:
    # Effect of promotions
    promo_effect = filtered_data.groupby('promotion')['total_sales'].sum() # Keep as Series with promotion as index
    # Check if both 'Discount' and 'None' exist in the index
    if 'Discount' in promo_effect.index and 'None' in promo_effect.index:
        promo_impact = promo_effect['Discount'] - promo_effect['None']
        if promo_impact > 0:
            insights.append(f"Promotions compared to no promotion show a positive impact, increasing sales by ${promo_impact:,.2f}.")
        elif promo_impact < 0:
             insights.append(f"Sales during promotions were lower than sales without promotions by ${abs(promo_impact):,.2f}.")
        else:
             insights.append(f"Sales during promotions were similar to sales without promotions.")
    elif 'Discount' in promo_effect.index:
        insights.append(f"Only 'Discount' promotions found in the filtered data. Total sales during discount: ${promo_effect['Discount']:,.2f}.")
    elif 'None' in promo_effect.index:
         insights.append(f"Only 'None' (no promotion) sales found in the filtered data. Total sales without promotion: ${promo_effect['None']:,.2f}.")

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

# Add new tab for the missing plots
with tab4:
    st.markdown('<div class="sub-header">Additional Analysis</div>', unsafe_allow_html=True)

    col_add1, col_add2 = st.columns(2)

    with col_add1:
        # Sales by Age Group
        st.markdown("### Sales by Dominant Age Group")
        if 'dominant_age_group' in filtered_data.columns:
            age_sales = filtered_data.groupby('dominant_age_group')['total_sales'].mean().reset_index().sort_values('total_sales', ascending=False)
            fig_age = px.bar(
                age_sales,
                x='dominant_age_group',
                y='total_sales',
                title="Average Sales by Dominant Age Group",
                color='dominant_age_group',
                template="plotly_white"
            )
            fig_age.update_layout(showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.warning("Column 'dominant_age_group' not found.")

        # Store Performance
        st.markdown("### Total Sales by Store")
        if 'store_id' in filtered_data.columns:
            store_sales = filtered_data.groupby('store_id')['total_sales'].sum().reset_index().sort_values('total_sales', ascending=False)
            fig_store = px.bar(
                store_sales,
                x='store_id',
                y='total_sales',
                title="Total Sales by Store",
                color='store_id',
                template="plotly_white"
            )
            fig_store.update_layout(showlegend=False)
            st.plotly_chart(fig_store, use_container_width=True)
        else:
            st.warning("Column 'store_id' not found.")


    with col_add2:
        # Return Rate by Category
        st.markdown("### Average Return Rate by Category")
        if 'return_rate' in filtered_data.columns and 'category' in filtered_data.columns:
            return_rate = filtered_data.groupby('category')['return_rate'].mean().reset_index().sort_values('return_rate', ascending=False)
            fig_return = px.bar(
                return_rate,
                x='category',
                y='return_rate',
                title="Average Return Rate by Category",
                color='category',
                template="plotly_white"
            )
            fig_return.update_layout(yaxis_tickformat='.2%') # Format as percentage
            fig_return.update_layout(showlegend=False)
            st.plotly_chart(fig_return, use_container_width=True)
        else:
            st.warning("Columns 'return_rate' or 'category' not found.")

        # Online vs In-store by Category (Stacked Bar)
        st.markdown("### Online vs In-Store Sales by Category")
        if 'category' in filtered_data.columns and 'online_sales' in filtered_data.columns and 'in_store_sales' in filtered_data.columns:
            channel_cat_data = filtered_data.groupby('category')[['in_store_sales', 'online_sales']].sum().reset_index()
            channel_cat_data_melted = pd.melt(
                channel_cat_data,
                id_vars=['category'],
                value_vars=['in_store_sales', 'online_sales'],
                var_name='Channel',
                value_name='Sales'
            )
            # Rename channels for clarity
            channel_cat_data_melted['Channel'] = channel_cat_data_melted['Channel'].replace({
                'in_store_sales': 'In-Store',
                'online_sales': 'Online'
            })

            fig_channel_cat = px.bar(
                channel_cat_data_melted,
                x='category',
                y='Sales',
                color='Channel',
                title="Online vs In-Store Sales by Category",
                barmode='stack', # Use stack instead of group
                template="plotly_white"
            )
            st.plotly_chart(fig_channel_cat, use_container_width=True)
        else:
            st.warning("Required columns for Online vs In-Store chart not found.")
