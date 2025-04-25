"""
Customer Segmentation Dashboard Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from dashboard.components.charts import (
    customer_segments_chart,
    create_kpi_row
)
from dashboard.components.filters import (
    category_filter,
    create_filter_sidebar
)

# Import data utilities
from src.utils.config import get_api_config
import requests
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation | Retail Analytics",
    page_icon="👥",
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
    .segment-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .segment-1 {
        background-color: rgba(30, 136, 229, 0.1);
        border-left: 5px solid #1E88E5;
    }
    .segment-2 {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 5px solid #FFC107;
    }
    .segment-3 {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_customer_data():
    """Load customer segmentation data from API or file"""
    try:
        # Try to load from API
        response = requests.get(f"{API_URL}/api/segmentation/data")
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not load data from API: {e}")

    # Fallback to file
    try:
        data = pd.read_csv("data/processed/customer_segments.csv")
        return data
    except Exception as e:
        st.error(f"Could not load data from file: {e}")
        # Return empty dataframe with expected columns
        return pd.DataFrame({
            'customer_id': [],
            'total_spend': [],
            'avg_transaction': [],
            'purchase_frequency': [],
            'days_since_last_purchase': [],
            'product_categories': [],
            'online_ratio': [],
            'cluster': []
        })


@st.cache_data(ttl=3600)
def get_segment_profiles():
    """Get segment profiles from API"""
    try:
        response = requests.get(f"{API_URL}/api/segmentation/profiles")
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting segment profiles: {e}")
        return None


# Load data
customer_data = load_customer_data()

# Main content
st.markdown('<div class="main-header">Customer Segmentation</div>', unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.markdown("## Segmentation Settings")

    # Number of clusters
    num_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=5,
        value=3
    )

    # Features to include
    st.markdown("### Features to Include")

    include_spend = st.checkbox("Spending Behavior", value=True)
    include_frequency = st.checkbox("Purchase Frequency", value=True)
    include_recency = st.checkbox("Purchase Recency", value=True)
    include_categories = st.checkbox("Product Categories", value=True)
    include_channel = st.checkbox("Purchase Channel", value=True)

    # Apply button
    apply_segmentation = st.button("Apply Segmentation")

# Initialize session state
if 'segmentation_applied' not in st.session_state:
    st.session_state.segmentation_applied = False
    st.session_state.segment_profiles = None

# Apply segmentation if button is clicked
if apply_segmentation:
    with st.spinner("Applying segmentation..."):
        # In a real app, we would call the API with the selected parameters
        # For now, we'll just use the existing data
        st.session_state.segmentation_applied = True
        st.session_state.segment_profiles = get_segment_profiles()

# Overview metrics
st.markdown('<div class="sub-header">Segmentation Overview</div>', unsafe_allow_html=True)

# Calculate segment sizes
if 'cluster' in customer_data.columns:
    segment_counts = customer_data['cluster'].value_counts().sort_index()
    total_customers = len(customer_data)

    # Create metrics
    metrics = []

    for i, count in enumerate(segment_counts):
        metrics.append({
            'label': f'Segment {i+1}',
            'value': count,
            'delta': f"{count/total_customers*100:.1f}",
            'delta_suffix': "%",
            'help_text': f'Number of customers in Segment {i+1}'
        })

    # Add total
    metrics.append({
        'label': 'Total Customers',
        'value': total_customers,
        'help_text': 'Total number of customers'
    })

    create_kpi_row(metrics)

# Segment profiles
st.markdown('<div class="sub-header">Segment Profiles</div>', unsafe_allow_html=True)

# Get segment profiles
segment_profiles = st.session_state.segment_profiles if st.session_state.segmentation_applied else get_segment_profiles()

if segment_profiles:
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Radar Chart", "Detailed Profiles"])

    with tab1:
        # Prepare data for radar chart
        features = list(segment_profiles[0]['profile'].keys())

        # Create dataframe for radar chart
        radar_data = []

        for segment in segment_profiles:
            segment_id = segment['segment_id']
            for feature, value in segment['profile'].items():
                radar_data.append({
                    'Segment': f"Segment {segment_id+1}",
                    'Feature': feature.replace('_', ' ').title(),
                    'Value': value
                })

        radar_df = pd.DataFrame(radar_data)

        # Create radar chart
        fig = px.line_polar(
            radar_df,
            r='Value',
            theta='Feature',
            color='Segment',
            line_close=True,
            title="Segment Profiles",
            template="plotly_white"
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1]
                )
            ),
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Display detailed profiles
        for i, segment in enumerate(segment_profiles):
            segment_id = segment['segment_id']
            segment_name = segment['name']
            segment_description = segment['description']
            segment_profile = segment['profile']
            segment_size = segment['size']

            st.markdown(f'<div class="segment-card segment-{segment_id+1}">', unsafe_allow_html=True)
            st.markdown(f"### Segment {segment_id+1}: {segment_name}")
            st.markdown(f"**Size:** {segment_size} customers ({segment['percentage']}%)")
            st.markdown(f"**Description:** {segment_description}")

            # Display profile as a horizontal bar chart
            profile_df = pd.DataFrame({
                'Feature': [k.replace('_', ' ').title() for k in segment_profile.keys()],
                'Value': list(segment_profile.values())
            })

            fig = px.bar(
                profile_df,
                x='Value',
                y='Feature',
                orientation='h',
                title=f"Segment {segment_id+1} Profile",
                template="plotly_white"
            )

            fig.update_layout(
                xaxis=dict(range=[0, 1]),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.markdown("**Recommendations:**")
            for rec in segment['recommendations']:
                st.markdown(f"- {rec}")

            st.markdown('</div>', unsafe_allow_html=True)

# Segment distribution
st.markdown('<div class="sub-header">Segment Distribution</div>', unsafe_allow_html=True)

if 'cluster' in customer_data.columns:
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Pie Chart", "Feature Distribution"])

    with tab1:
        # Create pie chart
        fig = px.pie(
            segment_counts.reset_index(),
            values='count',
            names='cluster',
            title="Customer Segment Distribution",
            hole=0.4,
            color_discrete_sequence=['#1E88E5', '#FFC107', '#4CAF50', '#F44336', '#9C27B0'][:len(segment_counts)],
            template="plotly_white"
        )

        # Update labels
        fig.update_traces(
            textinfo='percent+label',
            textposition='outside',
            texttemplate='Segment %{label}<br>%{percent}'
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Define available options based on actual columns, excluding IDs/cluster
        available_options = [
            col for col in customer_data.columns
            if col not in ['store_id', 'customer_id', 'cluster'] # Exclude potential ID columns and cluster
        ]

        # Define preferred default features
        preferred_defaults = [
            'total_sales', 'avg_transaction', 'num_customers',
            'online_ratio', 'price_per_customer', 'return_rate'
        ]

        # Filter preferred defaults to only include those available in the data
        actual_defaults = [
            col for col in preferred_defaults if col in available_options
        ]
        # If no preferred defaults are available, select the first few available options
        if not actual_defaults and available_options:
             actual_defaults = available_options[:min(3, len(available_options))]


        # Select features to display
        features = st.multiselect(
            "Select Features for Distribution Plots",
            options=available_options,
            default=actual_defaults # Use filtered defaults
        )

        if features:
            # Create distribution plots
            for feature in features:
                # Create box plot
                fig = px.box(
                    customer_data,
                    x='cluster',
                    y=feature,
                    title=f"{feature.replace('_', ' ').title()} by Segment",
                    color='cluster',
                    color_discrete_sequence=['#1E88E5', '#FFC107', '#4CAF50', '#F44336', '#9C27B0'][:len(segment_counts)],
                    template="plotly_white"
                )

                fig.update_layout(
                    xaxis_title="Segment",
                    yaxis_title=feature.replace('_', ' ').title(),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

# Segment comparison
st.markdown('<div class="sub-header">Segment Comparison</div>', unsafe_allow_html=True)

# Create scatter plot
if 'total_spend' in customer_data.columns and 'purchase_frequency' in customer_data.columns:
    # Select features for x and y axes
    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox(
            "X-Axis Feature",
            options=[col for col in customer_data.columns if col not in ['customer_id', 'cluster']],
            index=available_options.index('total_sales') if 'total_sales' in available_options else 0
        )

    with col2:
        y_feature = st.selectbox(
            "Y-Axis Feature",
            options=available_options,
            index=available_options.index('avg_transaction') if 'avg_transaction' in available_options else (1 if len(available_options) > 1 else 0) # Default to avg_transaction or second available
        )

    # Create scatter plot if features are selected
    if x_feature and y_feature:
        fig = px.scatter(
            customer_data,
            x=x_feature,
        y=y_feature,
        color='cluster',
        title=f"Customer Segments: {x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()}",
        color_discrete_sequence=['#1E88E5', '#FFC107', '#4CAF50', '#F44336', '#9C27B0'][:len(segment_counts)],
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title=x_feature.replace('_', ' ').title(),
        yaxis_title=y_feature.replace('_', ' ').title(),
        legend_title="Segment"
    )

    st.plotly_chart(fig, use_container_width=True)

# Segment insights
st.markdown('<div class="sub-header">Segment Insights</div>', unsafe_allow_html=True)

# Display insights
insights = [
    "**High-Value Customers (Segment 1)** represent 20% of customers but contribute 45% of revenue. They respond well to loyalty programs and premium product offerings.",
    "**Regular Shoppers (Segment 2)** have consistent purchasing patterns but lower average transaction values. Targeted promotions and cross-selling can increase their value.",
    "**Occasional Buyers (Segment 3)** are at risk of churn. Re-engagement campaigns and special offers can help convert them to more regular customers."
]

for insight in insights:
    st.markdown(insight)

# Download data button
st.markdown("### Download Segmentation Data")
csv = customer_data.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
)
