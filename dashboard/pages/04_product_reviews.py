"""
Product Reviews Dashboard Page
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
    sentiment_distribution_chart,
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
    page_title="Product Reviews | Retail Analytics",
    page_icon="📝",
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
    .positive {
        color: #4CAF50;
    }
    .neutral {
        color: #FFC107;
    }
    .negative {
        color: #F44336;
    }
    .review-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_review_data(file_path="data/raw/product_reviews.csv"):
    """Load review data directly from the raw CSV file."""
    try:
        data = pd.read_csv(file_path)
        st.success(f"Successfully loaded review data from {file_path}")
        # Basic cleaning needed after loading raw data
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if 'rating' in data.columns:
            data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
        # Add other necessary cleaning steps if needed, matching preprocessing.py
        return data
    except FileNotFoundError:
        st.error(f"Error: Raw review data file not found at {file_path}")
        return pd.DataFrame({
            'review_id': [], 'product': [], 'category': [], 'rating': [],
            'review_text': [], 'feature_mentioned': [], 'attribute_mentioned': [],
            'date': [], 'sentiment': []
        })
    except Exception as e:
        st.error(f"Error loading review data from {file_path}: {e}")
        return pd.DataFrame({
            'review_id': [], 'product': [], 'category': [], 'rating': [],
            'review_text': [], 'feature_mentioned': [], 'attribute_mentioned': [],
            'date': [], 'sentiment': []
        })


@st.cache_data(ttl=3600)
def get_product_summary(category=None, min_rating=None):
    """Get product summary data from API using /api/reviews/products endpoint"""
    try:
        params = {}
        if category:
            params["category"] = category
        if min_rating is not None:
            params["min_rating"] = min_rating

        api_endpoint = f"{API_URL}/api/reviews/products" # Corrected endpoint
        st.info(f"Fetching product summaries from: {api_endpoint} with params: {params}")
        response = requests.get(api_endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            # The dashboard expects a specific structure, adapt if necessary
            # For now, assume the API returns a list of product summaries directly
            st.success("Successfully fetched product summaries.")
            return data # Return the list of product dicts
        else:
            st.error(f"API Error fetching product summaries: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting product summary: {e}")
        return None


@st.cache_data(ttl=3600)
def get_feature_sentiment(products: list = None):
    """Get feature sentiment from API using /api/reviews/features endpoint"""
    if not products: # Need at least one product
        return None
    try:
        # Prepare request parameters - this endpoint expects a list of products
        params = {"products": products}

        api_endpoint = f"{API_URL}/api/reviews/features" # Corrected endpoint
        st.info(f"Fetching feature sentiment from: {api_endpoint} for products: {products}")
        # Note: GET requests don't typically send lists in query params easily.
        # If the API expects a POST or different format, this needs adjustment.
        # Assuming the API handles list query params correctly for now.
        response = requests.get(api_endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            # The dashboard expects a flat list of features, but API returns dict keyed by product.
            # We need to adapt the data structure or how it's used later.
            # For now, return the raw dict.
            st.success("Successfully fetched feature sentiment.")
            return data # Returns dict like {"product1": [features...], "product2": [features...]}
        else:
            st.error(f"API Error fetching feature sentiment: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting feature sentiment: {e}")
        return None


# Load data
review_data = load_review_data()

# Convert date column to datetime
if 'date' in review_data.columns:
    review_data['date'] = pd.to_datetime(review_data['date'])

# Main content
st.markdown('<div class="main-header">Product Reviews Analysis</div>', unsafe_allow_html=True)

# Apply filters
filtered_data = create_filter_sidebar(
    review_data,
    date_column='date',
    category_columns=['product', 'category', 'sentiment'],
    numeric_columns=['rating']
)

# Calculate KPIs
total_reviews = len(filtered_data)
avg_rating = filtered_data['rating'].mean() if 'rating' in filtered_data.columns else 0
sentiment_counts = filtered_data['sentiment'].value_counts() if 'sentiment' in filtered_data.columns else pd.Series()
positive_pct = sentiment_counts.get('positive', 0) / total_reviews * 100 if total_reviews > 0 else 0

# Create KPI row
kpi_metrics = [
    {
        'label': 'Total Reviews',
        'value': f"{total_reviews:,}",
        'help_text': 'Total number of reviews'
    },
    {
        'label': 'Average Rating',
        'value': f"{avg_rating:.1f}",
        'suffix': '/5',
        'help_text': 'Average product rating'
    },
    {
        'label': 'Positive Sentiment',
        'value': f"{positive_pct:.1f}",
        'suffix': '%',
        'help_text': 'Percentage of reviews with positive sentiment'
    },
    {
        'label': 'Most Mentioned Feature',
        'value': filtered_data['feature_mentioned'].value_counts().index[0] if 'feature_mentioned' in filtered_data.columns and len(filtered_data) > 0 else 'N/A',
        'help_text': 'Most frequently mentioned product feature'
    }
]

create_kpi_row(kpi_metrics)

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Feature Analysis", "Review Explorer"])

with tab1:
    st.markdown('<div class="sub-header">Sentiment Distribution</div>', unsafe_allow_html=True)

    # Create sentiment distribution chart
    if 'sentiment' in filtered_data.columns and 'product' in filtered_data.columns:
        fig = sentiment_distribution_chart(filtered_data, entity_col='product', sentiment_col='sentiment')
        st.plotly_chart(fig, use_container_width=True)

    # Rating distribution
    st.markdown("### Rating Distribution")

    if 'rating' in filtered_data.columns:
        # Create rating histogram
        fig = px.histogram(
            filtered_data,
            x='rating',
            title="Rating Distribution",
            nbins=5,
            color_discrete_sequence=['#1E88E5'],
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Count",
            bargap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)

    # Sentiment over time
    st.markdown("### Sentiment Trend")

    if 'date' in filtered_data.columns and 'sentiment' in filtered_data.columns:
        # Group by date and sentiment
        sentiment_trend = filtered_data.groupby([pd.Grouper(key='date', freq='M'), 'sentiment']).size().reset_index(name='count')

        # Create line chart
        fig = px.line(
            sentiment_trend,
            x='date',
            y='count',
            color='sentiment',
            title="Sentiment Trend Over Time",
            color_discrete_map={
                'positive': '#4CAF50',
                'neutral': '#FFC107',
                'negative': '#F44336'
            },
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            legend_title="Sentiment"
        )

        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="sub-header">Feature Analysis</div>', unsafe_allow_html=True)

    # Get feature sentiment
    feature_sentiment = get_feature_sentiment()

    if feature_sentiment:
        # Create feature sentiment dataframe
        feature_df = pd.DataFrame(feature_sentiment)

        # Create heatmap
        fig = px.imshow(
            feature_df.pivot(index='feature', columns='product', values='sentiment_score'),
            title="Feature Sentiment by Product",
            color_continuous_scale=['#F44336', '#FFC107', '#4CAF50'],
            zmin=-1,
            zmax=1,
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Product",
            yaxis_title="Feature"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature sentiment bar chart
        st.markdown("### Top Features by Sentiment")

        # Group by feature
        feature_avg = feature_df.groupby('feature')['sentiment_score'].mean().reset_index()
        feature_avg = feature_avg.sort_values('sentiment_score', ascending=False)

        # Create bar chart
        fig = px.bar(
            feature_avg,
            x='feature',
            y='sentiment_score',
            title="Average Feature Sentiment",
            color='sentiment_score',
            color_continuous_scale=['#F44336', '#FFC107', '#4CAF50'],
            range_color=[-1, 1],
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Sentiment Score",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Feature co-occurrence
    st.markdown("### Feature Co-occurrence")

    if 'feature_mentioned' in filtered_data.columns and 'attribute_mentioned' in filtered_data.columns:
        # Create co-occurrence matrix
        feature_counts = filtered_data.groupby(['feature_mentioned', 'attribute_mentioned']).size().reset_index(name='count')

        # Create heatmap
        fig = px.density_heatmap(
            feature_counts,
            x='feature_mentioned',
            y='attribute_mentioned',
            z='count',
            title="Feature-Attribute Co-occurrence",
            color_continuous_scale='Viridis',
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Attribute"
        )

        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="sub-header">Review Explorer</div>', unsafe_allow_html=True)

    # Product selector
    if 'product' in filtered_data.columns:
        products = ['All'] + sorted(filtered_data['product'].unique().tolist())
        selected_product = st.selectbox("Select Product", options=products)

        # Filter by product
        if selected_product != 'All':
            product_reviews = filtered_data[filtered_data['product'] == selected_product]
        else:
            product_reviews = filtered_data
    else:
        product_reviews = filtered_data

    # Sentiment filter
    if 'sentiment' in product_reviews.columns:
        sentiments = ['All'] + sorted(product_reviews['sentiment'].unique().tolist())
        selected_sentiment = st.selectbox("Select Sentiment", options=sentiments)

        # Filter by sentiment
        if selected_sentiment != 'All':
            product_reviews = product_reviews[product_reviews['sentiment'] == selected_sentiment]

    # Display reviews
    st.markdown("### Reviews")

    if len(product_reviews) > 0:
        # Sort by date
        if 'date' in product_reviews.columns:
            product_reviews = product_reviews.sort_values('date', ascending=False)

        # Display reviews
        for i, review in product_reviews.head(10).iterrows():
            st.markdown(f'<div class="review-card">', unsafe_allow_html=True)

            # Safely access data using .get() with default values
            product_name = review.get('product', 'N/A')
            review_date_str = review.get('date').strftime('%Y-%m-%d') if pd.notna(review.get('date')) else 'N/A'
            rating_val = review.get('rating', 0)
            review_text_val = review.get('review_text', '')
            feature = review.get('feature_mentioned', 'N/A')
            attribute = review.get('attribute_mentioned', 'N/A')
            sentiment = review.get('sentiment', 'N/A')

            # Review header
            st.markdown(f"**{product_name}** - {review_date_str}")

            # Rating
            rating = int(rating_val) if pd.notna(rating_val) else 0
            st.markdown(f"{'⭐' * rating}")

            # Review text
            st.markdown(f"{review_text_val}")

            # Features and sentiment
            sentiment_class = ''
            if sentiment == 'positive':
                sentiment_class = 'positive'
            elif sentiment == 'neutral':
                sentiment_class = 'neutral'
            elif sentiment == 'negative':
                sentiment_class = 'negative'

            st.markdown(f"**Feature:** {feature} | **Attribute:** {attribute} | **Sentiment:** <span class='{sentiment_class}'>{sentiment}</span>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No reviews found with the selected filters.")

    # Word cloud
    st.markdown("### Word Cloud")

    # Display placeholder word cloud
    st.image("https://placehold.co/800x400?text=Word+Cloud", use_column_width=True)

# Product summary
st.markdown('<div class="sub-header">Product Summary</div>', unsafe_allow_html=True)

# Get product summary
product_summary = get_product_summary()

if product_summary:
    # Display product summaries
    for product in product_summary:
        st.markdown(f"### {product['product']}")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Create gauge chart for average rating
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=product['average_rating'],
                title={'text': "Average Rating"},
                gauge={
                    'axis': {'range': [0, 5]},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 2.5], 'color': "#F44336"},
                        {'range': [2.5, 3.5], 'color': "#FFC107"},
                        {'range': [3.5, 5], 'color': "#4CAF50"}
                    ]
                }
            ))

            fig.update_layout(height=200)

            st.plotly_chart(fig, use_container_width=True)

            # Sentiment distribution
            # Use the correct key 'sentiment_distribution' returned by the API
            sentiment_dist = product.get('sentiment_distribution', {}) # Get the distribution dict
            sentiment_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Proportion': [ # Use .get() for safety, API returns proportions
                    sentiment_dist.get('positive', 0),
                    sentiment_dist.get('neutral', 0),
                    sentiment_dist.get('negative', 0)
                ]
            })
            # Note: The API returns proportions, not counts. The pie chart will show relative proportions.

            fig = px.pie(
                sentiment_data,
                values='Proportion', # Use Proportion for values
                names='Sentiment',
                title="Sentiment Distribution",
                color='Sentiment',
                color_discrete_map={
                    'Positive': '#4CAF50',
                    'Neutral': '#FFC107',
                    'Negative': '#F44336'
                },
                template="plotly_white"
            )

            fig.update_layout(height=300)

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # The /api/reviews/products endpoint currently only returns:
            # product, category, average_rating, review_count, sentiment_distribution
            # Therefore, we cannot display detailed summary, strengths/weaknesses, or feature sentiment here.
            # Displaying a placeholder or basic info instead.
            st.markdown("#### Product Details")
            st.markdown(f"**Category:** {product.get('category', 'N/A')}")
            st.markdown(f"**Review Count:** {product.get('review_count', 'N/A')}")
            # Add more details here if they become available from the API endpoint

# Download data button
st.markdown("### Download Review Data")
csv = filtered_data.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="product_reviews.csv",
    mime="text/csv"
)
