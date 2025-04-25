"""
Main Streamlit dashboard for Retail Analytics
"""
import os
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import requests

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import utilities and components
from src.utils.config import get_api_config
from src.utils.metrics import format_metrics
from dashboard.components.charts import sales_trend_chart

# Set page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
api_config = get_api_config()
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
        border-radius: 0.75rem; /* Increased border-radius */
        background-color: #ffffff; /* Changed background to white */
        box-shadow: 0 0.25rem 0.5rem rgba(0, 0, 0, 0.1); /* Increased box shadow */
        margin-bottom: 1.5rem; /* Increased bottom margin */
        transition: transform 0.3s ease-in-out; /* Add hover effect */
    }
    .card:hover {
        transform: translateY(-5px); /* Lift card on hover */
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        color: #9e9e9e;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://placehold.co/200x80?text=Retail+Analytics", width=200)
    st.markdown("## Navigation")
    st.markdown("Use the pages in the sidebar to navigate to different sections of the dashboard.")

    st.markdown("---")
    st.markdown("## Data Sources")
    st.markdown("- Sales Data: Updated daily")
    st.markdown("- Product Reviews: Updated weekly")

    st.markdown("---")
    st.markdown("## Settings")
    theme = st.selectbox("Theme", ["Light", "Dark"])

    st.markdown("---")
    st.markdown("## About")
    st.markdown("Retail Analytics Dashboard v0.1.0")
    st.markdown("© 2025 Zero Margin Limited")

# Main content
st.markdown('<div class="main-header">Retail Analytics Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
Welcome to the Retail Analytics Dashboard. This dashboard provides insights into retail sales data,
customer segmentation, product reviews, and sales forecasting.

Use the sidebar to navigate to different sections of the dashboard.
""")

# Overview metrics
st.markdown('<div class="sub-header">Overview</div>', unsafe_allow_html=True)

# Create metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">$1.2M</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Total Sales</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">15.3K</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Customers</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">4.2</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Avg. Rating</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">+8.5%</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Sales Growth</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Add section for recent activity or key highlights
st.markdown('<div class="sub-header">Recent Activity & Highlights</div>', unsafe_allow_html=True)
st.markdown("""
- Latest sales data updated on 2025-04-24.
- New customer segment "High-Value Loyal Customers" identified.
- Positive trend in product reviews for the electronics category.
""")

# Add sales trend chart
st.markdown('<div class="sub-header">Sales Trend</div>', unsafe_allow_html=True)

# Function to fetch sales data from the API
@st.cache_data
def fetch_sales_data(api_url):
    try:
        response = requests.get(f"{api_url}/sales/data")
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching sales data: {e}")
        return None

# Fetch sales data
sales_data = fetch_sales_data(API_URL)

if sales_data:
    # Convert data to DataFrame
    sales_df = pd.DataFrame(sales_data)
    # Assuming 'date' column exists and is in a format pandas can understand
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    # Assuming 'total_sales' column exists
    sales_df = sales_df.sort_values('date')

    # Display sales trend chart
    sales_fig = sales_trend_chart(sales_df, date_col='date', value_col='total_sales')
    st.plotly_chart(sales_fig, use_container_width=True)
else:
    st.warning("Could not load sales trend chart due to data fetching error.")

# Add Call to Action
st.markdown('<div class="sub-header">Explore Further</div>', unsafe_allow_html=True)
st.markdown("""
Dive deeper into your retail data by exploring the dedicated sections of the dashboard:
""")

col_cta1, col_cta2, col_cta3 = st.columns(3)

with col_cta1:
    st.link_button("Sales Analysis", url="/Sales_Analysis")

with col_cta2:
    st.link_button("Customer Segments", url="/Customer_Segments")

with col_cta3:
    st.link_button("Product Reviews", url="/Product_Reviews")


# Footer
st.markdown('<div class="footer">Data last updated: 2025-04-24</div>', unsafe_allow_html=True)

# Health check endpoint for Docker
if os.environ.get("DOCKER_ENV") == "true":
    @st.cache_data
    def healthz():
        return {"status": "ok"}
