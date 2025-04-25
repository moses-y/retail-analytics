"""
Main Streamlit dashboard for Retail Analytics
"""
import os
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import utilities
from src.utils.config import get_api_config
from src.utils.metrics import format_metrics

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
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
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

# Footer
st.markdown('<div class="footer">Data last updated: 2025-04-24</div>', unsafe_allow_html=True)

# Health check endpoint for Docker
if os.environ.get("DOCKER_ENV") == "true":
    @st.cache_data
    def healthz():
        return {"status": "ok"}
