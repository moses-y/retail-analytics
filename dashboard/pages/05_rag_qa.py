"""
RAG Q&A Dashboard Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import utilities
from src.utils.config import get_api_config
import requests
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Product Q&A | Retail Analytics",
    page_icon="🤖",
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
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        background-color: #E3F2FD;
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    .bot-message {
        background-color: #f8f9fa;
        margin-right: auto;
        border-bottom-left-radius: 0;
    }
    .source-reference {
        font-size: 0.8rem;
        color: #757575;
        margin-top: 0.5rem;
    }
    .product-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_product_data():
    """Load product data from API or file"""
    try:
        # Try to load from API
        response = requests.get(f"{API_URL}/api/products")
        if response.status_code == 200:
            data = response.json()
            return data
    except Exception as e:
        st.warning(f"Could not load data from API: {e}")

    # Fallback to hardcoded data
    return [
        {
            "id": "P001",
            "name": "TechPro X20",
            "category": "Smartphones",
            "average_rating": 4.2,
            "price": 799.99,
            "features": ["5G", "6.7\" AMOLED Display", "Triple Camera", "128GB Storage"],
            "image_url": "https://placehold.co/300x300?text=TechPro+X20"
        },
        {
            "id": "P002",
            "name": "SmartWatch Pro",
            "category": "Wearables",
            "average_rating": 4.0,
            "price": 249.99,
            "features": ["Heart Rate Monitor", "GPS", "5-day Battery", "Water Resistant"],
            "image_url": "https://placehold.co/300x300?text=SmartWatch+Pro"
        },
        {
            "id": "P003",
            "name": "HomeConnect Hub",
            "category": "Smart Home",
            "average_rating": 3.8,
            "price": 129.99,
            "features": ["Voice Control", "Multi-device Support", "Energy Monitoring", "Easy Setup"],
            "image_url": "https://placehold.co/300x300?text=HomeConnect+Hub"
        }
    ]


@st.cache_data(ttl=60)
def get_rag_response(query, product_id=None):
    """Get RAG response from API"""
    try:
        # Prepare request payload - always include product_id
        payload = {"query": query, "product_id": product_id}

        # Make API request with corrected URL
        api_endpoint = f"{API_URL}/api/rag/query" # Added /api prefix back
        st.info(f"Sending RAG query to: {api_endpoint}")
        response = requests.post(
            api_endpoint,
            json=payload
        )

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {
                "answer": "Sorry, I couldn't process your request. Please try again later.",
                "sources": [],
                "related_products": []
            }
    except Exception as e:
        st.error(f"Error getting RAG response: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "related_products": []
        }


@st.cache_data(ttl=3600)
def get_product_comparison(product_ids):
    """Get product comparison from API"""
    try:
        # Make API request
        response = requests.post(
            f"{API_URL}/api/products/compare",
            json={"product_ids": product_ids}
        )

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting product comparison: {e}")
        return None


# Load product data
products = load_product_data()

# Main content
st.markdown('<div class="main-header">Product Q&A Assistant</div>', unsafe_allow_html=True)

st.markdown("""
Ask questions about our products and get AI-powered answers based on customer reviews and product information.
The assistant can help with product features, comparisons, recommendations, and more.
""")

# Sidebar
with st.sidebar:
    st.markdown("## Products")

    # Product selection
    selected_product = st.selectbox(
        "Filter by Product",
        options=["All Products"] + [p["name"] for p in products],
        index=0
    )

    # Get selected product ID
    selected_product_id = None
    if selected_product != "All Products":
        for p in products:
            if p["name"] == selected_product:
                selected_product_id = p["id"]
                break

    st.markdown("## Example Questions")

    example_questions = [
        "What are the best features of the TechPro X20?",
        "How does the SmartWatch Pro battery life compare to competitors?",
        "What do customers say about the HomeConnect Hub setup process?",
        "Which smartphone has the best camera quality?",
        "What are the common issues with the SmartWatch Pro?"
    ]

    for question in example_questions:
        if st.button(question):
            st.session_state.user_query = question

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize user query
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# Chat interface
st.markdown('<div class="sub-header">Ask a Question</div>', unsafe_allow_html=True)

# Query input
user_query = st.text_input(
    "Enter your question",
    value=st.session_state.user_query,
    key="query_input"
)

# Submit button
submit = st.button("Submit")

# Process query
if submit and user_query:
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Get RAG response
    with st.spinner("Generating answer..."):
        response = get_rag_response(user_query, selected_product_id)

    # Add bot response to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", []),
        "related_products": response.get("related_products", [])
    })

    # Clear input
    st.session_state.user_query = ""

# Display chat history
st.markdown('<div class="sub-header">Conversation</div>', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.info("Ask a question to start the conversation.")
else:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message["content"]}', unsafe_allow_html=True)

            # Display sources
            if "sources" in message and message["sources"]:
                st.markdown('<div class="source-reference"><strong>Sources:</strong></div>', unsafe_allow_html=True)
                for source in message["sources"]:
                    st.markdown(f'<div class="source-reference">- {source["text"]} (Review ID: {source["id"]})</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Display related products
            if "related_products" in message and message["related_products"]:
                st.markdown("#### Related Products")

                cols = st.columns(min(len(message["related_products"]), 3))

                for i, product_id in enumerate(message["related_products"]):
                    # Find product details
                    product = next((p for p in products if p["id"] == product_id), None)

                    if product:
                        with cols[i % 3]:
                            st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
                            st.image(product["image_url"], width=150)
                            st.markdown(f"**{product['name']}**")
                            st.markdown(f"${product['price']:.2f} | ⭐ {product['average_rating']:.1f}")
                            st.markdown("**Features:**")
                            for feature in product["features"]:
                                st.markdown(f"- {feature}")
                            st.markdown('</div>', unsafe_allow_html=True)

# Product comparison
if len(st.session_state.chat_history) > 0:
    last_message = st.session_state.chat_history[-1]
    if last_message["role"] == "assistant" and "related_products" in last_message and len(last_message["related_products"]) > 1:
        st.markdown('<div class="sub-header">Product Comparison</div>', unsafe_allow_html=True)

        # Get products to compare
        products_to_compare = last_message["related_products"][:3]  # Limit to 3 products

        # Get comparison data
        comparison = get_product_comparison(products_to_compare)

        if comparison:
            # Create comparison table
            comparison_data = []

            for feature in comparison["features"]:
                row = {"Feature": feature}

                for product_id, values in comparison["comparison"].items():
                    # Find product name
                    product_name = next((p["name"] for p in products if p["id"] == product_id), product_id)
                    row[product_name] = values.get(feature, "N/A")

                comparison_data.append(row)

            # Display comparison table
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

            # Display radar chart for numeric features
            numeric_features = []
            for feature in comparison["features"]:
                is_numeric = True
                for product_id, values in comparison["comparison"].items():
                    value = values.get(feature, "N/A")
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        is_numeric = False
                        break

                if is_numeric:
                    numeric_features.append(feature)

            if numeric_features:
                st.markdown("### Feature Comparison")

                # Prepare data for radar chart
                radar_data = []

                for product_id, values in comparison["comparison"].items():
                    # Find product name
                    product_name = next((p["name"] for p in products if p["id"] == product_id), product_id)

                    for feature in numeric_features:
                        value = values.get(feature, 0)
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            value = 0

                        radar_data.append({
                            "Product": product_name,
                            "Feature": feature,
                            "Value": value
                        })

                # Create radar chart
                fig = px.line_polar(
                    pd.DataFrame(radar_data),
                    r="Value",
                    theta="Feature",
                    color="Product",
                    line_close=True,
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)

# Clear conversation button
if st.session_state.chat_history:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()
