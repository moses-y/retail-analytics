"""
Reusable chart components for the dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Union, Any

# Import visualization utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.visualization.plots import (
    # plot_sales_trend, # Removed - Function does not exist in plots.py
    plot_sales_by_category,
    # plot_customer_segments, # Removed - Function does not exist in plots.py
    plot_feature_importance,
    plot_sentiment_distribution
)


def sales_trend_chart(
    data: pd.DataFrame,
    date_col: str = 'date',
    value_col: str = 'total_sales',
    title: str = 'Sales Trend',
    height: int = 400
    # use_plotly: bool = True # Removed - Only Plotly implementation available
) -> go.Figure:
    """
    Create a sales trend chart (Plotly version)

    Args:
        data: DataFrame with sales data
        date_col: Column name for dates
        value_col: Column name for values
        title: Chart title
        height: Chart height
        # use_plotly: Whether to use Plotly (True) or Matplotlib (False) # Removed

    Returns:
        Plotly figure
    """
    # Always use Plotly implementation
    fig = px.line(
        data,
        x=date_col,
        y=value_col,
        title=title,
        height=height,
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        legend_title="Legend",
        hovermode="x unified"
    )

    return fig
    # else: # Removed - plot_sales_trend does not exist
    #     # Use matplotlib version from visualization module
    #     fig = plot_sales_trend(data, date_column=date_col, value_column=value_col, title=title)
    #     return fig


def sales_by_category_chart(
    data: pd.DataFrame,
    category_col: str = 'category',
    value_col: str = 'total_sales',
    title: str = 'Sales by Category',
    height: int = 400,
    use_plotly: bool = True
) -> go.Figure:
    """
    Create a sales by category chart

    Args:
        data: DataFrame with sales data
        category_col: Column name for categories
        value_col: Column name for values
        title: Chart title
        height: Chart height
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plotly figure
    """
    if use_plotly:
        # Group by category if needed
        if data.shape[0] > len(data[category_col].unique()):
            data = data.groupby(category_col)[value_col].sum().reset_index()

        fig = px.bar(
            data,
            x=category_col,
            y=value_col,
            title=title,
            height=height,
            color=category_col,
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Sales ($)",
            showlegend=False,
            hovermode="x unified"
        )

        return fig
    else:
        # Use matplotlib version from visualization module
        fig = plot_sales_by_category(data, category_column=category_col, value_column=value_col, title=title)
        return fig


def customer_segments_chart(
    data: pd.DataFrame,
    features: List[str],
    cluster_col: str = 'cluster',
    title: str = 'Customer Segments',
    height: int = 500
    # use_plotly: bool = True # Removed - Only Plotly implementation available
) -> go.Figure:
    """
    Create a customer segments chart (Plotly version)

    Args:
        data: DataFrame with customer data and cluster assignments
        features: List of features to display
        cluster_col: Column name for cluster assignments
        title: Chart title
        height: Chart height
        # use_plotly: Whether to use Plotly (True) or Matplotlib (False) # Removed

    Returns:
        Plotly figure
    """
    # Always use Plotly implementation
    # Calculate mean values for each cluster and feature
    cluster_profiles = data.groupby(cluster_col)[features].mean().reset_index()

    # Melt the dataframe for easier plotting
    melted = pd.melt(
        cluster_profiles,
        id_vars=[cluster_col],
        value_vars=features,
        var_name='Feature',
        value_name='Value'
    )

    # Create radar chart
    fig = px.line_polar(
        melted,
        r='Value',
        theta='Feature',
        color=cluster_col,
        line_close=True,
        title=title,
        height=height,
        template="plotly_white"
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, melted['Value'].max() * 1.1]
            )
        ),
        showlegend=True
    )

    return fig
    # else: # Removed - plot_customer_segments does not exist
    #     # Use matplotlib version from visualization module
    #     fig = plot_customer_segments(data, features=features, cluster_column=cluster_col, title=title)
    #     return fig


def feature_importance_chart(
    feature_names: List[str],
    importance_values: List[float],
    title: str = 'Feature Importance',
    height: int = 400,
    use_plotly: bool = True
) -> go.Figure:
    """
    Create a feature importance chart

    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        title: Chart title
        height: Chart height
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plotly figure
    """
    if use_plotly:
        # Create dataframe
        data = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        })

        # Sort by importance
        data = data.sort_values('Importance', ascending=True)

        fig = px.bar(
            data,
            x='Importance',
            y='Feature',
            title=title,
            height=height,
            orientation='h',
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Feature",
            showlegend=False,
            hovermode="y unified"
        )

        return fig
    else:
        # Use matplotlib version from visualization module
        fig = plot_feature_importance(feature_names, importance_values, title=title)
        return fig


def sentiment_distribution_chart(
    data: pd.DataFrame,
    entity_col: str,
    sentiment_col: str = 'sentiment',
    title: str = 'Sentiment Distribution',
    height: int = 400,
    use_plotly: bool = True
) -> go.Figure:
    """
    Create a sentiment distribution chart

    Args:
        data: DataFrame with sentiment data
        entity_col: Column name for entities (products, features, etc.)
        sentiment_col: Column name for sentiment values
        title: Chart title
        height: Chart height
        use_plotly: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plotly figure
    """
    if use_plotly:
        # Group by entity and calculate sentiment counts
        sentiment_counts = data.groupby([entity_col, sentiment_col]).size().reset_index(name='count')

        fig = px.bar(
            sentiment_counts,
            x=entity_col,
            y='count',
            color=sentiment_col,
            title=title,
            height=height,
            barmode='group',
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title=entity_col.capitalize(),
            yaxis_title="Count",
            legend_title="Sentiment",
            hovermode="x unified"
        )

        return fig
    else:
        # Use matplotlib version from visualization module
        fig = plot_sentiment_distribution(data, entity_column=entity_col, sentiment_column=sentiment_col, title=title)
        return fig


def forecast_chart(
    actual_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    date_col: str = 'date',
    actual_col: str = 'total_sales',
    forecast_col: str = 'forecast',
    title: str = 'Sales Forecast',
    height: int = 400
) -> go.Figure:
    """
    Create a forecast chart with actual and predicted values

    Args:
        actual_data: DataFrame with actual data
        forecast_data: DataFrame with forecast data
        date_col: Column name for dates
        actual_col: Column name for actual values
        forecast_col: Column name for forecast values
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Add actual data
    fig.add_trace(
        go.Scatter(
            x=actual_data[date_col],
            y=actual_data[actual_col],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        )
    )

    # Add forecast data
    fig.add_trace(
        go.Scatter(
            x=forecast_data[date_col],
            y=forecast_data[forecast_col],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        legend_title="Legend",
        hovermode="x unified",
        height=height,
        template="plotly_white"
    )

    return fig


def metric_card(
    label: str,
    value: Union[float, int, str],
    delta: Optional[Union[float, int]] = None,
    delta_suffix: str = "%",
    prefix: str = "",
    suffix: str = "",
    help_text: Optional[str] = None
) -> None:
    """
    Display a metric card in Streamlit

    Args:
        label: Metric label
        value: Metric value
        delta: Delta value (optional)
        delta_suffix: Suffix for delta value
        prefix: Prefix for value
        suffix: Suffix for value
        help_text: Help text to display on hover
    """
    if delta is not None:
        st.metric(
            label=label,
            value=f"{prefix}{value}{suffix}",
            delta=f"{delta}{delta_suffix}",
            help=help_text
        )
    else:
        st.metric(
            label=label,
            value=f"{prefix}{value}{suffix}",
            help=help_text
        )


def create_kpi_row(metrics: List[Dict[str, Any]]) -> None:
    """
    Create a row of KPI metrics

    Args:
        metrics: List of dictionaries with metric information
    """
    cols = st.columns(len(metrics))

    for i, metric in enumerate(metrics):
        with cols[i]:
            metric_card(
                label=metric.get('label', ''),
                value=metric.get('value', 0),
                delta=metric.get('delta'),
                delta_suffix=metric.get('delta_suffix', '%'),
                prefix=metric.get('prefix', ''),
                suffix=metric.get('suffix', ''),
                help_text=metric.get('help_text')
            )


def create_comparison_chart(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    labels: List[str],
    title: str = 'Comparison',
    height: int = 400
) -> go.Figure:
    """
    Create a comparison chart with multiple series

    Args:
        data: DataFrame with data
        x_col: Column name for x-axis
        y_cols: List of column names for y-axis
        labels: List of labels for each y-column
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for y_col, label in zip(y_cols, labels):
        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines',
                name=label
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col.capitalize(),
        yaxis_title="Value",
        legend_title="Legend",
        hovermode="x unified",
        height=height,
        template="plotly_white"
    )

    return fig
