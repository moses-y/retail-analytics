"""
Reusable filter components for the dashboard
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any, Callable


def date_range_filter(
    label: str = "Date Range",
    default_days: int = 30,
    key: Optional[str] = None
) -> Tuple[datetime, datetime]:
    """
    Create a date range filter

    Args:
        label: Filter label
        default_days: Default number of days to show
        key: Unique key for the filter

    Returns:
        Tuple of (start_date, end_date)
    """
    # Calculate default dates
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=default_days)

    # Create filter
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            f"{label} Start",
            value=start_date,
            key=f"{key}_start" if key else None
        )

    with col2:
        end_date = st.date_input(
            f"{label} End",
            value=end_date,
            key=f"{key}_end" if key else None
        )

    return start_date, end_date


def category_filter(
    data: pd.DataFrame,
    column: str,
    label: str = "Category",
    default: str = "All",
    key: Optional[str] = None
) -> str:
    """
    Create a category filter

    Args:
        data: DataFrame with data
        column: Column name to filter on
        label: Filter label
        default: Default value
        key: Unique key for the filter

    Returns:
        Selected category
    """
    # Get unique categories
    categories = data[column].unique().tolist()
    categories.sort()

    # Add "All" option
    if default == "All":
        categories = ["All"] + categories

    # Create filter
    selected = st.selectbox(
        label,
        options=categories,
        index=categories.index(default) if default in categories else 0,
        key=key
    )

    return selected


def multi_category_filter(
    data: pd.DataFrame,
    column: str,
    label: str = "Categories",
    default: Optional[List[str]] = None,
    key: Optional[str] = None
) -> List[str]:
    """
    Create a multi-category filter

    Args:
        data: DataFrame with data
        column: Column name to filter on
        label: Filter label
        default: Default values
        key: Unique key for the filter

    Returns:
        List of selected categories
    """
    # Get unique categories
    categories = data[column].unique().tolist()
    categories.sort()

    # Set default
    if default is None:
        default = categories

    # Create filter
    selected = st.multiselect(
        label,
        options=categories,
        default=default,
        key=key
    )

    return selected


def numeric_range_filter(
    data: pd.DataFrame,
    column: str,
    label: str = "Range",
    key: Optional[str] = None
) -> Tuple[float, float]:
    """
    Create a numeric range filter

    Args:
        data: DataFrame with data
        column: Column name to filter on
        label: Filter label
        key: Unique key for the filter

    Returns:
        Tuple of (min_value, max_value)
    """
    # Get min and max values
    min_val = float(data[column].min())
    max_val = float(data[column].max())

    # Create filter
    values = st.slider(
        label,
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        key=key
    )

    return values


def apply_filters(
    data: pd.DataFrame,
    filters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply filters to a DataFrame

    Args:
        data: DataFrame with data
        filters: Dictionary of filters to apply

    Returns:
        Filtered DataFrame
    """
    filtered_data = data.copy()

    for column, filter_value in filters.items():
        if filter_value is None:
            continue

        # Handle different filter types
        if isinstance(filter_value, tuple) and len(filter_value) == 2:
            # Range filter
            if isinstance(filter_value[0], datetime):
                # Date range
                filtered_data = filtered_data[
                    (filtered_data[column] >= pd.Timestamp(filter_value[0])) &
                    (filtered_data[column] <= pd.Timestamp(filter_value[1]))
                ]
            else:
                # Numeric range
                filtered_data = filtered_data[
                    (filtered_data[column] >= filter_value[0]) &
                    (filtered_data[column] <= filter_value[1])
                ]
        elif isinstance(filter_value, list):
            # Multi-select filter
            if filter_value:  # Only apply if list is not empty
                filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
        elif filter_value != "All":
            # Single-select filter
            filtered_data = filtered_data[filtered_data[column] == filter_value]

    return filtered_data


def create_filter_sidebar(
    data: pd.DataFrame,
    date_column: Optional[str] = None,
    category_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    apply_button: bool = True
) -> pd.DataFrame:
    """
    Create a filter sidebar and return filtered data

    Args:
        data: DataFrame with data
        date_column: Column name for date filter
        category_columns: List of column names for category filters
        numeric_columns: List of column names for numeric filters
        apply_button: Whether to include an apply button

    Returns:
        Filtered DataFrame
    """
    with st.sidebar:
        st.markdown("## Filters")

        filters = {}

        # Date filter
        if date_column:
            st.markdown("### Date Range")
            start_date, end_date = date_range_filter(
                label="Select date range",
                key="date_filter"
            )
            filters[date_column] = (start_date, end_date)

        # Category filters
        if category_columns:
            st.markdown("### Categories")
            for column in category_columns:
                selected = category_filter(
                    data,
                    column=column,
                    label=column.replace('_', ' ').title(),
                    key=f"cat_filter_{column}"
                )
                filters[column] = selected

        # Numeric filters
        if numeric_columns:
            st.markdown("### Numeric Filters")
            for column in numeric_columns:
                values = numeric_range_filter(
                    data,
                    column=column,
                    label=column.replace('_', ' ').title(),
                    key=f"num_filter_{column}"
                )
                filters[column] = values

        # Apply button
        if apply_button:
            if st.button("Apply Filters"):
                filtered_data = apply_filters(data, filters)
                st.session_state.filtered_data = filtered_data
        else:
            filtered_data = apply_filters(data, filters)
            st.session_state.filtered_data = filtered_data

        # Reset button
        if st.button("Reset Filters"):
            st.session_state.filtered_data = data

    # Return filtered data
    if hasattr(st.session_state, 'filtered_data'):
        return st.session_state.filtered_data
    else:
        st.session_state.filtered_data = data
        return data