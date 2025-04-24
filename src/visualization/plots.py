"""
Visualization functions for retail analytics
"""
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set default style for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Default colors for consistent visualizations
DEFAULT_COLORS = px.colors.qualitative.Plotly


def set_plot_style(style: str = 'seaborn-v0_8-whitegrid') -> None:
    """
    Set global plot style

    Args:
        style: Matplotlib style name
    """
    plt.style.use(style)


def save_plot(
    fig: Any,
    output_path: str,
    dpi: int = 300,
    bbox_inches: str = 'tight',
    format: Optional[str] = None
) -> None:
    """
    Save plot to file

    Args:
        fig: Matplotlib figure
        output_path: Path to save the plot
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box in inches
        format: File format (optional)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, format=format)
    logger.info(f"Plot saved to {output_path}")


def plot_sales_by_category(
    df: pd.DataFrame,
    date_column: str = 'date',
    category_column: str = 'category',
    sales_column: str = 'total_sales',
    title: str = 'Total Sales by Category',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot sales by category

    Args:
        df: DataFrame with sales data
        date_column: Column containing dates
        category_column: Column containing categories
        sales_column: Column containing sales values
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting sales by category")

    # Group by date and category
    sales_by_category = df.groupby([date_column, category_column])[sales_column].sum().reset_index()

    if interactive:
        # Create Plotly figure
        fig = px.line(
            sales_by_category,
            x=date_column,
            y=sales_column,
            color=category_column,
            title=title,
            labels={
                date_column: 'Date',
                sales_column: 'Total Sales',
                category_column: 'Category'
            }
        )

        fig.update_layout(
            legend_title_text='Category',
            xaxis_title='Date',
            yaxis_title='Total Sales',
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(12, 6))

        # Plot each category
        for category, group in sales_by_category.groupby(category_column):
            plt.plot(group[date_column], group[sales_column], label=category)

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.legend(title='Category')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_sales_distribution(
    df: pd.DataFrame,
    sales_column: str = 'total_sales',
    group_column: Optional[str] = None,
    bins: int = 30,
    title: str = 'Sales Distribution',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot sales distribution

    Args:
        df: DataFrame with sales data
        sales_column: Column containing sales values
        group_column: Column to group by (optional)
        bins: Number of histogram bins
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting sales distribution")

    if interactive:
        # Create Plotly figure
        if group_column and group_column in df.columns:
            fig = px.histogram(
                df,
                x=sales_column,
                color=group_column,
                nbins=bins,
                title=title,
                opacity=0.7,
                barmode='overlay',
                labels={sales_column: 'Sales'}
            )
        else:
            fig = px.histogram(
                df,
                x=sales_column,
                nbins=bins,
                title=title,
                labels={sales_column: 'Sales'}
            )

        fig.update_layout(
            xaxis_title='Sales',
            yaxis_title='Count',
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(10, 6))

        if group_column and group_column in df.columns:
            # Plot for each group
            for group, data in df.groupby(group_column):
                sns.histplot(data[sales_column], bins=bins, label=group, alpha=0.5, kde=True)
            plt.legend(title=group_column)
        else:
            # Plot overall distribution
            sns.histplot(df[sales_column], bins=bins, kde=True)

        plt.title(title)
        plt.xlabel('Sales')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_sales_heatmap(
    df: pd.DataFrame,
    date_column: str = 'date',
    category_column: str = 'category',
    sales_column: str = 'total_sales',
    title: str = 'Sales Heatmap',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot sales heatmap by date and category

    Args:
        df: DataFrame with sales data
        date_column: Column containing dates
        category_column: Column containing categories
        sales_column: Column containing sales values
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting sales heatmap")

    # Group by date and category
    pivot_data = df.pivot_table(
        index=date_column,
        columns=category_column,
        values=sales_column,
        aggfunc='sum'
    )

    if interactive:
        # Create Plotly figure
        fig = px.imshow(
            pivot_data,
            title=title,
            labels=dict(x=category_column, y=date_column, color=sales_column),
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title=category_column,
            yaxis_title=date_column,
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(12, 8))

        sns.heatmap(
            pivot_data,
            cmap='viridis',
            annot=False,
            fmt='.0f',
            linewidths=0.5,
            cbar_kws={'label': sales_column}
        )

        plt.title(title)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_online_vs_instore_sales(
    df: pd.DataFrame,
    date_column: str = 'date',
    online_column: str = 'online_sales',
    instore_column: str = 'in_store_sales',
    title: str = 'Online vs In-Store Sales',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot online vs in-store sales over time

    Args:
        df: DataFrame with sales data
        date_column: Column containing dates
        online_column: Column containing online sales
        instore_column: Column containing in-store sales
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting online vs in-store sales")

    # Group by date
    sales_by_date = df.groupby(date_column)[[online_column, instore_column]].sum().reset_index()

    if interactive:
        # Create Plotly figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sales_by_date[date_column],
            y=sales_by_date[online_column],
            mode='lines',
            name='Online Sales',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=sales_by_date[date_column],
            y=sales_by_date[instore_column],
            mode='lines',
            name='In-Store Sales',
            line=dict(color='green', width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Sales',
            legend_title='Channel',
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(12, 6))

        plt.plot(sales_by_date[date_column], sales_by_date[online_column], label='Online Sales', linewidth=2)
        plt.plot(sales_by_date[date_column], sales_by_date[instore_column], label='In-Store Sales', linewidth=2)

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_sales_by_weather(
    df: pd.DataFrame,
    weather_column: str = 'weather',
    sales_column: str = 'total_sales',
    title: str = 'Average Sales by Weather Condition',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot average sales by weather condition

    Args:
        df: DataFrame with sales data
        weather_column: Column containing weather conditions
        sales_column: Column containing sales values
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting sales by weather condition")

    # Group by weather
    sales_by_weather = df.groupby(weather_column)[sales_column].mean().reset_index()

    if interactive:
        # Create Plotly figure
        fig = px.bar(
            sales_by_weather,
            x=weather_column,
            y=sales_column,
            title=title,
            color=weather_column,
            labels={
                weather_column: 'Weather Condition',
                sales_column: 'Average Sales'
            }
        )

        fig.update_layout(
            xaxis_title='Weather Condition',
            yaxis_title='Average Sales',
            showlegend=False,
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(10, 6))

        sns.barplot(x=weather_column, y=sales_column, data=sales_by_weather)

        plt.title(title)
        plt.xlabel('Weather Condition')
        plt.ylabel('Average Sales')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    title: str = 'Feature Importance',
    top_n: int = 20,
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot feature importance

    Args:
        importance: Array of feature importance values
        feature_names: List of feature names
        title: Plot title
        top_n: Number of top features to show
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting feature importance")

    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    # Sort by importance and get top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)

    if interactive:
        # Create Plotly figure
        fig = px.bar(
            importance_df,
            y='Feature',
            x='Importance',
            title=title,
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(10, max(6, top_n * 0.3)))

        sns.barplot(x='Importance', y='Feature', data=importance_df)

        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting confusion matrix")

    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    if interactive:
        # Create Plotly figure
        fig = px.imshow(
            cm,
            x=classes,
            y=classes,
            color_continuous_scale='Viridis',
            labels=dict(x='Predicted', y='True', color='Count'),
            title=title
        )

        # Add text annotations
        annotations = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(color='white' if cm[i, j] > cm.max() / 2 else 'black')
                    )
                )

        fig.update_layout(
            annotations=annotations,
            xaxis_title='Predicted',
            yaxis_title='True',
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='viridis',
            xticklabels=classes,
            yticklabels=classes
        )

        plt.title(title)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = 'ROC Curve',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot ROC curve

    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates of the positive class)
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting ROC curve")

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    if interactive:
        # Create Plotly figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white',
            legend=dict(x=0.7, y=0.1)
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(8, 6))

        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random')

        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = 'Precision-Recall Curve',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot precision-recall curve

    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates of the positive class)
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting precision-recall curve")

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    if interactive:
        # Create Plotly figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR curve (AUC = {pr_auc:.3f})',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Recall',
            yaxis_title='Precision',
            template='plotly_white',
            legend=dict(x=0.7, y=0.1)
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(8, 6))

        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')

        plt.title(title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_sentiment_distribution(
    df: pd.DataFrame,
    sentiment_column: str = 'sentiment',
    group_column: Optional[str] = None,
    title: str = 'Sentiment Distribution',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot sentiment distribution

    Args:
        df: DataFrame with sentiment data
        sentiment_column: Column containing sentiment labels
        group_column: Column to group by (optional)
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting sentiment distribution")

    if group_column and group_column in df.columns:
        # Group by sentiment and group column
        sentiment_counts = df.groupby([group_column, sentiment_column]).size().reset_index(name='count')
    else:
        # Group by sentiment only
        sentiment_counts = df[sentiment_column].value_counts().reset_index()
        sentiment_counts.columns = [sentiment_column, 'count']

    if interactive:
        # Create Plotly figure
        if group_column and group_column in df.columns:
            fig = px.bar(
                sentiment_counts,
                x=sentiment_column,
                y='count',
                color=group_column,
                title=title,
                barmode='group',
                labels={
                    sentiment_column: 'Sentiment',
                    'count': 'Count',
                    group_column: group_column
                }
            )
        else:
            fig = px.bar(
                sentiment_counts,
                x=sentiment_column,
                y='count',
                title=title,
                color=sentiment_column,
                labels={
                    sentiment_column: 'Sentiment',
                    'count': 'Count'
                }
            )

        fig.update_layout(
            xaxis_title='Sentiment',
            yaxis_title='Count',
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(10, 6))

        if group_column and group_column in df.columns:
            # Plot for each group
            sns.countplot(x=sentiment_column, hue=group_column, data=df)
        else:
            # Plot overall distribution
            sns.countplot(x=sentiment_column, data=df)

        plt.title(title)
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_feature_sentiment(
    feature_sentiment: pd.DataFrame,
    feature_column: str = 'feature',
    sentiment_column: str = 'sentiment_score',
    count_column: str = 'count',
    title: str = 'Feature Sentiment Analysis',
    top_n: int = 15,
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot feature sentiment analysis

    Args:
        feature_sentiment: DataFrame with feature sentiment data
        feature_column: Column containing feature names
        sentiment_column: Column containing sentiment scores
        count_column: Column containing mention counts
        title: Plot title
        top_n: Number of top features to show
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting feature sentiment analysis")

    # Sort by count and get top N
    top_features = feature_sentiment.sort_values(count_column, ascending=False).head(top_n)

    if interactive:
        # Create Plotly figure
        fig = px.bar(
            top_features,
            y=feature_column,
            x=sentiment_column,
            color=sentiment_column,
            color_continuous_scale='RdYlGn',
            title=title,
            orientation='h',
            text=count_column,
            labels={
                feature_column: 'Feature',
                sentiment_column: 'Sentiment Score',
                count_column: 'Mentions'
            }
        )

        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='Sentiment Score',
            yaxis_title='Feature',
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(10, max(6, top_n * 0.3)))

        # Create colormap
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(top_features[sentiment_column].min(), top_features[sentiment_column].max())

        # Plot bars
        bars = plt.barh(top_features[feature_column], top_features[sentiment_column])

        # Color bars by sentiment
        for i, bar in enumerate(bars):
            bar.set_color(cmap(norm(top_features[sentiment_column].iloc[i])))

        # Add count labels
        for i, (feature, score, count) in enumerate(zip(
            top_features[feature_column],
            top_features[sentiment_column],
            top_features[count_column]
        )):
            plt.text(
                score + (0.05 if score >= 0 else -0.05),
                i,
                f"{count}",
                va='center',
                ha='left' if score >= 0 else 'right'
            )

        plt.title(title)
        plt.xlabel('Sentiment Score')
        plt.ylabel('Feature')
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_rating_distribution(
    df: pd.DataFrame,
    rating_column: str = 'rating',
    group_column: Optional[str] = None,
    title: str = 'Rating Distribution',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot rating distribution

    Args:
        df: DataFrame with rating data
        rating_column: Column containing ratings
        group_column: Column to group by (optional)
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting rating distribution")

    if interactive:
        # Create Plotly figure
        if group_column and group_column in df.columns:
            fig = px.histogram(
                df,
                x=rating_column,
                color=group_column,
                title=title,
                barmode='group',
                labels={
                    rating_column: 'Rating',
                    group_column: group_column
                }
            )
        else:
            fig = px.histogram(
                df,
                x=rating_column,
                title=title,
                labels={
                    rating_column: 'Rating'
                }
            )

        fig.update_layout(
            xaxis_title='Rating',
            yaxis_title='Count',
            template='plotly_white'
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(10, 6))

        if group_column and group_column in df.columns:
            # Plot for each group
            sns.countplot(x=rating_column, hue=group_column, data=df)
        else:
            # Plot overall distribution
            sns.countplot(x=rating_column, data=df)

        plt.title(title)
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_forecast_vs_actual(
    dates: List[Any],
    actual: np.ndarray,
    forecast: np.ndarray,
    title: str = 'Forecast vs Actual',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot forecast vs actual values

    Args:
        dates: List of dates
        actual: Array of actual values
        forecast: Array of forecasted values
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting forecast vs actual values")

    if interactive:
        # Create Plotly figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=dates,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            legend=dict(x=0.01, y=0.99)
        )

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure
        plt.figure(figsize=(12, 6))

        plt.plot(dates, actual, label='Actual', linewidth=2)
        plt.plot(dates, forecast, label='Forecast', linewidth=2)

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig = plt.gcf()

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


def plot_forecast_components(
    dates: List[Any],
    trend: np.ndarray,
    seasonal: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    title: str = 'Forecast Components',
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Any:
    """
    Plot forecast components (trend, seasonal, residual)

    Args:
        dates: List of dates
        trend: Array of trend values
        seasonal: Array of seasonal values (optional)
        residual: Array of residual values (optional)
        title: Plot title
        output_path: Path to save the plot (optional)
        interactive: Whether to use Plotly (True) or Matplotlib (False)

    Returns:
        Plot figure
    """
    logger.info("Plotting forecast components")

    if interactive:
        # Create Plotly figure with subplots
        n_plots = 1 + (seasonal is not None) + (residual is not None)
        fig = make_subplots(rows=n_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05)

        # Add trend
        fig.add_trace(
            go.Scatter(x=dates, y=trend, mode='lines', name='Trend', line=dict(color='blue')),
            row=1, col=1
        )

        # Add seasonal if provided
        if seasonal is not None:
            fig.add_trace(
                go.Scatter(x=dates, y=seasonal, mode='lines', name='Seasonal', line=dict(color='green')),
                row=2, col=1
            )

        # Add residual if provided
        if residual is not None:
            fig.add_trace(
                go.Scatter(x=dates, y=residual, mode='lines', name='Residual', line=dict(color='red')),
                row=n_plots, col=1
            )

        fig.update_layout(
            title=title,
            template='plotly_white',
            height=300 * n_plots,
            showlegend=True
        )

        # Update y-axis titles
        fig.update_yaxes(title_text='Trend', row=1, col=1)
        if seasonal is not None:
            fig.update_yaxes(title_text='Seasonal', row=2, col=1)
        if residual is not None:
            fig.update_yaxes(title_text='Residual', row=n_plots, col=1)

        # Update x-axis title
        fig.update_xaxes(title_text='Date', row=n_plots, col=1)

        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    else:
        # Create Matplotlib figure with subplots
        n_plots = 1 + (seasonal is not None) + (residual is not None)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)

        if n_plots == 1:
            axes = [axes]

        # Plot trend
        axes[0].plot(dates, trend, label='Trend', color='blue')
        axes[0].set_ylabel('Trend')
        axes[0].grid(True, alpha=0.3)

        # Plot seasonal if provided
        if seasonal is not None:
            idx = 1
            axes[idx].plot(dates, seasonal, label='Seasonal', color='green')
            axes[idx].set_ylabel('Seasonal')
            axes[idx].grid(True, alpha=0.3)

        # Plot residual if provided
        if residual is not None:
            idx = 1 + (seasonal is not None)
            axes[idx].plot(dates, residual, label='Residual', color='red')
            axes[idx].set_ylabel('Residual')
            axes[idx].grid(True, alpha=0.3)

        # Set title and x-axis label
        fig.suptitle(title)
        axes[-1].set_xlabel('Date')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Save if output path is provided
        if output_path:
            save_plot(fig, output_path)

        plt.close()

    return fig


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Generate visualizations for retail analytics')
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--output_dir', required=True, help='Directory to save visualizations')
    parser.add_argument('--interactive', action='store_true', help='Generate interactive Plotly visualizations')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate visualizations
    if 'category' in df.columns and 'total_sales' in df.columns:
        plot_sales_by_category(
            df,
            output_path=os.path.join(args.output_dir, 'sales_by_category.html' if args.interactive else 'sales_by_category.png'),
            interactive=args.interactive
        )

    if 'total_sales' in df.columns:
        plot_sales_distribution(
            df,
            output_path=os.path.join(args.output_dir, 'sales_distribution.html' if args.interactive else 'sales_distribution.png'),
            interactive=args.interactive
        )

    if 'online_sales' in df.columns and 'in_store_sales' in df.columns:
        plot_online_vs_instore_sales(
            df,
            output_path=os.path.join(args.output_dir, 'online_vs_instore.html' if args.interactive else 'online_vs_instore.png'),
            interactive=args.interactive
        )

    if 'weather' in df.columns and 'total_sales' in df.columns:
        plot_sales_by_weather(
            df,
            output_path=os.path.join(args.output_dir, 'sales_by_weather.html' if args.interactive else 'sales_by_weather.png'),
            interactive=args.interactive
        )

    logger.info(f"Visualizations saved to {args.output_dir}")