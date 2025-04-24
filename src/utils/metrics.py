"""
Evaluation metrics for retail analytics models
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multioutput: str = 'uniform_average'
) -> Dict[str, float]:
    """
    Calculate regression metrics

    Args:
        y_true: True values
        y_pred: Predicted values
        multioutput: Parameter for multioutput regression

    Returns:
        Dictionary of metrics
    """
    logger.info("Calculating regression metrics")

    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred, multioutput=multioutput)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, multioutput=multioutput)
    
    # Handle zero values for MAPE
    if np.any(y_true == 0):
        logger.warning("Zero values detected in y_true, using masked MAPE calculation")
        mask = y_true != 0
        if np.any(mask):
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            mape = np.nan
    else:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    
    r2 = r2_score(y_true, y_pred, multioutput=multioutput)

    # Calculate additional metrics
    # Mean Bias Error
    mbe = np.mean(y_pred - y_true)
    
    # Normalized RMSE
    if np.mean(y_true) != 0:
        nrmse = rmse / np.mean(y_true)
    else:
        nrmse = np.nan

    # Return metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'mbe': mbe,
        'nrmse': nrmse
    }

    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'weighted',
    labels: Optional[List] = None
) -> Dict[str, Any]:
    """
    Calculate classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        average: Averaging method for multi-class metrics
        labels: List of labels

    Returns:
        Dictionary of metrics
    """
    logger.info("Calculating classification metrics")

    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique classes
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate classification report
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    # Calculate ROC AUC if probabilities are provided
    auc = None
    if y_prob is not None:
        try:
            # For binary classification
            if len(labels) == 2:
                # Ensure y_prob is for the positive class
                if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                    pos_idx = np.where(labels == 1)[0]
                    if len(pos_idx) > 0:
                        y_prob_pos = y_prob[:, pos_idx[0]]
                    else:
                        y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob
                
                auc = roc_auc_score(y_true, y_prob_pos)
            # For multi-class
            else:
                # Convert y_true to one-hot encoding
                y_true_bin = np.zeros((len(y_true), len(labels)))
                for i, label in enumerate(labels):
                    y_true_bin[:, i] = (y_true == label)
                
                auc = roc_auc_score(y_true_bin, y_prob, average=average, multi_class='ovr')
        except Exception as e:
            logger.warning(f"Error calculating ROC AUC: {e}")
            auc = None

    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }

    if auc is not None:
        metrics['roc_auc'] = auc

    return metrics


def calculate_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate clustering metrics

    Args:
        X: Feature matrix
        labels: Cluster labels
        centroids: Cluster centroids (optional)

    Returns:
        Dictionary of metrics
    """
    logger.info("Calculating clustering metrics")

    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score
    )

    # Ensure arrays are numpy arrays
    X = np.array(X)
    labels = np.array(labels)

    # Calculate metrics
    metrics = {}

    # Silhouette score
    try:
        silhouette = silhouette_score(X, labels)
        metrics['silhouette'] = silhouette
    except Exception as e:
        logger.warning(f"Error calculating silhouette score: {e}")

    # Calinski-Harabasz index
    try:
        calinski_harabasz = calinski_harabasz_score(X, labels)
        metrics['calinski_harabasz'] = calinski_harabasz
    except Exception as e:
        logger.warning(f"Error calculating Calinski-Harabasz index: {e}")

    # Davies-Bouldin index
    try:
        davies_bouldin = davies_bouldin_score(X, labels)
        metrics['davies_bouldin'] = davies_bouldin
    except Exception as e:
        logger.warning(f"Error calculating Davies-Bouldin index: {e}")

    # Inertia (sum of squared distances to closest centroid)
    if centroids is not None:
        try:
            inertia = 0
            for i, label in enumerate(labels):
                inertia += np.sum((X[i] - centroids[label]) ** 2)
            metrics['inertia'] = inertia
        except Exception as e:
            logger.warning(f"Error calculating inertia: {e}")

    return metrics


def calculate_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate forecast-specific metrics

    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Dates corresponding to values (optional)

    Returns:
        Dictionary of metrics
    """
    logger.info("Calculating forecast metrics")

    # Get basic regression metrics
    metrics = calculate_regression_metrics(y_true, y_pred)

    # Calculate additional forecast-specific metrics
    
    # Mean Absolute Scaled Error (MASE)
    # For MASE, we need the seasonal difference of y_true
    # We'll use a simple lag-1 difference as the naive forecast
    if len(y_true) > 1:
        naive_errors = np.abs(np.diff(y_true))
        if np.sum(naive_errors) > 0:
            mae = metrics['mae']
            mase = mae / np.mean(naive_errors)
            metrics['mase'] = mase
    
    # Tracking Signal (cumulative sum of errors / MAD)
    errors = y_pred - y_true
    mad = np.mean(np.abs(errors))
    if mad > 0:
        tracking_signal = np.sum(errors) / (len(errors) * mad)
        metrics['tracking_signal'] = tracking_signal
    
    # Theil's U statistic
    if np.sum(y_true ** 2) > 0:
        theil_u = np.sqrt(np.sum((y_pred - y_true) ** 2) / np.sum(y_true ** 2))
        metrics['theil_u'] = theil_u
    
    # Calculate metrics by time period if dates are provided
    if dates is not None:
        try:
            # Convert dates to pandas datetime if they're not already
            if not isinstance(dates, pd.DatetimeIndex):
                dates = pd.to_datetime(dates)
            
            # Create DataFrame with dates, true values, and predictions
            df = pd.DataFrame({
                'date': dates,
                'y_true': y_true,
                'y_pred': y_pred,
                'error': y_pred - y_true,
                'abs_error': np.abs(y_pred - y_true),
                'squared_error': (y_pred - y_true) ** 2
            })
            
            # Add time components
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['weekday'] = df['date'].dt.weekday
            
            # Calculate metrics by month
            monthly_metrics = df.groupby('month').agg({
                'abs_error': 'mean',
                'squared_error': 'mean'
            }).rename(columns={
                'abs_error': 'mae',
                'squared_error': 'mse'
            })
            monthly_metrics['rmse'] = np.sqrt(monthly_metrics['mse'])
            
            # Calculate metrics by weekday
            weekday_metrics = df.groupby('weekday').agg({
                'abs_error': 'mean',
                'squared_error': 'mean'
            }).rename(columns={
                'abs_error': 'mae',
                'squared_error': 'mse'
            })
            weekday_metrics['rmse'] = np.sqrt(weekday_metrics['mse'])
            
            metrics['monthly_metrics'] = monthly_metrics.to_dict()
            metrics['weekday_metrics'] = weekday_metrics.to_dict()
            
        except Exception as e:
            logger.warning(f"Error calculating time-based metrics: {e}")
    
    return metrics


def calculate_sentiment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    text_data: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate sentiment analysis metrics

    Args:
        y_true: True sentiment labels
        y_pred: Predicted sentiment labels
        y_prob: Predicted probabilities (optional)
        text_data: Original text data (optional)

    Returns:
        Dictionary of metrics
    """
    logger.info("Calculating sentiment analysis metrics")

    # Get basic classification metrics
    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
    
    # Calculate additional sentiment-specific metrics
    
    # Sentiment accuracy by length
    if text_data is not None:
        try:
            # Create DataFrame with text, true labels, and predictions
            df = pd.DataFrame({
                'text': text_data,
                'y_true': y_true,
                'y_pred': y_pred,
                'correct': y_true == y_pred
            })
            
            # Calculate text length
            df['length'] = df['text'].apply(len)
            
            # Bin text length
            df['length_bin'] = pd.cut(df['length'], bins=[0, 50, 100, 200, 500, np.inf], 
                                      labels=['0-50', '51-100', '101-200', '201-500', '500+'])
            
            # Calculate accuracy by length bin
            accuracy_by_length = df.groupby('length_bin')['correct'].mean().to_dict()
            metrics['accuracy_by_length'] = accuracy_by_length
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment metrics by text length: {e}")
    
    return metrics


def calculate_recommendation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 5
) -> Dict[str, float]:
    """
    Calculate recommendation system metrics

    Args:
        y_true: True ratings or relevance scores
        y_pred: Predicted ratings or relevance scores
        k: Number of top items to consider for precision/recall

    Returns:
        Dictionary of metrics
    """
    logger.info("Calculating recommendation metrics")

    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate basic regression metrics
    basic_metrics = calculate_regression_metrics(y_true, y_pred)
    
    # Initialize metrics dictionary with basic metrics
    metrics = {
        'rmse': basic_metrics['rmse'],
        'mae': basic_metrics['mae']
    }
    
    # Calculate recommendation-specific metrics
    
    # Precision@k and Recall@k
    # For these metrics, we need to convert ratings to binary relevance
    # We'll consider items with ratings above the mean as relevant
    try:
        threshold = np.mean(y_true)
        y_true_binary = y_true > threshold
        
        # Get indices of top-k predicted items
        top_k_indices = np.argsort(y_pred)[-k:]
        
        # Calculate precision@k
        precision_at_k = np.sum(y_true_binary[top_k_indices]) / k
        metrics['precision_at_k'] = precision_at_k
        
        # Calculate recall@k
        if np.sum(y_true_binary) > 0:
            recall_at_k = np.sum(y_true_binary[top_k_indices]) / np.sum(y_true_binary)
            metrics['recall_at_k'] = recall_at_k
        
        # Calculate F1@k
        if precision_at_k + recall_at_k > 0:
            f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
            metrics['f1_at_k'] = f1_at_k
    
    except Exception as e:
        logger.warning(f"Error calculating precision/recall@k: {e}")
    
    # Normalized Discounted Cumulative Gain (NDCG)
    try:
        # Sort items by predicted ratings
        pred_indices = np.argsort(y_pred)[::-1]
        
        # Calculate DCG
        dcg = y_true[pred_indices[0]]
        for i in range(1, len(pred_indices)):
            dcg += y_true[pred_indices[i]] / np.log2(i + 1 + 1)
        
        # Calculate ideal DCG
        ideal_indices = np.argsort(y_true)[::-1]
        idcg = y_true[ideal_indices[0]]
        for i in range(1, len(ideal_indices)):
            idcg += y_true[ideal_indices[i]] / np.log2(i + 1 + 1)
        
        # Calculate NDCG
        if idcg > 0:
            ndcg = dcg / idcg
            metrics['ndcg'] = ndcg
    
    except Exception as e:
        logger.warning(f"Error calculating NDCG: {e}")
    
    return metrics


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'regression',
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate model performance

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        task_type: Type of task ('regression', 'classification', 'clustering', 'forecast', 'sentiment', 'recommendation')
        **kwargs: Additional arguments for specific metrics

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating model for {task_type} task")

    # Make predictions
    if task_type == 'clustering':
        # For clustering, we just need the cluster labels
        y_pred = model.predict(X_test)
        centroids = model.cluster_centers_ if hasattr(model, 'cluster_centers_') else None
        metrics = calculate_clustering_metrics(X_test, y_pred, centroids)

    else:
        # For other tasks, make predictions
        y_pred = model.predict(X_test)

        # Get prediction probabilities for classification tasks
        y_prob = None
        if task_type in ['classification', 'sentiment'] and hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception as e:
                logger.warning(f"Error getting prediction probabilities: {e}")

        # Calculate metrics based on task type
        if task_type == 'regression':
            metrics = calculate_regression_metrics(y_test, y_pred)

        elif task_type == 'classification':
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)

        elif task_type == 'forecast':
            dates = kwargs.get('dates', None)
            metrics = calculate_forecast_metrics(y_test, y_pred, dates)

        elif task_type == 'sentiment':
            text_data = kwargs.get('text_data', None)
            metrics = calculate_sentiment_metrics(y_test, y_pred, y_prob, text_data)

        elif task_type == 'recommendation':
            k = kwargs.get('k', 5)
            metrics = calculate_recommendation_metrics(y_test, y_pred, k)

        else:
            logger.warning(f"Unknown task type: {task_type}, using regression metrics")
            metrics = calculate_regression_metrics(y_test, y_pred)

    return metrics


def format_metrics(metrics: Dict[str, Any], task_type: str = 'regression') -> str:
    """
    Format metrics as a string for display

    Args:
        metrics: Dictionary of metrics
        task_type: Type of task

    Returns:
        Formatted string
    """
    lines = [f"=== {task_type.upper()} METRICS ==="]

    if task_type == 'regression' or task_type == 'forecast':
        lines.extend([
            f"RMSE: {metrics.get('rmse', 'N/A'):.4f}",
            f"MAE: {metrics.get('mae', 'N/A'):.4f}",
            f"MAPE: {metrics.get('mape', 'N/A'):.4f}",
            f"R²: {metrics.get('r2', 'N/A'):.4f}"
        ])

        if 'mase' in metrics:
            lines.append(f"MASE: {metrics['mase']:.4f}")

        if 'theil_u' in metrics:
            lines.append(f"Theil's U: {metrics['theil_u']:.4f}")

    elif task_type == 'classification' or task_type == 'sentiment':
        lines.extend([
            f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}",
            f"Precision: {metrics.get('precision', 'N/A'):.4f}",
            f"Recall: {metrics.get('recall', 'N/A'):.4f}",
            f"F1 Score: {metrics.get('f1', 'N/A'):.4f}"
        ])

        if 'roc_auc' in metrics:
            lines.append(f"ROC AUC: {metrics['roc_auc']:.4f}")

    elif task_type == 'clustering':
        if 'silhouette' in metrics:
            lines.append(f"Silhouette Score: {metrics['silhouette']:.4f}")

        if 'calinski_harabasz' in metrics:
            lines.append(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f}")

        if 'davies_bouldin' in metrics:
            lines.append(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")

        if 'inertia' in metrics:
            lines.append(f"Inertia: {metrics['inertia']:.4f}")

    elif task_type == 'recommendation':
        lines.extend([
            f"RMSE: {metrics.get('rmse', 'N/A'):.4f}",
            f"MAE: {metrics.get('mae', 'N/A'):.4f}"
        ])

        if 'precision_at_k' in metrics:
            lines.append(f"Precision@k: {metrics['precision_at_k']:.4f}")

        if 'recall_at_k' in metrics:
            lines.append(f"Recall@k: {metrics['recall_at_k']:.4f}")

        if 'ndcg' in metrics:
            lines.append(f"NDCG: {metrics['ndcg']:.4f}")

    return "\n".join(lines)


def log_metrics(metrics: Dict[str, Any], task_type: str = 'regression') -> None:
    """
    Log metrics

    Args:
        metrics: Dictionary of metrics
        task_type: Type of task
    """
    formatted_metrics = format_metrics(metrics, task_type)
    for line in formatted_metrics.split('\n'):
        logger.info(line)


if __name__ == "__main__":
    # Example usage
    import argparse
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.cluster import KMeans

    parser = argparse.ArgumentParser(description='Test evaluation metrics')
    parser.add_argument('--task', choices=['regression', 'classification', 'clustering'],
                        default='regression', help='Task type')

    args = parser.parse_args()

    if args.task == 'regression':
        # Generate regression data
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, task_type='regression')
        log_metrics(metrics, task_type='regression')

    elif args.task == 'classification':
        # Generate classification data
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, task_type='classification')
        log_metrics(metrics, task_type='classification')

    elif args.task == 'clustering':
        # Generate clustering data
        X, _ = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)

        # Train model
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)

        # Evaluate model
        metrics = evaluate_model(model, X, None, task_type='clustering')
        log_metrics(metrics, task_type='clustering')