"""
Sentiment analysis models for retail analytics
"""
import os
import logging
import pickle
import yaml
import re
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config() -> Dict:
    """
    Load model configuration from YAML file

    Returns:
        Dictionary containing model configuration
    """
    config_path = os.path.join("config", "model_config.yml")

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config.get("sentiment", {})


def preprocess_text(text: str) -> str:
    """
    Preprocess text for sentiment analysis

    Args:
        text: Raw text

    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def prepare_sentiment_data(
    df: pd.DataFrame,
    text_column: str = 'review_text',
    sentiment_column: Optional[str] = 'sentiment',
    rating_column: Optional[str] = 'rating',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for sentiment analysis

    Args:
        df: DataFrame with review data
        text_column: Column containing review text
        sentiment_column: Column containing sentiment labels (optional)
        rating_column: Column containing ratings (optional)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Preparing data for sentiment analysis")

    # Create a copy to avoid modifying the original
    data = df.copy()

    # Preprocess text
    data['processed_text'] = data[text_column].apply(preprocess_text)

    # Remove rows with empty text
    data = data[data['processed_text'].str.strip() != '']

    # Determine sentiment labels
    if sentiment_column in data.columns:
        # Use existing sentiment column
        data['sentiment_label'] = data[sentiment_column]
    elif rating_column in data.columns:
        # Create sentiment labels from ratings
        data['sentiment_label'] = pd.cut(
            data[rating_column],
            bins=[0, 2, 3, 5],
            labels=['negative', 'neutral', 'positive']
        )
    else:
        logger.error("No sentiment or rating column found")
        raise ValueError("Either sentiment_column or rating_column must be present in the data")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_text'],
        data['sentiment_label'],
        test_size=test_size,
        random_state=random_state,
        stratify=data['sentiment_label']
    )

    logger.info(f"Prepared sentiment data: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def extract_features(
    X_train: pd.Series,
    X_test: pd.Series,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Extract features from text using TF-IDF

    Args:
        X_train: Training text data
        X_test: Test text data
        max_features: Maximum number of features
        ngram_range: Range of n-grams to consider

    Returns:
        Tuple of (X_train_features, X_test_features, vectorizer)
    """
    logger.info("Extracting features from text")

    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )

    # Transform text to features
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    logger.info(f"Extracted {X_train_features.shape[1]} features")
    return X_train_features, X_test_features, vectorizer


def train_sentiment_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    model_type: str = 'logistic',
    params: Optional[Dict] = None
) -> Any:
    """
    Train sentiment analysis model

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('logistic' or 'random_forest')
        params: Model parameters (optional)

    Returns:
        Trained sentiment model
    """
    logger.info(f"Training {model_type} sentiment model")

    # Load default parameters from config
    config = load_config()
    default_params = config.get(model_type, {}).get("params", {})

    # Use provided parameters or defaults
    if params is None:
        params = default_params

    # Create and train model
    if model_type == 'logistic':
        model = LogisticRegression(**params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**params)
    else:
        logger.warning(f"Unsupported model type: {model_type}, using logistic regression")
        model = LogisticRegression()

    model.fit(X_train, y_train)

    logger.info(f"{model_type} model training completed")
    return model


def evaluate_sentiment_model(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate sentiment analysis model

    Args:
        model: Trained sentiment model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating sentiment model")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log results
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    classes: List[str],
    output_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix

    Args:
        conf_matrix: Confusion matrix
        classes: Class labels
        output_path: Path to save the plot (optional)
    """
    logger.info("Plotting confusion matrix")

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Confusion matrix plot saved to {output_path}")

    plt.close()


def plot_feature_importance(
    model: Any,
    vectorizer: TfidfVectorizer,
    top_n: int = 20,
    output_path: Optional[str] = None
) -> None:
    """
    Plot feature importance for sentiment model

    Args:
        model: Trained sentiment model
        vectorizer: TF-IDF vectorizer
        top_n: Number of top features to display
        output_path: Path to save the plot (optional)
    """
    logger.info("Plotting feature importance")

    # Check if model supports feature importance
    if not hasattr(model, 'coef_') and not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not support feature importance")
        return

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get feature importance
    if hasattr(model, 'coef_'):
        # For linear models like logistic regression
        importance = model.coef_[0]
    else:
        # For tree-based models like random forest
        importance = model.feature_importances_

    # Create DataFrame with feature names and importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    # Sort by absolute importance
    feature_importance['abs_importance'] = feature_importance['importance'].abs()
    feature_importance = feature_importance.sort_values('abs_importance', ascending=False)

    # Plot top features
    plt.figure(figsize=(12, 8))

    top_features = feature_importance.head(top_n)
    colors = ['red' if imp < 0 else 'green' for imp in top_features['importance']]

    plt.barh(
        range(top_n),
        top_features['importance'],
        color=colors
    )

    plt.yticks(range(top_n), top_features['feature'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Features for Sentiment Analysis')
    plt.tight_layout()

    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")

    plt.close()


def save_sentiment_model(
    model: Any,
    vectorizer: TfidfVectorizer,
    output_path: str,
    model_type: str = 'logistic'
) -> None:
    """
    Save trained sentiment model and vectorizer

    Args:
        model: Trained sentiment model
        vectorizer: TF-IDF vectorizer
        output_path: Path to save the model
        model_type: Type of model
    """
    logger.info(f"Saving {model_type} sentiment model to {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save model and vectorizer
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'model_type': model_type
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {output_path}")


def load_sentiment_model(model_path: str) -> Tuple[Any, TfidfVectorizer, str]:
    """
    Load trained sentiment model and vectorizer

    Args:
        model_path: Path to the saved model

    Returns:
        Tuple of (model, vectorizer, model_type)
    """
    logger.info(f"Loading sentiment model from {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    vectorizer = model_data['vectorizer']
    model_type = model_data.get('model_type', 'logistic')

    logger.info(f"Loaded {model_type} model from {model_path}")
    return model, vectorizer, model_type


def predict_sentiment(
    texts: List[str],
    model: Any,
    vectorizer: TfidfVectorizer
) -> List[str]:
    """
    Predict sentiment for new texts

    Args:
        texts: List of texts to analyze
        model: Trained sentiment model
        vectorizer: TF-IDF vectorizer

    Returns:
        List of predicted sentiment labels
    """
    logger.info(f"Predicting sentiment for {len(texts)} texts")

    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]

    # Transform texts to features
    features = vectorizer.transform(processed_texts)

    # Predict sentiment
    predictions = model.predict(features)

    return predictions.tolist()


def analyze_feature_sentiment(
    df: pd.DataFrame,
    text_column: str = 'review_text',
    feature_column: str = 'feature_mentioned',
    sentiment_column: str = 'sentiment'
) -> pd.DataFrame:
    """
    Analyze sentiment by product feature

    Args:
        df: DataFrame with review data
        text_column: Column containing review text
        feature_column: Column containing mentioned features
        sentiment_column: Column containing sentiment labels

    Returns:
        DataFrame with feature sentiment analysis
    """
    logger.info("Analyzing feature sentiment")

    # Create a copy to avoid modifying the original
    data = df.copy()

    # Ensure sentiment column exists
    if sentiment_column not in data.columns:
        logger.warning(f"Sentiment column '{sentiment_column}' not found")
        return pd.DataFrame()

    # Ensure feature column exists
    if feature_column not in data.columns:
        logger.warning(f"Feature column '{feature_column}' not found")
        return pd.DataFrame()

    # Remove rows with missing features
    data = data.dropna(subset=[feature_column])

    # Calculate sentiment counts by feature
    feature_sentiment = data.groupby([feature_column, sentiment_column]).size().unstack(fill_value=0)

    # Calculate total mentions and sentiment scores
    feature_sentiment['total_mentions'] = feature_sentiment.sum(axis=1)

    # Calculate sentiment percentages
    for sentiment in feature_sentiment.columns:
        if sentiment != 'total_mentions':
            feature_sentiment[f'{sentiment}_pct'] = feature_sentiment[sentiment] / feature_sentiment['total_mentions']

    # Calculate overall sentiment score (-1 for negative, 0 for neutral, 1 for positive)
    sentiment_weights = {'negative': -1, 'neutral': 0, 'positive': 1}

    feature_sentiment['sentiment_score'] = 0
    for sentiment, weight in sentiment_weights.items():
        if sentiment in feature_sentiment.columns:
            feature_sentiment['sentiment_score'] += feature_sentiment[sentiment] * weight / feature_sentiment['total_mentions']

    # Sort by total mentions
    feature_sentiment = feature_sentiment.sort_values('total_mentions', ascending=False)

    logger.info(f"Analyzed sentiment for {len(feature_sentiment)} features")
    return feature_sentiment


def plot_feature_sentiment(
    feature_sentiment: pd.DataFrame,
    top_n: int = 10,
    output_path: Optional[str] = None
) -> None:
    """
    Plot sentiment analysis by product feature

    Args:
        feature_sentiment: DataFrame with feature sentiment analysis
        top_n: Number of top features to display
        output_path: Path to save the plot (optional)
    """
    logger.info(f"Plotting sentiment for top {top_n} features")

    # Select top features by total mentions
    top_features = feature_sentiment.head(top_n)

    # Create plot
    plt.figure(figsize=(12, 8))

    # Create stacked bar chart
    sentiments = ['positive', 'neutral', 'negative']
    colors = ['green', 'gray', 'red']

    # Check which sentiments are available in the data
    available_sentiments = [s for s in sentiments if s in top_features.columns]
    available_colors = [colors[sentiments.index(s)] for s in available_sentiments]

    # Create stacked bars
    bottom = np.zeros(len(top_features))
    for sentiment, color in zip(available_sentiments, available_colors):
        if sentiment in top_features.columns:
            plt.barh(
                range(len(top_features)),
                top_features[f'{sentiment}_pct'],
                left=bottom,
                color=color,
                label=sentiment
            )
            bottom += top_features[f'{sentiment}_pct']

    # Add feature names and total mentions
    feature_labels = [f"{feature} ({mentions})" for feature, mentions in
                     zip(top_features.index, top_features['total_mentions'])]

    plt.yticks(range(len(top_features)), feature_labels)
    plt.xlabel('Proportion of Reviews')
    plt.ylabel('Feature (Total Mentions)')
    plt.title('Sentiment Analysis by Product Feature')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Feature sentiment plot saved to {output_path}")

    plt.close()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--input', required=True, help='Path to input data file')
    parser.add_argument('--output', required=True, help='Path to save trained model')
    parser.add_argument('--model', choices=['logistic', 'random_forest'],
                        default='logistic', help='Type of model to train')
    parser.add_argument('--text_column', default='review_text', help='Column containing review text')
    parser.add_argument('--sentiment_column', help='Column containing sentiment labels (optional)')
    parser.add_argument('--rating_column', default='rating', help='Column containing ratings (optional)')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_sentiment_data(
        df,
        text_column=args.text_column,
        sentiment_column=args.sentiment_column,
        rating_column=args.rating_column
    )

    # Extract features
    X_train_features, X_test_features, vectorizer = extract_features(X_train, X_test)

    # Train model
    model = train_sentiment_model(X_train_features, y_train, model_type=args.model)

    # Evaluate model
    metrics = evaluate_sentiment_model(model, X_test_features, y_test)

    # Plot confusion matrix
    conf_matrix_path = os.path.join(os.path.dirname(args.output), f"{args.model}_confusion_matrix.png")
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        classes=list(metrics['classification_report'].keys())[:-3],  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        output_path=conf_matrix_path
    )

    # Plot feature importance
    importance_path = os.path.join(os.path.dirname(args.output), f"{args.model}_feature_importance.png")
    plot_feature_importance(model, vectorizer, output_path=importance_path)

    # Save model
    save_sentiment_model(model, vectorizer, args.output, args.model)

    # Analyze feature sentiment if feature column exists
    if 'feature_mentioned' in df.columns and (args.sentiment_column in df.columns or args.rating_column in df.columns):
        feature_sentiment = analyze_feature_sentiment(
            df,
            text_column=args.text_column,
            feature_column='feature_mentioned',
            sentiment_column=args.sentiment_column if args.sentiment_column else 'sentiment_label'
        )

        # Plot feature sentiment
        sentiment_path = os.path.join(os.path.dirname(args.output), "feature_sentiment.png")
        plot_feature_sentiment(feature_sentiment, output_path=sentiment_path)