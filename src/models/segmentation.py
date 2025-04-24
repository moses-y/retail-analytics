"""
Customer segmentation models for retail analytics
"""
import os
import logging
import pickle
import yaml
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
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
    
    return config.get("segmentation", {})


def prepare_segmentation_data(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    id_column: str = 'store_id',
    scale_data: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Prepare data for customer segmentation
    
    Args:
        df: DataFrame with customer data
        feature_columns: List of columns to use for segmentation
        id_column: Column containing customer/store IDs
        scale_data: Whether to standardize the features
        
    Returns:
        Tuple of (features_df, ids_df, scaler)
    """
    logger.info("Preparing data for segmentation")
    
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Use default features if not specified
    if feature_columns is None:
        # Load from config if available
        config = load_config()
        feature_columns = config.get("features", [])
        
        # Fall back to numeric columns if not in config
        if not feature_columns:
            feature_columns = data.select_dtypes(include=['number']).columns.tolist()
            # Exclude ID column and any obvious non-feature columns
            exclude_cols = [id_column, 'date', 'year', 'month', 'day']
            feature_columns = [col for col in feature_columns if col not in exclude_cols]
    
    logger.info(f"Using features: {feature_columns}")
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_columns if col not in data.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in data.columns]
    
    # Extract features and IDs
    features_df = data[feature_columns].copy()
    ids_df = data[[id_column]].copy()
    
    # Handle missing values
    features_df = features_df.fillna(features_df.mean())
    
    # Scale features if requested
    scaler = None
    if scale_data:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        features_df = pd.DataFrame(features_scaled, columns=feature_columns, index=features_df.index)
    
    logger.info(f"Prepared segmentation data: {features_df.shape}")
    return features_df, ids_df, scaler


def find_optimal_clusters(
    features: pd.DataFrame,
    max_clusters: int = 10,
    method: str = 'elbow',
    random_state: int = 42
) -> int:
    """
    Find the optimal number of clusters
    
    Args:
        features: DataFrame with features for clustering
        max_clusters: Maximum number of clusters to consider
        method: Method to use ('elbow', 'silhouette', or 'gap')
        random_state: Random seed for reproducibility
        
    Returns:
        Optimal number of clusters
    """
    logger.info(f"Finding optimal number of clusters using {method} method")
    
    if method == 'elbow':
        # Elbow method
        inertia = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            kmeans.fit(features)
            inertia.append(kmeans.inertia_)
        
        # Calculate rate of change
        deltas = np.diff(inertia)
        delta_deltas = np.diff(deltas)
        
        # Find the elbow point (where the rate of change in inertia slows down)
        elbow_point = np.argmax(delta_deltas) + 2
        
        logger.info(f"Optimal number of clusters (elbow method): {elbow_point}")
        return elbow_point
    
    elif method == 'silhouette':
        # Silhouette method
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            silhouette_scores.append(score)
        
        # Find the k with highest silhouette score
        optimal_k = np.argmax(silhouette_scores) + 2
        
        logger.info(f"Optimal number of clusters (silhouette method): {optimal_k}")
        return optimal_k
    
    else:
        # Default to 3 clusters if method not recognized
        logger.warning(f"Unrecognized method: {method}, defaulting to 3 clusters")
        return 3


def train_kmeans_model(
    features: pd.DataFrame,
    n_clusters: Optional[int] = None,
    params: Optional[Dict] = None,
    random_state: int = 42
) -> KMeans:
    """
    Train KMeans clustering model
    
    Args:
        features: DataFrame with features for clustering
        n_clusters: Number of clusters (if None, will be determined automatically)
        params: Additional parameters for KMeans
        random_state: Random seed for reproducibility
        
    Returns:
        Trained KMeans model
    """
    logger.info("Training KMeans clustering model")
    
    # Load default parameters from config
    config = load_config()
    default_params = config.get("kmeans", {}).get("params", {})
    
    # Use provided parameters or defaults
    if params is None:
        params = default_params.copy()  # Make a copy to avoid modifying the original
    else:
        params = params.copy()
    
    # Remove n_clusters and random_state from params since we'll set them explicitly
    params.pop('n_clusters', None)
    params.pop('random_state', None)
    
    # Determine number of clusters if not provided
    if n_clusters is None:
        n_clusters = find_optimal_clusters(features)
    
    # Create and train model
    model = KMeans(n_clusters=n_clusters, random_state=random_state, **params)
    model.fit(features)
    
    logger.info(f"KMeans model trained with {n_clusters} clusters")
    return model


def train_dbscan_model(
    features: pd.DataFrame,
    params: Optional[Dict] = None
) -> DBSCAN:
    """
    Train DBSCAN clustering model
    
    Args:
        features: DataFrame with features for clustering
        params: Parameters for DBSCAN
        
    Returns:
        Trained DBSCAN model
    """
    logger.info("Training DBSCAN clustering model")
    
    # Load default parameters from config
    config = load_config()
    default_params = config.get("dbscan", {}).get("params", {})
    
    # Use provided parameters or defaults
    if params is None:
        params = default_params
    
    # Create and train model
    model = DBSCAN(**params)
    model.fit(features)
    
    # Log number of clusters found
    n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    logger.info(f"DBSCAN model trained, found {n_clusters} clusters")
    
    return model


def train_hierarchical_model(
    features: pd.DataFrame,
    n_clusters: Optional[int] = None,
    params: Optional[Dict] = None
) -> AgglomerativeClustering:
    """
    Train hierarchical clustering model
    
    Args:
        features: DataFrame with features for clustering
        n_clusters: Number of clusters (if None, will be determined automatically)
        params: Parameters for AgglomerativeClustering
        
    Returns:
        Trained hierarchical clustering model
    """
    logger.info("Training hierarchical clustering model")
    
    # Load default parameters from config
    config = load_config()
    default_params = config.get("hierarchical", {}).get("params", {})
    
    # Use provided parameters or defaults
    if params is None:
        params = default_params
    
    # Determine number of clusters if not provided
    if n_clusters is None:
        n_clusters = find_optimal_clusters(features)
    
    # Create and train model
    model = AgglomerativeClustering(n_clusters=n_clusters, **params)
    model.fit(features)
    
    logger.info(f"Hierarchical clustering model trained with {n_clusters} clusters")
    return model


def evaluate_clustering(
    features: pd.DataFrame,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate clustering results
    
    Args:
        features: DataFrame with features used for clustering
        labels: Cluster labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating clustering results")
    
    metrics = {}
    
    # Skip evaluation if only one cluster or if any cluster has only one sample
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or min(np.bincount(labels[labels >= 0])) <= 1:
        logger.warning("Cannot evaluate clustering: too few clusters or samples per cluster")
        return metrics
    
    # Calculate silhouette score
    try:
        silhouette = silhouette_score(features, labels)
        metrics['silhouette_score'] = silhouette
        logger.info(f"Silhouette score: {silhouette:.4f}")
    except Exception as e:
        logger.warning(f"Error calculating silhouette score: {e}")
    
    # Calculate Calinski-Harabasz index
    try:
        ch_score = calinski_harabasz_score(features, labels)
        metrics['calinski_harabasz_score'] = ch_score
        logger.info(f"Calinski-Harabasz score: {ch_score:.4f}")
    except Exception as e:
        logger.warning(f"Error calculating Calinski-Harabasz score: {e}")
    
    # Calculate basic cluster statistics
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    cluster_sizes = np.bincount(labels[labels >= 0])
    
    metrics['n_clusters'] = n_clusters
    metrics['avg_cluster_size'] = np.mean(cluster_sizes)
    metrics['min_cluster_size'] = np.min(cluster_sizes)
    metrics['max_cluster_size'] = np.max(cluster_sizes)
    
    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Cluster sizes: min={metrics['min_cluster_size']}, avg={metrics['avg_cluster_size']:.1f}, max={metrics['max_cluster_size']}")
    
    return metrics


def analyze_clusters(
    features: pd.DataFrame,
    labels: np.ndarray,
    original_data: pd.DataFrame,
    id_column: str = 'store_id'
) -> pd.DataFrame:
    """
    Analyze cluster characteristics
    
    Args:
        features: DataFrame with features used for clustering
        labels: Cluster labels
        original_data: Original DataFrame with all columns
        id_column: Column containing customer/store IDs
        
    Returns:
        DataFrame with cluster analysis
    """
    logger.info("Analyzing cluster characteristics")
    
    # Add cluster labels to original data
    data_with_clusters = original_data.copy()
    data_with_clusters['cluster'] = labels
    
    # Calculate cluster profiles (mean values for each feature by cluster)
    cluster_profiles = data_with_clusters.groupby('cluster').mean()
    
    # Calculate feature importance for each cluster
    # (how much each feature deviates from the overall mean)
    overall_means = data_with_clusters.mean()
    
    feature_importance = pd.DataFrame()
    for cluster in sorted(set(labels)):
        if cluster == -1:  # Skip noise points in DBSCAN
            continue
            
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster]
        cluster_means = cluster_data.mean()
        
        # Calculate z-scores relative to overall distribution
        importance = (cluster_means - overall_means) / overall_means.std()
        feature_importance[f'Cluster {cluster}'] = importance
    
    # Get top features for each cluster
    top_features = {}
    for cluster in feature_importance.columns:
        # Sort features by absolute importance
        sorted_features = feature_importance[cluster].abs().sort_values(ascending=False)
        # Get top 5 features (excluding non-feature columns)
        exclude_cols = ['cluster', id_column, 'date', 'year', 'month', 'day']
        top = sorted_features[~sorted_features.index.isin(exclude_cols)].head(5)
        top_features[cluster] = top
    
    logger.info("Cluster analysis completed")
    return cluster_profiles, feature_importance, top_features


def plot_clusters_2d(
    features: pd.DataFrame,
    labels: np.ndarray,
    method: str = 'pca',
    output_path: Optional[str] = None
) -> None:
    """
    Plot clusters in 2D

    Args:
        features: DataFrame with features used for clustering
        labels: Cluster labels
        method: Dimensionality reduction method ('pca' or 'tsne')
        output_path: Path to save the plot (optional)
    """
    logger.info(f"Plotting clusters in 2D using {method}")

    # Apply dimensionality reduction
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(features)

        # Get explained variance
        explained_variance = reducer.explained_variance_ratio_
        explained_variance_sum = sum(explained_variance)
        title = f'PCA Cluster Visualization (Explained Variance: {explained_variance_sum:.2%})'

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
        title = 'TSNE Cluster Visualization'

    else:
        logger.warning(f"Unsupported method: {method}, using PCA")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(features)
        title = 'PCA Cluster Visualization'

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot each cluster
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            c=[color],
            label=f'Cluster {label}' if label != -1 else 'Noise',
            alpha=0.7
        )

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Cluster plot saved to {output_path}")

    plt.close()


def plot_cluster_profiles(
    cluster_profiles: pd.DataFrame,
    top_n_features: int = 10,
    output_path: Optional[str] = None
) -> None:
    """
    Plot cluster profiles showing key characteristics

    Args:
        cluster_profiles: DataFrame with cluster profiles
        top_n_features: Number of top features to display
        output_path: Path to save the plot (optional)
    """
    logger.info("Plotting cluster profiles")

    # Select top features based on variance across clusters
    feature_variance = cluster_profiles.var(axis=0).sort_values(ascending=False)
    top_features = feature_variance.head(top_n_features).index.tolist()

    # Create plot
    plt.figure(figsize=(12, 8))

    # Create heatmap of cluster profiles
    sns.heatmap(
        cluster_profiles[top_features].T,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.5
    )

    plt.title('Cluster Profiles')
    plt.ylabel('Features')
    plt.xlabel('Cluster')
    plt.tight_layout()

    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Cluster profiles plot saved to {output_path}")

    plt.close()


def save_segmentation_model(
    model: Any,
    scaler: Any,
    output_path: str,
    model_type: str = 'kmeans'
) -> None:
    """
    Save trained segmentation model and scaler

    Args:
        model: Trained segmentation model
        scaler: Feature scaler
        output_path: Path to save the model
        model_type: Type of model
    """
    logger.info(f"Saving {model_type} model to {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_type': model_type
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {output_path}")


def load_segmentation_model(model_path: str) -> Tuple[Any, Any, str]:
    """
    Load trained segmentation model and scaler

    Args:
        model_path: Path to the saved model

    Returns:
        Tuple of (model, scaler, model_type)
    """
    logger.info(f"Loading segmentation model from {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    scaler = model_data['scaler']
    model_type = model_data.get('model_type', 'kmeans')

    logger.info(f"Loaded {model_type} model from {model_path}")
    return model, scaler, model_type


def assign_segments(
    df: pd.DataFrame,
    model: Any,
    scaler: Any,
    feature_columns: List[str],
    model_type: str = 'kmeans'
) -> pd.DataFrame:
    """
    Assign segments to new data

    Args:
        df: DataFrame with customer data
        model: Trained segmentation model
        scaler: Feature scaler
        feature_columns: List of columns to use for segmentation
        model_type: Type of model

    Returns:
        DataFrame with segment assignments
    """
    logger.info("Assigning segments to data")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Extract features
    features = result_df[feature_columns].copy()

    # Handle missing values
    features = features.fillna(features.mean())

    # Scale features if scaler is provided
    if scaler is not None:
        features_scaled = scaler.transform(features)
        features = pd.DataFrame(features_scaled, columns=feature_columns, index=features.index)

    # Assign segments
    if model_type == 'kmeans':
        result_df['segment'] = model.predict(features)
    elif model_type == 'dbscan':
        result_df['segment'] = model.fit_predict(features)
    elif model_type == 'hierarchical':
        result_df['segment'] = model.fit_predict(features)
    else:
        logger.warning(f"Unsupported model type: {model_type}")
        return result_df

    logger.info(f"Assigned segments to {len(result_df)} records")
    return result_df


def generate_segment_descriptions(
    cluster_profiles: pd.DataFrame,
    feature_importance: pd.DataFrame,
    top_features: Dict
) -> Dict[int, str]:
    """
    Generate human-readable descriptions for each segment

    Args:
        cluster_profiles: DataFrame with cluster profiles
        feature_importance: DataFrame with feature importance
        top_features: Dictionary with top features for each cluster

    Returns:
        Dictionary mapping segment IDs to descriptions
    """
    logger.info("Generating segment descriptions")

    descriptions = {}

    for cluster in cluster_profiles.index:
        if cluster == -1:  # Skip noise points in DBSCAN
            descriptions[cluster] = "Noise/Outliers: Points that don't fit well into any cluster"
            continue

        # Get top positive and negative features
        cluster_col = f'Cluster {cluster}'
        if cluster_col in feature_importance.columns:
            top_pos = feature_importance[cluster_col].nlargest(3)
            top_neg = feature_importance[cluster_col].nsmallest(3)

            # Create description
            desc_parts = []

            # Add positive features
            pos_features = []
            for feature, value in top_pos.items():
                if value > 0.5:  # Only include significant features
                    pos_features.append(f"high {feature}")

            if pos_features:
                desc_parts.append("characterized by " + ", ".join(pos_features))

            # Add negative features
            neg_features = []
            for feature, value in top_neg.items():
                if value < -0.5:  # Only include significant features
                    neg_features.append(f"low {feature}")

            if neg_features:
                if desc_parts:
                    desc_parts.append("and")
                desc_parts.append(", ".join(neg_features))

            # Combine description
            if desc_parts:
                descriptions[cluster] = f"Segment {cluster}: " + " ".join(desc_parts)
            else:
                descriptions[cluster] = f"Segment {cluster}: Average across most dimensions"
        else:
            descriptions[cluster] = f"Segment {cluster}"

    logger.info("Generated segment descriptions")
    return descriptions


def train_segmentation_model(data: pd.DataFrame, feature_cols: List[str], n_clusters: int = 3) -> Tuple[Any, np.ndarray]:
    """
    Wrapper function to train a segmentation model (defaults to KMeans)
    
    Args:
        data: DataFrame with customer data
        feature_cols: List of columns to use for segmentation
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (trained_model, cluster_centers)
    """
    # Prepare the features
    features_df, _, scaler = prepare_segmentation_data(
        data, 
        feature_columns=feature_cols
    )
    
    # Train KMeans model
    model = train_kmeans_model(features_df, n_clusters=n_clusters)
    
    return model, model.cluster_centers_


def predict_segment(model: Any, data: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Predict segments for new data
    
    Args:
        model: Trained segmentation model
        data: DataFrame with customer data
        feature_cols: List of feature columns used for segmentation
        
    Returns:
        Array of segment assignments
    """
    # Prepare the features
    features_df = data[feature_cols].copy()
    features_df = features_df.fillna(features_df.mean())
    
    # Make predictions
    return model.predict(features_df)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Train customer segmentation model')
    parser.add_argument('--input', required=True, help='Path to input data file')
    parser.add_argument('--output', required=True, help='Path to save trained model')
    parser.add_argument('--model', choices=['kmeans', 'dbscan', 'hierarchical'],
                        default='kmeans', help='Type of model to train')
    parser.add_argument('--id_column', default='store_id', help='Column containing customer/store IDs')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters (optional)')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Prepare data
    features_df, ids_df, scaler = prepare_segmentation_data(df, id_column=args.id_column)

    # Train model
    if args.model == 'kmeans':
        model = train_kmeans_model(features_df, n_clusters=args.n_clusters)
        labels = model.labels_
    elif args.model == 'dbscan':
        model = train_dbscan_model(features_df)
        labels = model.labels_
    elif args.model == 'hierarchical':
        model = train_hierarchical_model(features_df, n_clusters=args.n_clusters)
        labels = model.labels_

    # Evaluate clustering
    metrics = evaluate_clustering(features_df, labels)

    # Analyze clusters
    cluster_profiles, feature_importance, top_features = analyze_clusters(
        features_df, labels, df, id_column=args.id_column
    )

    # Plot clusters
    plot_path = os.path.join(os.path.dirname(args.output), f"{args.model}_clusters.png")
    plot_clusters_2d(features_df, labels, output_path=plot_path)

    # Plot cluster profiles
    profiles_path = os.path.join(os.path.dirname(args.output), f"{args.model}_profiles.png")
    plot_cluster_profiles(cluster_profiles, output_path=profiles_path)

    # Save model
    save_segmentation_model(model, scaler, args.output, args.model)