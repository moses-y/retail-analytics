import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import logging
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_customer_segments(input_path: str, output_path: str):
    """
    Generates customer segments based on store-level features and saves the results.

    Args:
        input_path: Path to the processed retail sales data CSV.
        output_path: Path to save the customer segments CSV.
    """
    logger.info(f"Loading processed retail data from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    retail_df = pd.read_csv(input_path)

    # Convert date column if it's not already datetime
    if 'date' in retail_df.columns and not pd.api.types.is_datetime64_any_dtype(retail_df['date']):
         retail_df['date'] = pd.to_datetime(retail_df['date'])

    # Add is_weekend if not present (might be missing if preprocessing script changed)
    if 'date' in retail_df.columns and 'is_weekend' not in retail_df.columns:
         retail_df['day_of_week'] = retail_df['date'].dt.dayofweek
         retail_df['is_weekend'] = retail_df['day_of_week'].isin([5, 6]).astype(int)

    # Add online_ratio if not present
    if 'online_sales' in retail_df.columns and 'total_sales' in retail_df.columns and 'online_ratio' not in retail_df.columns:
        # Avoid division by zero
        retail_df['online_ratio'] = np.where(
            retail_df['total_sales'] > 0,
            retail_df['online_sales'] / retail_df['total_sales'],
            0
        )

    # Add price_per_customer if not present
    if 'total_sales' in retail_df.columns and 'num_customers' in retail_df.columns and 'price_per_customer' not in retail_df.columns:
         # Avoid division by zero
         retail_df['price_per_customer'] = np.where(
             retail_df['num_customers'] > 0,
             retail_df['total_sales'] / retail_df['num_customers'],
             0
         )


    logger.info("Aggregating data at store level for segmentation")
    # Aggregate data at the store level
    store_features = retail_df.groupby('store_id').agg({
        'total_sales': 'mean',
        'online_sales': 'mean',
        'in_store_sales': 'mean',
        'num_customers': 'mean',
        'avg_transaction': 'mean',
        'return_rate': 'mean',
        'online_ratio': 'mean',
        'price_per_customer': 'mean',
        'is_weekend': 'mean'  # Proportion of weekend sales days in data
    }).reset_index()

    # Calculate additional KPIs
    # Avoid division by zero for online_to_instore_ratio
    store_features['online_to_instore_ratio'] = np.where(
        store_features['in_store_sales'] > 0,
        store_features['online_sales'] / store_features['in_store_sales'],
        np.inf # Assign infinity if in_store_sales is zero but online_sales is positive
    )
    # Handle cases where both are zero
    store_features.loc[(store_features['online_sales'] == 0) & (store_features['in_store_sales'] == 0), 'online_to_instore_ratio'] = 0


    logger.info("Preparing data for clustering")
    # Select features for clustering, handle potential missing columns gracefully
    cluster_feature_cols = [
        'total_sales', 'online_sales', 'in_store_sales', 'num_customers',
        'avg_transaction', 'return_rate', 'online_ratio', 'price_per_customer',
        'is_weekend', 'online_to_instore_ratio'
    ]
    available_features = [col for col in cluster_feature_cols if col in store_features.columns]
    X_cluster = store_features[available_features].fillna(0) # Fill NaNs that might arise from aggregation

    # Replace infinite values resulting from division by zero
    X_cluster.replace([np.inf, -np.inf], 0, inplace=True)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    logger.info("Determining optimal number of clusters using silhouette score")
    silhouette_scores = []
    # Check for sufficient samples for clustering range
    max_k = min(6, X_scaled.shape[0] - 1) # K must be less than n_samples
    if max_k < 2:
        logger.warning("Not enough samples to perform clustering. Skipping segmentation.")
        # Save an empty file or file with just store_id if needed by dashboard
        store_features[['store_id']].to_csv(output_path, index=False)
        logger.info(f"Saved basic store list to {output_path}")
        return

    K = range(2, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Set n_init explicitly
        labels = kmeans.fit_predict(X_scaled)
        # Ensure there's more than one cluster label before calculating silhouette score
        if len(set(labels)) > 1:
             silhouette_scores.append(silhouette_score(X_scaled, labels))
        else:
             silhouette_scores.append(-1) # Assign a low score if only one cluster is formed


    if not silhouette_scores or max(silhouette_scores) == -1:
         logger.warning("Could not determine optimal K via silhouette score. Defaulting to K=3.")
         optimal_k = 3
    else:
         optimal_k = K[np.argmax(silhouette_scores)]
    logger.info(f"Optimal number of clusters: {optimal_k}")

    logger.info(f"Applying K-means clustering with k={optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10) # Set n_init explicitly
    store_features['cluster'] = kmeans.fit_predict(X_scaled)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Saving customer segments data to {output_path}")
    store_features.to_csv(output_path, index=False)
    logger.info("Customer segmentation completed and saved.")


if __name__ == "__main__":
    input_file = "data/processed/retail_sales_data.csv"
    output_file = "data/processed/customer_segments.csv"
    generate_customer_segments(input_file, output_file)
