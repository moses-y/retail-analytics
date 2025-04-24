"""
Vector search functionality for retail analytics
"""
import os
import logging
import pickle
import yaml
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

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

    return config.get("vector_search", {})


class VectorSearchIndex:
    """Vector search index for efficient similarity search"""

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = 'flat',
        metric: str = 'cosine',
        use_gpu: bool = False
    ):
        """
        Initialize vector search index

        Args:
            embedding_dim: Dimensionality of embeddings
            index_type: Type of index ('flat', 'ivf', or 'hnsw')
            metric: Distance metric ('cosine' or 'l2')
            use_gpu: Whether to use GPU for search
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu
        self.index = None
        self.ids = None
        self.metadata = None

        logger.info(f"Initializing {index_type} vector search index with {embedding_dim} dimensions")

        # Create index
        self._create_index()

    def _create_index(self) -> None:
        """Create FAISS index based on configuration"""
        # Determine index type
        if self.metric == 'cosine':
            # For cosine similarity, normalize vectors during search
            index_flat = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        else:
            # For L2 distance
            index_flat = faiss.IndexFlatL2(self.embedding_dim)

        # Create specific index type
        if self.index_type == 'flat':
            self.index = index_flat
        elif self.index_type == 'ivf':
            # IVF index for faster search with slight accuracy trade-off
            nlist = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(index_flat, self.embedding_dim, nlist)
        elif self.index_type == 'hnsw':
            # HNSW index for even faster search
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 neighbors
        else:
            logger.warning(f"Unsupported index type: {self.index_type}, using flat index")
            self.index = index_flat

        # Use GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            logger.info("Using GPU for vector search")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[Any]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Add embeddings to the index

        Args:
            embeddings: Array of embeddings
            ids: List of IDs corresponding to embeddings (optional)
            metadata: List of metadata dictionaries (optional)
        """
        # Ensure embeddings are in the right format
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)

        # Store IDs and metadata
        if ids is None:
            ids = list(range(len(embeddings)))

        self.ids = ids
        self.metadata = metadata

        # Train index if needed
        if self.index_type == 'ivf' and not self.index.is_trained:
            self.index.train(embeddings)

        # Add embeddings to index
        self.index.add(embeddings)
        logger.info(f"Added {len(embeddings)} embeddings to index")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for similar embeddings

        Args:
            query_embedding: Query embedding
            k: Number of results to return

        Returns:
            Tuple of (distances, indices, metadata)
        """
        # Ensure query is in the right format
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(query_embedding)

        # Perform search
        distances, indices = self.index.search(query_embedding, k)

        # Get metadata for results
        result_metadata = []
        if self.metadata is not None:
            for idx in indices[0]:
                if 0 <= idx < len(self.metadata):
                    result_metadata.append(self.metadata[idx])
                else:
                    result_metadata.append({})
        else:
            result_metadata = [{} for _ in range(len(indices[0]))]

        # Get IDs for results
        result_ids = []
        for idx in indices[0]:
            if 0 <= idx < len(self.ids):
                result_ids.append(self.ids[idx])
            else:
                result_ids.append(None)

        return distances[0], result_ids, result_metadata

    def save(self, output_path: str) -> None:
        """
        Save index to file

        Args:
            output_path: Path to save the index
        """
        logger.info(f"Saving vector search index to {output_path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert GPU index to CPU if needed
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            index_cpu = self.index

        # Save index data
        index_data = {
            'index_type': self.index_type,
            'metric': self.metric,
            'embedding_dim': self.embedding_dim,
            'ids': self.ids,
            'metadata': self.metadata
        }

        with open(output_path, 'wb') as f:
            pickle.dump(index_data, f)

        # Save FAISS index
        faiss_path = output_path + '.faiss'
        faiss.write_index(index_cpu, faiss_path)

        logger.info(f"Index saved to {output_path}")

    @classmethod
    def load(cls, input_path: str, use_gpu: bool = False) -> 'VectorSearchIndex':
        """
        Load index from file

        Args:
            input_path: Path to load the index from
            use_gpu: Whether to use GPU for search

        Returns:
            Loaded VectorSearchIndex
        """
        logger.info(f"Loading vector search index from {input_path}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Index file not found: {input_path}")

        # Load index data
        with open(input_path, 'rb') as f:
            index_data = pickle.load(f)

        # Load FAISS index
        faiss_path = input_path + '.faiss'
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index file not found: {faiss_path}")

        index = faiss.read_index(faiss_path)

        # Create instance
        instance = cls(
            embedding_dim=index_data['embedding_dim'],
            index_type=index_data['index_type'],
            metric=index_data['metric'],
            use_gpu=use_gpu
        )

        # Replace index
        if use_gpu and faiss.get_num_gpus() > 0:
            instance.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        else:
            instance.index = index

        # Set IDs and metadata
        instance.ids = index_data['ids']
        instance.metadata = index_data['metadata']

        logger.info(f"Loaded index with {instance.index.ntotal} embeddings")
        return instance


def create_vector_search_index(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    id_column: Optional[str] = None,
    metadata_columns: Optional[List[str]] = None,
    index_type: str = 'flat',
    metric: str = 'cosine',
    use_gpu: bool = False
) -> VectorSearchIndex:
    """
    Create vector search index from embeddings and DataFrame

    Args:
        embeddings: Array of embeddings
        df: DataFrame with metadata
        id_column: Column to use as ID (optional)
        metadata_columns: Columns to include in metadata (optional)
        index_type: Type of index ('flat', 'ivf', or 'hnsw')
        metric: Distance metric ('cosine' or 'l2')
        use_gpu: Whether to use GPU for search

    Returns:
        VectorSearchIndex
    """
    logger.info("Creating vector search index")

    # Get embedding dimension
    embedding_dim = embeddings.shape[1]

    # Create index
    index = VectorSearchIndex(
        embedding_dim=embedding_dim,
        index_type=index_type,
        metric=metric,
        use_gpu=use_gpu
    )

    # Prepare IDs
    ids = None
    if id_column is not None and id_column in df.columns:
        ids = df[id_column].tolist()

    # Prepare metadata
    metadata = None
    if metadata_columns is not None:
        # Use only columns that exist in the DataFrame
        valid_columns = [col for col in metadata_columns if col in df.columns]
        if valid_columns:
            metadata = df[valid_columns].to_dict('records')

    # Add embeddings to index
    index.add_embeddings(embeddings, ids, metadata)

    return index


def search_similar_items(
    index: VectorSearchIndex,
    query_embedding: np.ndarray,
    k: int = 5,
    df: Optional[pd.DataFrame] = None,
    id_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Search for similar items

    Args:
        index: Vector search index
        query_embedding: Query embedding
        k: Number of results to return
        df: DataFrame with item data (optional)
        id_column: Column containing item IDs (optional)

    Returns:
        DataFrame with search results
    """
    logger.info(f"Searching for {k} similar items")

    # Perform search
    distances, result_ids, result_metadata = index.search(query_embedding, k)

    # Create results DataFrame
    results = pd.DataFrame({
        'id': result_ids,
        'distance': distances,
        'similarity': 1 - (distances / 2) if index.metric == 'cosine' else 1 / (1 + distances)
    })

    # Add metadata
    for i, metadata in enumerate(result_metadata):
        for key, value in metadata.items():
            if key not in results.columns:
                results[key] = None
            results.at[i, key] = value

    # Join with original DataFrame if provided
    if df is not None and id_column is not None:
        # Convert IDs to the same type as in the DataFrame
        id_type = df[id_column].dtype
        results['id'] = results['id'].astype(id_type)

        # Join with original DataFrame
        results = results.merge(
            df,
            left_on='id',
            right_on=id_column,
            how='left',
            suffixes=('', '_original')
        )

    return results


def batch_search(
    index: VectorSearchIndex,
    query_embeddings: np.ndarray,
    k: int = 5
) -> List[pd.DataFrame]:
    """
    Perform batch search for multiple queries

    Args:
        index: Vector search index
        query_embeddings: Array of query embeddings
        k: Number of results per query

    Returns:
        List of DataFrames with search results
    """
    logger.info(f"Performing batch search for {len(query_embeddings)} queries")

    results = []
    for i, embedding in enumerate(query_embeddings):
        # Search for similar items
        distances, result_ids, result_metadata = index.search(embedding.reshape(1, -1), k)

        # Create results DataFrame
        result_df = pd.DataFrame({
            'query_id': i,
            'result_id': result_ids,
            'distance': distances,
            'similarity': 1 - (distances / 2) if index.metric == 'cosine' else 1 / (1 + distances)
        })

        # Add metadata
        for j, metadata in enumerate(result_metadata):
            for key, value in metadata.items():
                if key not in result_df.columns:
                    result_df[key] = None
                result_df.at[j, key] = value

        results.append(result_df)

    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    from src.nlp.embeddings import load_embeddings

    parser = argparse.ArgumentParser(description='Create and search vector index')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings file')
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--output', required=True, help='Path to save index')
    parser.add_argument('--id_column', help='Column to use as ID')
    parser.add_argument('--metadata', nargs='+', help='Columns to include in metadata')
    parser.add_argument('--index_type', choices=['flat', 'ivf', 'hnsw'],
                        default='flat', help='Type of index')
    parser.add_argument('--metric', choices=['cosine', 'l2'],
                        default='cosine', help='Distance metric')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for search')

    args = parser.parse_args()

    # Load embeddings
    embeddings, _, _ = load_embeddings(args.embeddings)

    # Load data
    df = pd.read_csv(args.data)

    # Create index
    index = create_vector_search_index(
        embeddings,
        df,
        id_column=args.id_column,
        metadata_columns=args.metadata,
        index_type=args.index_type,
        metric=args.metric,
        use_gpu=args.gpu
    )

    # Save index
    index.save(args.output)

    # Example search
    if len(embeddings) > 0:
        query_embedding = embeddings[0]
        results = search_similar_items(index, query_embedding, k=5, df=df, id_column=args.id_column)
        print("Example search results:")
        print(results)