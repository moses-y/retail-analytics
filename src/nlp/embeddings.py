"""
Text embedding generation module for retail analytics
"""
import os
import logging
import pickle
import yaml
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

    return config.get("embeddings", {})


def preprocess_text_for_embedding(text: str) -> str:
    """
    Preprocess text for embedding generation

    Args:
        text: Raw text

    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def generate_tfidf_embeddings(
    texts: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Generate TF-IDF embeddings for texts

    Args:
        texts: List of texts to embed
        max_features: Maximum number of features
        ngram_range: Range of n-grams to consider

    Returns:
        Tuple of (embeddings, vectorizer)
    """
    logger.info(f"Generating TF-IDF embeddings for {len(texts)} texts")

    # Preprocess texts
    processed_texts = [preprocess_text_for_embedding(text) for text in texts]

    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )

    # Generate embeddings
    embeddings = vectorizer.fit_transform(processed_texts)

    logger.info(f"Generated TF-IDF embeddings with {embeddings.shape[1]} dimensions")
    return embeddings, vectorizer


def generate_sentence_transformer_embeddings(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings using Sentence Transformers

    Args:
        texts: List of texts to embed
        model_name: Name of the Sentence Transformer model
        batch_size: Batch size for embedding generation

    Returns:
        Array of embeddings
    """
    logger.info(f"Generating Sentence Transformer embeddings for {len(texts)} texts using {model_name}")

    # Preprocess texts
    processed_texts = [preprocess_text_for_embedding(text) for text in texts]

    # Load model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(
        processed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    logger.info(f"Generated Sentence Transformer embeddings with {embeddings.shape[1]} dimensions")
    return embeddings


def generate_huggingface_embeddings(
    texts: List[str],
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    batch_size: int = 32,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Generate embeddings using Hugging Face transformers

    Args:
        texts: List of texts to embed
        model_name: Name of the Hugging Face model
        batch_size: Batch size for embedding generation
        device: Device to use ('cpu' or 'cuda')

    Returns:
        Array of embeddings
    """
    logger.info(f"Generating Hugging Face embeddings for {len(texts)} texts using {model_name}")

    # Preprocess texts
    processed_texts = [preprocess_text_for_embedding(text) for text in texts]

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Move model to device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()

    # Generate embeddings in batches
    embeddings = []
    for i in range(0, len(processed_texts), batch_size):
        batch_texts = processed_texts[i:i+batch_size]

        # Tokenize
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        # Generate embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Use mean pooling to get sentence embeddings
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.sum(input_mask_expanded, 1)
        batch_embeddings = sum_embeddings / sum_mask

        embeddings.append(batch_embeddings.cpu().numpy())

    # Combine batches
    embeddings = np.vstack(embeddings)

    logger.info(f"Generated Hugging Face embeddings with {embeddings.shape[1]} dimensions")
    return embeddings


def generate_embeddings(
    df: pd.DataFrame,
    text_column: str = 'review_text',
    method: str = 'sentence_transformer',
    model_name: Optional[str] = None,
    batch_size: int = 32
) -> Tuple[np.ndarray, Any]:
    """
    Generate embeddings for texts in a DataFrame

    Args:
        df: DataFrame with text data
        text_column: Column containing text
        method: Embedding method ('tfidf', 'sentence_transformer', or 'huggingface')
        model_name: Name of the model to use (optional)
        batch_size: Batch size for embedding generation

    Returns:
        Tuple of (embeddings, model/vectorizer)
    """
    logger.info(f"Generating embeddings using {method} method")

    # Load configuration
    config = load_config()

    # Get texts
    texts = df[text_column].fillna("").tolist()

    # Generate embeddings based on method
    if method == 'tfidf':
        # Get parameters from config or use defaults
        max_features = config.get("tfidf", {}).get("max_features", 5000)
        ngram_range = tuple(config.get("tfidf", {}).get("ngram_range", [1, 2]))

        embeddings, vectorizer = generate_tfidf_embeddings(
            texts,
            max_features=max_features,
            ngram_range=ngram_range
        )
        return embeddings, vectorizer

    elif method == 'sentence_transformer':
        # Get model name from config, parameter, or use default
        if model_name is None:
            model_name = config.get("sentence_transformer", {}).get("model_name", 'all-MiniLM-L6-v2')

        embeddings = generate_sentence_transformer_embeddings(
            texts,
            model_name=model_name,
            batch_size=batch_size
        )
        return embeddings, model_name

    elif method == 'huggingface':
        # Get model name from config, parameter, or use default
        if model_name is None:
            model_name = config.get("huggingface", {}).get("model_name", 'sentence-transformers/all-MiniLM-L6-v2')

        device = config.get("huggingface", {}).get("device", 'cpu')

        embeddings = generate_huggingface_embeddings(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            device=device
        )
        return embeddings, model_name

    else:
        logger.warning(f"Unsupported method: {method}, using sentence_transformer")
        model_name = 'all-MiniLM-L6-v2' if model_name is None else model_name
        embeddings = generate_sentence_transformer_embeddings(
            texts,
            model_name=model_name,
            batch_size=batch_size
        )
        return embeddings, model_name


def save_embeddings(
    embeddings: np.ndarray,
    model_info: Any,
    output_path: str,
    method: str = 'sentence_transformer'
) -> None:
    """
    Save embeddings and model information

    Args:
        embeddings: Array of embeddings
        model_info: Model or vectorizer used for embedding generation
        output_path: Path to save embeddings
        method: Embedding method
    """
    logger.info(f"Saving embeddings to {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save data
    embedding_data = {
        'embeddings': embeddings,
        'model_info': model_info,
        'method': method
    }

    with open(output_path, 'wb') as f:
        pickle.dump(embedding_data, f)

    logger.info(f"Saved embeddings with shape {embeddings.shape}")


def load_embeddings(input_path: str) -> Tuple[np.ndarray, Any, str]:
    """
    Load embeddings and model information

    Args:
        input_path: Path to load embeddings from

    Returns:
        Tuple of (embeddings, model_info, method)
    """
    logger.info(f"Loading embeddings from {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Embeddings file not found: {input_path}")

    with open(input_path, 'rb') as f:
        embedding_data = pickle.load(f)

    embeddings = embedding_data['embeddings']
    model_info = embedding_data['model_info']
    method = embedding_data['method']

    logger.info(f"Loaded embeddings with shape {embeddings.shape}")
    return embeddings, model_info, method


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: Optional[List[Any]] = None,
    method: str = 'pca',
    n_components: int = 2,
    output_path: Optional[str] = None
) -> None:
    """
    Visualize embeddings in 2D or 3D

    Args:
        embeddings: Array of embeddings
        labels: Labels for coloring points (optional)
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of components (2 or 3)
        output_path: Path to save the visualization (optional)
    """
    logger.info(f"Visualizing embeddings using {method} with {n_components} components")

    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        logger.warning(f"Unsupported method: {method}, using PCA")
        reducer = PCA(n_components=n_components)

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Create plot
    fig = plt.figure(figsize=(10, 8))

    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')

        if labels is not None:
            # Plot with labels
            unique_labels = set(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = np.array(labels) == label
                ax.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    reduced_embeddings[mask, 2],
                    c=[color],
                    label=str(label),
                    alpha=0.7
                )
        else:
            # Plot without labels
            ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                reduced_embeddings[:, 2],
                alpha=0.7
            )

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

    else:  # 2D plot
        if labels is not None:
            # Plot with labels
            unique_labels = set(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = np.array(labels) == label
                plt.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    c=[color],
                    label=str(label),
                    alpha=0.7
                )
        else:
            # Plot without labels
            plt.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                alpha=0.7
            )

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

    plt.title(f'Embedding Visualization ({method.upper()})')

    if labels is not None:
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Embedding visualization saved to {output_path}")

    plt.close()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Generate text embeddings')
    parser.add_argument('--input', required=True, help='Path to input data file')
    parser.add_argument('--output', required=True, help='Path to save embeddings')
    parser.add_argument('--text_column', default='review_text', help='Column containing text')
    parser.add_argument('--method', choices=['tfidf', 'sentence_transformer', 'huggingface'],
                        default='sentence_transformer', help='Embedding method')
    parser.add_argument('--model', help='Model name (optional)')
    parser.add_argument('--visualize', action='store_true', help='Visualize embeddings')
    parser.add_argument('--label_column', help='Column to use for visualization labels (optional)')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Generate embeddings
    embeddings, model_info = generate_embeddings(
        df,
        text_column=args.text_column,
        method=args.method,
        model_name=args.model
    )

    # Save embeddings
    save_embeddings(embeddings, model_info, args.output, args.method)

    # Visualize embeddings if requested
    if args.visualize:
        labels = None
        if args.label_column and args.label_column in df.columns:
            labels = df[args.label_column].tolist()

        vis_path = os.path.join(os.path.dirname(args.output), "embedding_visualization.png")
        visualize_embeddings(embeddings, labels, output_path=vis_path)