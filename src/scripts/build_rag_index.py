"""
Script to build and populate the RAG vector database (ChromaDB)
"""
import os
# Set environment variable to prevent TensorFlow loading attempt by transformers
os.environ['USE_TF'] = '0'
import sys
import logging
import pandas as pd
import numpy as np

# Add project root to path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from api.dependencies import get_embedding_model, get_vector_db, get_model_config
from src.data.preprocessing import load_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("build_rag_index")

def populate_vector_db():
    """Loads processed reviews, generates embeddings, and populates ChromaDB."""
    logger.info("Starting RAG index build process...")

    processed_reviews_path = os.path.join(project_root, "data", "processed", "product_reviews_processed.csv")

    if not os.path.exists(processed_reviews_path):
        logger.error(f"Processed reviews file not found at: {processed_reviews_path}")
        logger.error("Please run the preprocessing script first (e.g., python src/data/preprocessing.py --input data/raw/product_reviews.csv --output data/processed/product_reviews_processed.csv --type product_reviews)")
        return

    try:
        # Load processed data
        logger.info(f"Loading processed reviews from {processed_reviews_path}")
        reviews_df = pd.read_csv(processed_reviews_path, sep=',') # Ensure correct separator
        logger.info(f"Loaded {len(reviews_df)} reviews.")

        # Ensure necessary columns exist
        required_cols = ['review_id', 'product_id', 'product', 'rating', 'review_text']
        if not all(col in reviews_df.columns for col in required_cols):
            logger.error(f"Missing required columns in processed reviews file. Needed: {required_cols}, Found: {reviews_df.columns.tolist()}")
            return

        # Drop rows with missing review text
        reviews_df.dropna(subset=['review_text'], inplace=True)
        reviews_df = reviews_df[reviews_df['review_text'].str.strip() != '']
        if reviews_df.empty:
            logger.error("No valid reviews found after cleaning.")
            return
        logger.info(f"Processing {len(reviews_df)} reviews after cleaning.")

        # Get embedding model
        logger.info("Loading embedding model...")
        embedding_model = get_embedding_model()
        logger.info("Embedding model loaded.")

        # Generate embeddings
        logger.info("Generating embeddings for review text...")
        texts_to_embed = reviews_df['review_text'].tolist()
        embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=True)
        logger.info(f"Generated {len(embeddings)} embeddings.")

        # Prepare data for ChromaDB
        ids = reviews_df['review_id'].astype(str).tolist()
        documents = reviews_df['review_text'].tolist()
        metadata = reviews_df[['product_id', 'product', 'rating']].astype(str).to_dict(orient='records')

        # Get ChromaDB collection
        logger.info("Connecting to vector database and getting collection...")
        vector_db_collection = get_vector_db() # This will create if not exists
        logger.info(f"Using collection: {vector_db_collection.name}")

        # Add data to ChromaDB collection in batches
        batch_size = 100 # Adjust batch size as needed
        num_batches = int(np.ceil(len(ids) / batch_size))
        logger.info(f"Adding data to collection in {num_batches} batches of size {batch_size}...")

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(ids))
            logger.info(f"Adding batch {i+1}/{num_batches} (indices {start_idx}-{end_idx-1})")

            batch_ids = ids[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx].tolist() # ChromaDB expects lists
            batch_documents = documents[start_idx:end_idx]
            batch_metadata = metadata[start_idx:end_idx]

            try:
                vector_db_collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadata
                )
            except Exception as batch_error:
                 logger.error(f"Error adding batch {i+1}: {batch_error}")
                 # Optionally decide whether to continue or stop on batch error
                 # continue

        logger.info(f"Successfully added {len(ids)} items to the vector database collection '{vector_db_collection.name}'.")

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found during RAG index build: {fnf_error}")
    except KeyError as ke_error:
        logger.error(f"Missing expected column during RAG index build: {ke_error}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during RAG index build: {e}")

if __name__ == "__main__":
    populate_vector_db()
