"""
RAG (Retrieval-Augmented Generation) implementation with Google Gemini
"""
import os
import logging
import yaml
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

from src.nlp.embeddings import generate_sentence_transformer_embeddings, preprocess_text_for_embedding
from src.nlp.vector_search import VectorSearchIndex, search_similar_items

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


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

    return config.get("rag", {})


def setup_gemini_api() -> None:
    """
    Setup Google Gemini API with API key
    """
    logger.info("Setting up Google Gemini API")

    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    # Configure Gemini API
    genai.configure(api_key=api_key)


def get_available_models() -> List[str]:
    """
    Get list of available Gemini models

    Returns:
        List of model names
    """
    logger.info("Getting available Gemini models")

    try:
        models = genai.list_models()
        model_names = [model.name for model in models]
        logger.info(f"Available models: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return []


def create_gemini_model(model_name: Optional[str] = None) -> Any:
    """
    Create Gemini model instance

    Args:
        model_name: Name of the model to use (optional)

    Returns:
        Gemini model instance
    """
    # Load configuration
    config = load_config()

    # Use provided model name or get from config
    if model_name is None:
        model_name = config.get("model_name", "gemini-pro")

    logger.info(f"Creating Gemini model: {model_name}")

    # Create model
    model = genai.GenerativeModel(model_name)

    return model


def generate_text(
    model: Any,
    prompt: str,
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
    top_p: float = 0.95,
    top_k: int = 40
) -> str:
    """
    Generate text using Gemini model

    Args:
        model: Gemini model instance
        prompt: Text prompt
        temperature: Sampling temperature
        max_output_tokens: Maximum number of tokens to generate
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter

    Returns:
        Generated text
    """
    logger.info("Generating text with Gemini")

    try:
        # Set generation config
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }

        # Generate text
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Extract text from response
        if hasattr(response, 'text'):
            return response.text
        else:
            return str(response)

    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return f"Error generating text: {str(e)}"


def prepare_product_context(
    df: pd.DataFrame,
    product_column: str = 'product',
    category_column: str = 'category',
    review_column: str = 'review_text',
    rating_column: str = 'rating',
    feature_column: Optional[str] = 'feature_mentioned',
    attribute_column: Optional[str] = 'attribute_mentioned',
    sentiment_column: Optional[str] = 'sentiment'
) -> Dict[str, Dict]:
    """
    Prepare product context from review data

    Args:
        df: DataFrame with review data
        product_column: Column containing product names
        category_column: Column containing product categories
        review_column: Column containing review text
        rating_column: Column containing ratings
        feature_column: Column containing mentioned features (optional)
        attribute_column: Column containing mentioned attributes (optional)
        sentiment_column: Column containing sentiment labels (optional)

    Returns:
        Dictionary mapping product names to context dictionaries
    """
    logger.info("Preparing product context from review data")

    # Group by product
    product_context = {}

    for product, group in df.groupby(product_column):
        # Get product category
        category = group[category_column].iloc[0] if category_column in group.columns else "Unknown"

        # Get average rating
        avg_rating = group[rating_column].mean() if rating_column in group.columns else None

        # Get review count
        review_count = len(group)

        # Get sentiment distribution
        sentiment_dist = None
        if sentiment_column in group.columns:
            sentiment_dist = group[sentiment_column].value_counts().to_dict()

        # Get feature mentions
        feature_mentions = None
        if feature_column in group.columns:
            feature_mentions = group[feature_column].value_counts().to_dict()

        # Get attribute mentions
        attribute_mentions = None
        if attribute_column in group.columns:
            attribute_mentions = group[attribute_column].value_counts().to_dict()

        # Get sample reviews (3 highest rated, 3 lowest rated)
        if rating_column in group.columns:
            high_rated = group.nlargest(3, rating_column)[review_column].tolist()
            low_rated = group.nsmallest(3, rating_column)[review_column].tolist()
        else:
            # Random samples if no rating
            sample_reviews = group[review_column].sample(min(6, len(group))).tolist()
            high_rated = sample_reviews[:len(sample_reviews)//2]
            low_rated = sample_reviews[len(sample_reviews)//2:]

        # Create context dictionary
        product_context[product] = {
            "product": product,
            "category": category,
            "avg_rating": avg_rating,
            "review_count": review_count,
            "sentiment_distribution": sentiment_dist,
            "feature_mentions": feature_mentions,
            "attribute_mentions": attribute_mentions,
            "high_rated_reviews": high_rated,
            "low_rated_reviews": low_rated
        }

    logger.info(f"Prepared context for {len(product_context)} products")
    return product_context


def create_vector_index_from_reviews(
    df: pd.DataFrame,
    review_column: str = 'review_text',
    product_column: str = 'product',
    embedding_model: str = 'all-MiniLM-L6-v2',
    index_type: str = 'flat'
) -> Tuple[VectorSearchIndex, np.ndarray]:
    """
    Create vector search index from review data

    Args:
        df: DataFrame with review data
        review_column: Column containing review text
        product_column: Column containing product names
        embedding_model: Name of the embedding model
        index_type: Type of index

    Returns:
        Tuple of (vector_index, embeddings)
    """
    logger.info("Creating vector search index from reviews")

    # Get review texts
    reviews = df[review_column].fillna("").tolist()

    # Generate embeddings
    embeddings = generate_sentence_transformer_embeddings(
        reviews,
        model_name=embedding_model
    )

    # Prepare metadata columns
    metadata_columns = [col for col in df.columns if col != review_column]

    # Create index
    from src.nlp.vector_search import create_vector_search_index

    index = create_vector_search_index(
        embeddings,
        df,
        id_column=None,  # Use default sequential IDs
        metadata_columns=metadata_columns,
        index_type=index_type,
        metric='cosine'
    )

    logger.info(f"Created vector index with {len(reviews)} reviews")
    return index, embeddings


def retrieve_relevant_reviews(
    index: VectorSearchIndex,
    query: str,
    embedding_model: str = 'all-MiniLM-L6-v2',
    k: int = 5,
    df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Retrieve relevant reviews for a query

    Args:
        index: Vector search index
        query: Query text
        embedding_model: Name of the embedding model
        k: Number of results to return
        df: Original DataFrame (optional)

    Returns:
        DataFrame with relevant reviews
    """
    logger.info(f"Retrieving {k} relevant reviews for query: {query}")

    # Generate query embedding
    query_embedding = generate_sentence_transformer_embeddings(
        [query],
        model_name=embedding_model
    )[0]

    # Search for similar reviews
    results = search_similar_items(
        index,
        query_embedding,
        k=k,
        df=df
    )

    return results


def format_context_for_rag(
    product_context: Dict,
    relevant_reviews: Optional[pd.DataFrame] = None,
    product_name: Optional[str] = None
) -> str:
    """
    Format context for RAG prompt

    Args:
        product_context: Product context dictionary
        relevant_reviews: DataFrame with relevant reviews (optional)
        product_name: Name of the product to focus on (optional)

    Returns:
        Formatted context string
    """
    logger.info("Formatting context for RAG")

    context_parts = []

    # Add product-specific context if product name is provided
    if product_name and product_name in product_context:
        product_info = product_context[product_name]

        context_parts.append(f"PRODUCT INFORMATION:")
        context_parts.append(f"Product: {product_info['product']}")
        context_parts.append(f"Category: {product_info['category']}")

        if product_info['avg_rating'] is not None:
            context_parts.append(f"Average Rating: {product_info['avg_rating']:.1f}/5 ({product_info['review_count']} reviews)")

        # Add sentiment distribution if available
        if product_info['sentiment_distribution']:
            sentiment_str = ", ".join([f"{k}: {v}" for k, v in product_info['sentiment_distribution'].items()])
            context_parts.append(f"Sentiment Distribution: {sentiment_str}")

        # Add feature mentions if available
        if product_info['feature_mentions']:
            top_features = sorted(product_info['feature_mentions'].items(), key=lambda x: x[1], reverse=True)[:5]
            feature_str = ", ".join([f"{k} ({v})" for k, v in top_features])
            context_parts.append(f"Top Mentioned Features: {feature_str}")

        # Add attribute mentions if available
        if product_info['attribute_mentions']:
            top_attrs = sorted(product_info['attribute_mentions'].items(), key=lambda x: x[1], reverse=True)[:5]
            attr_str = ", ".join([f"{k} ({v})" for k, v in top_attrs])
            context_parts.append(f"Top Mentioned Attributes: {attr_str}")

        # Add sample reviews
        context_parts.append("\nPOSITIVE REVIEWS:")
        for i, review in enumerate(product_info['high_rated_reviews'], 1):
            context_parts.append(f"{i}. {review}")

        context_parts.append("\nCRITICAL REVIEWS:")
        for i, review in enumerate(product_info['low_rated_reviews'], 1):
            context_parts.append(f"{i}. {review}")

    # Add relevant reviews if provided
    if relevant_reviews is not None and not relevant_reviews.empty:
        context_parts.append("\nRELEVANT REVIEWS:")

        for i, (_, row) in enumerate(relevant_reviews.iterrows(), 1):
            review_text = row.get('review_text', '')
            product = row.get('product', 'Unknown Product')
            rating = row.get('rating', None)

            review_header = f"{i}. Product: {product}"
            if rating is not None:
                review_header += f", Rating: {rating}/5"

            context_parts.append(review_header)
            context_parts.append(f"   {review_text}")

    # Combine all context parts
    formatted_context = "\n".join(context_parts)

    return formatted_context


def create_rag_prompt(
    query: str,
    context: str,
    prompt_template: Optional[str] = None
) -> str:
    """
    Create RAG prompt with query and context

    Args:
        query: User query
        context: Context information
        prompt_template: Custom prompt template (optional)

    Returns:
        Complete RAG prompt
    """
    logger.info("Creating RAG prompt")

    # Load configuration
    config = load_config()

    # Use provided template or default
    if prompt_template is None:
        prompt_template = config.get("prompt_template", """
You are a retail analytics assistant that helps customers understand product reviews and make informed decisions.
Use ONLY the information provided in the context below to answer the question.
If you don't know the answer based on the context, say "I don't have enough information to answer that question."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""")

    # Fill in template
    prompt = prompt_template.format(
        context=context,
        query=query
    )

    return prompt


def answer_product_question(
    query: str,
    df: pd.DataFrame,
    vector_index: VectorSearchIndex,
    product_context: Dict[str, Dict],
    model: Any,
    embedding_model: str = 'all-MiniLM-L6-v2',
    k: int = 5
) -> str:
    """
    Answer product-related question using RAG

    Args:
        query: User query
        df: DataFrame with review data
        vector_index: Vector search index
        product_context: Product context dictionary
        model: Gemini model instance
        embedding_model: Name of the embedding model
        k: Number of relevant reviews to retrieve

    Returns:
        Generated answer
    """
    logger.info(f"Answering product question: {query}")

    # Extract product name from query if possible
    product_name = None
    for product in product_context.keys():
        if product.lower() in query.lower():
            product_name = product
            break

    # Retrieve relevant reviews
    relevant_reviews = retrieve_relevant_reviews(
        vector_index,
        query,
        embedding_model=embedding_model,
        k=k,
        df=df
    )

    # Format context
    context = format_context_for_rag(
        product_context,
        relevant_reviews,
        product_name
    )

    # Create RAG prompt
    prompt = create_rag_prompt(query, context)

    # Generate answer
    answer = generate_text(model, prompt)

    return answer


def compare_products(
    product_names: List[str],
    product_context: Dict[str, Dict],
    model: Any,
    comparison_aspects: Optional[List[str]] = None
) -> str:
    """
    Compare multiple products using RAG

    Args:
        product_names: List of product names to compare
        product_context: Product context dictionary
        model: Gemini model instance
        comparison_aspects: Aspects to compare (optional)

    Returns:
        Generated comparison
    """
    logger.info(f"Comparing products: {', '.join(product_names)}")

    # Filter to only include products that exist in the context
    valid_products = [p for p in product_names if p in product_context]

    if not valid_products:
        return "None of the specified products were found in the database."

    # Default comparison aspects
    if comparison_aspects is None:
        comparison_aspects = [
            "Overall rating and sentiment",
            "Key features and their performance",
            "Common praise points",
            "Common criticism points",
            "Value for money",
            "Reliability and durability"
        ]

    # Create context for each product
    context_parts = []

    for product in valid_products:
        product_info = product_context[product]

        product_context_str = format_context_for_rag(
            {product: product_info},
            None,
            product
        )

        context_parts.append(product_context_str)

    # Combine contexts
    combined_context = "\n\n".join(context_parts)

    # Create comparison prompt
    aspects_str = "\n".join([f"- {aspect}" for aspect in comparison_aspects])

    comparison_prompt = f"""
You are a retail analytics assistant that helps customers compare products based on reviews.
Use ONLY the information provided in the context below to compare the products.
If you don't have enough information for a particular aspect, indicate that.

CONTEXT:
{combined_context}

TASK:
Compare the following products: {', '.join(valid_products)}

Please compare them on these aspects:
{aspects_str}

For each aspect, indicate which product performs better and why, based on the review data.
Conclude with an overall recommendation based on different customer needs.

COMPARISON:
"""

    # Generate comparison
    comparison = generate_text(model, comparison_prompt)

    return comparison


def generate_product_summary(
    product_name: str,
    product_context: Dict[str, Dict],
    model: Any
) -> str:
    """
    Generate comprehensive product summary using RAG

    Args:
        product_name: Name of the product
        product_context: Product context dictionary
        model: Gemini model instance

    Returns:
        Generated product summary
    """
    logger.info(f"Generating summary for product: {product_name}")

    if product_name not in product_context:
        return f"Product '{product_name}' not found in the database."

    # Get product context
    product_info = product_context[product_name]

    # Format context
    context = format_context_for_rag(
        {product_name: product_info},
        None,
        product_name
    )

    # Create summary prompt
    summary_prompt = f"""
You are a retail analytics assistant that creates comprehensive product summaries based on customer reviews.
Use ONLY the information provided in the context below to create the summary.

CONTEXT:
{context}

TASK:
Create a comprehensive summary for {product_name} that includes:
1. A brief product overview
2. Key strengths based on positive reviews
3. Areas for improvement based on critical reviews
4. Most discussed features and how they perform
5. Who this product is best suited for
6. Overall recommendation

FORMAT:
- Use markdown formatting with headers and bullet points
- Be specific and reference actual review content
- Keep the summary concise but comprehensive

SUMMARY:
"""

    # Generate summary
    summary = generate_text(model, summary_prompt)

    return summary


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='RAG implementation for product reviews')
    parser.add_argument('--data', required=True, help='Path to review data file')
    parser.add_argument('--query', help='Query to answer')
    parser.add_argument('--compare', nargs='+', help='Products to compare')
    parser.add_argument('--summary', help='Product to summarize')
    parser.add_argument('--model', default='gemini-pro', help='Gemini model name')
    parser.add_argument('--embedding_model', default='all-MiniLM-L6-v2', help='Embedding model name')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)

    # Setup Gemini API
    setup_gemini_api()

    # Create Gemini model
    model = create_gemini_model(args.model)

    # Prepare product context
    product_context = prepare_product_context(df)

    # Create vector index
    vector_index, embeddings = create_vector_index_from_reviews(
        df,
        embedding_model=args.embedding_model
    )

    # Process based on arguments
    if args.query:
        # Answer question
        answer = answer_product_question(
            args.query,
            df,
            vector_index,
            product_context,
            model,
            embedding_model=args.embedding_model
        )
        print("\nQUESTION:")
        print(args.query)
        print("\nANSWER:")
        print(answer)

    elif args.compare:
        # Compare products
        comparison = compare_products(
            args.compare,
            product_context,
            model
        )
        print("\nPRODUCT COMPARISON:")
        print(comparison)

    elif args.summary:
        # Generate product summary
        summary = generate_product_summary(
            args.summary,
            product_context,
            model
        )
        print("\nPRODUCT SUMMARY:")
        print(summary)

    else:
        print("Please specify --query, --compare, or --summary")