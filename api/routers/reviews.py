"""
Router for product review analysis endpoints
"""
import logging
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from api.models import (
    ReviewAnalysisRequest,
    ReviewAnalysisResponse,
    ReviewAnalysisResult,
    FeatureSentiment,
    ProductSummary,
    Sentiment,
    RAGQuery,
    RAGResponse,
    ErrorResponse
)
from api.dependencies import (
    get_sentiment_model,
    get_embedding_model,
    get_vector_db,
    get_rag_model,
    get_review_data,
    verify_api_key
)

# Setup logging
logger = logging.getLogger("api.reviews")

# Create router
router = APIRouter()


@router.post(
    "/reviews/analyze",
    response_model=ReviewAnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Analyze product reviews",
    description="Analyze product reviews for sentiment, feature extraction, and summarization"
)
async def analyze_reviews(
    request: ReviewAnalysisRequest,
    api_key_valid: bool = Depends(verify_api_key),
    sentiment_model=Depends(get_sentiment_model)
):
    """Analyze product reviews for sentiment, feature extraction, and summarization"""
    try:
        # Check if we have reviews to analyze
        if not request.reviews:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No reviews provided for analysis"
            )
        
        # Initialize results list
        results = []
        
        # Process each review
        for review in request.reviews:
            # Analyze sentiment if requested
            sentiment_score = None
            sentiment = None
            features = None
            
            if request.include_sentiment:
                # Get sentiment from model
                sentiment_result = sentiment_model(review.review_text)
                
                if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                    # Map sentiment label to our enum
                    label = sentiment_result[0]["label"]
                    score = sentiment_result[0]["score"]
                    
                    if label == "POSITIVE":
                        sentiment_score = score
                        sentiment = Sentiment.POSITIVE
                    elif label == "NEGATIVE":
                        sentiment_score = -score
                        sentiment = Sentiment.NEGATIVE
                    else:
                        sentiment_score = 0.0
                        sentiment = Sentiment.NEUTRAL
            
            # Extract features if requested
            if request.include_features:
                features = extract_features(review.review_text, sentiment_model)
            
            # Create result
            result = ReviewAnalysisResult(
                review_id=review.review_id or f"REV{random.randint(10000, 99999)}",
                product=review.product,
                category=review.category,
                rating=review.rating,
                sentiment_score=sentiment_score,
                sentiment=sentiment,
                features=features
            )
            
            results.append(result)
        
        # Generate product summaries if requested
        product_summaries = None
        if request.include_summary:
            product_summaries = generate_product_summaries(results)
        
        # Create response
        response = ReviewAnalysisResponse(
            results=results,
            product_summaries=product_summaries,
            model_version="distilbert-v1.0.0",
            created_at=datetime.now()
        )
        
        return response
    
    except Exception as e:
        logger.exception("Error analyzing reviews")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing reviews: {str(e)}"
        )


@router.post(
    "/reviews/rag",
    response_model=RAGResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Query product reviews with RAG",
    description="Use Retrieval-Augmented Generation to answer queries about products based on reviews"
)
async def query_reviews_rag(
    query: RAGQuery,
    api_key_valid: bool = Depends(verify_api_key),
    embedding_model=Depends(get_embedding_model),
    vector_db=Depends(get_vector_db),
    rag_model=Depends(get_rag_model),
    review_data=Depends(get_review_data)
):
    """Query product reviews using RAG"""
    try:
        # Check if we have a query
        if not query.query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No query provided"
            )
        
        # Generate embedding for the query
        query_embedding = embedding_model.encode(query.query)
        
        # Prepare filters
        filter_dict = {}
        if query.products:
            filter_dict["product"] = {"$in": query.products}
        if query.categories:
            filter_dict["category"] = {"$in": query.categories}
        
        # Query the vector database
        try:
            results = vector_db.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=query.max_results,
                where=filter_dict if filter_dict else None
            )
            
            # If no results from vector DB, fall back to random sampling
            if not results or len(results.get("ids", [[]])[0]) == 0:
                logger.warning("No results from vector DB, falling back to random sampling")
                
                # Filter review data
                filtered_reviews = review_data
                if query.products:
                    filtered_reviews = filtered_reviews[filtered_reviews["product"].isin(query.products)]
                if query.categories:
                    filtered_reviews = filtered_reviews[filtered_reviews["category"].isin(query.categories)]
                
                # Sample reviews
                if len(filtered_reviews) > 0:
                    sampled_reviews = filtered_reviews.sample(min(query.max_results, len(filtered_reviews)))
                    
                    # Create sources
                    sources = []
                    for _, review in sampled_reviews.iterrows():
                        sources.append({
                            "review_id": review.get("review_id", f"REV{random.randint(10000, 99999)}"),
                            "product": review["product"],
                            "rating": review["rating"],
                            "review_text": review["review_text"],
                            "relevance_score": 0.5  # Placeholder score
                        })
                else:
                    # No reviews available
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No reviews available for the specified filters"
                    )
            else:
                # Process vector DB results
                sources = []
                for i, doc_id in enumerate(results["ids"][0]):
                    # Get metadata and document
                    metadata = results["metadatas"][0][i] if "metadatas" in results else {}
                    document = results["documents"][0][i] if "documents" in results else ""
                    
                    # Create source entry
                    sources.append({
                        "review_id": metadata.get("review_id", f"REV{random.randint(10000, 99999)}"),
                        "product": metadata.get("product", "Unknown"),
                        "rating": metadata.get("rating", 3),
                        "review_text": document,
                        "relevance_score": float(results["distances"][0][i]) if "distances" in results else 0.5
                    })
        except Exception as e:
            logger.exception(f"Error querying vector database: {e}")
            
            # Fall back to random sampling from review data
            sampled_reviews = review_data.sample(min(query.max_results, len(review_data)))
            
            # Create sources
            sources = []
            for _, review in sampled_reviews.iterrows():
                sources.append({
                    "review_id": review.get("review_id", f"REV{random.randint(10000, 99999)}"),
                    "product": review["product"],
                    "rating": review["rating"],
                    "review_text": review["review_text"],
                    "relevance_score": 0.5  # Placeholder score
                })
        
        # Extract products mentioned in sources
        products_mentioned = list(set(source["product"] for source in sources))
        
        # Generate RAG prompt
        prompt = generate_rag_prompt(query.query, sources)
        
        # Generate answer using RAG model
        try:
            response = rag_model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            logger.exception(f"Error generating RAG response: {e}")
            
            # Fall back to a simple answer
            answer = generate_fallback_answer(query.query, sources)
        
        # Create response
        response = RAGResponse(
            answer=answer,
            sources=sources,
            products_mentioned=products_mentioned,
            created_at=datetime.now()
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing RAG query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing RAG query: {str(e)}"
        )


@router.get(
    "/reviews/products",
    response_model=List[Dict[str, Any]],
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Get product information",
    description="Get information about available products and their reviews"
)
async def get_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_rating: Optional[float] = Query(None, description="Filter by minimum rating", ge=1, le=5),
    api_key_valid: bool = Depends(verify_api_key),
    review_data=Depends(get_review_data)
):
    """Get information about available products and their reviews"""
    try:
        # Filter by category if provided
        filtered_data = review_data
        if category:
            filtered_data = filtered_data[filtered_data["category"] == category]
        
        # Filter by minimum rating if provided
        if min_rating is not None:
            filtered_data = filtered_data[filtered_data["rating"] >= min_rating]
        
        # Group by product
        product_info = []
        for product, group in filtered_data.groupby("product"):
            # Get product category
            product_category = group["category"].iloc[0]
            
            # Calculate metrics
            avg_rating = group["rating"].mean()
            review_count = len(group)
            
            # Count sentiment distribution if available
            sentiment_distribution = {}
            if "sentiment" in group.columns:
                for sentiment, count in group["sentiment"].value_counts().items():
                    sentiment_distribution[sentiment] = count / review_count
            
            # Create product info
            info = {
                "product": product,
                "category": product_category,
                "average_rating": round(avg_rating, 2),
                "review_count": review_count,
                "sentiment_distribution": sentiment_distribution or None
            }
            
            product_info.append(info)
        
        return product_info
    
    except Exception as e:
        logger.exception("Error retrieving product information")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving product information: {str(e)}"
        )


@router.get(
    "/reviews/features",
    response_model=Dict[str, List[Dict[str, Any]]],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Get product features",
    description="Get features mentioned in reviews for specific products"
)
async def get_product_features(
    products: List[str] = Query(..., description="List of products to analyze"),
    api_key_valid: bool = Depends(verify_api_key),
    review_data=Depends(get_review_data),
    sentiment_model=Depends(get_sentiment_model)
):
    """Get features mentioned in reviews for specific products"""
    try:
        # Check if we have products to analyze
        if not products:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No products specified"
            )
        
        # Filter reviews for the specified products
        filtered_data = review_data[review_data["product"].isin(products)]
        
        # Check if we have reviews after filtering
        if filtered_data.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No reviews found for the specified products"
            )
        
        # Initialize result dictionary
        result = {}
        
        # Process each product
        for product, group in filtered_data.groupby("product"):
            # Extract features from all reviews
            all_features = []
            for _, review in group.iterrows():
                features = extract_features(review["review_text"], sentiment_model)
                if features:
                    all_features.extend(features)
            
            # Aggregate features
            feature_dict = {}
            for feature in all_features:
                if feature.feature in feature_dict:
                    # Update existing feature
                    existing = feature_dict[feature.feature]
                    existing["mentions"] += feature.mentions
                    existing["sentiment_score"] = (existing["sentiment_score"] * existing["mentions"] + 
                                                 feature.sentiment_score * feature.mentions) / (existing["mentions"] + feature.mentions)
                    
                    # Update sentiment based on new score
                    if existing["sentiment_score"] > 0.1:
                        existing["sentiment"] = "positive"
                    elif existing["sentiment_score"] < -0.1:
                        existing["sentiment"] = "negative"
                    else:
                        existing["sentiment"] = "neutral"
                else:
                    # Add new feature
                    feature_dict[feature.feature] = {
                        "feature": feature.feature,
                        "sentiment_score": feature.sentiment_score,
                        "sentiment": feature.sentiment,
                        "mentions": feature.mentions
                    }
            
            # Sort features by mentions
            sorted_features = sorted(
                feature_dict.values(),
                key=lambda x: x["mentions"],
                reverse=True
            )
            
            # Add to result
            result[product] = sorted_features
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving product features")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving product features: {str(e)}"
        )


def extract_features(review_text: str, sentiment_model) -> List[FeatureSentiment]:
    """Extract features and their sentiment from review text"""
    # List of common product features to look for
    common_features = [
        "battery life", "camera quality", "screen", "display", "performance",
        "design", "price", "durability", "sound quality", "connectivity",
        "user interface", "storage", "comfort", "build quality", "reliability",
        "app integration", "voice recognition", "portability", "weight", "setup",
        "ease of use", "features", "value", "customer service", "warranty"
    ]
    
    # Initialize results
    features = []
    
    # Convert to lowercase for matching
    text_lower = review_text.lower()
    
    # Check for each feature
    for feature in common_features:
        if feature in text_lower:
            # Extract the sentence containing the feature
            sentences = text_lower.split('.')
            feature_sentences = [s for s in sentences if feature in s]
            
            if feature_sentences:
                # Analyze sentiment for the feature
                try:
                    sentiment_result = sentiment_model(feature_sentences[0])
                    
                    if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                        # Map sentiment label to our enum
                        label = sentiment_result[0]["label"]
                        score = sentiment_result[0]["score"]
                        
                        if label == "POSITIVE":
                            sentiment_score = score
                            sentiment = Sentiment.POSITIVE
                        elif label == "NEGATIVE":
                            sentiment_score = -score
                            sentiment = Sentiment.NEGATIVE
                        else:
                            sentiment_score = 0.0
                            sentiment = Sentiment.NEUTRAL
                        
                        # Create feature sentiment
                        feature_sentiment = FeatureSentiment(
                            feature=feature,
                            sentiment_score=sentiment_score,
                            sentiment=sentiment,
                            mentions=1
                        )
                        
                        features.append(feature_sentiment)
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for feature '{feature}': {e}")
    
    return features


def generate_product_summaries(results: List[ReviewAnalysisResult]) -> List[ProductSummary]:
    """Generate product summaries from review analysis results"""
    # Group results by product
    product_groups = {}
    for result in results:
        if result.product not in product_groups:
            product_groups[result.product] = []

        product_groups[result.product].append(result)

    # Generate summaries
    summaries = []
    for product, group in product_groups.items():
        # Get product category
        category = group[0].category

        # Calculate average rating
        average_rating = sum(r.rating for r in group) / len(group)

        # Count sentiment distribution
        sentiment_counts = {s: 0 for s in Sentiment}
        for result in group:
            if result.sentiment:
                sentiment_counts[result.sentiment] += 1

        # Calculate sentiment distribution percentages
        total_with_sentiment = sum(sentiment_counts.values())
        sentiment_distribution = {}
        if total_with_sentiment > 0:
            for sentiment, count in sentiment_counts.items():
                sentiment_distribution[sentiment] = count / total_with_sentiment

        # Collect all features
        all_features = []
        for result in group:
            if result.features:
                all_features.extend(result.features)

        # Aggregate features
        feature_dict = {}
        for feature in all_features:
            if feature.feature in feature_dict:
                # Update existing feature
                existing = feature_dict[feature.feature]
                existing.mentions += feature.mentions
                existing.sentiment_score = (existing.sentiment_score * (existing.mentions - feature.mentions) +
                                           feature.sentiment_score * feature.mentions) / existing.mentions

                # Update sentiment based on new score
                if existing.sentiment_score > 0.1:
                    existing.sentiment = Sentiment.POSITIVE
                elif existing.sentiment_score < -0.1:
                    existing.sentiment = Sentiment.NEGATIVE
                else:
                    existing.sentiment = Sentiment.NEUTRAL
            else:
                # Add new feature
                feature_dict[feature.feature] = feature

        # Get top positive and negative features
        features_list = list(feature_dict.values())
        positive_features = sorted(
            [f for f in features_list if f.sentiment == Sentiment.POSITIVE],
            key=lambda x: (x.sentiment_score * x.mentions),
            reverse=True
        )[:5]  # Top 5 positive features

        negative_features = sorted(
            [f for f in features_list if f.sentiment == Sentiment.NEGATIVE],
            key=lambda x: (x.sentiment_score * x.mentions)
        )[:5]  # Top 5 negative features

        # Generate summary text
        summary_text = generate_summary_text(product, average_rating, positive_features, negative_features)

        # Create product summary
        summary = ProductSummary(
            product=product,
            category=category,
            average_rating=round(average_rating, 2),
            review_count=len(group),
            sentiment_distribution=sentiment_distribution,
            top_positive_features=positive_features,
            top_negative_features=negative_features,
            summary=summary_text
        )

        summaries.append(summary)

    return summaries


def generate_summary_text(product: str, rating: float, positive_features: List[FeatureSentiment], negative_features: List[FeatureSentiment]) -> str:
    """Generate a summary text for a product based on its features"""
    # Start with product name and rating
    summary = f"The {product} has an average rating of {rating:.1f} out of 5. "

    # Add positive features
    if positive_features:
        summary += "Customers particularly like "
        if len(positive_features) == 1:
            summary += f"the {positive_features[0].feature}. "
        elif len(positive_features) == 2:
            summary += f"the {positive_features[0].feature} and {positive_features[1].feature}. "
        else:
            features_text = ", ".join(f.feature for f in positive_features[:-1])
            summary += f"the {features_text}, and {positive_features[-1].feature}. "

    # Add negative features
    if negative_features:
        summary += "However, some concerns were raised about "
        if len(negative_features) == 1:
            summary += f"the {negative_features[0].feature}. "
        elif len(negative_features) == 2:
            summary += f"the {negative_features[0].feature} and {negative_features[1].feature}. "
        else:
            features_text = ", ".join(f.feature for f in negative_features[:-1])
            summary += f"the {features_text}, and {negative_features[-1].feature}. "

    # Add recommendation based on rating
    if rating >= 4.5:
        summary += "Overall, this product is highly recommended."
    elif rating >= 4.0:
        summary += "Overall, this product is recommended."
    elif rating >= 3.0:
        summary += "Overall, this product has mixed reviews."
    else:
        summary += "Overall, customers have significant concerns about this product."

    return summary


def generate_rag_prompt(query: str, sources: List[Dict[str, Any]]) -> str:
    """Generate a prompt for the RAG model"""
    # Start with the system prompt
    prompt = "You are a retail product expert assistant. Answer the following question based only on the provided review information. Be specific and cite products by name when relevant.\n\n"

    # Add the query
    prompt += f"Question: {query}\n\n"

    # Add the sources
    prompt += "Review information:\n"
    for i, source in enumerate(sources, 1):
        prompt += f"Review {i}: Product: {source['product']}, Rating: {source['rating']}/5\n"
        prompt += f"Review text: {source['review_text']}\n\n"

    # Add instructions for the response
    prompt += "Please provide a helpful, accurate answer based solely on the information in these reviews. If the reviews don't contain relevant information to answer the question, say so."

    return prompt


def generate_fallback_answer(query: str, sources: List[Dict[str, Any]]) -> str:
    """Generate a fallback answer when the RAG model fails"""
    # Extract products mentioned in sources
    products = list(set(source["product"] for source in sources))

    # Extract average ratings
    product_ratings = {}
    for source in sources:
        product = source["product"]
        rating = source["rating"]

        if product not in product_ratings:
            product_ratings[product] = {"sum": 0, "count": 0}

        product_ratings[product]["sum"] += rating
        product_ratings[product]["count"] += 1

    # Calculate average ratings
    for product, data in product_ratings.items():
        data["average"] = data["sum"] / data["count"]

    # Generate a simple answer based on the query and sources
    if "best" in query.lower() or "recommend" in query.lower() or "top" in query.lower():
        # Sort products by rating
        sorted_products = sorted(product_ratings.items(), key=lambda x: x[1]["average"], reverse=True)

        if sorted_products:
            top_product = sorted_products[0][0]
            top_rating = sorted_products[0][1]["average"]

            answer = f"Based on the reviews, the {top_product} appears to be the highest rated with an average rating of {top_rating:.1f}/5. "

            # Add a mention of features if we can extract them
            features_mentioned = []
            for source in sources:
                if source["product"] == top_product:
                    text = source["review_text"].lower()
                    for feature in ["battery", "camera", "screen", "display", "performance", "design", "price"]:
                        if feature in text and feature not in features_mentioned:
                            features_mentioned.append(feature)

            if features_mentioned:
                answer += f"Customers particularly mentioned the {', '.join(features_mentioned)}."

            return answer
        else:
            return "I couldn't find enough information in the reviews to answer your question about the best products."

    elif "compare" in query.lower() or "difference" in query.lower() or "versus" in query.lower() or " vs " in query.lower():
        if len(products) >= 2:
            product1 = products[0]
            product2 = products[1]

            rating1 = product_ratings[product1]["average"]
            rating2 = product_ratings[product2]["average"]

            answer = f"Based on the reviews, the {product1} has an average rating of {rating1:.1f}/5, while the {product2} has an average rating of {rating2:.1f}/5. "

            if rating1 > rating2:
                answer += f"The {product1} appears to be rated higher by customers."
            elif rating2 > rating1:
                answer += f"The {product2} appears to be rated higher by customers."
            else:
                answer += "Both products have similar ratings from customers."

            return answer
        else:
            return "I couldn't find enough information to compare different products based on the reviews."

    else:
        # Generic answer for other types of queries
        return f"Based on the {len(sources)} reviews I analyzed, I found information about {', '.join(products)}. To get more specific insights, please ask about particular products or features you're interested in."