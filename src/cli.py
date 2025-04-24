"""
Command-line interface for retail analytics
"""
import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/cli.log')
        ]
    )
logger = logging.getLogger(name)

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

def train_models(args):
    """Train models based on arguments"""
    logger.info("Training models")

    if args.model == 'all' or args.model == 'forecasting':
        logger.info("Training forecasting model")
        from src.models.forecasting import train_forecasting_model
        train_forecasting_model()

    if args.model == 'all' or args.model == 'segmentation':
        logger.info("Training segmentation model")
        from src.models.segmentation import train_segmentation_model
        train_segmentation_model()

    if args.model == 'all' or args.model == 'sentiment':
        logger.info("Training sentiment model")
        from src.models.sentiment import train_sentiment_model
        train_sentiment_model()

    logger.info("Model training complete")

def evaluate_models(args):
    """Evaluate models based on arguments"""
    logger.info("Evaluating models")

    if args.model == 'all' or args.model == 'forecasting':
        logger.info("Evaluating forecasting model")
        from src.models.forecasting import evaluate_forecasting_model
        evaluate_forecasting_model()

    if args.model == 'all' or args.model == 'segmentation':
        logger.info("Evaluating segmentation model")
        from src.models.segmentation import evaluate_segmentation_model
        evaluate_segmentation_model()

    if args.model == 'all' or args.model == 'sentiment':
        logger.info("Evaluating sentiment model")
        from src.models.sentiment import evaluate_sentiment_model
        evaluate_sentiment_model()

    logger.info("Model evaluation complete")

def preprocess_data(args):
    """Preprocess data based on arguments"""
    logger.info("Preprocessing data")

    from src.data.preprocessing import preprocess_sales_data, preprocess_review_data

    if args.data_type == 'all' or args.data_type == 'sales':
        logger.info("Preprocessing sales data")
        preprocess_sales_data(args.input_path, args.output_path)

    if args.data_type == 'all' or args.data_type == 'reviews':
        logger.info("Preprocessing review data")
        preprocess_review_data(args.input_path, args.output_path)

    logger.info("Data preprocessing complete")

def generate_features(args):
    """Generate features based on arguments"""
    logger.info("Generating features")

    from src.data.feature_engineering import generate_sales_features, generate_review_features

    if args.data_type == 'all' or args.data_type == 'sales':
        logger.info("Generating sales features")
        generate_sales_features(args.input_path, args.output_path)

    if args.data_type == 'all' or args.data_type == 'reviews':
        logger.info("Generating review features")
        generate_review_features(args.input_path, args.output_path)

    logger.info("Feature generation complete")

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description='Retail Analytics CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', choices=['all', 'forecasting', 'segmentation', 'sentiment'],
                            default='all', help='Model to train')
    train_parser.set_defaults(func=train_models)

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    evaluate_parser.add_argument('--model', choices=['all', 'forecasting', 'segmentation', 'sentiment'],
                                default='all', help='Model to evaluate')
    evaluate_parser.set_defaults(func=evaluate_models)

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument('--data-type', choices=['all', 'sales', 'reviews'],
                                default='all', help='Type of data to preprocess')
    preprocess_parser.add_argument('--input-path', type=str, default='data/raw',
                                help='Path to input data')
    preprocess_parser.add_argument('--output-path', type=str, default='data/processed',
                                help='Path to output processed data')
    preprocess_parser.set_defaults(func=preprocess_data)

    # Feature engineering command
    feature_parser = subparsers.add_parser('features', help='Generate features')
    feature_parser.add_argument('--data-type', choices=['all', 'sales', 'reviews'],
                            default='all', help='Type of data to generate features for')
    feature_parser.add_argument('--input-path', type=str, default='data/processed',
                            help='Path to input processed data')
    feature_parser.add_argument('--output-path', type=str, default='data/features',
                            help='Path to output feature data')
    feature_parser.set_defaults(func=generate_features)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()