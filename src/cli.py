"""
Command-line interface for retail analytics
"""
import argparse
import argparse
import logging
import sys
import pandas as pd # Add pandas import
from pathlib import Path

# Add project root to path to allow importing project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/cli.log')
        ]
    )
logger = logging.getLogger(__name__) # Use __name__ for logger

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

def train_models(args):
    """Train models based on arguments"""
    logger.info("Training models")

    if args.model == 'all' or args.model == 'forecasting':
        logger.info("Training forecasting model")
        from src.models.forecasting import train_forecasting_model, prepare_forecasting_data
        # Load and prepare data
        data_path = project_root / "data" / "processed" / "retail_sales_data.csv"
        if not data_path.exists():
            logger.error(f"Forecasting training data not found at {data_path}. Run preprocessing first.")
        else:
            logger.info(f"Loading forecasting data from {data_path}")
            df = pd.read_csv(data_path)
            X_dict, y_dict, _ = prepare_forecasting_data(df) # Assuming default settings are okay
            if not X_dict['train']:
                 logger.error("No training data found after preparation.")
            else:
                X_train = pd.concat(X_dict['train'].values())
                y_train = pd.concat(y_dict['train'].values())
                # Train the model
                from src.models.forecasting import save_model # Import save function
                model, _ = train_forecasting_model(X_train, y_train) # Pass data, get model back
                # Save the model
                save_path = project_root / "models" / "forecasting_model.xgb"
                save_model(model, str(save_path), model_type='xgboost')

    if args.model == 'all' or args.model == 'segmentation':
        logger.info("Training segmentation model")
        from src.models.segmentation import train_segmentation_model, load_config as load_segmentation_config
        # Load data (using processed sales data as input for segmentation features)
        data_path = project_root / "data" / "processed" / "retail_sales_data.csv"
        if not data_path.exists():
             logger.error(f"Segmentation training data not found at {data_path}. Run preprocessing first.")
        else:
            logger.info(f"Loading segmentation data from {data_path}")
            df = pd.read_csv(data_path)
            # Get feature columns from config or defaults
            seg_config = load_segmentation_config()
            feature_cols = seg_config.get("kmeans", {}).get("features") # Assuming kmeans is default
            if not feature_cols:
                 logger.warning("Segmentation features not found in config, using defaults.")
                 # Use default logic from prepare_segmentation_data if needed, or define explicitly
                 # For now, let the function handle default feature selection if None is passed
                 feature_cols = None
            # Train the model
            from src.models.segmentation import save_segmentation_model, prepare_segmentation_data # Import save function and prepare data
            # Need scaler from prepare_segmentation_data to save
            features_df, _, scaler = prepare_segmentation_data(df, feature_columns=feature_cols, scale_data=True) # Prepare data with scaling
            # Assuming train_segmentation_model uses features_df implicitly or needs modification
            # Let's assume train_segmentation_model returns the model
            # We need to adjust train_segmentation_model call if it doesn't use prepared features
            # For now, assume it returns the model trained on appropriate data
            # Re-check train_segmentation_model signature if this fails
            model, _ = train_segmentation_model(df, feature_cols=feature_cols) # Pass data and features, get model back
            # Save the model and scaler
            save_path = project_root / "models" / "segmentation_model.pkl"
            save_segmentation_model(model, scaler, str(save_path), model_type='kmeans') # Assuming kmeans

    if args.model == 'all' or args.model == 'sentiment':
        logger.info("Training sentiment model")
        from src.models.sentiment import train_sentiment_model, prepare_sentiment_data, extract_features
        # Load data
        data_path = project_root / "data" / "processed" / "product_reviews_processed.csv"
        if not data_path.exists():
            logger.error(f"Sentiment training data not found at {data_path}. Run preprocessing first.")
        else:
            logger.info(f"Loading sentiment data from {data_path}")
            df = pd.read_csv(data_path)
            # Prepare data (assuming default text/rating columns)
            X_train, X_test, y_train, y_test = prepare_sentiment_data(df)
            # Extract features
            X_train_features, X_test_features, vectorizer = extract_features(X_train, X_test) # Get vectorizer back
            # Train the model
            from src.models.sentiment import save_sentiment_model # Import save function
            model = train_sentiment_model(X_train_features, y_train) # Pass features and labels, get model back
            # Save the model and vectorizer
            save_path = project_root / "models" / "sentiment_model.pkl"
            save_sentiment_model(model, vectorizer, str(save_path)) # Assuming default logistic model type

    logger.info("Model training complete")

def evaluate_models(args):
    """Evaluate models based on arguments"""
    logger.info("Evaluating models")

    if args.model == 'all' or args.model == 'forecasting':
        logger.info("Evaluating forecasting model")
        from src.models.forecasting import evaluate_forecast_model, prepare_forecasting_data, load_model
        from src.data.preprocessing import load_data # Need load_data

        # Define paths
        data_path = project_root / "data" / "processed" / "retail_sales_data.csv"
        model_path = project_root / "models" / "forecasting_model.xgb" # Assuming default XGBoost model path

        if not data_path.exists():
            logger.error(f"Forecasting evaluation data not found at {data_path}.")
        elif not model_path.exists():
            logger.error(f"Forecasting model not found at {model_path}.")
        else:
            try:
                logger.info(f"Loading forecasting data from {data_path}")
                df = load_data(str(data_path))

                logger.info("Preparing data for evaluation")
                # Prepare data to get the test split
                X_dict, y_dict, _ = prepare_forecasting_data(df) # Use default settings

                if not X_dict['test']:
                    logger.error("No test data found for forecasting evaluation.")
                else:
                    # Combine test data from all groups (consistent with training)
                    X_test = pd.concat(X_dict['test'].values())
                    y_test = pd.concat(y_dict['test'].values())

                    logger.info(f"Loading forecasting model from {model_path}")
                    # Assuming default model type is xgboost
                    model = load_model(str(model_path), model_type='xgboost')

                    logger.info("Evaluating forecasting model")
                    evaluate_forecast_model(model, X_test, y_test, model_type='xgboost')

            except Exception as e:
                logger.error(f"Error during forecasting model evaluation: {e}", exc_info=True)


    if args.model == 'all' or args.model == 'segmentation':
        logger.info("Evaluating segmentation model")
        from src.models.segmentation import evaluate_clustering, prepare_segmentation_data, load_segmentation_model, load_config as load_segmentation_config
        from src.data.preprocessing import load_data # Need load_data

        # Define paths
        data_path = project_root / "data" / "processed" / "retail_sales_data.csv"
        model_path = project_root / "models" / "segmentation_model.pkl" # Assuming default KMeans model path

        if not data_path.exists():
            logger.error(f"Segmentation evaluation data not found at {data_path}.")
        elif not model_path.exists():
            logger.error(f"Segmentation model not found at {model_path}.")
        else:
            try:
                logger.info(f"Loading segmentation data from {data_path}")
                df = load_data(str(data_path))

                # Get feature columns from config or defaults (consistent with training)
                seg_config = load_segmentation_config()
                feature_cols = seg_config.get("kmeans", {}).get("features")
                if not feature_cols:
                    logger.warning("Segmentation features not found in config for evaluation, using defaults.")
                    feature_cols = None # Let prepare_segmentation_data handle defaults

                logger.info("Preparing data for evaluation")
                # Prepare data (scaling is important for evaluation metrics)
                features_df, _, _ = prepare_segmentation_data(df, feature_columns=feature_cols, scale_data=True)

                logger.info(f"Loading segmentation model from {model_path}")
                # Assuming default model type is kmeans
                model, _, model_type = load_segmentation_model(str(model_path))

                if model_type != 'kmeans':
                     logger.warning(f"Loaded model type is {model_type}, but evaluation logic assumes KMeans prediction method.")
                     # Add logic here if other model types need different prediction calls

                logger.info("Predicting cluster labels for evaluation data")
                labels = model.predict(features_df)

                logger.info("Evaluating clustering results")
                evaluate_clustering(features_df, labels)

            except Exception as e:
                logger.error(f"Error during segmentation model evaluation: {e}", exc_info=True)


    if args.model == 'all' or args.model == 'sentiment':
        logger.info("Evaluating sentiment model")
        from src.models.sentiment import evaluate_sentiment_model, load_sentiment_model, prepare_sentiment_data, extract_features
        from src.data.preprocessing import load_data # Need load_data

        # Define paths
        data_path = project_root / "data" / "processed" / "product_reviews_processed.csv"
        model_path = project_root / "models" / "sentiment_model.pkl" # Assuming this is the save path

        if not data_path.exists():
            logger.error(f"Sentiment evaluation data not found at {data_path}.")
        elif not model_path.exists():
            logger.error(f"Sentiment model not found at {model_path}.")
        else:
            try:
                logger.info(f"Loading sentiment data from {data_path}")
                df = load_data(str(data_path))

                logger.info("Preparing data for evaluation")
                # Use prepare_sentiment_data to get the test split consistent with training
                _, X_test_text, _, y_test = prepare_sentiment_data(df) # We only need test data here

                logger.info(f"Loading sentiment model from {model_path}")
                model, vectorizer, _ = load_sentiment_model(str(model_path))

                logger.info("Extracting features for test data")
                # Need to pass dummy train data to extract_features or adapt it
                # For simplicity here, we'll re-vectorize only the test set
                # Note: A better approach might save the vectorizer during training
                # and load it here, but load_sentiment_model already does this.
                # We need X_train_features shape to match, but extract_features only needs X_test
                # Let's call extract_features in a way that only transforms X_test
                # We need the vectorizer from load_sentiment_model
                X_test_features = vectorizer.transform(X_test_text)
                logger.info(f"Extracted features for {X_test_features.shape[0]} test samples.")


                # Call evaluation function with required arguments
                evaluate_sentiment_model(model, X_test_features, y_test)
            except Exception as e:
                logger.error(f"Error during sentiment model evaluation: {e}", exc_info=True)


    logger.info("Model evaluation complete")

def preprocess_data(args):
    """Preprocess data based on arguments"""
    logger.info("Preprocessing data")

    # Corrected function name imports and added save function
    from src.data.preprocessing import clean_sales_data, clean_review_data, save_processed_data, load_data

    if args.data_type == 'all' or args.data_type == 'sales':
        input_file = Path(args.input_path) / "retail_sales_data.csv" # Assuming default filename
        output_file = Path(args.output_path) / "retail_sales_data.csv" # Assuming same filename
        if not input_file.exists():
            logger.error(f"Sales input file not found: {input_file}")
        else:
            logger.info(f"Preprocessing sales data from {input_file}")
            df = load_data(str(input_file))
            cleaned_df = clean_sales_data(df)
            save_processed_data(cleaned_df, str(output_file))

    if args.data_type == 'all' or args.data_type == 'reviews':
        input_file = Path(args.input_path) / "product_reviews.csv" # Assuming default filename
        output_file = Path(args.output_path) / "product_reviews_processed.csv" # Specific output name
        if not input_file.exists():
             logger.error(f"Reviews input file not found: {input_file}")
        else:
            logger.info(f"Preprocessing review data from {input_file}")
            df = load_data(str(input_file))
            cleaned_df = clean_review_data(df)
            save_processed_data(cleaned_df, str(output_file))

    logger.info("Data preprocessing complete")

def generate_features(args):
    """Generate features based on arguments"""
    logger.info("Generating features")

    # Corrected function name imports
    from src.data.feature_engineering import create_sales_features, create_review_features, save_feature_data

    # TODO: Need to load data before calling feature creation functions
    # This command needs implementation similar to 'train' and 'evaluate'

    if args.data_type == 'all' or args.data_type == 'sales':
        logger.info("Generating sales features")
        # Need to load data first, e.g., df = pd.read_csv(args.input_path)
        # result_df = create_sales_features(df)
        # save_feature_data(result_df, args.output_path)
        logger.warning("Sales feature generation needs data loading implementation in CLI.")


    if args.data_type == 'all' or args.data_type == 'reviews':
        logger.info("Generating review features")
        # Need to load data first, e.g., df = pd.read_csv(args.input_path)
        # result_df = create_review_features(df)
        # save_feature_data(result_df, args.output_path)
        logger.warning("Review feature generation needs data loading implementation in CLI.")

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
