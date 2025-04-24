"""
Sales forecasting models for retail analytics
"""
import os
import logging
import pickle
import yaml
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from prophet import Prophet

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

    return config.get("forecasting", {})


def prepare_forecasting_data(
    df: pd.DataFrame,
    target_column: str = 'total_sales',
    date_column: str = 'date',
    group_columns: List[str] = ['store_id', 'category'],
    test_size: float = 0.2
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str]]:
    """
    Prepare data for forecasting models

    Args:
        df: DataFrame with time series data
        target_column: Name of the target column
        date_column: Name of the date column
        group_columns: Columns to group by
        test_size: Proportion of data to use for testing

    Returns:
        Tuple of (X_dict, y_dict, feature_names)
    """
    logger.info("Preparing data for forecasting")

    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort by date
    df = df.sort_values(by=[*group_columns, date_column])

    # Drop rows with missing target
    df = df.dropna(subset=[target_column])

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove target and group columns from features, but keep date
    exclude_cols = [target_column] + group_columns
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    categorical_features = [col for col in categorical_cols if col not in exclude_cols and col != date_column]

    # Create dummy variables for categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)

    # Get all feature columns
    feature_cols = [col for col in df_encoded.columns
                   if col not in [target_column] + group_columns]

    # Split data by time
    split_date = df[date_column].max() - pd.Timedelta(days=int(len(df[date_column].unique()) * test_size))

    train_df = df_encoded[df_encoded[date_column] < split_date].copy()
    test_df = df_encoded[df_encoded[date_column] >= split_date].copy()

    logger.info(f"Train data: {train_df.shape}, Test data: {test_df.shape}")

    # Create X and y dictionaries for each group
    X_dict = {'train': {}, 'test': {}}
    y_dict = {'train': {}, 'test': {}}

    # Process each group
    for name, group in train_df.groupby(group_columns):
        # Create group key
        if len(group_columns) == 1:
            group_key = name
        else:
            group_key = "_".join(str(n) for n in name)

        # Store X and y for training
        X_dict['train'][group_key] = group[feature_cols].copy()
        y_dict['train'][group_key] = group[target_column].copy()

    for name, group in test_df.groupby(group_columns):
        # Create group key
        if len(group_columns) == 1:
            group_key = name
        else:
            group_key = "_".join(str(n) for n in name)

        # Store X and y for testing
        X_dict['test'][group_key] = group[feature_cols].copy()
        y_dict['test'][group_key] = group[target_column].copy()

    return X_dict, y_dict, feature_cols


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
    cv: int = 5
) -> xgb.XGBRegressor:
    """
    Train XGBoost model for sales forecasting

    Args:
        X_train: Training features
        y_train: Training target
        params: XGBoost parameters (optional)
        cv: Number of cross-validation folds

    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model")

    # Load default parameters from config
    config = load_config()
    default_params = config.get("xgboost", {}).get("params", {})

    # Use provided parameters or defaults
    if params is None:
        params = default_params

    # Remove datetime columns and store feature columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'bool']).columns
    X_train_numeric = X_train[numeric_features]
    
    # Store feature columns in model for prediction
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train_numeric,
        y_train,
        eval_set=[(X_train_numeric, y_train)],
        verbose=False
    )
    
    # Store feature columns as an attribute
    model.feature_columns_ = numeric_features.tolist()

    logger.info("XGBoost model training completed")
    return model


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
    cv: int = 5
) -> LGBMRegressor:
    """
    Train LightGBM model for sales forecasting

    Args:
        X_train: Training features
        y_train: Training target
        params: LightGBM parameters (optional)
        cv: Number of cross-validation folds

    Returns:
        Trained LightGBM model
    """
    logger.info("Training LightGBM model")

    # Load default parameters from config
    config = load_config()
    default_params = config.get("lightgbm", {}).get("params", {})

    # Use provided parameters or defaults
    if params is None:
        params = default_params

    # Create model
    model = LGBMRegressor(**params)

    # Train model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )

    logger.info("LightGBM model training completed")
    return model


def train_prophet_model(
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'total_sales',
    params: Optional[Dict] = None
) -> Prophet:
    """
    Train Prophet model for sales forecasting

    Args:
        df: DataFrame with time series data
        date_column: Name of the date column
        target_column: Name of the target column
        params: Prophet parameters (optional)

    Returns:
        Trained Prophet model
    """
    logger.info("Training Prophet model")

    # Load default parameters from config
    config = load_config()
    default_params = config.get("prophet", {}).get("params", {})

    # Use provided parameters or defaults
    if params is None:
        params = default_params

    # Prepare data for Prophet
    prophet_df = df[[date_column, target_column]].copy()
    prophet_df.columns = ['ds', 'y']

    # Create and train model
    model = Prophet(**params)

    # Add regressors if available in the data
    for regressor in ['is_holiday', 'is_weekend']:
        if regressor in df.columns:
            model.add_regressor(regressor)

    # Fit model
    model.fit(prophet_df)

    logger.info("Prophet model training completed")
    return model


def evaluate_forecast_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = 'xgboost'
) -> Dict[str, float]:
    """
    Evaluate forecasting model performance

    Args:
        model: Trained forecasting model
        X_test: Test features
        y_test: Test target
        model_type: Type of model ('xgboost', 'lightgbm', or 'prophet')

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_type} model")

    if model_type in ['xgboost', 'lightgbm']:
        # Use stored feature columns for prediction
        if hasattr(model, 'feature_columns_'):
            X_test = X_test[model.feature_columns_]
        # Make predictions
        y_pred = model.predict(X_test)
    elif model_type == 'prophet':
        # For Prophet, X_test should be a DataFrame with 'ds' column
        future = model.make_future_dataframe(periods=len(X_test))
        forecast = model.predict(future)
        y_pred = forecast.tail(len(X_test))['yhat'].values
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Log metrics
    logger.info(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}, MAPE: {mape:.2f}%")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    model_type: str = 'xgboost',
    output_path: Optional[str] = None
) -> None:
    """
    Plot feature importance for the forecasting model

    Args:
        model: Trained forecasting model
        feature_names: List of feature names
        top_n: Number of top features to display
        model_type: Type of model ('xgboost' or 'lightgbm')
        output_path: Path to save the plot (optional)
    """
    logger.info(f"Plotting feature importance for {model_type} model")

    plt.figure(figsize=(12, 8))

    if model_type == 'xgboost':
        # Get feature importance from XGBoost model and match with actual features used
        if hasattr(model, 'feature_columns_'):
            feature_names = model.feature_columns_
        importance = model.feature_importances_
        n_features = min(len(importance), len(feature_names))
        top_n = min(top_n, n_features)
        indices = np.argsort(importance)[-top_n:]

        plt.barh(range(top_n), importance[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])

    elif model_type == 'lightgbm':
        # Get feature importance from LightGBM model
        importance = model.feature_importances_
        n_features = min(len(importance), len(feature_names))
        top_n = min(top_n, n_features)
        indices = np.argsort(importance)[-top_n:]

        plt.barh(range(top_n), importance[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])

    else:
        logger.warning(f"Feature importance not supported for model type: {model_type}")
        return

    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()

    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")

    plt.close()


def plot_forecast_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series,
    title: str = 'Forecast vs Actual',
    output_path: Optional[str] = None
) -> None:
    """
    Plot forecast values against actual values

    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Dates corresponding to the values
        title: Plot title
        output_path: Path to save the plot (optional)
    """
    logger.info("Plotting forecast vs actual values")

    plt.figure(figsize=(12, 6))

    plt.plot(dates, y_true, label='Actual', marker='o')
    plt.plot(dates, y_pred, label='Forecast', marker='x')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Forecast plot saved to {output_path}")

    plt.close()


def save_model(model: Any, output_path: str, model_type: str = 'xgboost') -> None:
    """
    Save trained forecasting model

    Args:
        model: Trained forecasting model
        output_path: Path to save the model
        model_type: Type of model
    """
    logger.info(f"Saving {model_type} model to {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if model_type == 'xgboost':
        model.save_model(output_path)
    elif model_type == 'lightgbm':
        model.booster_.save_model(output_path)
    elif model_type == 'prophet':
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)

    logger.info(f"Model saved to {output_path}")


def load_model(model_path: str, model_type: str = 'xgboost') -> Any:
    """
    Load trained forecasting model

    Args:
        model_path: Path to the saved model
        model_type: Type of model

    Returns:
        Loaded model
    """
    logger.info(f"Loading {model_type} model from {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_type == 'xgboost':
        model = xgb.XGBRegressor()
        model.load_model(model_path)
    elif model_type == 'lightgbm':
        model = LGBMRegressor()
        model.booster_ = model.booster_.load_model(model_path)
    elif model_type == 'prophet':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    logger.info(f"Model loaded from {model_path}")
    return model


def generate_forecast(
    model: Any,
    X_future: pd.DataFrame,
    model_type: str = 'xgboost',
    prediction_intervals: bool = True,
    alpha: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Generate forecasts using the trained model

    Args:
        model: Trained forecasting model
        X_future: Future features for forecasting
        model_type: Type of model
        prediction_intervals: Whether to generate prediction intervals
        alpha: Significance level for prediction intervals

    Returns:
        Dictionary with forecast results
    """
    logger.info(f"Generating forecasts using {model_type} model")

    result = {}

    if model_type in ['xgboost', 'lightgbm']:
        # Use stored feature columns if available
        if hasattr(model, 'feature_columns_'):
            X_future = X_future[model.feature_columns_]
            
        # Generate point forecasts
        result['forecast'] = model.predict(X_future)

        # Generate prediction intervals if requested
        if prediction_intervals:
            if model_type == 'xgboost' and hasattr(model, 'predict'):
                # For XGBoost, use quantile regression if available
                try:
                    result['lower'] = model.predict(X_future, ntree_limit=model.best_ntree_limit, iteration_range=(0, model.best_ntree_limit), quantile=alpha/2)
                    result['upper'] = model.predict(X_future, ntree_limit=model.best_ntree_limit, iteration_range=(0, model.best_ntree_limit), quantile=1-alpha/2)
                except Exception:
                    # Fall back to simple intervals
                    std_dev = np.std(result['forecast']) * 1.96  # Approximate 95% interval
                    result['lower'] = result['forecast'] - std_dev
                    result['upper'] = result['forecast'] + std_dev
            else:
                # Simple intervals based on historical error
                std_dev = np.std(result['forecast']) * 1.96  # Approximate 95% interval
                result['lower'] = result['forecast'] - std_dev
                result['upper'] = result['forecast'] + std_dev

    elif model_type == 'prophet':
        # For Prophet, X_future should be a DataFrame with future dates
        future = model.make_future_dataframe(periods=len(X_future))
        forecast = model.predict(future)
        forecast = forecast.tail(len(X_future))

        result['forecast'] = forecast['yhat'].values

        if prediction_intervals:
            result['lower'] = forecast['yhat_lower'].values
            result['upper'] = forecast['yhat_upper'].values

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Generated forecasts for {len(X_future)} periods")
    return result


def predict_sales(model: Any, X: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Make sales predictions using the trained model

    Args:
        model: Trained forecasting model
        X: Features for prediction
        feature_cols: List of feature column names to use for prediction

    Returns:
        Array of predicted sales values
    """
    logger.info("Making predictions")

    if hasattr(model, 'feature_columns_'):
        # Use stored feature columns if available
        X = X[model.feature_columns_]
    else:
        # Otherwise use provided feature columns
        X = X[feature_cols]

    # Ensure predictions are numeric
    predictions = model.predict(X)
    return predictions.astype(np.float64)


def train_forecasting_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[dict] = None,
    cv: int = 5
) -> Tuple[xgb.XGBRegressor, np.ndarray]:
    """Train XGBoost model for sales forecasting (wrapper for compatibility).
    
    Returns:
        Tuple of (trained model, feature importance array)
    """
    model = train_xgboost_model(X_train, y_train, params=params, cv=cv)
    importance = model.feature_importances_
    return model, importance


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Train sales forecasting model')
    parser.add_argument('--input', required=True, help='Path to input data file')
    parser.add_argument('--output', required=True, help='Path to save trained model')
    parser.add_argument('--model', choices=['xgboost', 'lightgbm', 'prophet'],
                        default='xgboost', help='Type of model to train')
    parser.add_argument('--target', default='total_sales', help='Target column for forecasting')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Prepare data
    X_dict, y_dict, feature_names = prepare_forecasting_data(df, target_column=args.target)

    # Combine all training data
    X_train = pd.concat([X_dict['train'][key] for key in X_dict['train']])
    y_train = pd.concat([y_dict['train'][key] for key in y_dict['train']])

    # Train model
    if args.model == 'xgboost':
        model, importance = train_forecasting_model(X_train, y_train)
    elif args.model == 'lightgbm':
        model = train_lightgbm_model(X_train, y_train)
    elif args.model == 'prophet':
        model = train_prophet_model(df, target_column=args.target)

    # Save model
    save_model(model, args.output, args.model)

    # Plot feature importance if applicable
    if args.model in ['xgboost', 'lightgbm']:
        plot_path = os.path.join(os.path.dirname(args.output), f"{args.model}_feature_importance.png")
        plot_feature_importance(model, feature_names, model_type=args.model, output_path=plot_path)