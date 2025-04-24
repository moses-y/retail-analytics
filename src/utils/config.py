"""
Configuration utilities for retail analytics
"""
import os
import logging
import yaml
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def load_yaml_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    logger.info(f"Loading YAML configuration from {config_path}")

    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration with {len(config)} top-level keys")
        return config

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def load_json_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary containing configuration
    """
    logger.info(f"Loading JSON configuration from {config_path}")

    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info(f"Loaded configuration with {len(config)} top-level keys")
        return config

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def save_yaml_config(config: Dict, config_path: str) -> bool:
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        config_path: Path to save YAML configuration file

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Saving YAML configuration to {config_path}")

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Saved configuration with {len(config)} top-level keys")
        return True

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False


def save_json_config(config: Dict, config_path: str, indent: int = 2) -> bool:
    """
    Save configuration to JSON file

    Args:
        config: Configuration dictionary
        config_path: Path to save JSON configuration file
        indent: Indentation level for JSON formatting

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Saving JSON configuration to {config_path}")

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=indent)

        logger.info(f"Saved configuration with {len(config)} top-level keys")
        return True

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False


def get_config(config_path: str) -> Dict:
    """
    Load configuration from file based on extension

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    if config_path.endswith('.yml') or config_path.endswith('.yaml'):
        return load_yaml_config(config_path)
    elif config_path.endswith('.json'):
        return load_json_config(config_path)
    else:
        logger.warning(f"Unknown configuration file format: {config_path}")
        return {}


def get_model_config() -> Dict:
    """
    Load model configuration

    Returns:
        Dictionary containing model configuration
    """
    config_path = os.path.join("config", "model_config.yml")
    return get_config(config_path)


def get_api_config() -> Dict:
    """
    Load API configuration

    Returns:
        Dictionary containing API configuration
    """
    config_path = os.path.join("config", "api_config.yml")
    return get_config(config_path)


def get_monitoring_config() -> Dict:
    """
    Load monitoring configuration

    Returns:
        Dictionary containing monitoring configuration
    """
    config_path = os.path.join("config", "monitoring_config.yml")
    return get_config(config_path)


def get_env_variable(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable

    Args:
        name: Name of environment variable
        default: Default value if not found

    Returns:
        Value of environment variable or default
    """
    return os.environ.get(name, default)


def get_api_key(service: str) -> Optional[str]:
    """
    Get API key for a service

    Args:
        service: Service name (e.g., 'google', 'huggingface')

    Returns:
        API key or None if not found
    """
    # Map service names to environment variable names
    service_map = {
        'google': 'GOOGLE_API_KEY',
        'huggingface': 'HUGGINGFACE_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'azure': 'AZURE_API_KEY',
        'aws': 'AWS_API_KEY'
    }

    # Get environment variable name
    env_var = service_map.get(service.lower())

    if not env_var:
        logger.warning(f"Unknown service: {service}")
        return None

    # Get API key
    api_key = get_env_variable(env_var)

    if not api_key:
        logger.warning(f"API key not found for service: {service}")
        return None

    return api_key


def merge_configs(*configs: Dict) -> Dict:
    """
    Merge multiple configuration dictionaries

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    merged = {}

    for config in configs:
        if not isinstance(config, dict):
            logger.warning(f"Skipping non-dictionary config: {type(config)}")
            continue

        # Deep merge dictionaries
        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = merge_configs(merged[key], value)
            else:
                # Replace or add value
                merged[key] = value

    return merged


def validate_config(config: Dict, required_keys: List[str]) -> bool:
    """
    Validate configuration by checking for required keys

    Args:
        config: Configuration dictionary
        required_keys: List of required keys

    Returns:
        True if all required keys are present, False otherwise
    """
    missing_keys = []

    for key in required_keys:
        # Handle nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            current = config
            found = True

            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    found = False
                    break
                current = current[part]

            if not found:
                missing_keys.append(key)

        # Handle top-level keys
        elif key not in config:
            missing_keys.append(key)

    if missing_keys:
        logger.warning(f"Missing required configuration keys: {', '.join(missing_keys)}")
        return False

    return True


def get_nested_config(config: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation

    Args:
        config: Configuration dictionary
        key_path: Key path using dot notation (e.g., 'database.host')
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    if not key_path:
        return default

    parts = key_path.split('.')
    current = config

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Test configuration utilities')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--save', help='Path to save configuration file')
    parser.add_argument('--validate', nargs='+', help='Required keys to validate')

    args = parser.parse_args()

    if args.config:
        # Load configuration
        config = get_config(args.config)
        print(f"Loaded configuration: {config}")

        # Validate configuration
        if args.validate:
            is_valid = validate_config(config, args.validate)
            print(f"Configuration is valid: {is_valid}")

        # Save configuration
        if args.save:
            if args.save.endswith('.yml') or args.save.endswith('.yaml'):
                success = save_yaml_config(config, args.save)
            elif args.save.endswith('.json'):
                success = save_json_config(config, args.save)
            else:
                print(f"Unknown file format: {args.save}")
                success = False

            print(f"Configuration saved: {success}")

    else:
        # Example configuration
        example_config = {
            'model': {
                'name': 'xgboost',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5
                }
            },
            'data': {
                'train_path': 'data/train.csv',
                'test_path': 'data/test.csv'
            }
        }

        # Save example configuration
        if args.save:
            if args.save.endswith('.yml') or args.save.endswith('.yaml'):
                success = save_yaml_config(example_config, args.save)
            elif args.save.endswith('.json'):
                success = save_json_config(example_config, args.save)
            else:
                print(f"Unknown file format: {args.save}")
                success = False

            print(f"Example configuration saved: {success}")
        else:
            print("No configuration file specified")