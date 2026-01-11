"""
Utility functions for the fraud detection system
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import json


def setup_logging(log_file: str = "fraud_detector.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        "data/raw/twitter",
        "data/raw/reddit",
        "data/raw/legitimate",
        "data/processed/train",
        "data/processed/val",
        "data/processed/test",
        "data/models/checkpoints",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully!")


if __name__ == "__main__":
    create_directory_structure()