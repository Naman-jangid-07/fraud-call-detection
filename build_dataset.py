"""
Build the dataset from raw data
"""

import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.dataset_builder import DatasetBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Build the complete dataset"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Build dataset
    builder = DatasetBuilder(config)
    builder.build_complete_dataset()

if __name__ == "__main__":
    main()