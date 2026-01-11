"""
Generate initial dataset for training
"""

import sys
sys.path.append('src')

from src.data_collection.synthetic_generator import SyntheticDataGenerator
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Generate initial synthetic dataset"""
    
    # Create directories
    Path("data/raw/synthetic").mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    logger.info("Generating synthetic conversations...")
    generator = SyntheticDataGenerator()
    
    df = generator.generate_dataset(
        num_fraud=1000,
        num_legitimate=1000,
        save_path="data/raw/synthetic/conversations.csv"
    )
    
    logger.info(f"Generated {len(df)} conversations")
    print("\nSample conversations:")
    print("="*80)
    print("\nFRAUD EXAMPLE:")
    print(df[df['is_fraud']==1].iloc[0]['conversation'])
    print("\nLEGITIMATE EXAMPLE:")
    print(df[df['is_fraud']==0].iloc[0]['conversation'])
    print("="*80)

if __name__ == "__main__":
    main()