"""
Train the fraud detection model
"""

import sys
import yaml
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.model_trainer import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

def main():
    """Train the fraud detection model"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model directory
    Path(config['data']['model_path']).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    print("="*50)
    print("FRAUD DETECTION MODEL TRAINING")
    print("="*50)
    
    trainer = ModelTrainer(config, use_simple_model=True)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(epochs=3)  # Start with 3 epochs for testing
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"Model saved to: {config['data']['model_path']}")

if __name__ == "__main__":
    main()