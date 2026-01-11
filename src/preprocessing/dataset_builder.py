"""
Build and prepare datasets for training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build training, validation, and test datasets"""
    
    def __init__(self, config: dict):
        """
        Initialize dataset builder
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.text_cleaner = TextCleaner()
        self.feature_extractor = FeatureExtractor()
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load all raw data from various sources
        
        Returns:
            Combined DataFrame with all data
        """
        data_frames = []
        
        # Load synthetic data
        synthetic_path = Path("data/raw/synthetic/conversations.csv")
        if synthetic_path.exists():
            df_synthetic = pd.read_csv(synthetic_path)
            df_synthetic['source'] = 'synthetic'
            data_frames.append(df_synthetic)
            logger.info(f"Loaded {len(df_synthetic)} synthetic conversations")
        
        # Load Twitter data
        twitter_path = Path("data/raw/twitter/fraud_tweets.csv")
        if twitter_path.exists():
            df_twitter = pd.read_csv(twitter_path)
            df_twitter['conversation'] = df_twitter['text']
            df_twitter['is_fraud'] = 1
            df_twitter['label'] = 'fraud'
            df_twitter['source'] = 'twitter'
            data_frames.append(df_twitter[['conversation', 'is_fraud', 'label', 'source']])
            logger.info(f"Loaded {len(df_twitter)} Twitter posts")
        
        # Load Reddit data
        reddit_path = Path("data/raw/reddit/fraud_posts.csv")
        if reddit_path.exists():
            df_reddit = pd.read_csv(reddit_path)
            df_reddit['conversation'] = df_reddit['title'] + ' ' + df_reddit['text']
            df_reddit['is_fraud'] = 1
            df_reddit['label'] = 'fraud'
            df_reddit['source'] = 'reddit'
            data_frames.append(df_reddit[['conversation', 'is_fraud', 'label', 'source']])
            logger.info(f"Loaded {len(df_reddit)} Reddit posts")
        
        # Combine all data
        if data_frames:
            df_combined = pd.concat(data_frames, ignore_index=True)
            logger.info(f"Total data points: {len(df_combined)}")
            return df_combined
        else:
            logger.warning("No data files found!")
            return pd.DataFrame()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['conversation'])
        
        # Remove null values
        df = df.dropna(subset=['conversation'])
        
        # Clean text
        df['cleaned_text'] = df['conversation'].apply(self.text_cleaner.clean_text)
        
        # Remove very short texts
        df = df[df['cleaned_text'].str.len() > 20]
        
        # Extract features
        logger.info("Extracting features...")
        features_df = self.feature_extractor.extract_features_batch(df['cleaned_text'].tolist())
        df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        logger.info(f"Preprocessed {len(df)} samples")
        return df
    
    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_size = self.config['data']['train_split']
        val_size = self.config['data']['val_split']
        
        # First split: train and temp (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=42,
            stratify=df['is_fraud']
        )
        
        # Second split: val and test
        val_relative_size = val_size / (val_size + self.config['data']['test_split'])
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_relative_size,
            random_state=42,
            stratify=temp_df['is_fraud']
        )
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Save datasets to disk
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
        """
        # Create directories
        Path("data/processed/train").mkdir(parents=True, exist_ok=True)
        Path("data/processed/val").mkdir(parents=True, exist_ok=True)
        Path("data/processed/test").mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_df.to_csv("data/processed/train/train.csv", index=False)
        val_df.to_csv("data/processed/val/val.csv", index=False)
        test_df.to_csv("data/processed/test/test.csv", index=False)
        
        logger.info("Datasets saved successfully!")
    
    def build_complete_dataset(self):
        """Build complete dataset from raw data"""
        logger.info("Starting dataset building process...")
        
        # Load raw data
        df = self.load_raw_data()
        
        if df.empty:
            logger.error("No data to process!")
            return
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Split
        train_df, val_df, test_df = self.split_dataset(df)
        
        # Save
        self.save_datasets(train_df, val_df, test_df)
        
        logger.info("Dataset building complete!")
        
        # Print statistics
        print("\n" + "="*50)
        print("Dataset Statistics:")
        print("="*50)
        print(f"Training samples: {len(train_df)}")
        print(f"  - Fraud: {sum(train_df['is_fraud'])}")
        print(f"  - Legitimate: {len(train_df) - sum(train_df['is_fraud'])}")
        print(f"\nValidation samples: {len(val_df)}")
        print(f"  - Fraud: {sum(val_df['is_fraud'])}")
        print(f"  - Legitimate: {len(val_df) - sum(val_df['is_fraud'])}")
        print(f"\nTest samples: {len(test_df)}")
        print(f"  - Fraud: {sum(test_df['is_fraud'])}")
        print(f"  - Legitimate: {len(test_df) - sum(test_df['is_fraud'])}")
        print("="*50)