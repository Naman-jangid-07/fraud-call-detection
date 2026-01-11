"""
Feature extraction for fraud detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import re
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from conversation text"""
    
    def __init__(self, fraud_keywords: Dict[str, List[str]] = None):
        """
        Initialize feature extractor
        
        Args:
            fraud_keywords: Dictionary of fraud-related keywords by category
        """
        self.fraud_keywords = fraud_keywords or self._load_default_keywords()
        
    def _load_default_keywords(self) -> Dict[str, List[str]]:
        """Load default fraud detection keywords"""
        return {
            'urgency': [
                'immediately', 'right now', 'urgent', 'emergency', 'act fast',
                'limited time', 'expire', 'deadline', 'asap', 'hurry'
            ],
            'financial': [
                'bank account', 'credit card', 'social security', 'ssn',
                'wire transfer', 'gift card', 'bitcoin', 'cryptocurrency',
                'payment', 'money', 'cash', 'account number', 'pin', 'password'
            ],
            'impersonation': [
                'irs', 'tax department', 'microsoft', 'apple', 'amazon',
                'tech support', 'police', 'fbi', 'government', 'official',
                'authority', 'agent', 'officer'
            ],
            'threats': [
                'arrest', 'warrant', 'lawsuit', 'legal action', 'court',
                'jail', 'prison', 'suspended', 'frozen', 'blocked',
                'penalty', 'fine', 'prosecute'
            ],
            'verification': [
                'verify', 'confirm', 'provide', 'share', 'enter',
                'authenticate', 'validate', 'prove', 'identification'
            ]
        }
    
    def extract_keyword_features(self, text: str) -> Dict[str, int]:
        """
        Extract keyword-based features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of keyword counts by category
        """
        text_lower = text.lower()
        features = {}
        
        for category, keywords in self.fraud_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'{category}_count'] = count
            features[f'has_{category}'] = 1 if count > 0 else 0
        
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of linguistic features
        """
        features = {}
        
        # Text length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0
        
        # Sentence features
        sentences = text.split('.')
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        
        # Capital letters
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Number features
        numbers = re.findall(r'\d+', text)
        features['number_count'] = len(numbers)
        features['has_phone_number'] = 1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) else 0
        features['has_ssn_pattern'] = 1 if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text) else 0
        features['has_dollar_amount'] = 1 if re.search(r'\$\d+', text) else 0
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentiment features
        """
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            features = {
                'polarity': sentiment.polarity,  # -1 to 1
                'subjectivity': sentiment.subjectivity,  # 0 to 1
                'is_negative': 1 if sentiment.polarity < -0.1 else 0,
                'is_positive': 1 if sentiment.polarity > 0.1 else 0,
                'is_neutral': 1 if -0.1 <= sentiment.polarity <= 0.1 else 0
            }
        except Exception as e:
            logger.error(f"Error extracting sentiment: {str(e)}")
            features = {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'is_negative': 0,
                'is_positive': 0,
                'is_neutral': 1
            }
        
        return features
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Extract all features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Extract different feature types
        features.update(self.extract_keyword_features(text))
        features.update(self.extract_linguistic_features(text))
        features.update(self.extract_sentiment_features(text))
        
        return features
    
    def extract_features_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract features from multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            DataFrame containing features for all texts
        """
        features_list = []
        
        for text in texts:
            features = self.extract_all_features(text)
            features_list.append(features)
        
        return pd.DataFrame(features_list)


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Test with fraud text
    fraud_text = "This is an urgent call from the IRS. You must pay immediately or face arrest."
    features = extractor.extract_all_features(fraud_text)
    
    print("Fraud Text Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Test with legitimate text
    legit_text = "Hello, this is a reminder about your appointment tomorrow at 2 PM."
    features = extractor.extract_all_features(legit_text)
    
    print("Legitimate Text Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")