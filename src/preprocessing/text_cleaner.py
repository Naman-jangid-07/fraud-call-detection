"""
Text cleaning and normalization utilities
"""

import re
import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextCleaner:
    """Clean and normalize text data"""
    
    def __init__(self, remove_stopwords: bool = False, lowercase: bool = True):
        """
        Initialize text cleaner
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lowercase: Whether to convert to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words]
            text = ' '.join(tokens)
        
        return text
    
    def normalize_numbers(self, text: str) -> str:
        """Replace numbers with a generic token"""
        return re.sub(r'\d+', '<NUM>', text)
    
    def remove_repeated_chars(self, text: str) -> str:
        """Remove repeated characters (e.g., 'hellooo' -> 'hello')"""
        return re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process a batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]


# Example usage
if __name__ == "__main__":
    cleaner = TextCleaner()
    
    sample_text = "URGENT!!! This is John from IRS. Call 555-123-4567 NOW!!! http://scam.com"
    cleaned = cleaner.clean_text(sample_text)
    
    print("Original:", sample_text)
    print("Cleaned:", cleaned)