"""
XLNet-based fraud detection model
"""

import torch
import torch.nn as nn
from transformers import XLNetModel, XLNetTokenizer
import logging

logger = logging.getLogger(__name__)


class FraudDetectorModel(nn.Module):
    """XLNet-based model for fraud detection"""
    
    def __init__(self, model_name: str = "xlnet-base-cased", num_labels: int = 2, dropout: float = 0.3):
        """
        Initialize the fraud detector model
        
        Args:
            model_name: Pre-trained XLNet model name
            num_labels: Number of output labels (2 for binary classification)
            dropout: Dropout rate
        """
        super(FraudDetectorModel, self).__init__()
        
        # Load pre-trained XLNet
        self.xlnet = XLNetModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.xlnet.config.hidden_size, num_labels)
        
        logger.info(f"Initialized FraudDetectorModel with {model_name}")
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Logits for classification
        """
        # Get XLNet outputs
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Pool the output (take the first token [CLS])
        pooled_output = last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class SimpleFraudDetector(nn.Module):
    """Simpler BERT-based model (faster alternative to XLNet)"""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2, dropout: float = 0.3):
        """
        Initialize simple fraud detector
        
        Args:
            model_name: Pre-trained BERT model name
            num_labels: Number of output labels
            dropout: Dropout rate
        """
        super(SimpleFraudDetector, self).__init__()
        
        from transformers import BertModel
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        logger.info(f"Initialized SimpleFraudDetector with {model_name}")
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """Forward pass"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


def load_model(model_path: str, device: str = "cpu") -> FraudDetectorModel:
    """
    Load a saved model
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = FraudDetectorModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


def save_model(model: nn.Module, save_path: str):
    """
    Save model to disk
    
    Args:
        model: Model to save
        save_path: Path to save model
    """
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved model to {save_path}")