"""
Model evaluation utilities
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate fraud detection model"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def predict_text(self, text: str) -> dict:
        """
        Predict fraud probability for a single text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            
        fraud_prob = probs[0][1].item()
        prediction = 1 if fraud_prob > 0.5 else 0
        
        return {
            'text': text,
            'is_fraud': bool(prediction),
            'fraud_probability': fraud_prob,
            'legitimate_probability': probs[0][0].item()
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved confusion matrix to {save_path}")
    
    def plot_roc_curve(self, y_true, y_probs, save_path='roc_curve.png'):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved ROC curve to {save_path}")
    
    def generate_report(self, y_true, y_pred, y_probs=None, save_dir='reports'):
        """Generate complete evaluation report"""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud'])
        logger.info(f"\nClassification Report:\n{report}")
        
        with open(f'{save_dir}/classification_report.txt', 'w') as f:
            f.write(report)
        
        # Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, f'{save_dir}/confusion_matrix.png')
        
        # ROC curve
        if y_probs is not None:
            self.plot_roc_curve(y_true, y_probs, f'{save_dir}/roc_curve.png')
        
        logger.info(f"Evaluation report saved to {save_dir}")