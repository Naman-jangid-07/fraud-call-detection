"""
Model training pipeline
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, XLNetTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .fraud_detector_model import FraudDetectorModel, SimpleFraudDetector

logger = logging.getLogger(__name__)


class FraudDataset(Dataset):
    """Dataset for fraud detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize dataset
        
        Args:
            texts: List of text samples
            labels: List of labels
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class ModelTrainer:
    """Train fraud detection model"""
    
    def __init__(self, config: Dict, use_simple_model: bool = True):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            use_simple_model: Use BERT instead of XLNet (faster)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        if use_simple_model:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = SimpleFraudDetector(model_name='bert-base-uncased')
        else:
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            self.model = FraudDetectorModel(model_name='xlnet-base-cased')
        
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load training, validation, and test data
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load datasets
        train_df = pd.read_csv('data/processed/train/train.csv')
        val_df = pd.read_csv('data/processed/val/val.csv')
        test_df = pd.read_csv('data/processed/test/test.csv')
        
        logger.info(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
        
        # Create datasets
        train_dataset = FraudDataset(
            train_df['cleaned_text'].values,
            train_df['is_fraud'].values,
            self.tokenizer,
            max_length=self.config['model']['max_length']
        )
        
        val_dataset = FraudDataset(
            val_df['cleaned_text'].values,
            val_df['is_fraud'].values,
            self.tokenizer,
            max_length=self.config['model']['max_length']
        )
        
        test_dataset = FraudDataset(
            test_df['cleaned_text'].values,
            test_df['is_fraud'].values,
            self.tokenizer,
            max_length=self.config['model']['max_length']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler, criterion):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader, criterion):
        """Evaluate model"""
        self.model.eval()
        
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self, epochs: int = None):
        """
        Train the model
        
        Args:
            epochs: Number of epochs (uses config if not specified)
        """
        if epochs is None:
            epochs = self.config['model']['epochs']
        
        # Load data
        train_loader, val_loader, test_loader = self.load_data()
        
        # Setup optimizer and scheduler
      # Convert learning_rate to float if it's a string
        lr = float(self.config['model']['learning_rate'])
        weight_decay = float(self.config['model']['weight_decay'])

        optimizer = AdamW(
             self.model.parameters(),
              lr=lr,
             weight_decay=weight_decay
)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['model']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        best_model_path = Path(self.config['data']['model_path']) / 'best_model.pt'
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, scheduler, criterion
            )
            
            # Evaluate
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.evaluate(
                val_loader, criterion
            )
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"Saved best model with accuracy: {best_val_acc:.4f}")
        
        # Final evaluation on test set
        logger.info("\nEvaluating on test set...")
        test_loss, test_acc, test_precision, test_recall, test_f1 = self.evaluate(
            test_loader, criterion
        )
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1: {test_f1:.4f}")
        
        # Save final model
        final_model_path = Path(self.config['data']['model_path']) / 'final_model.pt'
        torch.save(self.model.state_dict(), final_model_path)
        
        # Save tokenizer
        tokenizer_path = Path(self.config['data']['model_path']) / 'tokenizer'
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info("Training complete!")
        
        return self.history