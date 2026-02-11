"""
Sequence Builder for LSTM Models
Builds sequential transaction data for deep learning
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FraudSequenceBuilder:
    """Build sequential data for LSTM/RNN models"""
    
    def __init__(self, seq_length: int = 10, stride: int = 1):
        self.seq_length = seq_length
        self.stride = stride
        self.scaler = StandardScaler()
        self.feature_cols = None
    
    def build_sequences(
        self, 
        df: pd.DataFrame, 
        user_col: str = 'card1', 
        target_col: str = 'isFraud',
        min_transactions: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences of transactions per user
        
        Args:
            df: DataFrame with features
            user_col: Column identifying users
            target_col: Target column name
            min_transactions: Minimum transactions per user to include
            
        Returns:
            X: (n_samples, seq_length, n_features)
            y: (n_samples,)
            user_ids: (n_samples,)
        """
        # Sort by user and time
        df = df.sort_values([user_col, 'TransactionDT'])
        
        # Get feature columns (exclude identifiers and target)
        exclude_cols = [user_col, target_col, 'TransactionID', 
                       'TransactionDT', 'transaction_datetime']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Filter numeric columns only
        self.feature_cols = [col for col in self.feature_cols 
                            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        logger.info(f"Using {len(self.feature_cols)} features for sequences")
        
        sequences = []
        labels = []
        user_ids = []
        
        # Group by user
        for user, group in df.groupby(user_col):
            if len(group) < min_transactions:
                continue
                
            user_data = group[self.feature_cols].values
            user_labels = group[target_col].values
            
            # Create sliding windows
            for i in range(0, len(user_data) - self.seq_length, self.stride):
                seq = user_data[i:i + self.seq_length]
                # Predict the label of the LAST transaction in sequence
                label = user_labels[i + self.seq_length - 1]
                
                sequences.append(seq)
                labels.append(label)
                user_ids.append(user)
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        
        # Handle any infinite values
        X = np.nan_to_num(X, nan=-999, posinf=999, neginf=-999)
        
        logger.info(f"Created {len(sequences):,} sequences")
        logger.info(f"Sequence shape: {X.shape}")
        logger.info(f"Fraud rate in sequences: {y.mean():.4%}")
        
        return X, y, np.array(user_ids)
    
    def scale_sequences(
        self, 
        X_train: np.ndarray, 
        X_val: np.ndarray = None,
        X_test: np.ndarray = None
    ) -> Tuple:
        """
        Scale sequences using StandardScaler
        
        Args:
            X_train: Training sequences (n_samples, seq_length, n_features)
            X_val: Validation sequences
            X_test: Test sequences
            
        Returns:
            Scaled sequences
        """
        n_samples, seq_length, n_features = X_train.shape
        
        # Reshape for scaling: (samples * seq_length, features)
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        # Fit scaler on training data only
        self.scaler.fit(X_train_reshaped)
        
        # Transform training data
        X_train_scaled = self.scaler.transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, seq_length, n_features)
        
        result = [X_train_scaled]
        
        # Transform validation data
        if X_val is not None:
            X_val_reshaped = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(-1, seq_length, n_features)
            result.append(X_val_scaled)
        
        # Transform test data
        if X_test is not None:
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(-1, seq_length, n_features)
            result.append(X_test_scaled)
        
        logger.info("Sequences scaled successfully")
        return tuple(result) if len(result) > 1 else result[0]
    
    def get_scaler(self) -> StandardScaler:
        """Return the fitted scaler for later use"""
        return self.scaler
    
    def pad_sequences(
        self, 
        sequences: List[np.ndarray], 
        maxlen: int = None, 
        padding: str = 'pre'
    ) -> np.ndarray:
        """
        Pad sequences to same length (for users with few transactions)
        
        Args:
            sequences: List of variable-length sequences
            maxlen: Maximum length (uses self.seq_length if None)
            padding: 'pre' or 'post'
            
        Returns:
            Padded sequences array
        """
        maxlen = maxlen or self.seq_length
        
        n_features = sequences[0].shape[-1] if len(sequences) > 0 else 0
        padded = np.zeros((len(sequences), maxlen, n_features), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), maxlen)
            if padding == 'pre':
                padded[i, -seq_len:] = seq[-seq_len:]
            else:
                padded[i, :seq_len] = seq[:seq_len]
        
        return padded
