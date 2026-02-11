"""
Data Preprocessor Module
Handles cleaning, imputation, and train/val/test splitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses fraud detection data with strategies optimized for fraud signals
    """
    
    def __init__(self):
        self.categorical_cols = []
        self.numerical_cols = []
        self.missing_indicators = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data while preserving fraud signals
        
        Strategy:
        - Categorical missing -> 'missing' category (could be fraud signal!)
        - Numerical missing -> -999 (tree models handle this well)
        - Create missing indicators for important features
        """
        df_clean = df.copy()
        
        # Identify column types
        self.categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from numerical cols if present
        if 'isFraud' in self.numerical_cols:
            self.numerical_cols.remove('isFraud')
        if 'TransactionID' in self.numerical_cols:
            self.numerical_cols.remove('TransactionID')
        
        logger.info(f"Found {len(self.categorical_cols)} categorical columns")
        logger.info(f"Found {len(self.numerical_cols)} numerical columns")
        
        # Handle categorical missing values
        for col in self.categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna('missing')
                logger.debug(f"Filled missing values in {col} with 'missing'")
        
        # Handle numerical missing values
        for col in self.numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                # Create missing indicator (can be a fraud signal)
                indicator_col = f'{col}_missing'
                df_clean[indicator_col] = df_clean[col].isnull().astype(int)
                self.missing_indicators.append(indicator_col)
                
                # Fill with -999 (tree models handle this well)
                df_clean[col] = df_clean[col].fillna(-999)
                logger.debug(f"Created missing indicator for {col}")
        
        logger.info(f"Created {len(self.missing_indicators)} missing indicators")
        logger.info(f"Final shape after cleaning: {df_clean.shape}")
        
        return df_clean
    
    def time_based_split(
        self,
        df: pd.DataFrame,
        time_col: str = 'TransactionDT',
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (critical for fraud detection!)
        
        Never shuffle - use time-based split to avoid data leakage
        """
        assert abs(train_size + val_size + test_size - 1.0) < 0.01, \
            "Split sizes must sum to 1.0"
        
        # Sort by time
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        
        n = len(df_sorted)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        train_df = df_sorted[:train_end]
        val_df = df_sorted[train_end:val_end]
        test_df = df_sorted[val_end:]
        
        logger.info("Time-based split completed:")
        logger.info(f"  Train: {len(train_df):,} ({train_df['isFraud'].mean():.4%} fraud)")
        logger.info(f"  Val:   {len(val_df):,} ({val_df['isFraud'].mean():.4%} fraud)")
        logger.info(f"  Test:  {len(test_df):,} ({test_df['isFraud'].mean():.4%} fraud)")
        
        return train_df, val_df, test_df
    
    def encode_categoricals(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        method: str = 'label'
    ) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            categorical_cols: Columns to encode (uses detected if None)
            method: 'label' or 'onehot'
        """
        df_encoded = df.copy()
        cols_to_encode = categorical_cols or self.categorical_cols
        
        if method == 'label':
            for col in cols_to_encode:
                if col in df_encoded.columns:
                    df_encoded[col] = df_encoded[col].astype('category').cat.codes
        
        elif method == 'onehot':
            df_encoded = pd.get_dummies(
                df_encoded, 
                columns=cols_to_encode, 
                prefix=cols_to_encode
            )
        
        return df_encoded
    
    def remove_high_cardinality(
        self,
        df: pd.DataFrame,
        threshold: int = 100
    ) -> pd.DataFrame:
        """
        Remove or bin high cardinality categorical features
        """
        df_clean = df.copy()
        
        for col in self.categorical_cols:
            if col in df_clean.columns:
                n_unique = df_clean[col].nunique()
                if n_unique > threshold:
                    logger.warning(
                        f"Column {col} has {n_unique} unique values (> {threshold}). "
                        f"Consider frequency encoding or removing."
                    )
        
        return df_clean
    
    def get_feature_columns(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of feature columns (excluding target and identifiers)
        """
        default_exclude = [
            'TransactionID', 
            'isFraud', 
            'TransactionDT', 
            'transaction_datetime'
        ]
        
        exclude = set(default_exclude + (exclude_cols or []))
        
        feature_cols = [col for col in df.columns if col not in exclude]
        
        return feature_cols
    
    def prepare_for_training(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = 'isFraud'
    ) -> Tuple:
        """
        Prepare train/val/test splits for model training
        
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
        """
        feature_cols = self.get_feature_columns(train_df)
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        logger.info(f"Prepared {len(feature_cols)} features for training")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
