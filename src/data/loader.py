"""
Data Loader Module
Handles loading and initial validation of fraud detection datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads IEEE-CIS Fraud Detection dataset or synthetic data for development
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_ieee_cis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load IEEE-CIS Fraud Detection dataset
        
        Returns:
            Tuple of (transaction_df, identity_df)
        """
        transaction_path = self.data_dir / "train_transaction.csv"
        identity_path = self.data_dir / "train_identity.csv"
        
        if not transaction_path.exists():
            raise FileNotFoundError(
                f"Transaction data not found at {transaction_path}. "
                "Please download from Kaggle: kaggle competitions download -c ieee-fraud-detection"
            )
        
        logger.info("Loading transaction data...")
        transaction_df = pd.read_csv(transaction_path)
        logger.info(f"Loaded {len(transaction_df):,} transactions")
        
        if identity_path.exists():
            logger.info("Loading identity data...")
            identity_df = pd.read_csv(identity_path)
            logger.info(f"Loaded {len(identity_df):,} identity records")
        else:
            logger.warning("Identity data not found, using empty DataFrame")
            identity_df = pd.DataFrame()
        
        return transaction_df, identity_df
    
    def load_merged(self) -> pd.DataFrame:
        """
        Load and merge transaction + identity data
        """
        transaction_df, identity_df = self.load_ieee_cis()
        
        if not identity_df.empty:
            logger.info("Merging transaction and identity data...")
            merged = transaction_df.merge(
                identity_df, 
                on='TransactionID', 
                how='left'
            )
        else:
            merged = transaction_df
        
        logger.info(f"Final dataset shape: {merged.shape}")
        logger.info(f"Fraud rate: {merged['isFraud'].mean():.4%}")
        
        return merged
    
    def generate_synthetic_data(
        self,
        n_samples: int = 100000,
        fraud_rate: float = 0.035,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic fraud detection data for development
        
        Args:
            n_samples: Number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions
            random_state: Random seed for reproducibility
            
        Returns:
            Synthetic transaction DataFrame
        """
        np.random.seed(random_state)
        logger.info(f"Generating {n_samples:,} synthetic transactions...")
        
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud
        
        # Generate user IDs (some users will have multiple transactions)
        n_users = n_samples // 5
        user_ids = np.random.randint(1, n_users + 1, n_samples)
        
        # Generate merchant IDs
        n_merchants = n_samples // 20
        merchant_ids = np.random.randint(1, n_merchants + 1, n_samples)
        
        # Generate timestamps (30 days of data)
        start_time = 0
        end_time = 30 * 24 * 60 * 60  # 30 days in seconds
        timestamps = np.sort(np.random.uniform(start_time, end_time, n_samples))
        
        # Generate amounts - different distributions for fraud/legit
        amounts = np.zeros(n_samples)
        is_fraud = np.zeros(n_samples, dtype=int)
        
        # Assign fraud labels randomly
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        is_fraud[fraud_indices] = 1
        
        # Legitimate transactions: mostly small amounts
        legit_mask = is_fraud == 0
        amounts[legit_mask] = np.random.exponential(50, legit_mask.sum())
        
        # Fraud transactions: higher amounts, more variation
        fraud_mask = is_fraud == 1
        amounts[fraud_mask] = np.random.exponential(200, fraud_mask.sum()) + 100
        
        # Clip amounts
        amounts = np.clip(amounts, 1, 10000)
        
        # Product codes
        product_codes = np.random.choice(['W', 'H', 'C', 'S', 'R'], n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        
        # Card types
        card_types = np.random.choice(['visa', 'mastercard', 'american express', 'discover'], n_samples, p=[0.5, 0.3, 0.15, 0.05])
        
        # Device info
        devices = np.random.choice(['Windows', 'iOS Device', 'MacOS', 'Android', None], n_samples, p=[0.4, 0.25, 0.15, 0.15, 0.05])
        
        # Email domains
        email_domains = np.random.choice(
            ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'protonmail.com', None],
            n_samples,
            p=[0.4, 0.2, 0.15, 0.1, 0.05, 0.1]
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'TransactionID': range(1, n_samples + 1),
            'isFraud': is_fraud,
            'TransactionDT': timestamps,
            'TransactionAmt': amounts,
            'ProductCD': product_codes,
            'card1': user_ids,
            'card2': merchant_ids,
            'card4': card_types,
            'card6': np.random.choice(['debit', 'credit'], n_samples, p=[0.6, 0.4]),
            'P_emaildomain': email_domains,
            'R_emaildomain': np.where(np.random.random(n_samples) > 0.7, email_domains, None),
            'DeviceType': np.random.choice(['desktop', 'mobile'], n_samples, p=[0.6, 0.4]),
            'DeviceInfo': devices,
        })
        
        # Add some V columns (anonymous features) with different distributions
        for i in range(1, 20):
            if is_fraud.sum() > 0:
                # Create features with slight fraud signal
                fraud_mean = np.random.uniform(-1, 1)
                df[f'V{i}'] = np.where(
                    df['isFraud'] == 1,
                    np.random.normal(fraud_mean, 1, n_samples),
                    np.random.normal(0, 1, n_samples)
                )
            else:
                df[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        # Add some missing values (realistic)
        for col in ['DeviceInfo', 'P_emaildomain', 'R_emaildomain']:
            mask = np.random.random(n_samples) < 0.1
            df.loc[mask, col] = None
        
        logger.info(f"Generated synthetic data with shape: {df.shape}")
        logger.info(f"Fraud rate: {df['isFraud'].mean():.4%}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str, directory: str = None):
        """Save DataFrame to CSV"""
        if directory:
            save_path = Path(directory) / filename
        else:
            save_path = self.data_dir / filename
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved data to {save_path}")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary information about the dataset"""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'fraud_rate': df['isFraud'].mean() if 'isFraud' in df.columns else None,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        return info
