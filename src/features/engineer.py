"""
Feature Engineering Module
Comprehensive feature engineering pipeline for fraud detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """Extract time-based features critical for fraud detection"""
    
    def __init__(self, reference_date: datetime = None):
        # IEEE dataset starts around December 2017
        self.reference_date = reference_date or datetime(2017, 12, 1)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        df = df.copy()
        
        # Convert TransactionDT to datetime
        df['transaction_datetime'] = pd.to_datetime(
            df['TransactionDT'], 
            unit='s', 
            origin=self.reference_date
        )
        
        # Basic time features
        df['hour'] = df['transaction_datetime'].dt.hour
        df['day_of_week'] = df['transaction_datetime'].dt.dayofweek
        df['day_of_month'] = df['transaction_datetime'].dt.day
        df['week_of_year'] = df['transaction_datetime'].dt.isocalendar().week.astype(int)
        df['month'] = df['transaction_datetime'].dt.month
        
        # Cyclical encoding (critical for ML models)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary flags
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Month segments
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        logger.info("Created temporal features")
        return df


class UserBehaviorFeatureEngineer:
    """Aggregate user historical behavior - KEY for fraud detection"""
    
    def __init__(self, lookback_windows: List[int] = None):
        self.lookback_windows = lookback_windows or [1, 7, 30]
    
    def create_features(self, df: pd.DataFrame, user_col: str = 'card1') -> pd.DataFrame:
        """Create rolling aggregations of user behavior"""
        df = df.copy()
        df = df.sort_values([user_col, 'TransactionDT'])
        
        for window in self.lookback_windows:
            logger.info(f"Creating {window}-day user behavior features...")
            
            # Transaction count
            df[f'user_txn_count_{window}d'] = df.groupby(user_col).cumcount()
            
            # Amount statistics using expanding window (cumulative)
            df[f'user_total_amount_{window}d'] = df.groupby(user_col)['TransactionAmt'].transform(
                lambda x: x.expanding().sum()
            )
            
            df[f'user_avg_amount_{window}d'] = df.groupby(user_col)['TransactionAmt'].transform(
                lambda x: x.expanding().mean()
            )
            
            df[f'user_std_amount_{window}d'] = df.groupby(user_col)['TransactionAmt'].transform(
                lambda x: x.expanding().std()
            )
            
            df[f'user_max_amount_{window}d'] = df.groupby(user_col)['TransactionAmt'].transform(
                lambda x: x.expanding().max()
            )
            
            df[f'user_min_amount_{window}d'] = df.groupby(user_col)['TransactionAmt'].transform(
                lambda x: x.expanding().min()
            )
            
            # Deviation from normal behavior (POWERFUL feature)
            df[f'amount_deviation_{window}d'] = (
                (df['TransactionAmt'] - df[f'user_avg_amount_{window}d']) / 
                (df[f'user_std_amount_{window}d'].fillna(1) + 1)
            )
        
        # Time since last transaction
        df['time_since_last_txn'] = df.groupby(user_col)['TransactionDT'].diff()
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(0)
        
        # First transaction flag
        df['is_first_txn'] = (df.groupby(user_col).cumcount() == 0).astype(int)
        
        logger.info("Created user behavior features")
        return df


class VelocityFeatureEngineer:
    """Detect rapid bursts of activity - classic fraud pattern"""
    
    def create_features(self, df: pd.DataFrame, user_col: str = 'card1') -> pd.DataFrame:
        """Create velocity-based features"""
        df = df.copy()
        df = df.sort_values([user_col, 'TransactionDT'])
        
        # Calculate time differences
        df['time_diff'] = df.groupby(user_col)['TransactionDT'].diff().fillna(0)
        
        # Transactions in short time windows (using cumulative counts)
        df['txn_velocity_1h'] = df.groupby(user_col).cumcount()
        
        # Amount velocity
        df['amount_velocity'] = df.groupby(user_col)['TransactionAmt'].transform(
            lambda x: x.expanding().sum()
        ) / (df.groupby(user_col).cumcount() + 1)
        
        # Rate of spending
        df['spending_rate'] = df['TransactionAmt'] / (df['time_diff'].replace(0, 1) + 1)
        
        # Acceleration (change in spending rate)
        df['spending_acceleration'] = df.groupby(user_col)['spending_rate'].diff().fillna(0)
        
        logger.info("Created velocity features")
        return df


class MerchantFeatureEngineer:
    """Features based on merchant risk profile"""
    
    def create_features(
        self, 
        df: pd.DataFrame, 
        train_df: pd.DataFrame = None,
        merchant_col: str = 'card2'
    ) -> pd.DataFrame:
        """Create merchant-based features"""
        df = df.copy()
        
        if train_df is not None and 'isFraud' in train_df.columns:
            # Merchant fraud rate (calculated ONLY on training data)
            merchant_fraud_rate = train_df.groupby(merchant_col)['isFraud'].mean()
            df['merchant_fraud_rate'] = df[merchant_col].map(merchant_fraud_rate).fillna(0.035)
            
            # Product fraud rate
            if 'ProductCD' in train_df.columns:
                product_fraud_rate = train_df.groupby('ProductCD')['isFraud'].mean()
                df['product_fraud_rate'] = df['ProductCD'].map(product_fraud_rate).fillna(0.035)
            
            # Card type fraud rate
            if 'card4' in train_df.columns:
                card_fraud_rate = train_df.groupby('card4')['isFraud'].mean()
                df['card_type_fraud_rate'] = df['card4'].map(card_fraud_rate).fillna(0.035)
        
        # User-merchant interaction history
        df['user_merchant_txn_count'] = df.groupby(['card1', merchant_col]).cumcount()
        df['is_first_txn_merchant'] = (df['user_merchant_txn_count'] == 0).astype(int)
        
        # User-merchant amount statistics
        df['user_merchant_avg_amount'] = df.groupby(['card1', merchant_col])['TransactionAmt'].transform(
            lambda x: x.expanding().mean()
        )
        
        # Deviation from user-merchant norm
        df['amount_vs_user_merchant_avg'] = (
            df['TransactionAmt'] / (df['user_merchant_avg_amount'].fillna(df['TransactionAmt']) + 1)
        )
        
        logger.info("Created merchant features")
        return df


class DeviceFeatureEngineer:
    """Device fingerprinting features"""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract device-based fraud signals"""
        df = df.copy()
        
        if 'DeviceInfo' in df.columns:
            # Device ID counts (how many users per device - fraud indicator)
            df['device_user_count'] = df.groupby('DeviceInfo')['card1'].transform('nunique')
            
            # User device count (user with multiple devices - potential fraud)
            df['user_device_count'] = df.groupby('card1')['DeviceInfo'].transform('nunique')
            
            # New device flag
            df['is_new_device'] = (
                df.groupby(['card1', 'DeviceInfo']).cumcount() == 0
            ).astype(int)
        
        if 'P_emaildomain' in df.columns:
            # Email domain user count
            df['email_domain_user_count'] = df.groupby('P_emaildomain')['card1'].transform('nunique')
        
        if 'DeviceType' in df.columns:
            # Device type encoding
            df['is_mobile'] = (df['DeviceType'] == 'mobile').astype(int)
        
        logger.info("Created device features")
        return df


class AmountFeatureEngineer:
    """Transaction amount features"""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from transaction amounts"""
        df = df.copy()
        
        # Log transform (helps with skewness)
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
        
        # Decimal analysis
        df['TransactionAmt_decimal'] = df['TransactionAmt'] % 1
        df['has_decimal'] = (df['TransactionAmt_decimal'] > 0).astype(int)
        
        # Round number flag (fraud often uses round numbers)
        df['is_round_number'] = (df['TransactionAmt'] % 1 == 0).astype(int)
        df['is_round_10'] = (df['TransactionAmt'] % 10 == 0).astype(int)
        df['is_round_100'] = (df['TransactionAmt'] % 100 == 0).astype(int)
        
        # Amount percentile within user's history
        df['amount_percentile_user'] = df.groupby('card1')['TransactionAmt'].rank(pct=True)
        
        # Amount relative to global statistics
        global_mean = df['TransactionAmt'].mean()
        global_std = df['TransactionAmt'].std()
        df['amount_zscore'] = (df['TransactionAmt'] - global_mean) / (global_std + 1)
        
        # Amount bins
        df['amount_bin'] = pd.cut(
            df['TransactionAmt'], 
            bins=[-np.inf, 50, 100, 200, 500, 1000, 5000, np.inf],
            labels=[0, 1, 2, 3, 4, 5, 6]
        ).cat.codes.astype(int)
        
        logger.info("Created amount features")
        return df


class FraudFeatureEngineer:
    """Master feature engineering pipeline"""
    
    def __init__(self, lookback_windows: List[int] = None):
        self.lookback_windows = lookback_windows or [1, 7, 30]
        
        self.temporal = TemporalFeatureEngineer()
        self.user_behavior = UserBehaviorFeatureEngineer(self.lookback_windows)
        self.velocity = VelocityFeatureEngineer()
        self.merchant = MerchantFeatureEngineer()
        self.device = DeviceFeatureEngineer()
        self.amount = AmountFeatureEngineer()
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        train_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info(f"Starting feature engineering on {len(df):,} rows...")
        
        # Step 1: Temporal features
        logger.info("  Creating temporal features...")
        df = self.temporal.create_features(df)
        
        # Step 2: Amount features
        logger.info("  Creating amount features...")
        df = self.amount.create_features(df)
        
        # Step 3: User behavior features
        logger.info("  Creating user behavior features...")
        df = self.user_behavior.create_features(df)
        
        # Step 4: Velocity features
        logger.info("  Creating velocity features...")
        df = self.velocity.create_features(df)
        
        # Step 5: Merchant features
        logger.info("  Creating merchant features...")
        df = self.merchant.create_features(df, train_df)
        
        # Step 6: Device features
        logger.info("  Creating device features...")
        df = self.device.create_features(df)
        
        # Fill any remaining NaN values
        df = df.fillna(-999)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        logger.info(f"Total features: {df.shape[1]}")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of engineered feature names"""
        exclude_cols = [
            'TransactionID', 'isFraud', 'TransactionDT', 
            'transaction_datetime', 'card1', 'card2'
        ]
        return [col for col in df.columns if col not in exclude_cols]
