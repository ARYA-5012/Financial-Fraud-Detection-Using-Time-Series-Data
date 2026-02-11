"""
LightGBM Fraud Detection Model
Primary production model - fast and accurate
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any, Optional, List
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class LightGBMFraudDetector:
    """
    LightGBM model optimized for fraud detection
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or self.get_default_params()
        self.model = None
        self.feature_importance = None
        self.best_iteration = None
    
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        """
        Optimized parameters for fraud detection
        """
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': 100,  # Adjust based on fraud rate
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        categorical_features: List[str] = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50
    ):
        """
        Train LightGBM model with early stopping
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            categorical_features: List of categorical column names
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience
        """
        logger.info("Training LightGBM model...")
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Validation samples: {len(X_val):,}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Adjust scale_pos_weight based on actual fraud rate
        fraud_rate = y_train.mean()
        self.params['scale_pos_weight'] = (1 - fraud_rate) / fraud_rate
        logger.info(f"Fraud rate: {fraud_rate:.4%}, scale_pos_weight: {self.params['scale_pos_weight']:.2f}")
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            categorical_feature=categorical_features or 'auto'
        )
        
        val_data = lgb.Dataset(
            X_val, 
            label=y_val,
            reference=train_data,
            categorical_feature=categorical_features or 'auto'
        )
        
        # Train
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.best_iteration = self.model.best_iteration
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(),
            'importance_gain': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        val_pr_auc = average_precision_score(y_val, val_pred)
        
        logger.info(f"\nTraining Results:")
        logger.info(f"  Best iteration: {self.best_iteration}")
        logger.info(f"  Training AUC: {train_auc:.4f}")
        logger.info(f"  Validation AUC: {val_auc:.4f}")
        logger.info(f"  Validation PR-AUC: {val_pr_auc:.4f}")
        
        return {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_pr_auc': val_pr_auc,
            'best_iteration': self.best_iteration
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_with_threshold(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Binary prediction with custom threshold
        """
        proba = self.predict(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N important features
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet!")
        return self.feature_importance.head(top_n)
    
    def save(self, filepath: str):
        """Save model to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'params': self.params,
            'feature_importance': self.feature_importance,
            'best_iteration': self.best_iteration
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LightGBMFraudDetector':
        """Load model from disk"""
        data = joblib.load(filepath)
        
        detector = cls(params=data['params'])
        detector.model = data['model']
        detector.feature_importance = data['feature_importance']
        detector.best_iteration = data['best_iteration']
        
        logger.info(f"Model loaded from {filepath}")
        return detector
