"""
CatBoost Fraud Detection Model
Excellent for categorical features
"""

from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CatBoostFraudDetector:
    """
    CatBoost model - excellent for categorical features
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or self.get_default_params()
        self.model = None
        self.feature_importance = None
    
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'verbose': 100,
            'early_stopping_rounds': 50,
            'task_type': 'CPU'  # Change to 'GPU' if available
        }
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        categorical_features: List[str] = None
    ):
        """
        Train CatBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            categorical_features: List of categorical column names
        """
        logger.info("Training CatBoost model...")
        
        # Create Pools
        train_pool = Pool(
            X_train,
            y_train,
            cat_features=categorical_features
        )
        
        val_pool = Pool(
            X_val,
            y_val,
            cat_features=categorical_features
        )
        
        # Initialize and train model
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate
        train_pred = self.model.predict_proba(X_train)[:, 1]
        val_pred = self.model.predict_proba(X_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        val_pr_auc = average_precision_score(y_val, val_pred)
        
        logger.info(f"\nTraining Results:")
        logger.info(f"  Training AUC: {train_auc:.4f}")
        logger.info(f"  Validation AUC: {val_auc:.4f}")
        logger.info(f"  Validation PR-AUC: {val_pr_auc:.4f}")
        
        return {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_pr_auc': val_pr_auc
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)[:, 1]
    
    def predict_with_threshold(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction with custom threshold"""
        proba = self.predict(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N important features"""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet!")
        return self.feature_importance.head(top_n)
    
    def save(self, filepath: str):
        """Save model to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(filepath))
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CatBoostFraudDetector':
        """Load model from disk"""
        detector = cls()
        detector.model = CatBoostClassifier()
        detector.model.load_model(str(filepath))

        # Restore feature importance from loaded model
        try:
            names = detector.model.feature_names_
            if names:
                importances = detector.model.feature_importances_
                detector.feature_importance = pd.DataFrame({
                    'feature': names,
                    'importance': importances,
                }).sort_values('importance', ascending=False)
        except Exception:
            pass  # feature_importance stays None if model has none

        logger.info(f"Model loaded from {filepath}")
        return detector
