"""
XGBoost Fraud Detection Model
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any, Optional
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class XGBoostFraudDetector:
    """
    XGBoost model for fraud detection
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or self.get_default_params()
        self.model = None
        self.feature_importance = None
        self.best_iteration = None
    
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': 100,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50
    ):
        """
        Train XGBoost model with early stopping
        """
        logger.info("Training XGBoost model...")
        
        # Adjust scale_pos_weight based on actual fraud rate
        fraud_rate = y_train.mean()
        self.params['scale_pos_weight'] = (1 - fraud_rate) / fraud_rate
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds, save_best=True
                ),
                xgb.callback.EvaluationMonitor(period=100),
            ],
        )
        
        self.best_iteration = self.model.best_iteration
        
        # Feature importance
        importance = self.model.get_score(importance_type='weight')
        self.feature_importance = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        # Evaluate
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
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
            'val_pr_auc': val_pr_auc
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
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
        
        # Save XGBoost model separately
        self.model.save_model(str(filepath.with_suffix('.xgb')))
        
        # Save metadata
        joblib.dump({
            'params': self.params,
            'feature_importance': self.feature_importance,
            'best_iteration': self.best_iteration
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'XGBoostFraudDetector':
        """Load model from disk"""
        filepath = Path(filepath)
        
        detector = cls()
        
        # Load metadata
        data = joblib.load(filepath)
        detector.params = data['params']
        detector.feature_importance = data['feature_importance']
        detector.best_iteration = data['best_iteration']
        
        # Load XGBoost model
        detector.model = xgb.Booster()
        detector.model.load_model(str(filepath.with_suffix('.xgb')))
        
        logger.info(f"Model loaded from {filepath}")
        return detector
