"""
Ensemble Model for Fraud Detection
Combines tree-based models with LSTM using stacking
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class FraudEnsemble:
    """
    Ensemble of tree-based models + LSTM using stacking
    """
    
    def __init__(self, models_dict: Dict[str, Any] = None):
        """
        Args:
            models_dict: Dictionary of {'model_name': model_object}
        """
        self.models = models_dict or {}
        self.meta_model = None
        self.weights = None
        self.model_names = []
    
    def add_model(self, name: str, model: Any):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.model_names = list(self.models.keys())
        logger.info(f"Added model: {name}")
    
    def create_meta_features(
        self, 
        X_tabular: pd.DataFrame, 
        X_sequential: np.ndarray = None,
        sequential_models: List[str] = None
    ) -> np.ndarray:
        """
        Generate predictions from all base models
        
        Args:
            X_tabular: Tabular features for tree-based models
            X_sequential: Sequential features for LSTM
            sequential_models: Names of models that use sequential data
        """
        sequential_models = sequential_models or ['lstm', 'LSTM']
        meta_features = []
        
        for name, model in self.models.items():
            logger.debug(f"Getting predictions from {name}...")
            
            is_sequential = any(seq_name.lower() in name.lower() 
                              for seq_name in sequential_models)
            
            if is_sequential and X_sequential is not None:
                pred = model.predict(X_sequential)
            else:
                pred = model.predict(X_tabular)
            
            # Ensure predictions are 1D
            pred = np.asarray(pred).flatten()
            meta_features.append(pred.reshape(-1, 1))
        
        return np.hstack(meta_features)
    
    def train_meta_model(
        self, 
        X_train_tabular: pd.DataFrame,
        y_train: np.ndarray,
        X_val_tabular: pd.DataFrame,
        y_val: np.ndarray,
        X_train_seq: np.ndarray = None,
        X_val_seq: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Train meta-model (Level 1) on base model predictions
        
        Args:
            X_train_tabular: Training features (tabular)
            y_train: Training labels
            X_val_tabular: Validation features (tabular)
            y_val: Validation labels
            X_train_seq: Training features (sequential)
            X_val_seq: Validation features (sequential)
        """
        logger.info("Creating meta features for training...")
        meta_features_train = self.create_meta_features(X_train_tabular, X_train_seq)
        
        logger.info("Creating meta features for validation...")
        meta_features_val = self.create_meta_features(X_val_tabular, X_val_seq)
        
        # Train meta-model (Logistic Regression)
        logger.info("Training meta-model (Logistic Regression)...")
        self.meta_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        self.meta_model.fit(meta_features_train, y_train)
        
        # Evaluate
        val_pred = self.meta_model.predict_proba(meta_features_val)[:, 1]
        train_pred = self.meta_model.predict_proba(meta_features_train)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        val_pr_auc = average_precision_score(y_val, val_pred)
        
        # Extract weights (coefficients)
        self.weights = self.meta_model.coef_[0]
        
        logger.info(f"\nEnsemble Training Results:")
        logger.info(f"  Training AUC: {train_auc:.4f}")
        logger.info(f"  Validation AUC: {val_auc:.4f}")
        logger.info(f"  Validation PR-AUC: {val_pr_auc:.4f}")
        
        logger.info("\nModel Weights:")
        for name, weight in zip(self.models.keys(), self.weights):
            logger.info(f"  {name}: {weight:.4f}")
        
        return {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_pr_auc': val_pr_auc
        }
    
    def predict(
        self, 
        X_tabular: pd.DataFrame, 
        X_sequential: np.ndarray = None
    ) -> np.ndarray:
        """
        Make ensemble prediction using meta-model
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained yet!")
        
        meta_features = self.create_meta_features(X_tabular, X_sequential)
        return self.meta_model.predict_proba(meta_features)[:, 1]
    
    def predict_with_threshold(
        self, 
        X_tabular: pd.DataFrame,
        X_sequential: np.ndarray = None,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Binary prediction with custom threshold"""
        proba = self.predict(X_tabular, X_sequential)
        return (proba >= threshold).astype(int)
    
    def weighted_average_predict(
        self, 
        X_tabular: pd.DataFrame, 
        X_sequential: np.ndarray = None,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Alternative: Simple weighted average (no meta-model)
        
        Args:
            X_tabular: Tabular features
            X_sequential: Sequential features
            weights: Manual weights for each model
        """
        if weights is None:
            # Equal weights
            weights = {name: 1/len(self.models) for name in self.models.keys()}
        
        ensemble_pred = np.zeros(len(X_tabular))
        
        for name, model in self.models.items():
            if 'lstm' in name.lower() and X_sequential is not None:
                pred = model.predict(X_sequential)
            else:
                pred = model.predict(X_tabular)
            
            pred = np.asarray(pred).flatten()
            ensemble_pred += weights.get(name, 0) * pred
        
        return ensemble_pred
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get learned weights for each model"""
        if self.weights is None:
            return {}
        return dict(zip(self.models.keys(), self.weights))
    
    def compare_individual_models(
        self,
        X_tabular: pd.DataFrame,
        y_true: np.ndarray,
        X_sequential: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Compare individual model performance vs ensemble
        """
        results = []
        
        for name, model in self.models.items():
            if 'lstm' in name.lower() and X_sequential is not None:
                pred = model.predict(X_sequential)
            else:
                pred = model.predict(X_tabular)
            
            pred = np.asarray(pred).flatten()
            auc = roc_auc_score(y_true, pred)
            pr_auc = average_precision_score(y_true, pred)
            
            results.append({
                'Model': name,
                'ROC-AUC': auc,
                'PR-AUC': pr_auc
            })
        
        # Add ensemble
        if self.meta_model is not None:
            ensemble_pred = self.predict(X_tabular, X_sequential)
            ensemble_auc = roc_auc_score(y_true, ensemble_pred)
            ensemble_pr_auc = average_precision_score(y_true, ensemble_pred)
            
            results.append({
                'Model': 'ENSEMBLE',
                'ROC-AUC': ensemble_auc,
                'PR-AUC': ensemble_pr_auc
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('ROC-AUC', ascending=False)
        
        return df
    
    def save(self, filepath: str):
        """Save ensemble (meta-model and weights only, not base models)"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'meta_model': self.meta_model,
            'weights': self.weights,
            'model_names': list(self.models.keys())
        }, filepath)
        
        logger.info(f"Ensemble saved to {filepath}")
        logger.info("Note: Base models must be saved separately")
    
    @classmethod
    def load(cls, filepath: str, models_dict: Dict[str, Any]) -> 'FraudEnsemble':
        """
        Load ensemble meta-model and combine with base models
        
        Args:
            filepath: Path to saved ensemble
            models_dict: Dictionary of loaded base models
        """
        data = joblib.load(filepath)
        
        ensemble = cls(models_dict)
        ensemble.meta_model = data['meta_model']
        ensemble.weights = data['weights']
        
        logger.info(f"Ensemble loaded from {filepath}")
        return ensemble
