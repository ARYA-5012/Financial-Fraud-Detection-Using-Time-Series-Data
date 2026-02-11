"""
Evaluation Metrics Module
Comprehensive fraud detection model evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def evaluate_fraud_model(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    threshold: float = 0.5,
    cost_fraud_missed: float = 1000,
    cost_false_alarm: float = 10,
    print_report: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive fraud detection evaluation
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        cost_fraud_missed: Cost per missed fraud
        cost_false_alarm: Cost per false alarm
        print_report: Whether to print detailed report
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        results['roc_auc'] = roc_auc
    except ValueError:
        results['roc_auc'] = 0.0
    
    # PR-AUC (better for imbalanced data)
    try:
        pr_auc = average_precision_score(y_true, y_pred_proba)
        results['pr_auc'] = pr_auc
    except ValueError:
        results['pr_auc'] = 0.0
    
    # F1 Score
    results['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Accuracy
    results['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        tn, fp, fn, tp = 0, 0, 0, 0
    
    results['true_negatives'] = tn
    results['false_positives'] = fp
    results['false_negatives'] = fn
    results['true_positives'] = tp
    
    # Precision, Recall, Specificity
    results['precision'] = precision_score(y_true, y_pred, zero_division=0)
    results['recall'] = recall_score(y_true, y_pred, zero_division=0)
    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False Positive Rate
    results['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Cost analysis
    total_cost = (fn * cost_fraud_missed) + (fp * cost_false_alarm)
    results['estimated_cost'] = total_cost
    
    # Money saved (compared to no model - all transactions marked as legit)
    baseline_cost = y_true.sum() * cost_fraud_missed
    results['cost_savings'] = baseline_cost - total_cost
    results['cost_savings_percent'] = (results['cost_savings'] / baseline_cost * 100) if baseline_cost > 0 else 0
    
    if print_report:
        print("=" * 60)
        print("FRAUD DETECTION MODEL EVALUATION")
        print("=" * 60)
        print(f"\nThreshold: {threshold}")
        print(f"\nROC-AUC Score: {results['roc_auc']:.4f}")
        print(f"PR-AUC Score: {results['pr_auc']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn:,} | FP: {fp:,}")
        print(f"  FN: {fn:,} | TP: {tp:,}")
        print(f"\nPrecision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"Specificity: {results['specificity']:.4f}")
        print(f"False Positive Rate: {results['fpr']:.4f}")
        print(f"\nCost Analysis:")
        print(f"  Estimated Cost: ${total_cost:,.2f}")
        print(f"  Cost Savings: ${results['cost_savings']:,.2f} ({results['cost_savings_percent']:.1f}%)")
        print("=" * 60)
    
    return results


def find_optimal_threshold(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    metric: str = 'f1',
    cost_fraud_missed: float = 1000,
    cost_false_alarm: float = 10
) -> Tuple[float, Dict[str, Any]]:
    """
    Find optimal classification threshold
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: 'f1', 'cost', or 'recall'
        cost_fraud_missed: Cost per missed fraud
        cost_false_alarm: Cost per false alarm
        
    Returns:
        Optimal threshold and metrics at that threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    if metric == 'f1':
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        best_metrics = {
            'threshold': optimal_threshold,
            'f1': f1_scores[optimal_idx],
            'precision': precision[optimal_idx],
            'recall': recall[optimal_idx]
        }
        
    elif metric == 'cost':
        costs = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                cost = (fn * cost_fraud_missed) + (fp * cost_false_alarm)
            else:
                cost = float('inf')
            costs.append(cost)
        
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        
        best_metrics = {
            'threshold': optimal_threshold,
            'cost': costs[optimal_idx],
            'precision': precision[optimal_idx],
            'recall': recall[optimal_idx]
        }
        
    elif metric == 'recall':
        # Find threshold that achieves at least 90% recall
        target_recall = 0.90
        valid_indices = np.where(recall >= target_recall)[0]
        
        if len(valid_indices) > 0:
            # Among those with sufficient recall, maximize precision
            best_idx = valid_indices[np.argmax(precision[valid_indices])]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        else:
            optimal_threshold = 0.1  # Lower threshold to catch more fraud
            best_idx = 0
        
        best_metrics = {
            'threshold': optimal_threshold,
            'precision': precision[best_idx],
            'recall': recall[best_idx]
        }
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    logger.info(f"Optimal threshold ({metric}): {optimal_threshold:.4f}")
    return optimal_threshold, best_metrics


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compare multiple models
    
    Args:
        y_true: True labels
        predictions: Dictionary of {model_name: predicted_probabilities}
        threshold: Classification threshold
        
    Returns:
        DataFrame with model comparison
    """
    results = []
    
    for name, y_pred_proba in predictions.items():
        metrics = evaluate_fraud_model(y_true, y_pred_proba, threshold, print_report=False)
        
        results.append({
            'Model': name,
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'F1 Score': metrics['f1_score'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Est. Cost': metrics['estimated_cost']
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('ROC-AUC', ascending=False)
    
    return df


def get_threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray = None
) -> pd.DataFrame:
    """
    Analyze metrics across different thresholds
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        thresholds: Array of thresholds to analyze
        
    Returns:
        DataFrame with metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    for thresh in thresholds:
        metrics = evaluate_fraud_model(y_true, y_pred_proba, thresh, print_report=False)
        
        results.append({
            'Threshold': thresh,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'FP': metrics['false_positives'],
            'FN': metrics['false_negatives'],
            'Est. Cost': metrics['estimated_cost']
        })
    
    return pd.DataFrame(results)
