"""
Visualization Module
Charts and plots for fraud detection analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_performance_curves(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    model_name: str = "Model",
    save_path: str = None
) -> plt.Figure:
    """
    Plot ROC and Precision-Recall curves
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name for legend
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    axes[0].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.2)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    axes[1].plot(recall, precision, label=f'{model_name} (AP = {pr_auc:.4f})', linewidth=2)
    axes[1].fill_between(recall, precision, alpha=0.2)
    axes[1].axhline(y=y_true.mean(), color='r', linestyle='--', label=f'Baseline ({y_true.mean():.4f})', alpha=0.5)
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance curves to {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    normalize: bool = False,
    save_path: str = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        normalize: Whether to normalize values
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=['Legitimate', 'Fraud'],
        yticklabels=['Legitimate', 'Fraud'],
        ax=ax,
        annot_kws={'size': 14}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    return fig


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
    save_path: str = None
) -> plt.Figure:
    """
    Plot feature importance bar chart
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    # Sort and get top N
    df = feature_importance.nlargest(top_n, 'importance')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    
    bars = ax.barh(df['feature'], df['importance'], color=colors)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance to {save_path}")
    
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str = None
) -> plt.Figure:
    """
    Plot metrics across different thresholds
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    thresholds = np.linspace(0.01, 0.99, 50)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precisions, label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
    
    # Mark optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    ax.axvline(x=thresholds[optimal_idx], color='r', linestyle='--', 
               label=f'Optimal F1 ({thresholds[optimal_idx]:.2f})', alpha=0.7)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Threshold Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='center right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved threshold analysis to {save_path}")
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'ROC-AUC',
    save_path: str = None
) -> plt.Figure:
    """
    Plot model comparison bar chart
    
    Args:
        comparison_df: DataFrame from compare_models()
        metric: Metric to compare
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(comparison_df)))
    
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors)
    
    # Add value labels on bars
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x labels if needed
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison to {save_path}")
    
    return fig


def plot_training_history(
    history: Dict,
    save_path: str = None
) -> plt.Figure:
    """
    Plot LSTM/neural network training history
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    if 'loss' in history:
        axes[0, 0].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
    
    # AUC
    if 'auc' in history:
        axes[0, 1].plot(history['auc'], label='Train AUC')
        if 'val_auc' in history:
            axes[0, 1].plot(history['val_auc'], label='Val AUC')
        axes[0, 1].set_title('Model AUC', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
    
    # Precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Train Precision')
        if 'val_precision' in history:
            axes[1, 0].plot(history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # Recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Train Recall')
        if 'val_recall' in history:
            axes[1, 1].plot(history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history to {save_path}")
    
    return fig
