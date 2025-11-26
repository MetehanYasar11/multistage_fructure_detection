"""
Metrics for Binary Classification

Compute comprehensive metrics for dental fracture detection.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_metrics(targets, predictions, threshold=0.5):
    """
    Compute comprehensive classification metrics.
    
    Args:
        targets: Ground truth labels (0/1)
        predictions: Model predictions (probabilities or binary)
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    # Ensure binary targets
    targets = targets.astype(int)
    
    # Convert probabilities to binary if needed
    if predictions.dtype == float:
        binary_preds = (predictions >= threshold).astype(int)
    else:
        binary_preds = predictions.astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, binary_preds).ravel()
    
    # Basic metrics
    accuracy = accuracy_score(targets, binary_preds)
    precision = precision_score(targets, binary_preds, zero_division=0)
    recall = recall_score(targets, binary_preds, zero_division=0)
    f1 = f1_score(targets, binary_preds, zero_division=0)
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Dice score (same as F1 for binary classification)
    dice = f1
    
    # AUC (if predictions are probabilities)
    try:
        if predictions.dtype == float and len(np.unique(targets)) > 1:
            auc = roc_auc_score(targets, predictions)
        else:
            auc = None
    except:
        auc = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,
        'f1_score': f1,
        'dice_score': dice,
        'specificity': specificity,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'auc': auc
    }
