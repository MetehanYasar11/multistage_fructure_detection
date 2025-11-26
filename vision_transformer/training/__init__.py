"""
Training module for Dental X-Ray Fracture Detection.

This module provides training utilities, loss functions, and training loops.
"""

from .losses import (
    FocalLoss,
    CombinedLoss,
    DiceLoss,
    DiceBCELoss,
    WeightedBCELoss,
    get_loss_function
)

from .train import Trainer, MetricsTracker, EarlyStopping

__all__ = [
    # Loss functions
    'FocalLoss',
    'CombinedLoss',
    'DiceLoss',
    'DiceBCELoss',
    'WeightedBCELoss',
    'get_loss_function',
    # Training utilities
    'Trainer',
    'MetricsTracker',
    'EarlyStopping',
]
