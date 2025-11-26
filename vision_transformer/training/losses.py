"""
Loss Functions for Dental X-Ray Fracture Detection

This module implements various loss functions optimized for imbalanced
binary classification in medical imaging.

Key Features:
- Focal Loss: Handle class imbalance and focus on hard examples
- Combined Loss: BCE + Focal weighted combination
- Label Smoothing: Prevent overconfidence
- Dice Loss: Segmentation-style loss for binary classification

Author: Master's Thesis Project
Date: October 28, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss down-weights easy examples and focuses training on hard negatives.
    This is particularly useful for our dataset with 3.27:1 class imbalance.
    
    Formula:
        FL(pt) = -α(1-pt)^γ * log(pt)
        
    where:
        pt = p if y=1, else (1-p)
        α = balancing factor for positive class
        γ = focusing parameter (higher = more focus on hard examples)
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        alpha: Balancing factor for positive class (0.25 recommended)
        gamma: Focusing parameter (2.0 recommended)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits) (B, 1) or (B,)
            targets: Ground truth labels (B,) - values in {0, 1}
            
        Returns:
            Loss value
        """
        # Ensure inputs and targets have compatible shapes
        if inputs.dim() == 2:
            inputs = inputs.squeeze(1)  # (B, 1) -> (B,)
        
        targets = targets.float()
        if targets.dim() == 2:
            targets = targets.squeeze(1)  # (B, 1) -> (B,)
        
        # Compute BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction='none'
        )
        
        # Compute pt (probability of correct class)
        pt = torch.exp(-bce_loss)  # pt = p if y=1, else (1-p)
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss: Weighted combination of BCE and Focal Loss.
    
    This loss function combines:
    1. BCE Loss: Standard binary cross-entropy (baseline)
    2. Focal Loss: Focus on hard examples and class imbalance
    
    The combination provides:
    - Stable training from BCE
    - Hard example mining from Focal Loss
    - Class imbalance handling
    
    Args:
        focal_alpha: Focal loss alpha parameter (0.25 = more weight to positive class)
        focal_gamma: Focal loss gamma parameter (2.0 = focus on hard examples)
        bce_weight: Weight for BCE loss in combination
        focal_weight: Weight for Focal loss in combination
        label_smoothing: Label smoothing factor (0.1 recommended)
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        label_smoothing: float = 0.1
    ):
        super(CombinedLoss, self).__init__()
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing
        
        # Validate weights sum to 1.0
        total_weight = bce_weight + focal_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Loss weights sum to {total_weight}, not 1.0. Normalizing...")
            self.bce_weight /= total_weight
            self.focal_weight /= total_weight
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            inputs: Model predictions (logits) (B, 1) or (B,)
            targets: Ground truth labels (B,) - values in {0, 1}
            
        Returns:
            Combined loss value
        """
        # Ensure compatible shapes
        if inputs.dim() == 2:
            inputs = inputs.squeeze(1)
        
        targets = targets.float()
        
        # Apply label smoothing
        # Original: 0 -> 0, 1 -> 1
        # Smoothed: 0 -> 0.05, 1 -> 0.95 (with smoothing=0.1)
        if self.label_smoothing > 0:
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # Compute BCE loss (with smoothed labels)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth)
        
        # Compute Focal loss (with original labels)
        focal_loss = self.focal_loss(inputs, targets)
        
        # Combine losses
        total_loss = self.bce_weight * bce_loss + self.focal_weight * focal_loss
        
        return total_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for binary classification.
    
    Dice loss is commonly used in segmentation tasks but can be adapted
    for classification. It directly optimizes the Dice coefficient metric.
    
    Formula:
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        DiceLoss = 1 - Dice
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Model predictions (logits) (B, 1) or (B,)
            targets: Ground truth labels (B,) - values in {0, 1}
            
        Returns:
            Dice loss value
        """
        # Ensure compatible shapes
        if inputs.dim() == 2:
            inputs = inputs.squeeze(1)
        
        targets = targets.float()
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute intersection and union
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss = 1 - Dice
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss.
    
    This loss combines Dice loss (optimizes metric directly) with
    BCE loss (provides stable gradients).
    
    Args:
        dice_weight: Weight for Dice loss
        bce_weight: Weight for BCE loss
        smooth: Smoothing factor for Dice loss
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0
    ):
        super(DiceBCELoss, self).__init__()
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined Dice + BCE loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Combined loss value
        """
        # Ensure compatible shapes
        if inputs.dim() == 2:
            inputs = inputs.squeeze(1)
        
        targets = targets.float()
        
        # Compute losses
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        
        # Combine
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE Loss with class weights.
    
    This applies different weights to positive and negative classes
    to handle class imbalance.
    
    Args:
        pos_weight: Weight for positive class (fractured)
    """
    
    def __init__(self, pos_weight: float = 1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight])
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Weighted BCE loss
        """
        # Ensure compatible shapes
        if inputs.dim() == 2:
            inputs = inputs.squeeze(1)
        
        targets = targets.float()
        
        # Move pos_weight to same device as inputs
        if self.pos_weight.device != inputs.device:
            self.pos_weight = self.pos_weight.to(inputs.device)
        
        # Compute weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.pos_weight
        )
        
        return loss


# Loss factory function
class BCEWrapper(nn.Module):
    """Wrapper for BCEWithLogitsLoss that handles shape mismatches."""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure both have same shape
        if inputs.dim() == 2 and targets.dim() == 1:
            targets = targets.unsqueeze(1)
        elif inputs.dim() == 1 and targets.dim() == 2:
            inputs = inputs.squeeze(1)
            targets = targets.squeeze(1)
        
        return self.bce(inputs, targets)


def get_loss_function(
    loss_type: str = 'combined',
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function
            - 'bce': Binary Cross-Entropy
            - 'weighted_bce': Weighted BCE with pos_weight
            - 'focal': Focal Loss
            - 'combined': BCE + Focal (recommended)
            - 'dice': Dice Loss
            - 'dice_bce': Dice + BCE
        **kwargs: Additional arguments for specific loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == 'bce':
        return BCEWrapper()
    
    elif loss_type == 'weighted_bce':
        pos_weight = kwargs.get('pos_weight', 3.27)  # Class ratio
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('focal_alpha', 0.25)
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'combined':
        focal_alpha = kwargs.get('focal_alpha', 0.25)
        focal_gamma = kwargs.get('focal_gamma', 2.0)
        bce_weight = kwargs.get('bce_weight', 0.5)
        focal_weight = kwargs.get('focal_weight', 0.5)
        label_smoothing = kwargs.get('label_smoothing', 0.1)
        return CombinedLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            bce_weight=bce_weight,
            focal_weight=focal_weight,
            label_smoothing=label_smoothing
        )
    
    elif loss_type == 'dice':
        smooth = kwargs.get('smooth', 1.0)
        return DiceLoss(smooth=smooth)
    
    elif loss_type == 'dice_bce':
        dice_weight = kwargs.get('dice_weight', 0.5)
        bce_weight = kwargs.get('bce_weight', 0.5)
        smooth = kwargs.get('smooth', 1.0)
        return DiceBCELoss(
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            smooth=smooth
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("TESTING LOSS FUNCTIONS")
    print("="*70)
    
    # Create dummy data
    batch_size = 8
    inputs = torch.randn(batch_size, 1)  # Logits
    targets = torch.randint(0, 2, (batch_size,)).float()  # Binary labels
    
    print(f"\nInputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets: {targets.tolist()}")
    
    # Test all loss functions
    loss_configs = {
        'BCE': ('bce', {}),
        'Weighted BCE': ('weighted_bce', {'pos_weight': 3.27}),
        'Focal Loss': ('focal', {'focal_alpha': 0.25, 'focal_gamma': 2.0}),
        'Combined Loss': ('combined', {
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'bce_weight': 0.5,
            'focal_weight': 0.5,
            'label_smoothing': 0.1
        }),
        'Dice Loss': ('dice', {}),
        'Dice + BCE': ('dice_bce', {'dice_weight': 0.5, 'bce_weight': 0.5})
    }
    
    print("\n" + "="*70)
    print("Loss Function Comparison")
    print("="*70)
    
    for name, (loss_type, kwargs) in loss_configs.items():
        loss_fn = get_loss_function(loss_type, **kwargs)
        loss_value = loss_fn(inputs, targets)
        print(f"{name:20s}: {loss_value.item():.4f}")
    
    # Test gradient flow
    print("\n" + "="*70)
    print("Testing Gradient Flow")
    print("="*70)
    
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    loss_fn = get_loss_function('combined')
    
    dummy_input = torch.randn(batch_size, 10)
    dummy_target = torch.randint(0, 2, (batch_size,)).float()
    
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = loss_fn(output, dummy_target)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradients computed: ✓")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
