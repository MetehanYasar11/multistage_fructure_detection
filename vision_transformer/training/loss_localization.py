"""
Loss Functions for Patch-Level Localization with Weak Supervision

Multi-Task Loss combining:
1. Global Classification Loss: Standard BCE on image-level labels
2. Patch Localization Loss: Weakly-supervised patch-level learning

Weak Supervision Strategy:
- We only have IMAGE-level labels (fractured / healthy)
- We don't have PATCH-level labels (which specific patches are fractured)
- Solution: Use Multiple Instance Learning (MIL) approach

MIL Assumptions:
- If image is FRACTURED → At least SOME patches should predict "fracture"
- If image is HEALTHY → ALL patches should predict "healthy"

This encourages spatial localization without patch-level annotations!

Author: Master's Thesis Project
Date: October 28, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiTaskLocalizationLoss(nn.Module):
    """
    Multi-task loss for global classification + patch localization.
    
    Components:
    1. Global BCE: Standard binary cross-entropy on image-level prediction
    2. Patch MIL Loss: Multiple Instance Learning for patch localization
    3. Optional: Diversity Loss to encourage varied patch predictions
    
    Args:
        global_weight: Weight for global classification loss
        patch_weight: Weight for patch localization loss
        diversity_weight: Weight for diversity regularization
        use_focal_loss: Use focal loss instead of BCE for global classification
        focal_alpha: Focal loss alpha (class balance)
        focal_gamma: Focal loss gamma (focus on hard examples)
    """
    
    def __init__(
        self,
        global_weight: float = 1.0,
        patch_weight: float = 0.5,
        diversity_weight: float = 0.1,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        pos_weight: float = None  # For class imbalance
    ):
        super().__init__()
        
        self.global_weight = global_weight
        self.patch_weight = patch_weight
        self.diversity_weight = diversity_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Class imbalance weight: 260 fractured / 80 healthy = 3.25
        # So healthy class should be weighted 3.25x more
        self.pos_weight = pos_weight if pos_weight is not None else (80.0 / 260.0)  # 0.31
        
        print("="*70)
        print("MULTI-TASK LOCALIZATION LOSS")
        print("="*70)
        print(f"Global weight:    {global_weight}")
        print(f"Patch MIL weight: {patch_weight}")
        print(f"Diversity weight: {diversity_weight}")
        print(f"Pos weight (class balance): {self.pos_weight:.2f}")
        print(f"Global loss type: {'Focal' if use_focal_loss else 'BCE'}")
        if use_focal_loss:
            print(f"  - Focal alpha:  {focal_alpha}")
            print(f"  - Focal gamma:  {focal_gamma}")
        print("="*70)
    
    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.75,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Focal Loss for handling class imbalance.
        
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        
        Args:
            logits: (B, 1) raw predictions
            targets: (B, 1) binary labels
            alpha: Weight for positive class
            gamma: Focusing parameter
        """
        probs = torch.sigmoid(logits)
        
        # Binary focal loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        
        focal = alpha_t * (1 - p_t) ** gamma * bce
        
        return focal.mean()
    
    def global_classification_loss(
        self,
        global_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Global image-level classification loss.
        
        Args:
            global_logits: (B, 1) image predictions
            targets: (B,) or (B, 1) binary labels
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Create pos_weight tensor for class imbalance
        pos_weight = torch.tensor([1.0 / self.pos_weight], device=global_logits.device)  # Inverse for healthy emphasis
        
        if self.use_focal_loss:
            return self.focal_loss(
                global_logits,
                targets.float(),
                alpha=self.focal_alpha,
                gamma=self.focal_gamma
            )
        else:
            return F.binary_cross_entropy_with_logits(
                global_logits,
                targets.float(),
                pos_weight=pos_weight
            )
    
    def patch_mil_loss(
        self,
        patch_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Multiple Instance Learning loss for patch localization.
        
        MIL Strategy:
        - Positive bags (fractured images): At least one patch should predict fracture
        - Negative bags (healthy images): All patches should predict healthy
        
        Standard MIL formulation:
        - Positive: Use max pooling over logits (noisy-OR)
        - Negative: Use all patches
        
        Args:
            patch_logits: (B, num_patches) patch predictions (raw logits)
            targets: (B,) binary labels
        """
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        # For positive images (fractured):
        # Use max pooling (noisy-OR): at least one patch should be fracture
        positive_mask = targets == 1
        if positive_mask.any():
            # Max pooling before sigmoid (standard MIL)
            max_patch_logits = patch_logits[positive_mask].max(dim=1)[0]  # (num_pos,)
            positive_loss = F.binary_cross_entropy_with_logits(
                max_patch_logits,
                torch.ones_like(max_patch_logits),
                reduction='mean'
            )
        else:
            positive_loss = torch.tensor(0.0, device=patch_logits.device)
        
        # For negative images (healthy):
        # All patches should predict healthy (penalize all patches)
        negative_mask = targets == 0
        if negative_mask.any():
            # Apply loss to ALL patches individually
            neg_patch_logits = patch_logits[negative_mask]  # (num_neg, num_patches)
            negative_loss = F.binary_cross_entropy_with_logits(
                neg_patch_logits,
                torch.zeros_like(neg_patch_logits),
                reduction='mean'
            )
        else:
            negative_loss = torch.tensor(0.0, device=patch_logits.device)
        
        # Combine positive and negative losses
        mil_loss = positive_loss + negative_loss
        
        return mil_loss
    
    def diversity_loss(
        self,
        patch_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Diversity regularization to encourage spatial variance.
        
        Penalizes uniform patch predictions to encourage localization.
        Only applied to positive (fractured) images.
        
        Args:
            patch_logits: (B, num_patches)
            targets: (B,)
        """
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        positive_mask = targets == 1
        
        if not positive_mask.any():
            return torch.tensor(0.0, device=patch_logits.device)
        
        patch_probs = torch.sigmoid(patch_logits[positive_mask])  # (num_pos, num_patches)
        
        # Encourage variance: penalize low std
        patch_std = patch_probs.std(dim=1)  # (num_pos,)
        
        # Target: We want std > 0.1 (not too uniform)
        target_std = 0.1
        diversity_loss = F.relu(target_std - patch_std).mean()
        
        return diversity_loss
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            model_output: Dictionary from model.forward() containing:
                - 'global_logits': (B, 1)
                - 'patch_logits': (B, num_patches)
            targets: (B,) or (B, 1) binary labels
            
        Returns:
            Dictionary with:
                - 'total_loss': Combined loss for backprop
                - 'global_loss': Global classification component
                - 'patch_loss': Patch MIL component
                - 'diversity_loss': Diversity regularization component
        """
        global_logits = model_output['global_logits']
        patch_logits = model_output['patch_logits']
        
        # 1. Global classification loss
        loss_global = self.global_classification_loss(global_logits, targets)
        
        # 2. Patch MIL loss
        loss_patch = self.patch_mil_loss(patch_logits, targets)
        
        # 3. Diversity loss (optional)
        if self.diversity_weight > 0:
            loss_diversity = self.diversity_loss(patch_logits, targets)
        else:
            loss_diversity = torch.tensor(0.0, device=global_logits.device)
        
        # 4. Combine losses
        total_loss = (
            self.global_weight * loss_global +
            self.patch_weight * loss_patch +
            self.diversity_weight * loss_diversity
        )
        
        return {
            'total_loss': total_loss,
            'global_loss': loss_global,
            'patch_loss': loss_patch,
            'diversity_loss': loss_diversity
        }


class PatchSupervisionLoss(nn.Module):
    """
    Alternative: Patch-level supervision with pseudo-labels.
    
    If we can generate pseudo patch labels (e.g., using radiologist annotations
    or heuristics), we can use direct patch-level supervision.
    
    This is stronger than MIL but requires more annotation effort.
    """
    
    def __init__(
        self,
        global_weight: float = 1.0,
        patch_weight: float = 1.0,
        use_focal_loss: bool = True
    ):
        super().__init__()
        
        self.global_weight = global_weight
        self.patch_weight = patch_weight
        self.use_focal_loss = use_focal_loss
        
        if use_focal_loss:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        global_targets: torch.Tensor,
        patch_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute supervised patch loss.
        
        Args:
            model_output: Model predictions
            global_targets: (B,) image labels
            patch_targets: (B, num_patches) patch labels (if available)
        """
        global_logits = model_output['global_logits']
        patch_logits = model_output['patch_logits']
        
        # Global loss
        if global_targets.dim() == 1:
            global_targets = global_targets.unsqueeze(1)
        loss_global = self.criterion(global_logits, global_targets.float())
        
        # Patch loss (only if patch labels provided)
        if patch_targets is not None:
            loss_patch = self.criterion(patch_logits, patch_targets.float())
            total_loss = self.global_weight * loss_global + self.patch_weight * loss_patch
        else:
            loss_patch = torch.tensor(0.0, device=global_logits.device)
            total_loss = loss_global
        
        return {
            'total_loss': total_loss,
            'global_loss': loss_global,
            'patch_loss': loss_patch
        }


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING MULTI-TASK LOCALIZATION LOSS")
    print("="*70 + "\n")
    
    # Create loss function
    criterion = MultiTaskLocalizationLoss(
        global_weight=1.0,
        patch_weight=0.5,
        diversity_weight=0.1,
        use_focal_loss=True,
        focal_alpha=0.75,
        focal_gamma=2.0
    )
    
    # Simulate model output
    batch_size = 4
    num_patches = 392
    
    model_output = {
        'global_logits': torch.randn(batch_size, 1),
        'patch_logits': torch.randn(batch_size, num_patches)
    }
    
    # Create targets (2 fractured, 2 healthy)
    targets = torch.tensor([1, 1, 0, 0])
    
    print(f"📥 Input:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num patches: {num_patches}")
    print(f"  - Targets: {targets.tolist()}")
    
    # Compute loss
    losses = criterion(model_output, targets)
    
    print(f"\n📊 Loss components:")
    for key, value in losses.items():
        print(f"  - {key}: {value.item():.4f}")
    
    print(f"\n✅ Loss test completed!")
    
    # Test gradients (need to make tensors require grad)
    model_output_grad = {
        'global_logits': torch.randn(batch_size, 1, requires_grad=True),
        'patch_logits': torch.randn(batch_size, num_patches, requires_grad=True)
    }
    
    losses_grad = criterion(model_output_grad, targets)
    total_loss_grad = losses_grad['total_loss']
    total_loss_grad.backward()
    
    print(f"\n✅ Backward pass successful!")
    print(f"  - Global logits grad: {model_output_grad['global_logits'].grad is not None}")
    print(f"  - Patch logits grad: {model_output_grad['patch_logits'].grad is not None}")
    print("="*70)
