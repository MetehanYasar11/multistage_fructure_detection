"""Test focal loss to see if it gives meaningful gradients"""

import torch
import torch.nn.functional as F


def focal_loss_with_logits(pred_logits, target, gamma=2.0, pos_weight=3.0):
    """
    Focal loss for binary classification with logits.
    
    Args:
        pred_logits: [B, 1, H, W] predicted logits (before sigmoid)
        target: [B, 1, H, W] binary target (0 or 1)
        gamma: Focusing parameter (default 2.0)
        pos_weight: Weight for positive class (default 3.0)
    """
    # Apply sigmoid to get probabilities
    pred_prob = torch.sigmoid(pred_logits)
    
    # Compute focal weight
    # For target=1: (1-p)^gamma weights hard examples more
    # For target=0: p^gamma weights hard examples more
    pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
    focal_weight = (1 - pt) ** gamma
    
    # Compute BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(
        pred_logits,
        target,
        reduction='none',
        pos_weight=torch.tensor([pos_weight]).to(pred_logits.device)
    )
    
    # Apply focal weight
    focal_loss = focal_weight * bce_loss
    
    return focal_loss.mean()


def main():
    print("=== Testing Focal Loss ===\n")
    
    # Simulate attention map (14x28)
    B, H, W = 1, 14, 28
    
    # Create GT bbox mask (4 patches at center)
    gt_mask = torch.zeros(B, 1, H, W)
    gt_mask[0, 0, 6:8, 13:15] = 1.0  # 2x2 region
    print(f"GT mask: {gt_mask.sum().item()} active patches out of {H*W}")
    
    # Test 1: Random predictions (untrained model)
    pred_random = torch.randn(B, 1, H, W) * 0.1  # Small random logits
    loss_random = focal_loss_with_logits(pred_random, gt_mask, gamma=2.0, pos_weight=3.0)
    print(f"\nTest 1 - Random predictions:")
    print(f"  Loss: {loss_random.item():.4f}")
    
    # Compute gradient
    pred_random.requires_grad = True
    loss_random = focal_loss_with_logits(pred_random, gt_mask, gamma=2.0, pos_weight=3.0)
    loss_random.backward()
    grad_norm = pred_random.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    # Test 2: All zeros (sigmoid = 0.5)
    pred_zeros = torch.zeros(B, 1, H, W)
    loss_zeros = focal_loss_with_logits(pred_zeros, gt_mask, gamma=2.0, pos_weight=3.0)
    print(f"\nTest 2 - All zeros (sigmoid=0.5):")
    print(f"  Loss: {loss_zeros.item():.4f}")
    
    pred_zeros.requires_grad = True
    loss_zeros = focal_loss_with_logits(pred_zeros, gt_mask, gamma=2.0, pos_weight=3.0)
    loss_zeros.backward()
    grad_norm = pred_zeros.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    # Check gradient at GT patches
    gt_patches_grad = pred_zeros.grad[gt_mask == 1].mean().item()
    other_patches_grad = pred_zeros.grad[gt_mask == 0].mean().item()
    print(f"  GT patches gradient mean: {gt_patches_grad:.6f}")
    print(f"  Other patches gradient mean: {other_patches_grad:.6f}")
    
    # Test 3: Perfect prediction
    pred_perfect = torch.zeros(B, 1, H, W)
    pred_perfect[gt_mask == 1] = 5.0  # High logit for GT patches
    pred_perfect[gt_mask == 0] = -5.0  # Low logit for others
    loss_perfect = focal_loss_with_logits(pred_perfect, gt_mask, gamma=2.0, pos_weight=3.0)
    print(f"\nTest 3 - Perfect prediction:")
    print(f"  Loss: {loss_perfect.item():.4f}")
    
    # Test 4: Wrong prediction (opposite)
    pred_wrong = torch.zeros(B, 1, H, W)
    pred_wrong[gt_mask == 1] = -5.0  # Low logit for GT patches (BAD!)
    pred_wrong[gt_mask == 0] = 5.0  # High logit for others (BAD!)
    loss_wrong = focal_loss_with_logits(pred_wrong, gt_mask, gamma=2.0, pos_weight=3.0)
    print(f"\nTest 4 - Wrong prediction (opposite):")
    print(f"  Loss: {loss_wrong.item():.4f}")
    
    pred_wrong.requires_grad = True
    loss_wrong = focal_loss_with_logits(pred_wrong, gt_mask, gamma=2.0, pos_weight=3.0)
    loss_wrong.backward()
    grad_norm = pred_wrong.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    # Test 5: Slightly better than random
    pred_better = torch.randn(B, 1, H, W) * 0.1
    pred_better[gt_mask == 1] += 0.5  # Slightly higher for GT
    loss_better = focal_loss_with_logits(pred_better, gt_mask, gamma=2.0, pos_weight=3.0)
    print(f"\nTest 5 - Slightly better than random:")
    print(f"  Loss: {loss_better.item():.4f}")
    
    # Compare with standard BCE
    print(f"\n=== Comparison with standard BCE ===")
    pred_test = torch.randn(B, 1, H, W) * 0.1
    
    focal = focal_loss_with_logits(pred_test, gt_mask, gamma=2.0, pos_weight=3.0)
    bce = F.binary_cross_entropy_with_logits(
        pred_test, gt_mask,
        pos_weight=torch.tensor([3.0])
    )
    
    print(f"Focal loss: {focal.item():.4f}")
    print(f"BCE loss: {bce.item():.4f}")
    print(f"Ratio: {focal.item() / bce.item():.3f}")


if __name__ == "__main__":
    main()
