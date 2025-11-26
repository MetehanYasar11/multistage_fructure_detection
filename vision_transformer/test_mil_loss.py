"""
Test updated MIL loss function
"""
import torch
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.loss_localization import MultiTaskLocalizationLoss

# Create loss
criterion = MultiTaskLocalizationLoss(
    global_weight=1.0,
    patch_weight=1.0,
    diversity_weight=0.0,
    use_focal_loss=True
)

# Create dummy data
B = 4
num_patches = 392

# Simulated model output
model_output = {
    'global_logits': torch.randn(B, 1, requires_grad=True),  # Random logits
    'patch_logits': torch.randn(B, num_patches, requires_grad=True),  # Random patch logits
}

# Targets: 2 fractured, 2 healthy
targets = torch.tensor([1, 1, 0, 0])

print("\nTest 1: Random initialization")
print("="*60)
losses = criterion(model_output, targets)
print(f"Total loss: {losses['total_loss'].item():.4f}")
print(f"Global loss: {losses['global_loss'].item():.4f}")
print(f"Patch MIL loss: {losses['patch_loss'].item():.4f}")
print(f"Diversity loss: {losses['diversity_loss'].item():.4f}")

# Test backward
print("\nTest 2: Backward pass")
print("="*60)
losses['total_loss'].backward()
print("✅ Backward pass successful")

# Test with perfect predictions
print("\nTest 3: Perfect predictions")
print("="*60)
perfect_output = {
    'global_logits': torch.tensor([[5.0], [5.0], [-5.0], [-5.0]]),  # High confidence
    'patch_logits': torch.cat([
        torch.full((2, num_patches), 5.0),   # Fractured: all high
        torch.full((2, num_patches), -5.0),  # Healthy: all low
    ], dim=0)
}
losses_perfect = criterion(perfect_output, targets)
print(f"Total loss: {losses_perfect['total_loss'].item():.4f}")
print(f"Global loss: {losses_perfect['global_loss'].item():.4f}")
print(f"Patch MIL loss: {losses_perfect['patch_loss'].item():.4f}")

# Test with worst predictions  
print("\nTest 4: Worst predictions")
print("="*60)
worst_output = {
    'global_logits': torch.tensor([[-5.0], [-5.0], [5.0], [5.0]]),  # Inverted
    'patch_logits': torch.cat([
        torch.full((2, num_patches), -5.0),  # Fractured: all low (WRONG!)
        torch.full((2, num_patches), 5.0),   # Healthy: all high (WRONG!)
    ], dim=0)
}
losses_worst = criterion(worst_output, targets)
print(f"Total loss: {losses_worst['total_loss'].item():.4f}")
print(f"Global loss: {losses_worst['global_loss'].item():.4f}")
print(f"Patch MIL loss: {losses_worst['patch_loss'].item():.4f}")

print("\n" + "="*60)
print("Expected behavior:")
print(f"  Random loss: ~{losses['total_loss'].item():.4f}")
print(f"  Perfect loss: ~{losses_perfect['total_loss'].item():.4f} (should be < 0.01)")
print(f"  Worst loss: ~{losses_worst['total_loss'].item():.4f} (should be > random)")
print("="*60)

if losses_perfect['total_loss'].item() < 0.01 and losses_worst['total_loss'].item() > losses['total_loss'].item():
    print("✅ Loss function working correctly!")
else:
    print("❌ Loss function has issues!")
