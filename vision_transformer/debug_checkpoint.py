"""
Simple debug: Load trained model and check predictions
"""
import torch
import numpy as np

# Load trained model
checkpoint_path = r"outputs\localization_model\best_model.pth"

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\nCheckpoint keys:", checkpoint.keys())
print("\nBest epoch:", checkpoint.get('epoch', 'N/A'))
print("Best F1:", checkpoint.get('f1', 'N/A'))

# Check if model learned anything
if 'history' in checkpoint:
    history = checkpoint['history']
    print("\nTraining History:")
    print(f"Train F1: {history['train_f1']}")
    print(f"Val F1: {history['val_f1']}")
    print(f"Train Loss: {history['train_loss']}")
    print(f"Val Loss: {history['val_loss']}")

# Load model weights
print("\nModel state dict keys (first 10):")
state_dict = checkpoint['model_state_dict']
for i, key in enumerate(list(state_dict.keys())[:10]):
    tensor = state_dict[key]
    print(f"  {key}: shape={tensor.shape}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")

# Check final layer weights
print("\nFinal layer weights:")
for key in state_dict.keys():
    if 'global_head' in key and 'weight' in key:
        tensor = state_dict[key]
        print(f"  {key}:")
        print(f"    Shape: {tensor.shape}")
        print(f"    Mean: {tensor.float().mean():.6f}")
        print(f"    Std: {tensor.float().std():.6f}")
        print(f"    Min: {tensor.float().min():.6f}")
        print(f"    Max: {tensor.float().max():.6f}")

print("\n" + "="*80)
print("DIAGNOSIS:")
if checkpoint.get('f1', 0) == 0:
    print("❌ Model did NOT learn (F1 = 0)")
    print("\nLikely causes:")
    print("  1. All predictions same class")
    print("  2. Threshold issue (all probs > 0.5 or all < 0.5)")
    print("  3. Loss function not training correctly")
else:
    print(f"✅ Model learned something (F1 = {checkpoint.get('f1', 0):.4f})")
