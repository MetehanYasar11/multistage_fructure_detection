"""
Quick test: Check if baseline model already works with attention

Test if we can extract attention from existing baseline model
"""

import torch
import yaml
from models.patch_transformer import PatchTransformerClassifier

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load checkpoint
checkpoint = torch.load(
    'checkpoints/patch_transformer_full/best.pth',
    map_location='cpu',
    weights_only=False
)

print("Checkpoint keys:")
for key in list(checkpoint['model_state_dict'].keys())[:10]:
    print(f"  {key}")

print("\n" + "="*80)
print("Summary:")
print(f"Total parameters: {len(checkpoint['model_state_dict'])}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best Val F1: {checkpoint['best_val_f1']:.4f}")
