"""Check what attention maps model is producing"""

import torch
import sys
sys.path.insert(0, '.')

from models.weakly_supervised_patch_transformer import WeaklySupervisedPatchTransformer
from train_weakly_supervised import DentalFractureDatasetWithBbox

# Load model
model = WeaklySupervisedPatchTransformer()
model.load_from_baseline('checkpoints/patch_transformer_full/best.pth')
model.eval()

# Try to load trained checkpoint if exists
import os
if os.path.exists('checkpoints/weakly_supervised_patch_transformer/last.pth'):
    ckpt = torch.load('checkpoints/weakly_supervised_patch_transformer/last.pth', 
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[*] Loaded checkpoint from epoch {ckpt['epoch']}")
else:
    print("[*] Using baseline (no training yet)")

# Load a sample
dataset = DentalFractureDatasetWithBbox(
    root_dir="c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset",
    split='train',
    split_file="outputs/splits/train_val_test_split.json"
)

# Get a fractured sample with bbox
img, label, bbox_mask = dataset[0]
img_batch = img.unsqueeze(0)  # [1, 3, H, W]

print(f"\nInput image shape: {img_batch.shape}")
print(f"Bbox mask shape: {bbox_mask.shape}")
print(f"Bbox active patches: {bbox_mask.sum().item()}")

# Forward pass
with torch.no_grad():
    logits, attention_map = model(img_batch, return_attention=True)

print(f"\nLogits shape: {logits.shape}")
print(f"Logits: {logits}")
print(f"Predicted class: {'Fractured' if logits.argmax() == 1 else 'Healthy'}")

print(f"\nAttention map shape: {attention_map.shape}")
attention = attention_map[0, 0]  # [14, 28]

print(f"Attention statistics:")
print(f"  Min: {attention.min().item():.6f}")
print(f"  Max: {attention.max().item():.6f}")
print(f"  Mean: {attention.mean().item():.6f}")
print(f"  Std: {attention.std().item():.6f}")

# Check if attention is uniform (all same value = not learning)
if attention.std() < 0.01:
    print("  ⚠ WARNING: Attention is nearly uniform! Model not learning localization!")
else:
    print("  ✓ Attention has variance, model is learning!")

# Show attention at GT bbox locations
if bbox_mask is not None:
    gt_attention = attention[bbox_mask > 0]
    other_attention = attention[bbox_mask == 0]
    
    print(f"\nAttention at GT bbox patches:")
    print(f"  Mean: {gt_attention.mean().item():.6f}")
    print(f"  Max: {gt_attention.max().item():.6f}")
    
    print(f"Attention at other patches:")
    print(f"  Mean: {other_attention.mean().item():.6f}")
    print(f"  Max: {other_attention.max().item():.6f}")
    
    diff = gt_attention.mean() - other_attention.mean()
    print(f"Difference (GT - Other): {diff.item():.6f}")
    
    if abs(diff) < 0.05:
        print("  ⚠ WARNING: No discrimination between GT and non-GT regions!")
    else:
        print("  ✓ Model shows some discrimination!")

print("\n✓ Done")
