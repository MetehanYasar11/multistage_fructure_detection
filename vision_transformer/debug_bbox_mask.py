"""Quick debug to check bbox mask creation"""

import sys
sys.path.insert(0, '.')

from train_weakly_supervised import DentalFractureDatasetWithBbox
import torch

# Load a sample
dataset = DentalFractureDatasetWithBbox(
    root_dir="c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset",
    split='train',
    split_file="outputs/splits/train_val_test_split.json",
    image_size=(1400, 2800),
    use_clahe=True
)

print(f"Dataset size: {len(dataset)}")

# Check first few fractured samples
count = 0
for i in range(len(dataset)):
    img, label, bbox_mask = dataset[i]
    
    if label == 1 and bbox_mask is not None:
        print(f"\nSample {i}:")
        print(f"  Image path: {dataset.image_paths[i]}")
        print(f"  Bbox mask shape: {bbox_mask.shape}")
        print(f"  Bbox mask sum: {bbox_mask.sum().item()}")
        print(f"  Bbox mask nonzero: {(bbox_mask > 0).sum().item()} / {bbox_mask.numel()} pixels")
        print(f"  Bbox mask min/max: {bbox_mask.min().item():.2f} / {bbox_mask.max().item():.2f}")
        
        # Show which patches are active
        active_patches = torch.nonzero(bbox_mask)
        print(f"  Active patches: {len(active_patches)}")
        if len(active_patches) > 0:
            print(f"  Patch coordinates (y, x): {active_patches[:5].tolist()}")
        
        count += 1
        if count >= 5:
            break

print("\n✓ Done")
