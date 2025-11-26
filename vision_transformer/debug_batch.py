"""Debug batch content to see why num_with_bbox is low"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from train_weakly_supervised import DentalFractureDatasetWithBbox, custom_collate_fn


def main():
    root_dir = Path("c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset")
    split_file = Path("outputs/splits/train_val_test_split.json")
    
    # Create train dataset
    train_dataset = DentalFractureDatasetWithBbox(
        root_dir=root_dir,
        split_file=split_file,
        split='train',
        image_size=(1400, 2800),
        use_clahe=False
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Test first 5 batches
    for batch_idx, (images, labels, bbox_masks) in enumerate(train_loader):
        if batch_idx >= 5:
            break
        
        print(f"\n=== Batch {batch_idx + 1} ===")
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Labels type: {type(labels)}, length: {len(labels)}")
        
        # Count fractured
        num_fractured = sum(1 for label in labels if label == 1)
        print(f"Fractured samples: {num_fractured}")
        
        # Check bbox_masks
        print(f"Bbox_masks type: {type(bbox_masks)}, length: {len(bbox_masks)}")
        num_not_none = sum(1 for mask in bbox_masks if mask is not None)
        print(f"Bbox_masks not None: {num_not_none}")
        
        # Check each sample
        for i, (label, bbox_mask) in enumerate(zip(labels, bbox_masks)):
            is_none = bbox_mask is None
            bbox_sum = 0 if is_none else bbox_mask.sum().item()
            print(f"  Sample {i}: label={label}, bbox_mask={'None' if is_none else f'Tensor(sum={bbox_sum:.1f})'}")
        
        # Count valid for localization loss
        num_valid = 0
        for label, bbox_mask in zip(labels, bbox_masks):
            if label == 1 and bbox_mask is not None:
                if bbox_mask.sum() > 0:
                    num_valid += 1
        
        print(f"Valid for localization loss: {num_valid}")


if __name__ == "__main__":
    main()
