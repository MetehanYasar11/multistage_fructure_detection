"""Debug current attention output during training"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from train_weakly_supervised import DentalFractureDatasetWithBbox, custom_collate_fn
from models.weakly_supervised_patch_transformer import WeaklySupervisedPatchTransformer


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load current checkpoint
    checkpoint_path = Path("outputs/weakly_supervised_patch_transformer/best_f1.pth")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    model = WeaklySupervisedPatchTransformer(
        image_size=(1400, 2800),
        patch_size=100,
        num_classes=2,
        feature_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Val F1: {checkpoint['val_metrics']['f1']:.4f}")
    print(f"Val IoU: {checkpoint['val_metrics']['mean_iou']:.4f}\n")
    
    # Load validation dataset
    root_dir = Path("c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset")
    split_file = Path("outputs/splits/train_val_test_split.json")
    
    val_dataset = DentalFractureDatasetWithBbox(
        root_dir=root_dir,
        split_file=split_file,
        split='val',
        image_size=(1400, 2800),
        use_clahe=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Test first batch with bbox
    print("Testing attention output on first batch with bbox...\n")
    
    with torch.no_grad():
        for images, labels, bbox_masks in val_loader:
            images = images.to(device)
            
            # Get predictions
            logits, attention_map = model(images, return_attention=True)
            attention_probs = torch.sigmoid(attention_map)
            
            # Check each sample
            for i, (label, bbox_mask) in enumerate(zip(labels, bbox_masks)):
                if label == 1 and bbox_mask is not None:
                    # Get attention for this sample
                    att = attention_probs[i, 0].cpu().numpy()  # [14, 28]
                    bbox = bbox_mask.cpu().numpy()  # [14, 28]
                    
                    print(f"Sample {i} (Fractured with bbox):")
                    print(f"  Bbox patches: {bbox.sum():.0f} active")
                    print(f"  Attention stats:")
                    print(f"    Min: {att.min():.6f}, Max: {att.max():.6f}")
                    print(f"    Mean: {att.mean():.6f}, Std: {att.std():.6f}")
                    
                    # Check attention at GT patches
                    gt_patches = bbox > 0.5
                    if gt_patches.sum() > 0:
                        gt_att = att[gt_patches].mean()
                        other_att = att[~gt_patches].mean()
                        print(f"    GT patches mean: {gt_att:.6f}")
                        print(f"    Other patches mean: {other_att:.6f}")
                        print(f"    Difference: {gt_att - other_att:.6f}")
                        
                        # Check if attention is discriminative
                        if abs(gt_att - other_att) < 0.01:
                            print(f"    ⚠️ WARNING: Almost no discrimination!")
                        elif gt_att < other_att:
                            print(f"    ⚠️ WARNING: WRONG direction (GT gets LESS attention)!")
                        else:
                            print(f"    ✓ Correct direction (GT gets MORE attention)")
                    
                    print()
            
            break  # Only first batch


if __name__ == "__main__":
    main()
