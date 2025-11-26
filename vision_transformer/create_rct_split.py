"""
Create Train/Val/Test Split for RCT Classification Dataset

Author: Master's Thesis Project
Date: November 23, 2025
"""

import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split


def create_rct_split(
    dataset_path: str = "RCT_classification_dataset",
    output_file: str = "RCT_classification_dataset/train_val_test_split.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Create stratified train/val/test split for RCT classification dataset
    
    Args:
        dataset_path: Path to RCT dataset
        output_file: Where to save split JSON
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    dataset_path = Path(dataset_path)
    
    # Load all image paths and labels
    fractured_dir = dataset_path / "fractured"
    healthy_dir = dataset_path / "healthy"
    
    fractured_images = sorted([f.name for f in fractured_dir.glob("*.jpg")])
    healthy_images = sorted([f.name for f in healthy_dir.glob("*.jpg")])
    
    print(f"Total fractured images: {len(fractured_images)}")
    print(f"Total healthy images: {len(healthy_images)}")
    
    # Create labels (1=fractured, 0=healthy)
    all_images = fractured_images + healthy_images
    all_labels = [1] * len(fractured_images) + [0] * len(healthy_images)
    
    print(f"\nTotal images: {len(all_images)}")
    print(f"Class distribution: Fractured={sum(all_labels)}, Healthy={len(all_labels) - sum(all_labels)}")
    
    # First split: train+val vs test
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        all_images,
        all_labels,
        test_size=test_ratio,
        stratify=all_labels,
        random_state=random_seed
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images,
        train_val_labels,
        test_size=val_ratio_adjusted,
        stratify=train_val_labels,
        random_state=random_seed
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("SPLIT STATISTICS")
    print("="*80)
    
    for split_name, images, labels in [
        ("TRAIN", train_images, train_labels),
        ("VAL", val_images, val_labels),
        ("TEST", test_images, test_labels)
    ]:
        n_total = len(images)
        n_fractured = sum(labels)
        n_healthy = n_total - n_fractured
        
        print(f"\n{split_name}:")
        print(f"  Total: {n_total} images ({n_total / len(all_images) * 100:.1f}%)")
        print(f"  Fractured: {n_fractured} ({n_fractured / n_total * 100:.1f}%)")
        print(f"  Healthy: {n_healthy} ({n_healthy / n_total * 100:.1f}%)")
    
    # Save split to JSON
    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
        "metadata": {
            "total_images": len(all_images),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
            "fractured_count": sum(all_labels),
            "healthy_count": len(all_labels) - sum(all_labels)
        }
    }
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Split saved to: {output_file}")
    print("="*80)


def main():
    """Main execution"""
    create_rct_split(
        dataset_path="RCT_classification_dataset",
        output_file="RCT_classification_dataset/train_val_test_split.json",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )


if __name__ == "__main__":
    main()
