"""
Dataset Splitting for Dental X-Ray Fracture Detection

Creates stratified train/val/test splits maintaining class balance.

Key Features:
- Stratified splitting (maintains 3.57:1 Fractured:Healthy ratio)
- Reproducible (fixed random seed)
- Save/load split indices from JSON
- Support for K-fold cross-validation

Split Ratios:
- Train: 70%
- Validation: 15%
- Test: 15%

Author: Master's Thesis Project
Date: October 28, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold


def create_train_val_test_split(
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, List[int]]:
    """
    Create stratified train/val/test split.
    
    Args:
        labels: List of labels (0=Healthy, 1=Fractured)
        train_ratio: Training set ratio (default: 0.70)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: val vs test
    temp_labels = [labels[i] for i in temp_indices]
    val_size = val_ratio / (val_ratio + test_ratio)
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=random_state
    )
    
    # Create split dictionary
    splits = {
        'train': train_indices.tolist(),
        'val': val_indices.tolist(),
        'test': test_indices.tolist()
    }
    
    # Print split statistics
    print("=== Dataset Split Statistics ===")
    for split_name, split_indices in splits.items():
        split_labels = [labels[i] for i in split_indices]
        n_fractured = sum(split_labels)
        n_healthy = len(split_labels) - n_fractured
        ratio = n_fractured / n_healthy if n_healthy > 0 else 0
        
        print(f"\n{split_name.upper()}:")
        print(f"  Total: {len(split_indices)} samples ({len(split_indices)/n_samples*100:.1f}%)")
        print(f"  Fractured: {n_fractured} ({n_fractured/len(split_indices)*100:.1f}%)")
        print(f"  Healthy: {n_healthy} ({n_healthy/len(split_indices)*100:.1f}%)")
        print(f"  Ratio (Fractured:Healthy): {ratio:.2f}:1")
    
    return splits


def create_kfold_splits(
    labels: List[int],
    n_folds: int = 5,
    random_state: int = 42
) -> List[Dict[str, List[int]]]:
    """
    Create K-fold cross-validation splits.
    
    Args:
        labels: List of labels (0=Healthy, 1=Fractured)
        n_folds: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of dictionaries, each with 'train' and 'val' keys
    """
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_splits = []
    
    print(f"=== {n_folds}-Fold Cross-Validation ===")
    
    for fold, (train_indices, val_indices) in enumerate(skf.split(indices, labels)):
        splits = {
            'train': train_indices.tolist(),
            'val': val_indices.tolist()
        }
        fold_splits.append(splits)
        
        # Print fold statistics
        train_labels = [labels[i] for i in train_indices]
        val_labels = [labels[i] for i in val_indices]
        
        train_fractured = sum(train_labels)
        train_healthy = len(train_labels) - train_fractured
        val_fractured = sum(val_labels)
        val_healthy = len(val_labels) - val_fractured
        
        print(f"\nFold {fold + 1}:")
        print(f"  Train: {len(train_indices)} samples (Fractured: {train_fractured}, Healthy: {train_healthy})")
        print(f"  Val: {len(val_indices)} samples (Fractured: {val_fractured}, Healthy: {val_healthy})")
    
    return fold_splits


def save_splits(splits: Dict[str, List[int]], output_path: str):
    """
    Save split indices to JSON file.
    
    Args:
        splits: Dictionary of split indices
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSaved splits to: {output_path}")


def load_splits(input_path: str) -> Dict[str, List[int]]:
    """
    Load split indices from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Dictionary of split indices
    """
    with open(input_path, 'r') as f:
        splits = json.load(f)
    
    print(f"Loaded splits from: {input_path}")
    
    return splits


def save_kfold_splits(fold_splits: List[Dict[str, List[int]]], output_dir: str):
    """
    Save K-fold splits to separate JSON files.
    
    Args:
        fold_splits: List of fold split dictionaries
        output_dir: Directory to save JSON files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fold, splits in enumerate(fold_splits):
        output_path = output_dir / f"fold_{fold + 1}.json"
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"Saved fold {fold + 1} to: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    from dataset import DentalXrayDataset
    
    # Dataset path
    root_dir = "c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset"
    
    print("=== Creating Dataset Splits ===\n")
    
    # Load full dataset to get labels
    dataset = DentalXrayDataset(
        root_dir=root_dir,
        split='all',
        transform=None
    )
    
    labels = dataset.labels
    print(f"Total samples: {len(labels)}")
    print(f"Fractured: {sum(labels)}")
    print(f"Healthy: {len(labels) - sum(labels)}")
    print(f"Class ratio (Fractured:Healthy): {sum(labels)/(len(labels) - sum(labels)):.2f}:1")
    
    # Create train/val/test split
    print("\n" + "="*50)
    print("Creating Train/Val/Test Split")
    print("="*50)
    
    splits = create_train_val_test_split(
        labels=labels,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    # Save splits
    output_path = "outputs/splits/train_val_test_split.json"
    save_splits(splits, output_path)
    
    # Test loading splits
    print("\n" + "="*50)
    print("Testing Split Loading")
    print("="*50)
    
    loaded_splits = load_splits(output_path)
    print(f"Loaded {len(loaded_splits)} splits")
    for split_name in loaded_splits:
        print(f"  {split_name}: {len(loaded_splits[split_name])} samples")
    
    # Create K-fold splits (optional)
    print("\n" + "="*50)
    print("Creating 5-Fold Cross-Validation Splits")
    print("="*50)
    
    fold_splits = create_kfold_splits(
        labels=labels,
        n_folds=5,
        random_state=42
    )
    
    # Save K-fold splits
    output_dir = "outputs/splits/kfold"
    save_kfold_splits(fold_splits, output_dir)
    
    print("\n=== Split creation COMPLETED ===")
