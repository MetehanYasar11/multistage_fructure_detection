"""
Test and visualize the data pipeline.

This script:
1. Loads train/val/test datasets
2. Visualizes sample images from each split
3. Tests augmentations
4. Verifies class balance
5. Tests DataLoader with batching

Author: Master's Thesis Project
Date: October 28, 2025
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    DentalXrayDataset,
    get_train_transforms,
    get_val_transforms,
    load_splits
)


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor for visualization.
    
    Args:
        img_tensor: Normalized image tensor (C, H, W)
        
    Returns:
        Numpy array (H, W, C) in [0, 255] range
    """
    # ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert to numpy and transpose
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img


def visualize_samples(dataset, split_name, num_samples=5, save_path=None):
    """
    Visualize random samples from dataset.
    
    Args:
        dataset: DentalXrayDataset instance
        split_name: Name of the split (for title)
        num_samples: Number of samples to visualize
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    # Get random indices (stratified by class)
    fractured_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    healthy_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    
    np.random.seed(42)
    fractured_samples = np.random.choice(fractured_indices, min(num_samples, len(fractured_indices)), replace=False)
    healthy_samples = np.random.choice(healthy_indices, min(num_samples, len(healthy_indices)), replace=False)
    
    # Plot Fractured samples
    for i, idx in enumerate(fractured_samples):
        image, label = dataset[idx]
        img_display = denormalize_image(image)
        
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f'Fractured #{idx}', fontsize=10)
        axes[0, i].axis('off')
    
    # Plot Healthy samples
    for i, idx in enumerate(healthy_samples):
        image, label = dataset[idx]
        img_display = denormalize_image(image)
        
        axes[1, i].imshow(img_display)
        axes[1, i].set_title(f'Healthy #{idx}', fontsize=10)
        axes[1, i].axis('off')
    
    fig.suptitle(f'{split_name.upper()} Set Sample Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_augmentations(dataset, num_augmentations=5, save_path=None):
    """
    Visualize augmentation effects on a single image.
    
    Args:
        dataset: DentalXrayDataset instance
        num_augmentations: Number of augmented versions to show
        save_path: Path to save figure (optional)
    """
    # Get one fractured image
    fractured_idx = [i for i, label in enumerate(dataset.labels) if label == 1][0]
    
    fig, axes = plt.subplots(1, num_augmentations + 1, figsize=((num_augmentations + 1) * 3, 3))
    
    # Original image (with val transforms - no augmentation)
    from data import get_val_transforms
    val_dataset = DentalXrayDataset(
        root_dir=dataset.root_dir,
        split='all',
        transform=get_val_transforms(image_size=640)
    )
    
    original_img, _ = val_dataset[fractured_idx]
    axes[0].imshow(denormalize_image(original_img))
    axes[0].set_title('Original', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Augmented versions
    for i in range(num_augmentations):
        image, label = dataset[fractured_idx]
        img_display = denormalize_image(image)
        
        axes[i + 1].imshow(img_display)
        axes[i + 1].set_title(f'Aug #{i+1}', fontsize=10)
        axes[i + 1].axis('off')
    
    fig.suptitle('Augmentation Examples (Same Image)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved augmentation visualization to: {save_path}")
    
    plt.show()


def test_dataloader(dataset, batch_size=8, weighted_sampling=False):
    """
    Test DataLoader with batching and optional weighted sampling.
    
    Args:
        dataset: DentalXrayDataset instance
        batch_size: Batch size
        weighted_sampling: Use WeightedRandomSampler for class balance
    """
    print(f"\n{'='*50}")
    print(f"Testing DataLoader (batch_size={batch_size}, weighted_sampling={weighted_sampling})")
    print(f"{'='*50}")
    
    # Create sampler if needed
    sampler = None
    if weighted_sampling:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,  # 0 for testing (Windows compatibility)
        pin_memory=True
    )
    
    # Test one batch
    images, labels = next(iter(loader))
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels.tolist()}")
    print(f"Labels distribution: Fractured={labels.sum().item()}, Healthy={len(labels) - labels.sum().item()}")
    print(f"Image dtype: {images.dtype}, range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test multiple batches for class distribution
    if weighted_sampling:
        print("\nTesting class distribution over 10 batches:")
        total_fractured = 0
        total_samples = 0
        
        for i, (images, labels) in enumerate(loader):
            if i >= 10:
                break
            total_fractured += labels.sum().item()
            total_samples += len(labels)
        
        print(f"  Total samples: {total_samples}")
        print(f"  Fractured: {total_fractured} ({total_fractured/total_samples*100:.1f}%)")
        print(f"  Healthy: {total_samples - total_fractured} ({(total_samples - total_fractured)/total_samples*100:.1f}%)")


def main():
    """Main test function."""
    print("="*70)
    print("DENTAL X-RAY DATA PIPELINE TEST")
    print("="*70)
    
    # Configuration
    root_dir = "c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset"
    split_file = "outputs/splits/train_val_test_split.json"
    image_size = 640
    
    # Load splits
    print("\nLoading splits...")
    splits = load_splits(split_file)
    
    # Create datasets
    print("\nCreating datasets...")
    
    train_dataset = DentalXrayDataset(
        root_dir=root_dir,
        split='train',
        transform=get_train_transforms(image_size=image_size),
        split_file=split_file
    )
    
    val_dataset = DentalXrayDataset(
        root_dir=root_dir,
        split='val',
        transform=get_val_transforms(image_size=image_size),
        split_file=split_file
    )
    
    test_dataset = DentalXrayDataset(
        root_dir=root_dir,
        split='test',
        transform=get_val_transforms(image_size=image_size),
        split_file=split_file
    )
    
    # Print dataset info
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")
    
    # Get class weights
    print("\n" + "="*70)
    print("CLASS WEIGHTS")
    print("="*70)
    class_weights = train_dataset.get_class_weights()
    print(f"Weights: {class_weights.tolist()}")
    
    # Test data loading
    print("\n" + "="*70)
    print("TESTING DATA LOADING")
    print("="*70)
    
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Label: {label} ({'Fractured' if label == 1 else 'Healthy'})")
    
    # Visualize samples from each split
    print("\n" + "="*70)
    print("VISUALIZING SAMPLES")
    print("="*70)
    
    visualize_samples(
        train_dataset,
        'train',
        num_samples=5,
        save_path='outputs/test_pipeline/train_samples.png'
    )
    
    visualize_samples(
        val_dataset,
        'validation',
        num_samples=5,
        save_path='outputs/test_pipeline/val_samples.png'
    )
    
    visualize_samples(
        test_dataset,
        'test',
        num_samples=5,
        save_path='outputs/test_pipeline/test_samples.png'
    )
    
    # Visualize augmentations
    print("\n" + "="*70)
    print("VISUALIZING AUGMENTATIONS")
    print("="*70)
    
    visualize_augmentations(
        train_dataset,
        num_augmentations=5,
        save_path='outputs/test_pipeline/augmentations.png'
    )
    
    # Test DataLoader
    test_dataloader(train_dataset, batch_size=8, weighted_sampling=False)
    test_dataloader(train_dataset, batch_size=8, weighted_sampling=True)
    
    print("\n" + "="*70)
    print("DATA PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
