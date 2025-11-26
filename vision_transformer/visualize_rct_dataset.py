"""
RCT Classification Dataset Visualizer

Visualize samples from the created RCT classification dataset
to verify correct cropping and labeling.

Author: Master's Thesis Project
Date: November 23, 2025
"""

import os
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def visualize_samples(
    dataset_path: str = "RCT_classification_dataset",
    num_samples: int = 6,
    save_path: str = "RCT_classification_dataset/sample_visualization.png"
):
    """
    Visualize random samples from RCT classification dataset
    
    Args:
        dataset_path: Path to RCT dataset
        num_samples: Number of samples to visualize (must be even)
        save_path: Where to save visualization
    """
    dataset_path = Path(dataset_path)
    
    # Get sample images
    fractured_dir = dataset_path / "fractured"
    healthy_dir = dataset_path / "healthy"
    
    fractured_images = list(fractured_dir.glob("*.jpg"))
    healthy_images = list(healthy_dir.glob("*.jpg"))
    
    print(f"Found {len(fractured_images)} fractured images")
    print(f"Found {len(healthy_images)} healthy images")
    
    # Sample random images
    num_per_class = num_samples // 2
    fractured_samples = random.sample(fractured_images, min(num_per_class, len(fractured_images)))
    healthy_samples = random.sample(healthy_images, min(num_per_class, len(healthy_images)))
    
    # Create figure
    fig, axes = plt.subplots(2, num_per_class, figsize=(15, 6))
    fig.suptitle('RCT Classification Dataset Samples', fontsize=16, fontweight='bold')
    
    # Plot fractured samples
    for idx, img_path in enumerate(fractured_samples):
        img = Image.open(img_path)
        
        # Load metadata
        metadata_path = dataset_path / "metadata" / f"{img_path.name}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        axes[0, idx].imshow(img, cmap='gray')
        axes[0, idx].set_title(
            f"FRACTURED\n{img_path.stem}\n"
            f"Conf: {metadata['confidence']:.2f}\n"
            f"Fractures: {len(metadata['fracture_lines_in_bbox'])}",
            fontsize=9,
            color='red'
        )
        axes[0, idx].axis('off')
    
    # Plot healthy samples
    for idx, img_path in enumerate(healthy_samples):
        img = Image.open(img_path)
        
        # Load metadata
        metadata_path = dataset_path / "metadata" / f"{img_path.name}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        axes[1, idx].imshow(img, cmap='gray')
        axes[1, idx].set_title(
            f"HEALTHY\n{img_path.stem}\n"
            f"Conf: {metadata['confidence']:.2f}",
            fontsize=9,
            color='green'
        )
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def print_dataset_stats(dataset_path: str = "RCT_classification_dataset"):
    """Print detailed dataset statistics"""
    dataset_path = Path(dataset_path)
    
    # Load statistics
    stats_path = dataset_path / "dataset_statistics.json"
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    print("\n" + "="*80)
    print("RCT CLASSIFICATION DATASET STATISTICS")
    print("="*80)
    print(f"Total original images processed: {stats['total_images']}")
    print(f"Total RCT teeth detected: {stats['total_rct_detected']}")
    print(f"Images skipped (no RCT): {stats['skipped_no_detection']}")
    print()
    print(f"FRACTURED teeth (with instrument): {stats['fractured_teeth']}")
    print(f"  Percentage: {stats['fractured_teeth'] / stats['total_rct_detected'] * 100:.1f}%")
    print()
    print(f"HEALTHY teeth (no instrument): {stats['healthy_teeth']}")
    print(f"  Percentage: {stats['healthy_teeth'] / stats['total_rct_detected'] * 100:.1f}%")
    print()
    print(f"Class ratio (Healthy:Fractured): {stats['healthy_teeth'] / stats['fractured_teeth']:.2f}:1")
    print("="*80)
    
    # Count actual files
    fractured_dir = dataset_path / "fractured"
    healthy_dir = dataset_path / "healthy"
    
    fractured_count = len(list(fractured_dir.glob("*.jpg")))
    healthy_count = len(list(healthy_dir.glob("*.jpg")))
    
    print(f"\nACTUAL FILES ON DISK:")
    print(f"  Fractured: {fractured_count} files")
    print(f"  Healthy: {healthy_count} files")
    print(f"  Total: {fractured_count + healthy_count} files")
    
    if fractured_count != stats['fractured_teeth']:
        print(f"\n⚠️  WARNING: Fractured count mismatch!")
        print(f"     Expected: {stats['fractured_teeth']}, Found: {fractured_count}")
    
    if healthy_count != stats['healthy_teeth']:
        print(f"\n⚠️  WARNING: Healthy count mismatch!")
        print(f"     Expected: {stats['healthy_teeth']}, Found: {healthy_count}")


def main():
    """Main execution"""
    print_dataset_stats()
    print("\nGenerating visualization...")
    visualize_samples(num_samples=6)


if __name__ == "__main__":
    main()
