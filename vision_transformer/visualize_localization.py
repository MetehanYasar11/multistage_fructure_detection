"""
Visualize Spatial Localization from Enhanced Model

This script visualizes patch-level predictions from the localization model.
Unlike the baseline, this model should show meaningful spatial variance.

Features:
- Side-by-side comparison: Original image + Heatmap
- Patch statistics (variance, entropy)
- Comparison with ground truth
- Export high-quality figures for thesis

Author: Master's Thesis Project
Date: October 28, 2025
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.patch_transformer_localization import PatchTransformerWithLocalization
from data.dataset import DentalXrayDataset


def load_model(checkpoint_path: str, device: torch.device) -> PatchTransformerWithLocalization:
    """Load trained localization model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = PatchTransformerWithLocalization(
        image_size=tuple(config['data']['target_size']),
        patch_size=config['model']['patch_size'],
        cnn_backbone=config['model']['cnn_backbone'],
        feature_dim=config['model']['feature_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        use_global_head=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   F1: {checkpoint.get('f1', 'N/A'):.4f}")
    
    return model, config


def visualize_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    patch_size: int,
    alpha: float = 0.5,
    cmap: str = 'jet'
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: (H, W, 3) RGB image [0-255]
        heatmap: (nh, nw) patch probabilities [0-1]
        patch_size: Size of each patch
        alpha: Overlay transparency
        cmap: Colormap name
    
    Returns:
        overlay: (H, W, 3) RGB image with heatmap overlay
    """
    H, W = image.shape[:2]
    nh, nw = heatmap.shape
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(
        heatmap,
        (nw * patch_size, nh * patch_size),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Crop to image size
    heatmap_resized = heatmap_resized[:H, :W]
    
    # Convert heatmap to color
    cmap_fn = plt.cm.get_cmap(cmap)
    heatmap_colored = (cmap_fn(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def draw_patch_grid(
    image: np.ndarray,
    heatmap: np.ndarray,
    patch_size: int,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Draw patch grid with color-coded borders.
    
    Green: Low fracture probability (< threshold)
    Red: High fracture probability (>= threshold)
    """
    img_draw = image.copy()
    nh, nw = heatmap.shape
    
    for i in range(nh):
        for j in range(nw):
            prob = heatmap[i, j]
            
            # Determine color based on probability
            if prob >= threshold:
                color = (0, 0, 255)  # Red (BGR)
                thickness = 2
            else:
                color = (0, 255, 0)  # Green (BGR)
                thickness = 1
            
            # Draw rectangle
            y1 = i * patch_size
            x1 = j * patch_size
            y2 = y1 + patch_size
            x2 = x1 + patch_size
            
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
    
    return img_draw


def visualize_single_image(
    model: PatchTransformerWithLocalization,
    image_tensor: torch.Tensor,
    image_np: np.ndarray,
    label: int,
    global_pred: float,
    patch_size: int,
    output_path: Path,
    threshold: float = 0.5
):
    """
    Create comprehensive visualization for one image.
    
    Layout:
    - Row 1: Original | Heatmap Overlay | Patch Grid
    - Row 2: Heatmap 2D | Statistics | Histogram
    """
    # Get predictions
    with torch.no_grad():
        heatmap, patch_probs = model.get_patch_heatmap(image_tensor.unsqueeze(0))
    
    heatmap = heatmap.squeeze(0).cpu().numpy()
    patch_probs = patch_probs.squeeze(0).cpu().numpy()
    
    # Statistics
    patch_mean = patch_probs.mean()
    patch_std = patch_probs.std()
    patch_min = patch_probs.min()
    patch_max = patch_probs.max()
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Row 1, Col 1: Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_np)
    ax1.set_title(f"Original Image\nGT: {'Fractured' if label == 1 else 'Healthy'}, Pred: {global_pred:.3f}", fontsize=12)
    ax1.axis('off')
    
    # Row 1, Col 2: Heatmap Overlay
    ax2 = fig.add_subplot(gs[0, 1])
    overlay = visualize_heatmap_overlay(image_np, heatmap, patch_size, alpha=0.6, cmap='RdYlGn_r')
    ax2.imshow(overlay)
    ax2.set_title(f"Heatmap Overlay\n(Red = Fracture, Green = Healthy)", fontsize=12)
    ax2.axis('off')
    
    # Row 1, Col 3: Patch Grid
    ax3 = fig.add_subplot(gs[0, 2])
    grid = draw_patch_grid(image_np, heatmap, patch_size, threshold=threshold)
    ax3.imshow(grid)
    ax3.set_title(f"Patch Grid (Threshold={threshold})", fontsize=12)
    ax3.axis('off')
    
    # Row 2, Col 1: 2D Heatmap
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        heatmap,
        ax=ax4,
        cmap='RdYlGn_r',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Fracture Probability'},
        square=True,
        annot=False
    )
    ax4.set_title(f"Patch Probability Map\n({heatmap.shape[0]}×{heatmap.shape[1]} patches)", fontsize=12)
    ax4.set_xlabel('Horizontal Patches')
    ax4.set_ylabel('Vertical Patches')
    
    # Row 2, Col 2: Statistics
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    stats_text = f"""
    PATCH STATISTICS
    ══════════════════
    
    Mean:      {patch_mean:.4f}
    Std Dev:   {patch_std:.4f}
    Min:       {patch_min:.4f}
    Max:       {patch_max:.4f}
    Range:     {patch_max - patch_min:.4f}
    
    ══════════════════
    
    Global Prediction: {global_pred:.4f}
    Ground Truth:      {label}
    
    Patches > {threshold}: {(patch_probs >= threshold).sum()}/{len(patch_probs)}
    Fraction High:     {(patch_probs >= threshold).mean():.2%}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', va='center')
    
    # Row 2, Col 3: Histogram
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(patch_probs, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax6.axvline(patch_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {patch_mean:.3f}')
    ax6.axvline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    ax6.set_xlabel('Fracture Probability', fontsize=11)
    ax6.set_ylabel('Number of Patches', fontsize=11)
    ax6.set_title('Patch Probability Distribution', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    print(f"   Variance: {patch_std**2:.6f}, Range: [{patch_min:.3f}, {patch_max:.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Visualize spatial localization")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')
    parser.add_argument('--output_dir', type=str, default='outputs/localization_vis', help='Output directory')
    parser.add_argument('--filter_type', type=str, default='all', 
                        choices=['all', 'tp', 'tn', 'fp', 'fn'],
                        help='Filter by prediction type')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"SPATIAL LOCALIZATION VISUALIZATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    patch_size = config['model']['patch_size']
    
    # Load dataset
    dataset = DentalXrayDataset(
        split=args.split,
        config=config,
        augment=False
    )
    
    print(f"\nDataset: {len(dataset)} images")
    
    # Get predictions for filtering
    print("\nGetting predictions...")
    all_preds = []
    all_labels = []
    
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        image = sample['image'].to(device)
        label = sample['label']
        
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            global_pred = torch.sigmoid(output['global_logits']).item()
        
        all_preds.append(global_pred)
        all_labels.append(label)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = (all_preds >= args.threshold).astype(int)
    
    # Filter indices
    if args.filter_type == 'tp':
        indices = np.where((all_labels == 1) & (binary_preds == 1))[0]
    elif args.filter_type == 'tn':
        indices = np.where((all_labels == 0) & (binary_preds == 0))[0]
    elif args.filter_type == 'fp':
        indices = np.where((all_labels == 0) & (binary_preds == 1))[0]
    elif args.filter_type == 'fn':
        indices = np.where((all_labels == 1) & (binary_preds == 0))[0]
    else:
        indices = np.arange(len(dataset))
    
    print(f"\nFiltered to {len(indices)} images ({args.filter_type})")
    
    # Visualize
    num_to_vis = min(args.num_samples, len(indices))
    selected_indices = np.random.choice(indices, num_to_vis, replace=False)
    
    print(f"\nVisualizing {num_to_vis} images...")
    
    for idx in tqdm(selected_indices):
        sample = dataset[idx]
        image_tensor = sample['image'].to(device)
        label = sample['label']
        image_path = sample['image_path']
        
        # Convert to numpy for visualization
        image_np = image_tensor.cpu().permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-7)
        image_np = (image_np * 255).astype(np.uint8)
        
        # Prediction
        global_pred = all_preds[idx]
        
        # Output path
        img_name = Path(image_path).stem
        output_path = output_dir / f"{img_name}_GT{label}_Pred{global_pred:.2f}.png"
        
        # Visualize
        visualize_single_image(
            model=model,
            image_tensor=image_tensor,
            image_np=image_np,
            label=label,
            global_pred=global_pred,
            patch_size=patch_size,
            output_path=output_path,
            threshold=args.threshold
        )
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
