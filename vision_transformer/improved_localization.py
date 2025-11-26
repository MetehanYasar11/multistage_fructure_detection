"""
Improved Localization: Multi-Scale Attention Extraction

Problems with previous GradCAM:
1. Wrong hook layer (transformer might not capture spatial info well)
2. Max aggregation loses spatial information
3. Need to hook earlier layers for spatial localization

New approach:
1. Hook patch encoder features (before aggregation)
2. Extract attention from multiple layers
3. Use class activation mapping (CAM) instead of GradCAM

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml

from models.patch_transformer import PatchTransformerClassifier


class ImprovedLocalizer:
    """
    Improved localization using patch-level predictions
    
    Key insight: Model predicts for each patch, then aggregates.
    We can extract per-patch predictions to see which patches are fractured!
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        print("\nLoading Patch Transformer...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Storage for intermediate features
        self.patch_features = None
        self.patch_logits = None
        
        print("\n[*] Improved Localizer ready!")
    
    def _load_model(self, checkpoint_path: str):
        """Load trained model"""
        model_config = self.config['model']
        image_config = self.config['image']
        
        model = PatchTransformerClassifier(
            image_size=tuple(image_config['default_size']),
            patch_size=model_config['patch_size'],
            aggregation=model_config['aggregation'],
            dropout=model_config['dropout']
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch {checkpoint['epoch']}")
            print(f"  Best Val F1: {checkpoint.get('best_val_f1', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def _register_hooks(self):
        """Register hooks to capture patch-level predictions"""
        
        def hook_fn(module, input, output):
            """Capture output from patch classifier"""
            self.patch_logits = output.detach()
        
        # Hook the patch classifier (outputs per-patch logits)
        # This is BEFORE aggregation (max/mean/attention)
        self.model.patch_classifier.register_forward_hook(hook_fn)
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Load and preprocess image"""
        # Load image (handle Unicode paths)
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply CLAHE
        if self.config['image'].get('apply_clahe', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        # Resize
        target_size = tuple(self.config['image']['default_size'])
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # To tensor
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor, original
    
    def extract_patch_attention(
        self,
        image_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Extract attention map from patch-level predictions
        
        Key idea: Each patch gets a prediction score.
        High scores = fracture patches!
        
        Returns:
            attention_map: (nh, nw) attention scores
        """
        # Use built-in method to get patch predictions
        with torch.no_grad():
            patch_logits, num_patches_h, num_patches_w = self.model.get_patch_predictions(image_tensor)
        
        # patch_logits shape: (B, num_patches, 1)
        patch_scores = patch_logits.squeeze(-1)  # (B, num_patches)
        
        # Apply sigmoid to get probabilities
        patch_probs = torch.sigmoid(patch_scores)
        
        # Reshape to 2D grid
        B = patch_probs.size(0)
        attention_map = patch_probs.view(B, num_patches_h, num_patches_w)
        
        return attention_map[0].cpu().numpy()
    
    def get_fracture_bboxes(
        self,
        attention_map: np.ndarray,
        threshold: float = 0.5,
        min_cluster_size: int = 2
    ) -> List[Tuple[int, int, int, int]]:
        """
        Convert attention map to bounding boxes
        
        Args:
            attention_map: (nh, nw) attention scores (0-1)
            threshold: Probability threshold (0.5 = confident fracture)
            min_cluster_size: Minimum connected patches
        """
        # Binary mask (patches with prob > threshold)
        binary_mask = (attention_map > threshold).astype(np.uint8)
        
        # Find connected components
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary_mask)
        
        patch_size = self.config['model']['patch_size']
        bboxes = []
        scores = []
        
        for i in range(1, num_features + 1):
            component = (labeled == i)
            if component.sum() < min_cluster_size:
                continue
            
            # Get bounding box
            rows, cols = np.where(component)
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            
            # Convert to pixel coordinates
            x1 = c_min * patch_size
            y1 = r_min * patch_size
            x2 = (c_max + 1) * patch_size
            y2 = (r_max + 1) * patch_size
            
            # Calculate average score for this region
            region_score = attention_map[component].mean()
            
            bboxes.append((x1, y1, x2, y2))
            scores.append(region_score)
        
        return bboxes, scores
    
    def predict(
        self,
        image_path: str,
        threshold: float = 0.5,
        min_cluster_size: int = 2
    ) -> Dict:
        """
        Complete prediction with localization
        """
        # Load image
        tensor, original = self.preprocess_image(image_path)
        
        # Classification (final prediction)
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()
            pred_class = int(prob > 0.5)
        
        # Extract patch-level attention
        attention_map = self.extract_patch_attention(tensor)
        
        # Get bboxes
        fracture_bboxes = []
        bbox_scores = []
        if pred_class == 1:  # Only for fractured
            fracture_bboxes, bbox_scores = self.get_fracture_bboxes(
                attention_map,
                threshold=threshold,
                min_cluster_size=min_cluster_size
            )
        
        # Statistics
        high_attention_patches = (attention_map > threshold).sum()
        total_patches = attention_map.size
        
        return {
            'classification': 'Fractured' if pred_class == 1 else 'Healthy',
            'confidence': prob if pred_class == 1 else (1 - prob),
            'attention_map': attention_map,
            'fracture_bboxes': fracture_bboxes,
            'bbox_scores': bbox_scores,
            'original_image': original,
            'stats': {
                'high_attention_patches': int(high_attention_patches),
                'total_patches': int(total_patches),
                'attention_ratio': float(high_attention_patches / total_patches),
                'max_attention': float(attention_map.max()),
                'mean_attention': float(attention_map.mean())
            }
        }
    
    def visualize_results(
        self,
        results: Dict,
        save_path: str
    ):
        """Visualize results with detailed statistics"""
        img = results['original_image']
        H, W = img.shape[:2]
        
        # Upsample attention map
        attention_map = results['attention_map']
        attn_resized = cv2.resize(
            attention_map,
            (W, H),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create figure with 4 subplots
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Original Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title("Original X-ray", fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # 2. Attention Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img)
        heatmap = ax2.imshow(attn_resized, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        ax2.set_title(f"Patch-Level Attention\n(Max: {results['stats']['max_attention']:.3f}, Mean: {results['stats']['mean_attention']:.3f})", 
                     fontsize=16, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. Binary Attention (threshold)
        ax3 = fig.add_subplot(gs[0, 2])
        binary_attn = (attn_resized > 0.5).astype(np.uint8) * 255
        ax3.imshow(img)
        ax3.imshow(binary_attn, cmap='Reds', alpha=0.5)
        ax3.set_title(f"High Attention Regions (>0.5)\n({results['stats']['attention_ratio']:.1%} of patches)", 
                     fontsize=16, fontweight='bold')
        ax3.axis('off')
        
        # 4. Detected Fracture Regions
        ax4 = fig.add_subplot(gs[1, :])
        img_bbox = img.copy()
        
        if results['fracture_bboxes']:
            for i, (bbox, score) in enumerate(zip(results['fracture_bboxes'], results['bbox_scores'])):
                x1, y1, x2, y2 = bbox
                
                # Color based on score (green to red)
                color_intensity = int(score * 255)
                color = (255, 255 - color_intensity, 0)  # Yellow to Red
                
                cv2.rectangle(img_bbox, (x1, y1), (x2, y2), color, 4)
                
                # Label
                label = f"Region {i+1}: {score:.2%}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img_bbox, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
                cv2.putText(
                    img_bbox,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        
        ax4.imshow(img_bbox)
        
        # Title with classification result
        title = f"{results['classification']} (Confidence: {results['confidence']:.2%})"
        if results['fracture_bboxes']:
            title += f"\n{len(results['fracture_bboxes'])} Fracture Region(s) Detected"
        else:
            if results['classification'] == 'Fractured':
                title += "\n⚠ Classified as Fractured but no high-confidence regions found"
        
        title_color = 'red' if results['classification'] == 'Fractured' else 'green'
        ax4.set_title(title, fontsize=18, fontweight='bold', color=title_color, pad=20)
        ax4.axis('off')
        
        # Save
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"\n[*] Visualization saved to {save_path}")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Fracture Localization")
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model', type=str, default='checkpoints/patch_transformer_full/best.pth')
    parser.add_argument('--output', type=str, default='outputs/localization/improved_result.png')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help="Probability threshold (0.5 = confident fracture)")
    parser.add_argument('--min_cluster', type=int, default=2,
                       help="Minimum connected patches to form a region")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("IMPROVED FRACTURE LOCALIZATION")
    print("="*80)
    
    # Initialize localizer
    localizer = ImprovedLocalizer(args.model)
    
    # Run prediction
    print(f"\nProcessing: {args.image}")
    results = localizer.predict(
        args.image,
        threshold=args.threshold,
        min_cluster_size=args.min_cluster
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Classification: {results['classification']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"\nAttention Statistics:")
    print(f"  Max attention: {results['stats']['max_attention']:.4f}")
    print(f"  Mean attention: {results['stats']['mean_attention']:.4f}")
    print(f"  High attention patches: {results['stats']['high_attention_patches']}/{results['stats']['total_patches']} ({results['stats']['attention_ratio']:.1%})")
    
    if results['fracture_bboxes']:
        print(f"\nDetected Regions: {len(results['fracture_bboxes'])}")
        for i, (bbox, score) in enumerate(zip(results['fracture_bboxes'], results['bbox_scores']), 1):
            print(f"  Region {i}: bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}), confidence={score:.2%}")
    else:
        if results['classification'] == 'Fractured':
            print(f"\n⚠ WARNING: Classified as Fractured but no regions found above threshold {args.threshold}")
            print(f"  Try lowering threshold (current: {args.threshold})")
        else:
            print(f"\nNo fracture regions detected (Healthy)")
    
    # Visualize
    localizer.visualize_results(results, args.output)
    
    print("\n" + "="*80)
    print("LOCALIZATION COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
