"""
Final Localization with Ground Truth Comparison

Reads ground truth annotations and compares with model predictions.
Shows:
1. Model's attention heatmap
2. Ground truth fracture locations (green)
3. Model's predicted regions (red)
4. Overlap analysis

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
from scipy import ndimage

from models.patch_transformer import PatchTransformerClassifier


class LocalizationWithGroundTruth:
    """Localization with GT comparison"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("\nLoading Patch Transformer...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print("[*] Localizer ready!")
    
    def _load_model(self, checkpoint_path: str):
        """Load model"""
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
            print(f"  Epoch {checkpoint['epoch']}, Val F1: {checkpoint.get('best_val_f1', 0):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def load_ground_truth(self, image_path: str) -> List[Tuple[int, int]]:
        """
        Load ground truth from .txt file
        
        Format: Each line is a point (x, y)
        Returns list of (x, y) coordinates
        """
        txt_path = Path(image_path).with_suffix('.txt')
        
        if not txt_path.exists():
            print(f"⚠ No ground truth found: {txt_path}")
            return []
        
        points = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append((int(x), int(y)))
        
        print(f"✓ Loaded {len(points)} ground truth points")
        return points
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, float, float]:
        """
        Load and preprocess, return scale factors
        
        Returns:
            tensor, original_image, scale_x, scale_y
        """
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load: {image_path}")
        
        original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        orig_h, orig_w = original.shape[:2]
        
        if self.config['image'].get('apply_clahe', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        target_size = tuple(self.config['image']['default_size'])
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        
        # Calculate scale factors
        scale_x = target_size[1] / orig_w
        scale_y = target_size[0] / orig_h
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device), original, scale_x, scale_y
    
    def extract_attention(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, Dict]:
        """Extract z-score normalized attention"""
        with torch.no_grad():
            patch_logits, nh, nw = self.model.get_patch_predictions(image_tensor)
        
        patch_probs = torch.sigmoid(patch_logits).squeeze(-1)
        attention_flat = patch_probs[0].cpu().numpy()
        
        # Z-score normalization
        mean_attn = attention_flat.mean()
        std_attn = attention_flat.std()
        z_scores = (attention_flat - mean_attn) / (std_attn + 1e-8)
        normalized = np.clip((z_scores + 3) / 6, 0, 1)
        
        attention_map = normalized.reshape(nh, nw)
        
        stats = {
            'mean': float(mean_attn),
            'std': float(std_attn),
            'min': float(attention_flat.min()),
            'max': float(attention_flat.max())
        }
        
        return attention_map, stats
    
    def get_predicted_regions(
        self,
        attention_map: np.ndarray,
        threshold: float = 0.7
    ) -> List[Dict]:
        """Get predicted fracture regions"""
        binary = (attention_map > threshold).astype(np.uint8)
        labeled, num = ndimage.label(binary)
        
        patch_size = self.config['model']['patch_size']
        regions = []
        
        for i in range(1, num + 1):
            component = (labeled == i)
            if component.sum() < 3:
                continue
            
            rows, cols = np.where(component)
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            
            x1, y1 = c_min * patch_size, r_min * patch_size
            x2, y2 = (c_max + 1) * patch_size, (r_max + 1) * patch_size
            
            regions.append({
                'bbox': (x1, y1, x2, y2),
                'score': float(attention_map[component].mean())
            })
        
        regions.sort(key=lambda x: x['score'], reverse=True)
        return regions
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def predict(
        self,
        image_path: str,
        threshold: float = 0.7
    ) -> Dict:
        """Complete prediction with GT comparison"""
        # Load GT
        gt_points = self.load_ground_truth(image_path)
        
        # Load and preprocess
        tensor, original, scale_x, scale_y = self.preprocess_image(image_path)
        
        # Classification
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()
            pred_class = int(prob > 0.5)
        
        # Attention
        attention_map, stats = self.extract_attention(tensor)
        
        # Predicted regions
        pred_regions = []
        if pred_class == 1:
            pred_regions = self.get_predicted_regions(attention_map, threshold)
        
        # Scale GT points to resized image
        gt_points_scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in gt_points]
        
        # Create GT bbox (bounding box of all GT points)
        gt_bbox = None
        if gt_points_scaled:
            xs = [p[0] for p in gt_points_scaled]
            ys = [p[1] for p in gt_points_scaled]
            margin = 50  # Add margin around points
            gt_bbox = (
                max(0, min(xs) - margin),
                max(0, min(ys) - margin),
                min(self.config['image']['default_size'][1], max(xs) + margin),
                min(self.config['image']['default_size'][0], max(ys) + margin)
            )
        
        # Calculate IoU with GT
        ious = []
        if gt_bbox and pred_regions:
            for region in pred_regions:
                iou = self.calculate_iou(region['bbox'], gt_bbox)
                ious.append(iou)
        
        return {
            'classification': 'Fractured' if pred_class == 1 else 'Healthy',
            'confidence': prob if pred_class == 1 else (1 - prob),
            'attention_map': attention_map,
            'pred_regions': pred_regions,
            'gt_points': gt_points_scaled,
            'gt_bbox': gt_bbox,
            'ious': ious,
            'original_image': original,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'statistics': stats
        }
    
    def visualize(self, results: Dict, save_path: str):
        """Visualize with GT comparison"""
        img = results['original_image']
        H, W = img.shape[:2]
        
        # Resize image to model size for overlay
        target_size = tuple(self.config['image']['default_size'])
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        
        attention_map = results['attention_map']
        attn_h = cv2.resize(attention_map, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Create figure
        fig = plt.figure(figsize=(28, 14))
        gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)
        
        # 1. Original with GT
        ax1 = fig.add_subplot(gs[0, 0])
        img_gt = img_resized.copy()
        
        # Draw GT points
        for x, y in results['gt_points']:
            cv2.circle(img_gt, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(img_gt, (x, y), 10, (255, 255, 255), 2)
        
        # Draw GT bbox
        if results['gt_bbox']:
            x1, y1, x2, y2 = results['gt_bbox']
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(img_gt, "GROUND TRUTH", (x1, y1 - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        ax1.imshow(img_gt)
        ax1.set_title("Ground Truth Annotation\n(Green = Actual Fracture Location)", 
                     fontsize=16, fontweight='bold', color='green')
        ax1.axis('off')
        
        # 2. Attention Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_resized)
        hm = ax2.imshow(attn_h, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        
        # Overlay GT points on heatmap
        for x, y in results['gt_points']:
            ax2.plot(x, y, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=2)
        
        title = f"Model Attention Heatmap\n"
        title += f"(White stars = GT locations)\n"
        title += f"Mean: {results['statistics']['mean']:.3f}, Std: {results['statistics']['std']:.3f}"
        ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(hm, ax=ax2, fraction=0.046)
        
        # 3. Predicted Regions vs GT
        ax3 = fig.add_subplot(gs[0, 2])
        img_comp = img_resized.copy()
        
        # Draw GT bbox first (green)
        if results['gt_bbox']:
            x1, y1, x2, y2 = results['gt_bbox']
            cv2.rectangle(img_comp, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(img_comp, "GT", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Draw predicted regions (red)
        for i, region in enumerate(results['pred_regions'][:3]):
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(img_comp, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            label = f"Pred #{i+1}"
            if i < len(results['ious']):
                label += f" (IoU: {results['ious'][i]:.2%})"
            
            cv2.putText(img_comp, label, (x1, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        ax3.imshow(img_comp)
        ax3.set_title("Prediction vs Ground Truth\n(Green=GT, Red=Predicted)", 
                     fontsize=16, fontweight='bold')
        ax3.axis('off')
        
        # 4. Attention Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(attention_map.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax4.axvline(results['statistics']['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {results['statistics']['mean']:.3f}")
        ax4.set_xlabel('Attention Score', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Attention Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. IoU Analysis
        ax5 = fig.add_subplot(gs[1, 1])
        if results['ious']:
            colors = ['red' if iou < 0.3 else 'orange' if iou < 0.5 else 'green' for iou in results['ious'][:5]]
            bars = ax5.bar(range(1, len(results['ious'][:5]) + 1), results['ious'][:5], color=colors, alpha=0.7)
            ax5.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Good Threshold (0.5)')
            ax5.set_xlabel('Predicted Region #', fontsize=12)
            ax5.set_ylabel('IoU with Ground Truth', fontsize=12)
            ax5.set_title('Localization Accuracy (IoU)', fontsize=14, fontweight='bold')
            ax5.set_ylim(0, 1)
            ax5.legend()
            ax5.grid(alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, 'No predictions to analyze', 
                    ha='center', va='center', fontsize=14, transform=ax5.transAxes)
            ax5.axis('off')
        
        # 6. Summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        summary_text = f"EVALUATION SUMMARY\n\n"
        summary_text += f"Classification: {results['classification']}\n"
        summary_text += f"Confidence: {results['confidence']:.1%}\n\n"
        
        if results['gt_bbox']:
            summary_text += f"Ground Truth: Present\n"
            summary_text += f"  Points: {len(results['gt_points'])}\n\n"
        else:
            summary_text += f"Ground Truth: Not Available\n\n"
        
        summary_text += f"Predicted Regions: {len(results['pred_regions'])}\n\n"
        
        if results['ious']:
            max_iou = max(results['ious'])
            summary_text += f"Best IoU: {max_iou:.1%}\n"
            if max_iou > 0.5:
                summary_text += f"✓ GOOD Localization\n"
            elif max_iou > 0.3:
                summary_text += f"⚠ MODERATE Localization\n"
            else:
                summary_text += f"✗ POOR Localization\n"
        else:
            summary_text += f"⚠ No overlap with GT\n"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=14, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Main title
        main_title = f"{results['classification']} ({results['confidence']:.1%})"
        if results['ious'] and max(results['ious']) < 0.3:
            main_title += " - ⚠ Model Failed to Localize Fracture Correctly!"
        
        fig.suptitle(main_title, fontsize=20, fontweight='bold', 
                    color='red' if results['classification'] == 'Fractured' else 'green')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"\n✓ Saved: {save_path}")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='checkpoints/patch_transformer_full/best.pth')
    parser.add_argument('--output', default='outputs/localization/gt_comparison.png')
    parser.add_argument('--threshold', type=float, default=0.7)
    
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LOCALIZATION WITH GROUND TRUTH COMPARISON")
    print("="*80)
    
    localizer = LocalizationWithGroundTruth(args.model)
    
    print(f"\nProcessing: {args.image}")
    results = localizer.predict(args.image, args.threshold)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Classification: {results['classification']} ({results['confidence']:.1%})")
    print(f"Ground Truth Points: {len(results['gt_points'])}")
    print(f"Predicted Regions: {len(results['pred_regions'])}")
    
    if results['ious']:
        print(f"\nIoU Scores:")
        for i, iou in enumerate(results['ious'], 1):
            status = "✓" if iou > 0.5 else "⚠" if iou > 0.3 else "✗"
            print(f"  Region {i}: {iou:.2%} {status}")
        
        max_iou = max(results['ious'])
        print(f"\nBest IoU: {max_iou:.1%}")
        if max_iou < 0.3:
            print("⚠ WARNING: Model failed to localize the fracture correctly!")
            print("  The high-attention regions do not overlap with ground truth.")
    
    localizer.visualize(results, args.output)
    
    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
