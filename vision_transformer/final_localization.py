"""
Final Localization: Relative Attention + Statistical Analysis

Key Insight:
- Max aggregation causes uniform attention across patches
- Need to find RELATIVE differences, not absolute scores
- Use statistical methods to identify outliers (high-attention patches)

Approach:
1. Get patch-level predictions
2. Normalize using z-scores to find outliers
3. Top-k selection: Select patches significantly above mean
4. Cluster adjacent high-attention patches into regions

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
from scipy import ndimage, stats

from models.patch_transformer import PatchTransformerClassifier


class FinalLocalizer:
    """Statistical attention-based localization"""
    
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
        
        print("\n[*] Final Localizer ready!")
    
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
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Load and preprocess"""
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load: {image_path}")
        
        original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if self.config['image'].get('apply_clahe', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        target_size = tuple(self.config['image']['default_size'])
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device), original
    
    def extract_relative_attention(
        self,
        image_tensor: torch.Tensor,
        method: str = 'zscore'  # 'zscore', 'percentile', 'topk'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract relative attention using statistical methods
        
        Returns:
            attention_map: (nh, nw) normalized scores
            stats: Dictionary with statistics
        """
        with torch.no_grad():
            patch_logits, nh, nw = self.model.get_patch_predictions(image_tensor)
        
        # Get probabilities
        patch_probs = torch.sigmoid(patch_logits).squeeze(-1)  # (B, N)
        attention_flat = patch_probs[0].cpu().numpy()  # (N,)
        
        # Statistical analysis
        mean_attn = attention_flat.mean()
        std_attn = attention_flat.std()
        median_attn = np.median(attention_flat)
        
        if method == 'zscore':
            # Z-score normalization: Find outliers
            z_scores = (attention_flat - mean_attn) / (std_attn + 1e-8)
            # Normalize to [0, 1], clamp outliers
            normalized = np.clip((z_scores + 3) / 6, 0, 1)  # 3 sigma range
        
        elif method == 'percentile':
            # Percentile-based: Relative ranking
            from scipy.stats import rankdata
            ranks = rankdata(attention_flat, method='average')
            normalized = ranks / len(ranks)
        
        elif method == 'topk':
            # Top-k selection
            k = int(len(attention_flat) * 0.1)  # Top 10%
            threshold = np.partition(attention_flat, -k)[-k]
            normalized = (attention_flat >= threshold).astype(float)
        
        else:
            # Default: min-max normalization
            normalized = (attention_flat - attention_flat.min()) / (attention_flat.max() - attention_flat.min() + 1e-8)
        
        # Reshape
        attention_map = normalized.reshape(nh, nw)
        
        statistics = {
            'mean': float(mean_attn),
            'std': float(std_attn),
            'median': float(median_attn),
            'min': float(attention_flat.min()),
            'max': float(attention_flat.max()),
            'range': float(attention_flat.max() - attention_flat.min()),
            'cv': float(std_attn / (mean_attn + 1e-8))  # Coefficient of variation
        }
        
        return attention_map, statistics
    
    def get_fracture_regions(
        self,
        attention_map: np.ndarray,
        threshold: float = 0.7,
        min_size: int = 3
    ) -> List[Dict]:
        """Get fracture regions with metadata"""
        binary = (attention_map > threshold).astype(np.uint8)
        labeled, num = ndimage.label(binary)
        
        patch_size = self.config['model']['patch_size']
        regions = []
        
        for i in range(1, num + 1):
            component = (labeled == i)
            if component.sum() < min_size:
                continue
            
            rows, cols = np.where(component)
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            
            x1, y1 = c_min * patch_size, r_min * patch_size
            x2, y2 = (c_max + 1) * patch_size, (r_max + 1) * patch_size
            
            region_attn = attention_map[component]
            
            regions.append({
                'bbox': (x1, y1, x2, y2),
                'score': float(region_attn.mean()),
                'max_score': float(region_attn.max()),
                'size': int(component.sum()),
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
            })
        
        # Sort by score
        regions.sort(key=lambda x: x['score'], reverse=True)
        return regions
    
    def predict(
        self,
        image_path: str,
        method: str = 'zscore',
        threshold: float = 0.7,
        min_size: int = 3
    ) -> Dict:
        """Complete prediction"""
        tensor, original = self.preprocess_image(image_path)
        
        # Classification
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()
            pred_class = int(prob > 0.5)
        
        # Relative attention
        attention_map, statistics = self.extract_relative_attention(tensor, method)
        
        # Get regions
        regions = []
        if pred_class == 1:
            regions = self.get_fracture_regions(attention_map, threshold, min_size)
        
        return {
            'classification': 'Fractured' if pred_class == 1 else 'Healthy',
            'confidence': prob if pred_class == 1 else (1 - prob),
            'attention_map': attention_map,
            'regions': regions,
            'original_image': original,
            'statistics': statistics,
            'method': method
        }
    
    def visualize(self, results: Dict, save_path: str):
        """Visualize with statistical analysis"""
        img = results['original_image']
        H, W = img.shape[:2]
        attention_map = results['attention_map']
        
        # Upsample
        attn_resized = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Create figure
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Original
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title("Original X-ray", fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # 2. Attention Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img)
        hm = ax2.imshow(attn_resized, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        title = f"Relative Attention ({results['method']})\n"
        title += f"Mean: {results['statistics']['mean']:.3f}, Std: {results['statistics']['std']:.3f}, Range: {results['statistics']['range']:.3f}"
        ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(hm, ax=ax2, fraction=0.046)
        
        # 3. Attention Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(attention_map.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax3.axvline(results['statistics']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['statistics']['mean']:.3f}")
        ax3.axvline(results['statistics']['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {results['statistics']['median']:.3f}")
        ax3.set_xlabel('Attention Score', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Attention Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Detected Regions
        ax4 = fig.add_subplot(gs[1, :])
        img_bbox = img.copy()
        
        if results['regions']:
            for i, region in enumerate(results['regions'][:5]):  # Top 5
                x1, y1, x2, y2 = region['bbox']
                score = region['score']
                
                # Color intensity
                color_val = int(score * 255)
                color = (255, 255 - color_val, 0)
                
                cv2.rectangle(img_bbox, (x1, y1), (x2, y2), color, 4)
                
                label = f"#{i+1}: {score:.2%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img_bbox, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
                cv2.putText(img_bbox, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ax4.imshow(img_bbox)
        
        title = f"{results['classification']} ({results['confidence']:.1%})"
        if results['regions']:
            title += f" - {len(results['regions'])} Region(s)"
        
        color = 'red' if results['classification'] == 'Fractured' else 'green'
        ax4.set_title(title, fontsize=18, fontweight='bold', color=color, pad=15)
        ax4.axis('off')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"\n[*] Saved: {save_path}")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='checkpoints/patch_transformer_full/best.pth')
    parser.add_argument('--output', default='outputs/localization/final_result.png')
    parser.add_argument('--method', default='zscore', choices=['zscore', 'percentile', 'topk'])
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--min_size', type=int, default=3)
    
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FINAL LOCALIZATION (Statistical Attention)")
    print("="*80)
    
    localizer = FinalLocalizer(args.model)
    
    print(f"\nProcessing: {args.image}")
    results = localizer.predict(args.image, args.method, args.threshold, args.min_size)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Classification: {results['classification']} ({results['confidence']:.1%})")
    print(f"\nStatistics (Raw Scores):")
    for k, v in results['statistics'].items():
        print(f"  {k}: {v:.4f}")
    
    if results['regions']:
        print(f"\nDetected {len(results['regions'])} region(s):")
        for i, r in enumerate(results['regions'][:5], 1):
            print(f"  #{i}: score={r['score']:.2%}, size={r['size']} patches, center={r['center']}")
    else:
        print(f"\nNo high-confidence regions found (threshold={args.threshold})")
    
    localizer.visualize(results, args.output)
    
    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
