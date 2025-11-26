"""
Comprehensive Test Set Evaluation

Evaluates model on entire test set:
1. Classification metrics (Accuracy, Precision, Recall, F1)
2. Localization metrics (IoU with ground truth)
3. Per-sample analysis
4. Confusion matrix
5. Save detailed results

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.patch_transformer import PatchTransformerClassifier


class TestSetEvaluator:
    """Complete test set evaluation"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Load splits
        split_file = self.config['data']['split_file']
        with open(split_file, 'r') as f:
            self.splits = json.load(f)
        
        print(f"\nLoaded splits from: {split_file}")
        print(f"  Train: {len(self.splits['train'])}")
        print(f"  Val: {len(self.splits['val'])}")
        print(f"  Test: {len(self.splits['test'])}")
        
        # Load model
        print("\nLoading Patch Transformer...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print("\n[*] Evaluator ready!")
    
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
        """Load GT points from .txt file"""
        txt_path = Path(image_path).with_suffix('.txt')
        
        if not txt_path.exists():
            return []
        
        points = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append((int(x), int(y)))
        
        return points
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, float, float]:
        """Load and preprocess, return scale factors"""
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load: {image_path}")
        
        orig_h, orig_w = img.shape[:2]
        
        if self.config['image'].get('apply_clahe', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        target_size = tuple(self.config['image']['default_size'])
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        
        # Scale factors
        scale_x = target_size[1] / orig_w
        scale_y = target_size[0] / orig_h
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device), scale_x, scale_y
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU"""
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
    
    def get_gt_bbox(self, gt_points: List[Tuple[int, int]], scale_x: float, scale_y: float):
        """Create GT bbox from points"""
        if not gt_points:
            return None
        
        gt_points_scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in gt_points]
        
        xs = [p[0] for p in gt_points_scaled]
        ys = [p[1] for p in gt_points_scaled]
        
        margin = 50
        target_size = tuple(self.config['image']['default_size'])
        
        gt_bbox = (
            max(0, min(xs) - margin),
            max(0, min(ys) - margin),
            min(target_size[1], max(xs) + margin),
            min(target_size[0], max(ys) + margin)
        )
        
        return gt_bbox
    
    def get_attention_bbox(self, image_tensor: torch.Tensor, threshold: float = 0.7):
        """Get highest attention region bbox"""
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
        
        # Find highest attention region
        binary = (attention_map > threshold).astype(np.uint8)
        from scipy import ndimage
        labeled, num = ndimage.label(binary)
        
        if num == 0:
            return None
        
        # Get largest region
        max_score = 0
        best_bbox = None
        patch_size = self.config['model']['patch_size']
        
        for i in range(1, num + 1):
            component = (labeled == i)
            if component.sum() < 3:
                continue
            
            score = attention_map[component].mean()
            
            if score > max_score:
                max_score = score
                rows, cols = np.where(component)
                r_min, r_max = rows.min(), rows.max()
                c_min, c_max = cols.min(), cols.max()
                
                x1, y1 = c_min * patch_size, r_min * patch_size
                x2, y2 = (c_max + 1) * patch_size, (r_max + 1) * patch_size
                best_bbox = (x1, y1, x2, y2)
        
        return best_bbox
    
    def evaluate_sample(self, image_path: str, true_label: int) -> Dict:
        """Evaluate single sample"""
        try:
            # Load GT
            gt_points = self.load_ground_truth(image_path)
            
            # Preprocess
            tensor, scale_x, scale_y = self.preprocess_image(image_path)
            
            # Predict
            with torch.no_grad():
                logits = self.model(tensor)
                prob = torch.sigmoid(logits).item()
                pred_class = int(prob > 0.5)
            
            # Localization (only for fractured)
            iou = None
            if true_label == 1 and gt_points:
                gt_bbox = self.get_gt_bbox(gt_points, scale_x, scale_y)
                pred_bbox = self.get_attention_bbox(tensor)
                
                if gt_bbox and pred_bbox:
                    iou = self.calculate_iou(pred_bbox, gt_bbox)
            
            return {
                'image': str(image_path),
                'true_label': true_label,
                'pred_label': pred_class,
                'probability': prob,
                'correct': pred_class == true_label,
                'has_gt': len(gt_points) > 0,
                'iou': iou,
                'success': True
            }
        
        except Exception as e:
            print(f"\n⚠ Error processing {image_path}: {str(e)}")
            return {
                'image': str(image_path),
                'true_label': true_label,
                'pred_label': -1,
                'probability': 0.0,
                'correct': False,
                'has_gt': False,
                'iou': None,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_test_set(self) -> Dict:
        """Evaluate entire test set"""
        print("\n" + "="*80)
        print("EVALUATING TEST SET")
        print("="*80)
        
        # Load test dataset directly
        from data.dataset import DentalXrayDataset
        test_dataset = DentalXrayDataset(
            root_dir=self.config['data']['root_dir'],
            split='test',
            split_file=self.config['data']['split_file']
        )
        
        results = []
        
        # Process test samples
        for idx in tqdm(range(len(test_dataset)), desc="Processing test set"):
            image_path = test_dataset.image_paths[idx]
            true_label = test_dataset.labels[idx]
            
            result = self.evaluate_sample(image_path, true_label)
            results.append(result)
        
        # Calculate metrics
        successful = [r for r in results if r['success']]
        
        y_true = [r['true_label'] for r in successful]
        y_pred = [r['pred_label'] for r in successful]
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Localization metrics
        fractured_with_gt = [r for r in successful if r['true_label'] == 1 and r['has_gt'] and r['iou'] is not None]
        ious = [r['iou'] for r in fractured_with_gt]
        
        localization_metrics = {
            'num_samples': len(fractured_with_gt),
            'mean_iou': np.mean(ious) if ious else 0.0,
            'median_iou': np.median(ious) if ious else 0.0,
            'good_localization': sum(1 for iou in ious if iou > 0.5) if ious else 0,
            'moderate_localization': sum(1 for iou in ious if 0.3 < iou <= 0.5) if ious else 0,
            'poor_localization': sum(1 for iou in ious if iou <= 0.3) if ious else 0
        }
        
        return {
            'total_samples': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'localization': localization_metrics,
            'per_sample_results': results
        }
    
    def visualize_results(self, eval_results: Dict, save_dir: str):
        """Create visualization of results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        cm = np.array(eval_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Healthy', 'Fractured'],
                   yticklabels=['Healthy', 'Fractured'])
        axes[0].set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # 2. Metrics Bar Chart
        metrics = {
            'Accuracy': eval_results['accuracy'],
            'Precision': eval_results['precision'],
            'Recall': eval_results['recall'],
            'F1 Score': eval_results['f1_score']
        }
        
        colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in metrics.values()]
        bars = axes[1].bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7)
        axes[1].set_ylim(0, 1)
        axes[1].axhline(0.9, color='green', linestyle='--', linewidth=2, label='Target (0.9)')
        axes[1].set_title('Classification Metrics', fontsize=16, fontweight='bold')
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'classification_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Localization Results
        if eval_results['localization']['num_samples'] > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            loc = eval_results['localization']
            categories = ['Good\n(IoU>0.5)', 'Moderate\n(0.3<IoU≤0.5)', 'Poor\n(IoU≤0.3)']
            values = [loc['good_localization'], loc['moderate_localization'], loc['poor_localization']]
            colors_loc = ['green', 'orange', 'red']
            
            bars = ax.bar(categories, values, color=colors_loc, alpha=0.7)
            ax.set_title(f"Localization Quality (Mean IoU: {loc['mean_iou']:.3f})", 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Number of Samples', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            
            # Add values
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'localization_quality.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\n✓ Visualizations saved to {save_dir}")
    
    def save_results(self, eval_results: Dict, output_path: str):
        """Save detailed results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"✓ Detailed results saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Set Evaluation")
    parser.add_argument('--model', default='checkpoints/patch_transformer_full/best.pth')
    parser.add_argument('--output_dir', default='outputs/test_evaluation')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    # Initialize evaluator
    evaluator = TestSetEvaluator(args.model)
    
    # Run evaluation
    results = evaluator.evaluate_test_set()
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nTotal samples: {results['total_samples']}")
    print(f"Successfully processed: {results['successful']}")
    print(f"Failed: {results['failed']}")
    
    print("\n" + "-"*80)
    print("CLASSIFICATION METRICS")
    print("-"*80)
    print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"F1 Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    
    print("\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(f"              Predicted")
    print(f"              Healthy  Fractured")
    print(f"True Healthy     {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"     Fractured   {cm[1,0]:3d}      {cm[1,1]:3d}")
    
    print("\n" + "-"*80)
    print("LOCALIZATION METRICS (Fractured samples with GT)")
    print("-"*80)
    loc = results['localization']
    if loc['num_samples'] > 0:
        print(f"Samples with GT: {loc['num_samples']}")
        print(f"Mean IoU: {loc['mean_iou']:.4f}")
        print(f"Median IoU: {loc['median_iou']:.4f}")
        print(f"\nLocalization Quality:")
        print(f"  Good (IoU > 0.5):     {loc['good_localization']} ({loc['good_localization']/loc['num_samples']*100:.1f}%)")
        print(f"  Moderate (0.3-0.5):   {loc['moderate_localization']} ({loc['moderate_localization']/loc['num_samples']*100:.1f}%)")
        print(f"  Poor (IoU ≤ 0.3):     {loc['poor_localization']} ({loc['poor_localization']/loc['num_samples']*100:.1f}%)")
    else:
        print("No samples with ground truth annotations")
    
    # Save results
    evaluator.save_results(results, output_dir / 'detailed_results.json')
    
    # Visualize
    evaluator.visualize_results(results, output_dir)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
