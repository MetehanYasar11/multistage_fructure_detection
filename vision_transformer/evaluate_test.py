"""
Test Set Evaluation - PatchTransformer Model

Evaluates the best trained model on the held-out test set.

Features:
- Load best checkpoint
- Evaluate on test set (74 images)
- Compute comprehensive metrics
- Generate confusion matrix
- Compute confidence intervals
- Save predictions for analysis

Author: Master's Thesis Project
Date: October 28, 2025
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DentalXrayDataset, get_val_transforms
from models import create_patch_transformer
from utils import set_seed, get_device


class TestEvaluator:
    """Evaluates model on test set."""
    
    def __init__(self, model, device, save_predictions=True):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to run on
            save_predictions: Whether to save predictions
        """
        self.model = model.to(device)
        self.device = device
        self.save_predictions = save_predictions
        
        # Storage
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.image_paths = []
        self.patch_predictions = []
    
    @torch.no_grad()
    def evaluate(self, test_loader):
        """
        Evaluate on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Batch size: {test_loader.batch_size}")
        print("="*70 + "\n")
        
        # Iterate over test set
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(self.device)
            targets = targets.cpu().numpy()
            
            # Forward pass
            outputs = self.model(images)
            
            # Get patch predictions if available
            if hasattr(self.model, 'get_patch_predictions'):
                patch_preds, nh, nw = self.model.get_patch_predictions(images)
                patch_preds = patch_preds.cpu().numpy()
                self.patch_predictions.extend(patch_preds)
            
            # Convert to probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            # Store results
            self.predictions.extend(preds.flatten())
            self.targets.extend(targets.flatten())
            self.probabilities.extend(probs.flatten())
            
            # Get image paths if available
            if hasattr(test_loader.dataset, 'image_paths'):
                batch_start = batch_idx * test_loader.batch_size
                batch_end = min(batch_start + len(images), len(test_loader.dataset))
                self.image_paths.extend(
                    test_loader.dataset.image_paths[batch_start:batch_end]
                )
        
        # Convert to numpy
        self.predictions = np.array(self.predictions)
        self.targets = np.array(self.targets)
        self.probabilities = np.array(self.probabilities)
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Print results
        self._print_results(metrics)
        
        return metrics
    
    def _compute_metrics(self):
        """Compute all evaluation metrics."""
        # Confusion matrix
        tp = np.sum((self.predictions == 1) & (self.targets == 1))
        tn = np.sum((self.predictions == 0) & (self.targets == 0))
        fp = np.sum((self.predictions == 1) & (self.targets == 0))
        fn = np.sum((self.predictions == 0) & (self.targets == 1))
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Balanced accuracy
        balanced_acc = (recall + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        # Confidence intervals (95%, bootstrap)
        ci_iterations = 1000
        bootstrap_f1 = []
        bootstrap_acc = []
        
        n = len(self.predictions)
        for _ in range(ci_iterations):
            indices = np.random.choice(n, n, replace=True)
            boot_preds = self.predictions[indices]
            boot_targets = self.targets[indices]
            
            boot_tp = np.sum((boot_preds == 1) & (boot_targets == 1))
            boot_tn = np.sum((boot_preds == 0) & (boot_targets == 0))
            boot_fp = np.sum((boot_preds == 1) & (boot_targets == 0))
            boot_fn = np.sum((boot_preds == 0) & (boot_targets == 1))
            
            boot_prec = boot_tp / (boot_tp + boot_fp) if (boot_tp + boot_fp) > 0 else 0
            boot_rec = boot_tp / (boot_tp + boot_fn) if (boot_tp + boot_fn) > 0 else 0
            boot_f1 = 2 * (boot_prec * boot_rec) / (boot_prec + boot_rec) if (boot_prec + boot_rec) > 0 else 0
            boot_acc = (boot_tp + boot_tn) / (boot_tp + boot_tn + boot_fp + boot_fn)
            
            bootstrap_f1.append(boot_f1)
            bootstrap_acc.append(boot_acc)
        
        f1_ci_lower = np.percentile(bootstrap_f1, 2.5)
        f1_ci_upper = np.percentile(bootstrap_f1, 97.5)
        acc_ci_lower = np.percentile(bootstrap_acc, 2.5)
        acc_ci_upper = np.percentile(bootstrap_acc, 97.5)
        
        return {
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn),
                'fp': int(fp), 'fn': int(fn)
            },
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1': float(f1),
            'dice': float(f1),  # Same as F1 for binary
            'npv': float(npv),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'balanced_accuracy': float(balanced_acc),
            'mcc': float(mcc),
            'confidence_intervals': {
                'f1': {'lower': float(f1_ci_lower), 'upper': float(f1_ci_upper)},
                'accuracy': {'lower': float(acc_ci_lower), 'upper': float(acc_ci_upper)}
            },
            'sample_counts': {
                'total': int(len(self.predictions)),
                'fractured': int(np.sum(self.targets == 1)),
                'healthy': int(np.sum(self.targets == 0))
            }
        }
    
    def _print_results(self, metrics):
        """Print evaluation results."""
        cm = metrics['confusion_matrix']
        
        print("\n" + "="*70)
        print("TEST SET RESULTS")
        print("="*70)
        
        print(f"\n📊 Sample Distribution:")
        print(f"   Total: {metrics['sample_counts']['total']}")
        print(f"   Fractured: {metrics['sample_counts']['fractured']}")
        print(f"   Healthy: {metrics['sample_counts']['healthy']}")
        
        print(f"\n🎯 Confusion Matrix:")
        print(f"                Predicted")
        print(f"                Fractured  Healthy")
        print(f"   Actual  ")
        print(f"   Fractured      {cm['tp']:<7}    {cm['fn']:<7}")
        print(f"   Healthy        {cm['fp']:<7}    {cm['tn']:<7}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Accuracy:    {metrics['accuracy']:.4f} (95% CI: [{metrics['confidence_intervals']['accuracy']['lower']:.4f}, {metrics['confidence_intervals']['accuracy']['upper']:.4f}])")
        print(f"   F1 Score:    {metrics['f1']:.4f} (95% CI: [{metrics['confidence_intervals']['f1']['lower']:.4f}, {metrics['confidence_intervals']['f1']['upper']:.4f}])")
        print(f"   Dice Score:  {metrics['dice']:.4f}")
        print(f"   Precision:   {metrics['precision']:.4f}")
        print(f"   Recall:      {metrics['recall']:.4f}")
        print(f"   Specificity: {metrics['specificity']:.4f}")
        print(f"   NPV:         {metrics['npv']:.4f}")
        print(f"   Balanced Acc:{metrics['balanced_accuracy']:.4f}")
        print(f"   MCC:         {metrics['mcc']:.4f}")
        
        print(f"\n🔍 Error Rates:")
        print(f"   False Positive Rate: {metrics['fpr']:.4f} ({cm['fp']}/{cm['fp']+cm['tn']})")
        print(f"   False Negative Rate: {metrics['fnr']:.4f} ({cm['fn']}/{cm['fn']+cm['tp']})")
        
        print("\n" + "="*70)
    
    def save_results(self, save_dir):
        """Save evaluation results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.save_predictions:
            # Save predictions
            results = {
                'predictions': self.predictions.tolist(),
                'targets': self.targets.tolist(),
                'probabilities': self.probabilities.tolist(),
                'image_paths': self.image_paths
            }
            
            with open(save_dir / 'test_predictions.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Predictions saved to: {save_dir / 'test_predictions.json'}")
            
            # Save patch predictions if available
            if self.patch_predictions:
                np.save(save_dir / 'test_patch_predictions.npy', 
                       np.array(self.patch_predictions))
                print(f"💾 Patch predictions saved to: {save_dir / 'test_patch_predictions.npy'}")


def main():
    """Main evaluation function."""
    print("="*70)
    print("PATCH TRANSFORMER - TEST SET EVALUATION")
    print("="*70)
    
    # Configuration
    seed = 42
    image_size = (1400, 2800)
    batch_size = 6
    checkpoint_path = "checkpoints/patch_transformer_full/best.pth"
    
    set_seed(seed)
    device = get_device()
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = DentalXrayDataset(
        root_dir=r"c:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset",
        split='test',
        split_file="outputs/splits/train_val_test_split.json",
        transform=get_val_transforms(image_size=image_size)
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    print("\nLoading best model...")
    model = create_patch_transformer(
        image_size=image_size,
        patch_size=100,
        model_size='base',
        aggregation='max'
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"   Best validation F1: {checkpoint['best_val_f1']:.4f}")
    print(f"   Best validation Dice: {checkpoint['best_val_dice']:.4f}")
    
    # Create evaluator
    evaluator = TestEvaluator(model, device, save_predictions=True)
    
    # Evaluate
    metrics = evaluator.evaluate(test_loader)
    
    # Save results
    save_dir = "outputs/test_evaluation"
    evaluator.save_results(save_dir)
    
    # Save metrics
    with open(f"{save_dir}/test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Metrics saved to: {save_dir}/test_metrics.json")
    
    # Compare with validation
    print("\n" + "="*70)
    print("VALIDATION vs TEST COMPARISON")
    print("="*70)
    print(f"Validation F1:   {checkpoint['best_val_f1']:.4f}")
    print(f"Test F1:         {metrics['f1']:.4f}")
    print(f"Difference:      {abs(checkpoint['best_val_f1'] - metrics['f1']):.4f}")
    
    if abs(checkpoint['best_val_f1'] - metrics['f1']) < 0.05:
        print("✅ Model generalizes well (difference < 5%)")
    else:
        print("⚠️  Significant difference between val and test")
    
    print("="*70)
    
    print("\n✅ Test evaluation complete!")
    
    return metrics


if __name__ == "__main__":
    metrics = main()
