"""
Analyze Training History and Generate Report

Reads training_history.json and generates detailed analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_history(checkpoint_dir):
    """Load training history."""
    history_file = Path(checkpoint_dir) / "training_history.json"
    with open(history_file, 'r') as f:
        return json.load(f)


def analyze_history(history):
    """Analyze training history and print summary."""
    train_hist = history['train']
    val_hist = history['val']
    
    print("="*70)
    print("TRAINING HISTORY ANALYSIS")
    print("="*70)
    
    # Best epoch
    best_epoch = np.argmax([h['f1'] for h in val_hist])
    best_val = val_hist[best_epoch]
    
    print(f"\n🏆 Best Epoch: {best_epoch + 1}")
    print(f"   Val F1: {best_val['f1']:.4f}")
    print(f"   Val Dice: {best_val['dice']:.4f}")
    print(f"   Val Accuracy: {best_val['accuracy']:.4f}")
    print(f"   Val Precision: {best_val['precision']:.4f}")
    print(f"   Val Recall: {best_val['recall']:.4f}")
    print(f"   Val Specificity: {best_val['specificity']:.4f}")
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix (Best Epoch):")
    print(f"   TP: {best_val['tp']}, TN: {best_val['tn']}")
    print(f"   FP: {best_val['fp']}, FN: {best_val['fn']}")
    
    # Final epoch
    final_val = val_hist[-1]
    print(f"\n📌 Final Epoch ({len(val_hist)}):")
    print(f"   Val F1: {final_val['f1']:.4f}")
    print(f"   Val Dice: {final_val['dice']:.4f}")
    print(f"   Val Accuracy: {final_val['accuracy']:.4f}")
    
    # Training progression
    print(f"\n📈 Training Progression:")
    
    # First 5 epochs
    print(f"\nFirst 5 Epochs:")
    print(f"{'Epoch':<8} {'Train F1':<12} {'Val F1':<12} {'Val Acc':<12}")
    print("-" * 50)
    for i in range(min(5, len(val_hist))):
        train_f1 = train_hist[i]['f1']
        val_f1 = val_hist[i]['f1']
        val_acc = val_hist[i]['accuracy']
        print(f"{i+1:<8} {train_f1:<12.4f} {val_f1:<12.4f} {val_acc:<12.4f}")
    
    # Last 5 epochs
    print(f"\nLast 5 Epochs:")
    print(f"{'Epoch':<8} {'Train F1':<12} {'Val F1':<12} {'Val Acc':<12}")
    print("-" * 50)
    start_idx = max(0, len(val_hist) - 5)
    for i in range(start_idx, len(val_hist)):
        train_f1 = train_hist[i]['f1']
        val_f1 = val_hist[i]['f1']
        val_acc = val_hist[i]['accuracy']
        marker = " 🏆" if i == best_epoch else ""
        print(f"{i+1:<8} {train_f1:<12.4f} {val_f1:<12.4f} {val_acc:<12.4f}{marker}")
    
    # Statistics
    val_f1s = [h['f1'] for h in val_hist]
    val_accs = [h['accuracy'] for h in val_hist]
    
    print(f"\n📊 Validation Statistics (50 epochs):")
    print(f"   F1    - Min: {min(val_f1s):.4f}, Max: {max(val_f1s):.4f}, "
          f"Mean: {np.mean(val_f1s):.4f}, Std: {np.std(val_f1s):.4f}")
    print(f"   Acc   - Min: {min(val_accs):.4f}, Max: {max(val_accs):.4f}, "
          f"Mean: {np.mean(val_accs):.4f}, Std: {np.std(val_accs):.4f}")
    
    # Overfitting analysis
    train_f1_final = train_hist[-1]['f1']
    val_f1_final = val_hist[-1]['f1']
    gap = train_f1_final - val_f1_final
    
    print(f"\n🔍 Overfitting Analysis:")
    print(f"   Train F1 (final): {train_f1_final:.4f}")
    print(f"   Val F1 (final): {val_f1_final:.4f}")
    print(f"   Gap: {gap:.4f}")
    if gap < 0.05:
        print(f"   ✅ Minimal overfitting (gap < 5%)")
    elif gap < 0.10:
        print(f"   ⚠️  Slight overfitting (gap < 10%)")
    else:
        print(f"   ❌ Significant overfitting (gap > 10%)")
    
    print("\n" + "="*70)
    
    return best_epoch, best_val, final_val


def plot_training_curves(history, save_path="outputs/training_curves.png"):
    """Plot training curves."""
    train_hist = history['train']
    val_hist = history['val']
    
    epochs = range(1, len(train_hist) + 1)
    
    # Extract metrics
    train_loss = [h['loss'] for h in train_hist]
    val_loss = [h['loss'] for h in val_hist]
    train_f1 = [h['f1'] for h in train_hist]
    val_f1 = [h['f1'] for h in val_hist]
    train_acc = [h['accuracy'] for h in train_hist]
    val_acc = [h['accuracy'] for h in val_hist]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - PatchTransformer Base', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 1].plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
    axes[0, 1].plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2)
    axes[0, 1].axhline(y=0.9091, color='g', linestyle='--', label='Best Val F1 (0.9091)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    axes[1, 0].plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation metrics breakdown
    val_precision = [h['precision'] for h in val_hist]
    val_recall = [h['recall'] for h in val_hist]
    val_spec = [h['specificity'] for h in val_hist]
    
    axes[1, 1].plot(epochs, val_f1, 'b-', label='F1', linewidth=2)
    axes[1, 1].plot(epochs, val_precision, 'g-', label='Precision', linewidth=2)
    axes[1, 1].plot(epochs, val_recall, 'r-', label='Recall', linewidth=2)
    axes[1, 1].plot(epochs, val_spec, 'orange', label='Specificity', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: {save_path}")
    
    plt.close()


def main():
    """Main function."""
    checkpoint_dir = "checkpoints/patch_transformer_full"
    
    # Load history
    print("Loading training history...")
    history = load_history(checkpoint_dir)
    
    # Analyze
    best_epoch, best_val, final_val = analyze_history(history)
    
    # Plot curves
    print("\nGenerating training curves...")
    plot_training_curves(history)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
