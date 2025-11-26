"""
Training Monitor

Monitors ongoing training progress by reading the latest checkpoint
and training history.

Usage:
    python monitor_training.py [checkpoint_dir]
"""

import json
import os
import sys
from pathlib import Path
import time


def find_latest_checkpoint_dir():
    """Find the most recent checkpoint directory."""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return None
    
    # Find all patch_transformer directories
    dirs = [d for d in checkpoints_dir.iterdir() 
            if d.is_dir() and d.name.startswith('patch_transformer_')]
    
    if not dirs:
        return None
    
    # Return the most recent one
    return max(dirs, key=lambda d: d.stat().st_mtime)


def load_training_history(checkpoint_dir):
    """Load training history from checkpoint directory."""
    history_file = checkpoint_dir / "training_history.json"
    
    if not history_file.exists():
        return None
    
    with open(history_file, 'r') as f:
        return json.load(f)


def print_progress(history):
    """Print training progress summary."""
    if not history:
        print("No training history found yet...")
        return
    
    train_history = history.get('train', [])
    val_history = history.get('val', [])
    
    if not train_history or not val_history:
        print("Training just started, no epochs completed yet...")
        return
    
    num_epochs = len(train_history)
    best_f1 = history.get('best_val_f1', 0)
    best_dice = history.get('best_val_dice', 0)
    
    print("\n" + "="*70)
    print(f"TRAINING PROGRESS - {num_epochs} EPOCHS COMPLETED")
    print("="*70)
    
    # Latest epoch
    latest_train = train_history[-1]
    latest_val = val_history[-1]
    
    print(f"\n📊 Latest Epoch ({num_epochs}):")
    print(f"  Train - Loss: {latest_train['loss']:.4f}, Acc: {latest_train['accuracy']:.4f}, F1: {latest_train['f1']:.4f}")
    print(f"  Val   - Loss: {latest_val['loss']:.4f}, Acc: {latest_val['accuracy']:.4f}, F1: {latest_val['f1']:.4f}")
    
    print(f"\n🏆 Best Results:")
    print(f"  Best Val F1: {best_f1:.4f}")
    print(f"  Best Val Dice: {best_dice:.4f}")
    
    # Show last 5 epochs
    print(f"\n📈 Last 5 Epochs:")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train F1':<10} {'Val Loss':<12} {'Val F1':<10}")
    print("-" * 70)
    
    start_idx = max(0, num_epochs - 5)
    for i in range(start_idx, num_epochs):
        train = train_history[i]
        val = val_history[i]
        epoch_num = i + 1
        
        marker = " 🏆" if abs(val['f1'] - best_f1) < 0.0001 else ""
        
        print(f"{epoch_num:<8} {train['loss']:<12.4f} {train['f1']:<10.4f} "
              f"{val['loss']:<12.4f} {val['f1']:<10.4f}{marker}")
    
    print("="*70)


def monitor(checkpoint_dir=None, watch=False, interval=10):
    """
    Monitor training progress.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (auto-detect if None)
        watch: Continuously watch for updates
        interval: Update interval in seconds (for watch mode)
    """
    if checkpoint_dir is None:
        checkpoint_dir = find_latest_checkpoint_dir()
        if checkpoint_dir is None:
            print("❌ No checkpoint directory found!")
            print("Training hasn't started yet or checkpoints are in a different location.")
            return
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    print(f"📂 Monitoring: {checkpoint_dir}")
    
    if watch:
        print(f"⏱️  Auto-updating every {interval} seconds (Ctrl+C to stop)")
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"📂 Monitoring: {checkpoint_dir}")
                print(f"⏱️  Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                history = load_training_history(checkpoint_dir)
                print_progress(history)
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
    else:
        history = load_training_history(checkpoint_dir)
        print_progress(history)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('checkpoint_dir', nargs='?', default=None,
                       help='Path to checkpoint directory (auto-detect if not provided)')
    parser.add_argument('-w', '--watch', action='store_true',
                       help='Continuously watch for updates')
    parser.add_argument('-i', '--interval', type=int, default=10,
                       help='Update interval in seconds (default: 10)')
    
    args = parser.parse_args()
    
    monitor(args.checkpoint_dir, args.watch, args.interval)


if __name__ == "__main__":
    main()
