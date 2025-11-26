"""
Minimal training test - 2 epochs, batch size 1

Tests the complete training pipeline.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from data import DentalXrayDataset, get_train_transforms, get_val_transforms
from models import create_patch_transformer
from training import Trainer
from utils import set_seed, get_device


def main():
    print("="*70)
    print("MINIMAL TRAINING TEST")
    print("="*70)
    
    # Config
    seed = 42
    image_size = (1400, 2800)
    batch_size = 1  # Minimal for speed
    epochs = 2
    
    set_seed(seed)
    device = get_device()
    
    # Datasets
    print("\nLoading datasets...")
    split_file = "outputs/splits/train_val_test_split.json"
    
    train_dataset = DentalXrayDataset(
        root_dir="c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset",
        split='train',
        split_file=split_file,
        transform=get_train_transforms(image_size=image_size)
    )
    
    val_dataset = DentalXrayDataset(
        root_dir="c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset",
        split='val',
        split_file=split_file,
        transform=get_val_transforms(image_size=image_size)
    )
    
    # Take only 10 samples for quick test
    train_dataset.image_paths = train_dataset.image_paths[:10]
    train_dataset.labels = train_dataset.labels[:10]
    val_dataset.image_paths = val_dataset.image_paths[:5]
    val_dataset.labels = val_dataset.labels[:5]
    
    print(f"Train samples: {len(train_dataset)} (subset for testing)")
    print(f"Val samples: {len(val_dataset)} (subset for testing)")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single process for debugging
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Model
    print("\nCreating model...")
    model = create_patch_transformer(
        image_size=image_size,
        patch_size=100,
        model_size='tiny',  # Fastest
        pretrained=True
    )
    
    # Training config
    config = {
        'loss_function': 'combined',
        'optimizer': {
            'name': 'adamw',
            'lr': 1e-4,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'name': 'none'  # No scheduler for quick test
        },
        'use_amp': True,
        'gradient_clip': 1.0,
        'early_stopping_patience': 10,
        'early_stopping_delta': 0.001,
        'epochs': epochs
    }
    
    # Trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir="checkpoints/test_run",
        use_wandb=False  # No WandB for test
    )
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    print("="*70)
    
    try:
        trainer.train(num_epochs=epochs)
        
        print("\n" + "="*70)
        print("✅ TRAINING TEST SUCCESSFUL!")
        print("="*70)
        print(f"\nBest validation F1: {trainer.best_val_f1:.4f}")
        print(f"Best validation Dice: {trainer.best_val_dice:.4f}")
        print("\n✅ Training pipeline is fully functional!")
        print("✅ Ready for full training runs!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
