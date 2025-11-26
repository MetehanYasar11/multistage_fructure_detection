"""
Training Script for RCT Classification

Train baseline Patch Transformer on single-tooth RCT dataset.

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import yaml
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.rct_dataset import RCTDataset
from data import get_train_transforms, get_val_transforms
from models import PatchTransformerClassifier, create_patch_transformer
from training import get_loss_function
from utils import set_seed, get_device


class MetricsTracker:
    """Track and compute metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch results."""
        probs = torch.sigmoid(preds).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        self.predictions.extend(probs.flatten())
        self.targets.extend(targets_np.flatten())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        binary_preds = (preds >= 0.5).astype(int)
        
        tp = np.sum((binary_preds == 1) & (targets == 1))
        tn = np.sum((binary_preds == 0) & (targets == 0))
        fp = np.sum((binary_preds == 1) & (targets == 0))
        fn = np.sum((binary_preds == 0) & (targets == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'loss': np.mean(self.losses),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'dice': f1,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    grad_clip: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            outputs = model(images)
            # Ensure output shape matches target
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        metrics.update(outputs, targets, loss.item())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return metrics.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    metrics = MetricsTracker()
    
    pbar = tqdm(dataloader, desc="Validation")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.float().unsqueeze(1).to(device)
        
        with autocast(device_type='cuda'):
            outputs = model(images)
            # Ensure output shape matches target
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, targets)
        
        metrics.update(outputs, targets, loss.item())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return metrics.compute()


def main():
    """Main training loop."""
    # Load config
    with open('config_rct.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("RCT CLASSIFICATION TRAINING")
    print("="*80)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Run: {config['run_name']}")
    
    # Set seed
    set_seed(config['seed'])
    device = get_device()
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = RCTDataset(
        root_dir=config['dataset']['root_dir'],
        split='train',
        transform=get_train_transforms(config['dataset']['image_size']),
        image_size=config['dataset']['image_size']
    )
    
    val_dataset = RCTDataset(
        root_dir=config['dataset']['root_dir'],
        split='val',
        transform=get_val_transforms(config['dataset']['image_size']),
        image_size=config['dataset']['image_size']
    )
    
    # Create weighted sampler for class imbalance
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_patch_transformer(
        num_patches_h=config['model']['num_patches_h'],
        num_patches_w=config['model']['num_patches_w'],
        patch_size=config['model']['patch_size'],
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        d_model=config['model']['transformer']['d_model'],
        nhead=config['model']['transformer']['nhead'],
        num_layers=config['model']['transformer']['num_layers'],
        dim_feedforward=config['model']['transformer']['dim_feedforward'],
        dropout=config['model']['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = get_loss_function(
        loss_type=config['training']['loss']['type'],
        alpha=config['training']['loss'].get('alpha', 0.75),
        gamma=config['training']['loss'].get('gamma', 2.0),
        pos_weight=torch.tensor([config['training']['loss'].get('pos_weight', 1.0)]).to(device)
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=config['training']['optimizer']['betas']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['training']['scheduler']['mode'],
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience'],
        min_lr=config['training']['scheduler']['min_lr']
    )
    
    # AMP scaler
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    # Training history
    history = {
        'train': [],
        'val': []
    }
    
    best_f1 = 0.0
    patience_counter = 0
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, config['training']['grad_clip']
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['f1'])
        
        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        # Save checkpoint
        if epoch % config['checkpointing']['save_freq'] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'history': history,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_path = output_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, best_path)
            print(f"Best model saved! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"\nEarly stopping triggered! No improvement for {patience_counter} epochs.")
            break
    
    # Save final history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
