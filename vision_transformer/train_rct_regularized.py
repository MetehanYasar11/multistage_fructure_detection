"""
Regularized Training for Dynamic Patch Transformer

Key improvements to fix overfitting:
1. Smaller model (256 dim, 4 layers vs 512 dim, 6 layers)
2. Heavy dropout (0.3 vs 0.1)
3. Strong weight decay (0.05 vs 0.01)
4. Better data augmentation
5. Lower learning rate

Previous: Train F1=0.99, Val F1=0.43 (overfitting!)
Target: Train F1=0.85, Val F1=0.75+ (good generalization)

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
import json
import math
import random

sys.path.append('.')

from data.rct_dataset import RCTDataset
from utils import set_seed, get_device
from models.dynamic_patch_transformer import DynamicPatchTransformer


# Simple augmentation functions (without albumentations)
def augment_image(image_tensor, training=True):
    """
    Apply augmentation to image tensor
    
    Args:
        image_tensor: (C, H, W) tensor
        training: If True, apply augmentation
        
    Returns:
        Augmented tensor
    """
    if not training:
        return image_tensor
    
    # Random horizontal flip
    if random.random() > 0.5:
        image_tensor = torch.flip(image_tensor, dims=[2])
    
    # Random vertical flip
    if random.random() > 0.5:
        image_tensor = torch.flip(image_tensor, dims=[1])
    
    # Random brightness
    if random.random() > 0.3:
        brightness_factor = 1.0 + random.uniform(-0.3, 0.3)
        image_tensor = torch.clamp(image_tensor * brightness_factor, 0, 1)
    
    # Random contrast
    if random.random() > 0.3:
        contrast_factor = 1.0 + random.uniform(-0.3, 0.3)
        mean = image_tensor.mean()
        image_tensor = torch.clamp((image_tensor - mean) * contrast_factor + mean, 0, 1)
    
    # Random noise
    if random.random() > 0.6:
        noise = torch.randn_like(image_tensor) * 0.02
        image_tensor = torch.clamp(image_tensor + noise, 0, 1)
    
    return image_tensor


class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds, targets, loss):
        probs = torch.sigmoid(preds).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        self.predictions.extend(probs.flatten())
        self.targets.extend(targets_np.flatten())
        self.losses.append(loss)
    
    def compute(self):
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
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'loss': np.mean(self.losses),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=1.0, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        targets = targets.float()
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets_smooth, reduction='none'
        )
        
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_weight = torch.where(targets == 1, self.alpha * self.pos_weight, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()


def collate_fn_variable_size(batch):
    images = []
    labels = []
    
    for img, label in batch:
        images.append(img)
        labels.append(label)
    
    labels = torch.tensor(labels, dtype=torch.float32)
    return images, labels


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, grad_clip=0.5):
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        batch_outputs = []
        batch_targets = []
        
        for img, target in zip(images, targets):
            # Apply augmentation
            img = augment_image(img, training=True)
            img = img.unsqueeze(0).to(device)
            batch_targets.append(target)
            
            with autocast(device_type='cuda'):
                output = model(img)
                batch_outputs.append(output)
        
        outputs = torch.cat(batch_outputs)
        targets = torch.stack(batch_targets).to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
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
def validate(model, dataloader, criterion, device):
    model.eval()
    metrics = MetricsTracker()
    
    pbar = tqdm(dataloader, desc="Validation")
    for images, targets in pbar:
        batch_outputs = []
        batch_targets = []
        
        for img, target in zip(images, targets):
            img = img.unsqueeze(0).to(device)
            batch_targets.append(target)
            
            with autocast(device_type='cuda'):
                output = model(img)
                batch_outputs.append(output)
        
        outputs = torch.cat(batch_outputs)
        targets = torch.stack(batch_targets).to(device)
        
        with autocast(device_type='cuda'):
            loss = criterion(outputs, targets)
        
        metrics.update(outputs, targets, loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return metrics.compute()


def main():
    with open('config_rct_dynamic_v2.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("REGULARIZED DYNAMIC PATCH TRANSFORMER TRAINING")
    print("="*80)
    print("Fix: Overfitting (Train F1=0.99, Val F1=0.43)")
    print("Solution: Smaller model + Heavy regularization + Augmentation")
    print("="*80)
    
    set_seed(config['seed'])
    device = get_device()
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("\nLoading datasets...")
    train_dataset = RCTDataset(
        root_dir=config['dataset']['root_dir'],
        split='train',
        transform=None,
        image_size=None
    )
    
    val_dataset = RCTDataset(
        root_dir=config['dataset']['root_dir'],
        split='val',
        transform=None,
        image_size=None
    )
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_variable_size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_variable_size
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    print("\nCreating SMALLER model (to prevent overfitting)...")
    model = DynamicPatchTransformer(
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        min_patch_size=config['model']['min_patch_size'],
        target_patch_density=config['model']['target_patch_density']
    ).to(device)
    
    criterion = FocalLossWithLabelSmoothing(
        alpha=config['training']['loss']['alpha'],
        gamma=config['training']['loss']['gamma'],
        pos_weight=config['training']['loss']['pos_weight'],
        smoothing=config['training']['loss']['label_smoothing']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=config['training']['optimizer']['betas']
    )
    
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config['training']['scheduler']['warmup_epochs'],
        total_epochs=config['training']['epochs'],
        min_lr=config['training']['scheduler']['min_lr'],
        base_lr=config['training']['optimizer']['lr']
    )
    
    scaler = GradScaler()
    
    history = {'train': [], 'val': []}
    best_f1 = 0.0
    patience_counter = 0
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print("-" * 40)
        
        current_lr = scheduler.step()
        print(f"Learning rate: {current_lr:.6f}")
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, config['training']['grad_clip']
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"Gap   - Train-Val F1: {train_metrics['f1'] - val_metrics['f1']:.4f}")
        
        if epoch % config['logging']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics
            }, output_dir / f"checkpoint_epoch_{epoch:03d}.pth")
            print(f"Checkpoint saved")
        
        if val_metrics['f1'] > best_f1:
            improvement = val_metrics['f1'] - best_f1
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics
            }, output_dir / "best_model.pth")
            print(f"[*] Best model saved! Val F1: {best_f1:.4f} (+{improvement:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if val_metrics['f1'] >= 0.75 and (train_metrics['f1'] - val_metrics['f1']) < 0.15:
            print(f"\n{'='*80}")
            print(f"GOOD GENERALIZATION! Val F1: {val_metrics['f1']:.4f}, Gap: {train_metrics['f1'] - val_metrics['f1']:.4f}")
            print(f"{'='*80}")
        
        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"\nEarly stopping triggered! No improvement for {patience_counter} epochs.")
            break
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Best Val F1 Score: {best_f1:.4f}")
    print(f"Target (>0.75): {'[ACHIEVED]' if best_f1 >= 0.75 else '[NOT REACHED]'}")
    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
