"""
Simple CNN Training for RCT Classification

Try simpler model - maybe Patch Transformer is too complex!

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

sys.path.append('.')

from data.rct_dataset import RCTDataset
from data import get_train_transforms, get_val_transforms
from utils import set_seed, get_device
import timm


class SimpleCNN(nn.Module):
    """Simple ResNet18 classifier"""
    
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1)
        )
        
        print(f"Simple CNN: ResNet18 + 2-layer head")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(1) if logits.dim() == 2 else logits


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


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, grad_clip=1.0):
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.float().to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            outputs = model(images)
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
        images = images.to(device)
        targets = targets.float().to(device)
        
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        metrics.update(outputs, targets, loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return metrics.compute()


def main():
    # Load config
    with open('config_rct_simple.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("RCT SIMPLE CNN TRAINING")
    print("="*80)
    
    set_seed(config['seed'])
    device = get_device()
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Datasets
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
    
    # Weighted sampler
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model
    print("\nCreating model...")
    model = SimpleCNN(
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Loss
    pos_weight = torch.tensor([config['training']['loss']['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-7
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
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, config['training']['grad_clip']
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['f1'])
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics
            }, output_dir / f"checkpoint_epoch_{epoch:03d}.pth")
            print(f"Checkpoint saved")
        
        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics
            }, output_dir / "best_model.pth")
            print(f"Best model saved! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            print(f"\nEarly stopping!")
            break
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
