"""
Training Script for Dental Fracture Detection

Supports both EfficientNet (baseline) and PatchTransformer models.
Features:
- Mixed Precision Training (AMP)
- Gradient Clipping
- Early Stopping
- WandB Logging
- Model Checkpointing
- Learning Rate Scheduling

Author: Master's Thesis Project
Date: October 28, 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
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

from data import DentalXrayDataset, get_train_transforms, get_val_transforms
from models import (
    EfficientNetClassifier, 
    create_efficientnet_classifier,
    PatchTransformerClassifier,
    create_patch_transformer
)
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
        # Convert logits to probabilities
        probs = torch.sigmoid(preds).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        self.predictions.extend(probs.flatten())
        self.targets.extend(targets_np.flatten())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Binary predictions (threshold 0.5)
        binary_preds = (preds >= 0.5).astype(int)
        
        # Confusion matrix elements
        tp = np.sum((binary_preds == 1) & (targets == 1))
        tn = np.sum((binary_preds == 0) & (targets == 0))
        fp = np.sum((binary_preds == 1) & (targets == 0))
        fn = np.sum((binary_preds == 0) & (targets == 1))
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Dice score (same as F1 for binary classification)
        dice = f1
        
        return {
            'loss': np.mean(self.losses),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'dice': dice,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """Training manager for dental fracture detection."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Training components
        self.criterion = get_loss_function(config['loss_function'])
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            min_delta=config.get('early_stopping_delta', 0.001),
            mode='max'  # Maximize validation F1
        )
        
        # Tracking
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.best_val_dice = 0.0
        self.train_history = []
        self.val_history = []
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config.get('optimizer', {})
        opt_name = opt_config.get('name', 'adamw').lower()
        lr = opt_config.get('lr', 1e-4)
        weight_decay = opt_config.get('weight_decay', 0.01)
        
        if opt_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        sched_config = self.config.get('scheduler', {})
        sched_name = sched_config.get('name', 'cosine').lower()
        
        if sched_name == 'none':
            return None
        elif sched_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_name == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 5),
                min_lr=sched_config.get('min_lr', 1e-6)
            )
        elif sched_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 10),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            
            # Mixed precision training
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', 0) > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update metrics
            tracker.update(outputs, targets, loss.item())
            
            # Update progress bar
            if batch_idx % 10 == 0:
                metrics = tracker.compute()
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'acc': f"{metrics['accuracy']:.4f}",
                    'f1': f"{metrics['f1']:.4f}"
                })
        
        return tracker.compute()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        tracker = MetricsTracker()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            tracker.update(outputs, targets, loss.item())
            
            # Update progress bar
            metrics = tracker.compute()
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'f1': f"{metrics['f1']:.4f}"
            })
        
        return tracker.compute()
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_f1': self.best_val_f1,
            'best_val_dice': self.best_val_dice
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (F1: {metrics['f1']:.4f}, Dice: {metrics['dice']:.4f})")
        
        # Save epoch checkpoint (every 5 epochs)
        if (self.current_epoch + 1) % 5 == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{self.current_epoch+1}.pth"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        print(f"📂 Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, resume: bool = False):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume: Whether to resume from checkpoint
        """
        # Resume from checkpoint if requested
        if resume:
            checkpoint_path = self.checkpoint_dir / "latest.pth"
            if checkpoint_path.exists():
                self.load_checkpoint(checkpoint_path)
        
        # Initialize WandB
        if self.use_wandb:
            wandb.watch(self.model, log='all', log_freq=100)
        
        print(f"\n{'='*70}")
        print(f"TRAINING STARTED")
        print(f"{'='*70}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Mixed precision: {self.scaler is not None}")
        print(f"{'='*70}\n")
        
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs} Summary")
            print(f"{'='*70}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"\nTrain Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  F1: {train_metrics['f1']:.4f}")
            print(f"  Dice: {train_metrics['dice']:.4f}")
            print(f"  Precision: {train_metrics['precision']:.4f}")
            print(f"  Recall: {train_metrics['recall']:.4f}")
            print(f"\nValidation Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  F1: {val_metrics['f1']:.4f}")
            print(f"  Dice: {val_metrics['dice']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            print(f"\nConfusion Matrix (Val):")
            print(f"  TP: {val_metrics['tp']}, TN: {val_metrics['tn']}")
            print(f"  FP: {val_metrics['fp']}, FN: {val_metrics['fn']}")
            print(f"{'='*70}\n")
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'lr': current_lr,
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'train/f1': train_metrics['f1'],
                    'train/dice': train_metrics['dice'],
                    'train/precision': train_metrics['precision'],
                    'train/recall': train_metrics['recall'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/f1': val_metrics['f1'],
                    'val/dice': val_metrics['dice'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall']
                })
            
            # Check if best model
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_dice = val_metrics['dice']
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best=is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['f1']):
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"Best F1: {self.best_val_f1:.4f}, Best Dice: {self.best_val_dice:.4f}")
                break
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print(f"Best Validation Dice: {self.best_val_dice:.4f}")
        print(f"{'='*70}\n")
        
        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history,
                'best_val_f1': self.best_val_f1,
                'best_val_dice': self.best_val_dice
            }, f, indent=2)


def main():
    """Main training function."""
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Training config
    train_config = config['training']
    model_config = config['model']
    data_config = config['data']
    
    # Image size
    image_size = config['image'].get('default_size', [1400, 2800])
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)
    
    print(f"Image size: {image_size}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DentalXrayDataset(
        root_dir=data_config['root_dir'],
        split='train',
        transform=get_train_transforms(image_size=image_size)
    )
    
    val_dataset = DentalXrayDataset(
        root_dir=data_config['root_dir'],
        split='val',
        transform=get_val_transforms(image_size=image_size)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    model_name = model_config['name']
    print(f"\nCreating model: {model_name}")
    
    if model_name == 'efficientnet':
        # Extract variant from backbone name (e.g., 'efficientnet_b0' -> 'b0')
        backbone = model_config.get('backbone', 'efficientnet_b0')
        variant = backbone.replace('efficientnet_', '') if 'efficientnet_' in backbone else 'b0'
        
        model = create_efficientnet_classifier(
            variant=variant,
            pretrained=model_config.get('pretrained', True),
            num_classes=model_config.get('num_classes', 1),
            dropout=model_config.get('dropout', 0.3)
        )
    elif model_name == 'patch_transformer':
        model = create_patch_transformer(
            image_size=image_size,
            patch_size=model_config.get('patch_size', 100),
            model_size=model_config.get('size', 'base'),
            pretrained=model_config.get('pretrained', True),
            aggregation=model_config.get('aggregation', 'max')
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/{model_name}_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    with open(f"{checkpoint_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Initialize WandB
    use_wandb = train_config.get('use_wandb', False)
    if use_wandb:
        wandb.init(
            project=train_config.get('wandb_project', 'dental-fracture-detection'),
            name=f"{model_name}_{timestamp}",
            config=config
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb
    )
    
    # Train
    trainer.train(
        num_epochs=train_config['epochs'],
        resume=train_config.get('resume', False)
    )
    
    # Finish WandB
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
