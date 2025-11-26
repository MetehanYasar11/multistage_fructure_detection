"""
Training Script for Patch Transformer with Localization

Train the enhanced model with multi-task learning:
- Global classification (image-level)
- Patch localization (weakly-supervised)

Key Features:
- Multi-task loss with MIL
- Patch variance tracking
- Spatial heatmap visualization during training
- Compare with baseline model

Author: Master's Thesis Project
Date: October 28, 2025
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.patch_transformer_localization import PatchTransformerWithLocalization
from training.loss_localization import MultiTaskLocalizationLoss
from data.dataset import DentalXrayDataset
from evaluation.metrics import compute_metrics


class LocalizationTrainer:
    """
    Trainer for patch localization model.
    
    Tracks both global metrics (F1, accuracy) and patch metrics (variance, entropy).
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training'].get('optimizer', {}).get('lr', 3e-4),  # Increased from 1e-4
            weight_decay=config['training'].get('optimizer', {}).get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=7  # Increased from 5 - be more patient
        )
        
        # AMP
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Tracking
        self.best_f1 = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'patch_variance': [],  # Track patch variance over epochs
            'patch_entropy': []    # Track patch prediction diversity
        }
        
        # Output directory
        output_base = config['training'].get('output_dir', 'outputs')
        self.output_dir = Path(output_base) / 'localization_model'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"LOCALIZATION TRAINER INITIALIZED")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")
    
    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_global_loss = 0.0
        total_patch_loss = 0.0
        total_diversity_loss = 0.0
        
        all_preds = []
        all_targets = []
        all_patch_variances = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast(device_type='cuda', enabled=self.use_amp):
                output = self.model(images)
                losses = self.criterion(output, targets)
                loss = losses['total_loss']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track losses
            total_loss += loss.item()
            total_global_loss += losses['global_loss'].item()
            total_patch_loss += losses['patch_loss'].item()
            total_diversity_loss += losses['diversity_loss'].item()
            
            # Track predictions
            global_preds = torch.sigmoid(output['global_logits']).detach().cpu().numpy()
            all_preds.extend(global_preds.flatten())
            all_targets.extend(targets.cpu().numpy())
            
            # Track patch variance (spatial diversity)
            patch_probs = output['patch_probs'].detach().cpu().numpy()
            patch_var = np.var(patch_probs, axis=1).mean()  # Mean variance across batch
            all_patch_variances.append(patch_var)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'patch_var': f"{patch_var:.4f}"
            })
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        metrics = compute_metrics(all_targets, all_preds, threshold=0.5)
        
        epoch_results = {
            'total_loss': total_loss / len(self.train_loader),
            'global_loss': total_global_loss / len(self.train_loader),
            'patch_loss': total_patch_loss / len(self.train_loader),
            'diversity_loss': total_diversity_loss / len(self.train_loader),
            'f1': metrics['f1_score'],
            'accuracy': metrics['accuracy'],
            'patch_variance': np.mean(all_patch_variances)
        }
        
        return epoch_results
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_global_loss = 0.0
        total_patch_loss = 0.0
        
        all_preds = []
        all_targets = []
        all_patch_variances = []
        all_patch_entropies = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for (images, targets) in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                with autocast(device_type='cuda', enabled=self.use_amp):
                    output = self.model(images)
                    losses = self.criterion(output, targets)
                    loss = losses['total_loss']
                
                # Track losses
                total_loss += loss.item()
                total_global_loss += losses['global_loss'].item()
                total_patch_loss += losses['patch_loss'].item()
                
                # Track predictions
                global_preds = torch.sigmoid(output['global_logits']).detach().cpu().numpy()
                all_preds.extend(global_preds.flatten())
                all_targets.extend(targets.cpu().numpy())
                
                # Track patch statistics
                patch_probs = output['patch_probs'].detach().cpu().numpy()
                
                # Variance (spatial diversity)
                patch_var = np.var(patch_probs, axis=1).mean()
                all_patch_variances.append(patch_var)
                
                # Entropy (prediction uncertainty)
                eps = 1e-7
                patch_entropy = -(
                    patch_probs * np.log(patch_probs + eps) +
                    (1 - patch_probs) * np.log(1 - patch_probs + eps)
                )
                all_patch_entropies.append(patch_entropy.mean())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'patch_var': f"{patch_var:.4f}"
                })
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        metrics = compute_metrics(all_targets, all_preds, threshold=0.5)
        
        epoch_results = {
            'total_loss': total_loss / len(self.val_loader),
            'global_loss': total_global_loss / len(self.val_loader),
            'patch_loss': total_patch_loss / len(self.val_loader),
            'f1': metrics['f1_score'],
            'accuracy': metrics['accuracy'],
            'dice': metrics['dice_score'],
            'patch_variance': np.mean(all_patch_variances),
            'patch_entropy': np.mean(all_patch_entropies)
        }
        
        return epoch_results
    
    def visualize_patches(self, epoch: int, num_samples: int = 4):
        """Visualize patch heatmaps during training."""
        self.model.eval()
        
        # Get a batch from validation - unpack tuple
        images, targets = next(iter(self.val_loader))
        images = images[:num_samples].to(self.device)
        targets = targets[:num_samples].cpu().numpy()
        
        with torch.no_grad():
            heatmap, patch_probs = self.model.get_patch_heatmap(images)
        
        heatmap = heatmap.cpu().numpy()
        
        # Plot
        fig, axes = plt.subplots(2, num_samples, figsize=(16, 8))
        
        for i in range(num_samples):
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"{'Fractured' if targets[i] == 1 else 'Healthy'}")
            axes[0, i].axis('off')
            
            # Heatmap
            sns.heatmap(
                heatmap[i],
                ax=axes[1, i],
                cmap='RdYlGn_r',
                vmin=0,
                vmax=1,
                cbar=True,
                square=True
            )
            var = np.var(heatmap[i])
            axes[1, i].set_title(f"Variance: {var:.4f}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'heatmap_epoch_{epoch:03d}.png', dpi=150)
        plt.close()
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\n{'='*80}")
        print(f"STARTING TRAINING - {num_epochs} EPOCHS")
        print(f"{'='*80}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_results = self.train_epoch(epoch)
            
            # Validate
            val_results = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_results['total_loss'])
            self.history['val_loss'].append(val_results['total_loss'])
            self.history['train_f1'].append(train_results['f1'])
            self.history['val_f1'].append(val_results['f1'])
            self.history['patch_variance'].append(val_results['patch_variance'])
            self.history['patch_entropy'].append(val_results['patch_entropy'])
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_results['total_loss']:.4f}, F1: {train_results['f1']:.4f}")
            print(f"  Val   - Loss: {val_results['total_loss']:.4f}, F1: {val_results['f1']:.4f}")
            print(f"  Patch - Var: {val_results['patch_variance']:.4f}, Entropy: {val_results['patch_entropy']:.4f}")
            
            # LR scheduler
            self.scheduler.step(val_results['f1'])
            
            # Save best model
            if val_results['f1'] > self.best_f1:
                self.best_f1 = val_results['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1': self.best_f1,
                    'config': self.config
                }, self.output_dir / 'best_model.pth')
                print(f"  ✅ Best model saved (F1: {self.best_f1:.4f})")
            
            # Visualize patches every 5 epochs
            if epoch % 5 == 0:
                self.visualize_patches(epoch)
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }, self.output_dir / f'checkpoint_epoch_{epoch:03d}.pth')
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED!")
        print(f"Best F1: {self.best_f1:.4f}")
        print(f"{'='*80}\n")
        
        # Save final history
        np.save(self.output_dir / 'history.npy', self.history)
        
        # Plot training curves
        self.plot_history()
    
    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # F1
        axes[0, 1].plot(self.history['train_f1'], label='Train')
        axes[0, 1].plot(self.history['val_f1'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Patch Variance
        axes[1, 0].plot(self.history['patch_variance'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Patch Variance')
        axes[1, 0].set_title('Patch Prediction Variance')
        axes[1, 0].grid(True)
        
        # Patch Entropy
        axes[1, 1].plot(self.history['patch_entropy'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Patch Entropy')
        axes[1, 1].set_title('Patch Prediction Entropy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        plt.close()


def main():
    """Main training function."""
    
    # Load config
    config_path = project_root / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    
    # Image size from config
    image_size = config.get('image', {}).get('default_size', [1400, 2800])
    print(f"Using image size: {image_size}")
    
    # Get split file path
    split_file = config.get('data', {}).get('split_file', 'outputs/splits/train_val_test_split.json')
    print(f"Using split file: {split_file}")
    
    # Dataset root - direct absolute path (config has c:/ which is already absolute)
    root_dir = r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset"
    print(f"Using root dir: {root_dir}")
    print(f"Root dir exists: {Path(root_dir).exists()}")
    
    train_dataset = DentalXrayDataset(
        root_dir=root_dir,
        split='train',
        transform=None,  # Will use dataset's default
        image_size=image_size[0],  # Use height
        use_clahe=config.get('image', {}).get('apply_clahe', False),
        split_file=split_file
    )
    
    val_dataset = DentalXrayDataset(
        root_dir=root_dir,
        split='val',
        transform=None,  # Will use dataset's default
        image_size=image_size[0],  # Use height
        use_clahe=config.get('image', {}).get('apply_clahe', False),
        split_file=split_file
    )
    
    # Create weighted sampler for class balance
    # Calculate sample weights: give more weight to minority class (healthy)
    train_labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    print(f"\nClass distribution: Fractured={class_counts[1]}, Healthy={class_counts[0]}")
    
    # Weight for each class: inverse of frequency
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 8),  # Increased from 4 to 8
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=0,  # Windows compatibility
        pin_memory=True,
        persistent_workers=False  # Can't use with num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 8),  # Increased from 4 to 8
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    image_size = config.get('image', {}).get('default_size', [1400, 2800])
    
    # Model parameters
    model_config = config.get('model', {})
    patch_size = model_config.get('patch_size', 100)
    cnn_backbone = model_config.get('backbone', 'resnet18')
    dropout = model_config.get('dropout', 0.1)
    
    # Transformer parameters (defaults for base model)
    feature_dim = 512
    num_heads = 8
    num_layers = 6
    
    model = PatchTransformerWithLocalization(
        image_size=tuple(image_size),
        patch_size=patch_size,
        cnn_backbone=cnn_backbone,
        feature_dim=feature_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_global_head=True
    ).to(device)
    
    # Create loss
    print("\nCreating loss function...")
    criterion = MultiTaskLocalizationLoss(
        global_weight=1.0,
        patch_weight=1.0,  # Increased from 0.5 - patch learning is important
        diversity_weight=0.0,  # Disabled - let model learn naturally
        use_focal_loss=True,
        focal_alpha=0.75,
        focal_gamma=2.0
    )
    
    # Create trainer
    trainer = LocalizationTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    num_epochs = config['training'].get('epochs', 50)
    trainer.train(num_epochs)
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
