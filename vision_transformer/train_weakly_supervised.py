"""
Training Script for Weakly-Supervised Patch Transformer

Multi-task learning with:
1. Classification loss (all samples): Fractured vs Healthy
2. Localization loss (fractured with GT): Attention map vs bbox

Key Features:
- Loads baseline model weights (transfer learning)
- Dual loss function with configurable weights (α, β)
- Tracks both classification (F1) and localization (IoU) metrics
- Saves best model based on combined metric

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not installed. Running without W&B logging.")

from models.weakly_supervised_patch_transformer import WeaklySupervisedPatchTransformer


def custom_collate_fn(batch):
    """
    Custom collate function to handle None bbox_masks.
    
    Args:
        batch: List of (image, label, bbox_mask) tuples
        
    Returns:
        images: [B, C, H, W] tensor
        labels: [B] list of labels
        bbox_masks: [B] list of masks (can contain None)
    """
    images = []
    labels = []
    bbox_masks = []
    
    for img, label, bbox_mask in batch:
        images.append(img)
        labels.append(label)
        bbox_masks.append(bbox_mask)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, labels, bbox_masks


class DentalFractureDatasetWithBbox(Dataset):
    """
    Dataset that loads images with both class labels and bbox annotations.
    
    For fractured images with .txt files:
        - Returns image, label=1, and bbox from .txt
    For fractured images without .txt:
        - Returns image, label=1, bbox=None
    For healthy images:
        - Returns image, label=0, bbox=None
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str,
        split_file: str,
        image_size: Tuple[int, int] = (1400, 2800),
        use_clahe: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.use_clahe = use_clahe
        
        # Load all images
        self.image_paths = []
        self.labels = []
        
        # Load Fractured
        fractured_dir = self.root_dir / "Fractured"
        if fractured_dir.exists():
            for img_file in fractured_dir.glob('*.jpg'):
                self.image_paths.append(str(img_file))
                self.labels.append(1)
        
        # Load Healthy
        healthy_dir = self.root_dir / "Healthy"
        if healthy_dir.exists():
            for img_file in healthy_dir.glob('*.jpg'):
                self.image_paths.append(str(img_file))
                self.labels.append(0)
        
        # Apply split filter
        with open(split_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        
        split_indices = splits[split]
        self.image_paths = [self.image_paths[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]
        
        print(f"\n[*] Loaded {split} dataset:")
        print(f"    Total: {len(self.image_paths)} images")
        print(f"    Fractured: {sum(self.labels)} | Healthy: {len(self.labels) - sum(self.labels)}")
        
        # Count how many have bbox annotations
        num_with_bbox = 0
        for img_path, label in zip(self.image_paths, self.labels):
            if label == 1:  # Fractured
                txt_path = Path(img_path).with_suffix('.txt')
                if txt_path.exists():
                    num_with_bbox += 1
        print(f"    With bbox annotations: {num_with_bbox}")
    
    def load_bbox(self, image_path: str) -> Optional[List[Tuple[int, int]]]:
        """Load bbox points from .txt file"""
        txt_path = Path(image_path).with_suffix('.txt')
        
        if not txt_path.exists():
            return None
        
        points = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append((int(x), int(y)))
        
        return points if len(points) > 0 else None
    
    def create_bbox_mask(
        self, 
        bbox_points: List[Tuple[int, int]], 
        orig_size: Tuple[int, int],
        grid_size: Tuple[int, int],
        margin: int = 50
    ) -> torch.Tensor:
        """
        Create a binary mask for the bbox region on the patch grid.
        
        Args:
            bbox_points: List of (x, y) coordinates
            orig_size: (orig_h, orig_w) of original image
            grid_size: (grid_h, grid_w) of patch grid
            margin: Pixels to expand bbox
        
        Returns:
            [grid_h, grid_w] binary mask
        """
        # Calculate bbox from points
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        
        x_min = max(0, min(xs) - margin)
        x_max = min(orig_size[1], max(xs) + margin)
        y_min = max(0, min(ys) - margin)
        y_max = min(orig_size[0], max(ys) + margin)
        
        # Scale to target image size
        orig_h, orig_w = orig_size
        target_h, target_w = self.image_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        x_min_scaled = int(x_min * scale_x)
        x_max_scaled = int(x_max * scale_x)
        y_min_scaled = int(y_min * scale_y)
        y_max_scaled = int(y_max * scale_y)
        
        # Convert to patch grid coordinates
        grid_h, grid_w = grid_size
        patch_h = target_h // grid_h
        patch_w = target_w // grid_w
        
        patch_x_min = x_min_scaled // patch_w
        patch_x_max = min(grid_w - 1, x_max_scaled // patch_w)
        patch_y_min = y_min_scaled // patch_h
        patch_y_max = min(grid_h - 1, y_max_scaled // patch_h)
        
        # Create mask
        mask = torch.zeros(grid_h, grid_w)
        mask[patch_y_min:patch_y_max+1, patch_x_min:patch_x_max+1] = 1.0
        
        return mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        with open(img_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Failed to load: {img_path}")
        
        orig_h, orig_w = img.shape[:2]
        
        # Apply CLAHE
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        # Resize
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # To tensor [C, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        # Load bbox if available
        bbox_mask = None
        if label == 1:  # Fractured
            bbox_points = self.load_bbox(img_path)
            if bbox_points is not None:
                bbox_mask = self.create_bbox_mask(
                    bbox_points,
                    orig_size=(orig_h, orig_w),
                    grid_size=(14, 28),  # Hard-coded for now
                    margin=50
                )
        
        return img_tensor, label, bbox_mask


def focal_loss_with_logits(pred_logits, target, gamma=2.0, pos_weight=3.0):
    """
    Focal loss for binary classification with logits.
    Focuses on hard examples and penalizes false negatives (missing GT patches).
    
    Args:
        pred_logits: [B, H, W] predicted logits
        target: [B, H, W] binary target (0 or 1)
        gamma: Focusing parameter (default 2.0)
        pos_weight: Weight for positive class (default 3.0 - moderate penalty)
    """
    # Get probabilities
    pred_prob = torch.sigmoid(pred_logits)
    
    # For positive samples (target=1): -pos_weight * (1-p)^gamma * log(p)
    # For negative samples (target=0): -(p)^gamma * log(1-p)
    pos_loss = -pos_weight * ((1 - pred_prob) ** gamma) * torch.log(pred_prob + 1e-8)
    neg_loss = -(pred_prob ** gamma) * torch.log(1 - pred_prob + 1e-8)
    
    # Combine
    loss = target * pos_loss + (1 - target) * neg_loss
    
    return loss.mean()


def contrastive_attention_loss(pred_logits, target, margin=0.5):
    """
    Contrastive loss that FORCES discrimination between GT and non-GT patches.
    
    Instead of just penalizing low attention on GT patches, this loss:
    1. Pushes GT patches to have HIGH attention (> 0.5 + margin)
    2. Pushes non-GT patches to have LOW attention (< 0.5 - margin)
    3. Creates a clear separation between GT and background
    
    This is more effective than BCE for learning discriminative attention maps.
    
    Args:
        pred_logits: [B, 1, H, W] predicted logits (before sigmoid)
        target: [B, 1, H, W] binary target (0 or 1)
        margin: Margin for separation (default 0.5 means GT > 0.5, background < 0.0)
    
    Returns:
        Scalar loss value
    """
    # Apply sigmoid to get probabilities
    pred_prob = torch.sigmoid(pred_logits)
    
    # Separate GT and non-GT patches
    gt_mask = (target > 0.5)
    bg_mask = (target <= 0.5)
    
    num_gt = gt_mask.sum().item()
    num_bg = bg_mask.sum().item()
    
    if num_gt == 0:
        return torch.tensor(0.0, device=pred_logits.device)
    
    # Loss for GT patches: push above 0.5 + margin
    gt_probs = pred_prob[gt_mask]
    gt_target = 0.5 + margin  # Want prob > 0.75
    gt_loss = torch.relu(gt_target - gt_probs).mean()  # Penalize if below target
    
    # Loss for background patches: push below 0.5 - margin  
    if num_bg > 0:
        bg_probs = pred_prob[bg_mask]
        bg_target = 0.5 - margin  # Want prob < 0.25
        bg_loss = torch.relu(bg_probs - bg_target).mean()  # Penalize if above target
    else:
        bg_loss = torch.tensor(0.0, device=pred_logits.device)
    
    # Combine: equal weight to GT and background
    # This forces the model to learn discrimination, not just uniform attention
    total_loss = gt_loss + bg_loss
    
    return total_loss


def combined_loss(
    logits: torch.Tensor,
    attention_map: torch.Tensor,
    labels: torch.Tensor,
    bbox_masks: List[Optional[torch.Tensor]],
    alpha: float = 0.7,
    beta: float = 0.3,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-task loss: Classification + Localization
    
    Args:
        logits: [B, 2] - classification logits
        attention_map: [B, 1, H, W] - predicted attention
        labels: [B] - class labels
        bbox_masks: List of [H, W] masks or None
        alpha: Weight for classification loss
        beta: Weight for localization loss
    
    Returns:
        total_loss, loss_dict
    """
    # Classification loss (all samples)
    cls_loss = F.cross_entropy(logits, labels)
    
    # Localization loss (only fractured with bbox)
    loc_loss = torch.tensor(0.0, device=device)
    num_with_bbox = 0
    
    # DEBUG: Check how many fractured samples and bbox masks we have
    num_fractured = sum(1 for label in labels if label == 1)
    num_bbox_not_none = sum(1 for mask in bbox_masks if mask is not None)
    
    for i, (label, bbox_mask) in enumerate(zip(labels, bbox_masks)):
        if label == 1 and bbox_mask is not None:
            # Move bbox_mask to device
            bbox_mask = bbox_mask.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            pred_attention = attention_map[i:i+1]  # [1, 1, H, W]
            
            # Check if bbox has any positive values
            if bbox_mask.sum() == 0:
                # DEBUG: This should NEVER happen - all fractured images have bbox
                print(f"[WARNING] Empty bbox mask for fractured sample {i} in batch!")
                continue  # Skip empty masks
            
            # Contrastive loss - forces discrimination between GT and background patches
            # This pushes GT patches HIGH and background patches LOW
            # margin=0.35 means: GT > 0.85, background < 0.15 (strong separation)
            sample_loss = contrastive_attention_loss(pred_attention, bbox_mask, margin=0.35)
            
            # Sanity check
            if not torch.isnan(sample_loss) and not torch.isinf(sample_loss):
                loc_loss += sample_loss
                num_with_bbox += 1
    
    if num_with_bbox > 0:
        loc_loss = loc_loss / num_with_bbox
    else:
        # No valid bbox in this batch - use small penalty to encourage exploration
        loc_loss = torch.tensor(0.0, device=device)
    
    # Combined loss
    total_loss = alpha * cls_loss + beta * loc_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'classification': cls_loss.item(),
        'localization': loc_loss.item() if num_with_bbox > 0 else 0.0,
        'num_with_bbox': num_with_bbox
    }
    
    return total_loss, loss_dict


def calculate_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Calculate IoU between predicted and GT masks"""
    # Apply sigmoid since model outputs logits now
    pred_prob = torch.sigmoid(pred_mask)
    pred_binary = (pred_prob > 0.5).float()
    
    intersection = (pred_binary * gt_mask).sum()
    union = pred_binary.sum() + gt_mask.sum() - intersection
    
    if union == 0:
        return 0.0
    
    return (intersection / union).item()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    alpha: float,
    beta: float
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    Returns both classification and localization metrics.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_ious = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels, bbox_masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            
            # Forward pass
            logits, attention_map = model(images, return_attention=True)
            
            # Loss
            loss, _ = combined_loss(logits, attention_map, labels_tensor, bbox_masks, alpha, beta, device)
            total_loss += loss.item()
            
            # Classification predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Localization IoU (for fractured with bbox)
            for i, (label, bbox_mask) in enumerate(zip(labels, bbox_masks)):
                if label == 1 and bbox_mask is not None:
                    pred_attention = attention_map[i, 0].cpu()  # [H, W]
                    iou = calculate_iou(pred_attention, bbox_mask)
                    all_ious.append(iou)
    
    # Classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Localization metrics
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    median_iou = np.median(all_ious) if len(all_ious) > 0 else 0.0
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'num_iou_samples': len(all_ious)
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    alpha: float,
    beta: float,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0
    total_with_bbox = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for images, labels, bbox_masks in pbar:
        images = images.to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        
        # Forward pass
        logits, attention_map = model(images, return_attention=True)
        
        # Calculate loss
        loss, loss_dict = combined_loss(
            logits, attention_map, labels_tensor, bbox_masks,
            alpha, beta, device
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate
        total_loss += loss_dict['total']
        total_cls_loss += loss_dict['classification']
        total_loc_loss += loss_dict['localization']
        total_with_bbox += loss_dict['num_with_bbox']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'cls': f"{loss_dict['classification']:.4f}",
            'loc': f"{loss_dict['localization']:.4f}",
            'n_bbox': loss_dict['num_with_bbox']
        })
    
    num_batches = len(dataloader)
    
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'loc_loss': total_loc_loss / num_batches,
        'avg_with_bbox': total_with_bbox / num_batches
    }


def train_weakly_supervised(
    config_path: str = "config.yaml",
    baseline_checkpoint: str = "checkpoints/patch_transformer_full/best.pth",
    alpha: float = 0.7,
    beta: float = 0.3,
    num_epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    use_wandb: bool = False
):
    """
    Main training function for weakly-supervised model.
    
    Args:
        config_path: Path to config.yaml
        baseline_checkpoint: Path to baseline model weights
        alpha: Weight for classification loss
        beta: Weight for localization loss
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        use_wandb: Whether to use Weights & Biases logging
    """
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("WEAKLY-SUPERVISED PATCH TRANSFORMER TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Alpha (cls): {alpha} | Beta (loc): {beta}")
    print(f"Batch size: {batch_size} | Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    
    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="dental-fracture-detection",
            name=f"weakly_supervised_a{alpha}_b{beta}",
            config={
                'alpha': alpha,
                'beta': beta,
                'batch_size': batch_size,
                'lr': lr,
                'num_epochs': num_epochs
            }
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("[WARNING] W&B logging requested but wandb not installed. Skipping.")
    
    # Create datasets
    train_dataset = DentalFractureDatasetWithBbox(
        root_dir=config['data']['root_dir'],
        split='train',
        split_file=config['data']['split_file'],
        image_size=tuple(config['image']['default_size']),
        use_clahe=config['image']['apply_clahe']
    )
    
    val_dataset = DentalFractureDatasetWithBbox(
        root_dir=config['data']['root_dir'],
        split='val',
        split_file=config['data']['split_file'],
        image_size=tuple(config['image']['default_size']),
        use_clahe=config['image']['apply_clahe']
    )
    
    # Create dataloaders with custom collate function
    # num_workers=0 to avoid multiprocessing issues on Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Create model
    model = WeaklySupervisedPatchTransformer(
        image_size=tuple(config['image']['default_size']),
        patch_size=config['model']['patch_size'],
        feature_dim=config['model'].get('feature_dim', 512),
        num_heads=config['model'].get('num_heads', 8),
        num_layers=config['model'].get('num_layers', 6),
        dropout=config['model']['dropout']
    )
    
    # Load baseline weights
    print(f"\n{'='*70}")
    model.load_from_baseline(baseline_checkpoint)
    print(f"{'='*70}\n")
    
    model = model.to(device)
    
    # Optimizer with separate learning rates
    # Higher LR for attention head (new), lower LR for pretrained parts
    attention_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'attention_head' in name:
            attention_params.append(param)
        else:
            other_params.append(param)
    
    print(f"\n[*] Optimizer setup:")
    print(f"    Attention head params: {len(attention_params)} tensors")
    print(f"    Other params: {len(other_params)} tensors")
    print(f"    Attention head LR: {lr * 5:.6f} (5x higher)")
    print(f"    Base model LR: {lr:.6f}")
    
    optimizer = optim.AdamW([
        {'params': attention_params, 'lr': lr * 5},  # 5x higher for attention head (more conservative)
        {'params': other_params, 'lr': lr}
    ], weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_f1 = 0.0
    best_combined_metric = 0.0
    
    output_dir = Path("checkpoints/weakly_supervised_patch_transformer")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_iou': []
    }
    
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, alpha, beta, epoch
        )
        
        print(f"\n[TRAIN] Loss: {train_metrics['loss']:.4f} "
              f"(Cls: {train_metrics['cls_loss']:.4f}, Loc: {train_metrics['loc_loss']:.4f})")
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device, alpha, beta)
        
        print(f"[VAL] Loss: {val_metrics['loss']:.4f}")
        print(f"[VAL] F1: {val_metrics['f1']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"[VAL] IoU: {val_metrics['mean_iou']:.4f} (median: {val_metrics['median_iou']:.4f}, n={val_metrics['num_iou_samples']})")
        
        # Combined metric: weighted by alpha/beta
        combined_metric = alpha * val_metrics['f1'] + beta * val_metrics['mean_iou']
        print(f"[VAL] Combined Metric: {combined_metric:.4f} ({alpha}*F1 + {beta}*IoU)")
        
        # Show improvement
        if epoch > 1:
            f1_change = val_metrics['f1'] - history['val_f1'][-1]
            iou_change = val_metrics['mean_iou'] - history['val_iou'][-1]
            print(f"[CHANGE] F1: {f1_change:+.4f} | IoU: {iou_change:+.4f}")
        
        # Update scheduler
        scheduler.step(val_metrics['f1'])
        
        # Log to W&B
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1'],
                'val_iou': val_metrics['mean_iou'],
                'combined_metric': combined_metric,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_iou'].append(val_metrics['mean_iou'])
        
        # Save checkpoints
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'alpha': alpha,
                'beta': beta
            }, output_dir / "best_f1.pth")
            print(f"[SAVE] Best F1 model: {best_f1:.4f}")
        
        if combined_metric > best_combined_metric:
            best_combined_metric = combined_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_combined_metric': best_combined_metric,
                'val_metrics': val_metrics,
                'alpha': alpha,
                'beta': beta
            }, output_dir / "best_combined.pth")
            print(f"[SAVE] Best combined model: {combined_metric:.4f}")
        
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'history': history
        }, output_dir / "last.pth")
    
    # Save training history
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best Combined Metric: {best_combined_metric:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    
    if use_wandb:
        wandb.finish()


def plot_training_curves(history: Dict, output_dir: Path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1
    axes[0, 1].plot(epochs, history['val_f1'], label='Val F1', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Classification F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[1, 0].plot(epochs, history['val_iou'], label='Val IoU', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean IoU')
    axes[1, 0].set_title('Localization IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined
    combined = [0.7*f1 + 0.3*iou for f1, iou in zip(history['val_f1'], history['val_iou'])]
    axes[1, 1].plot(epochs, combined, label='Combined (0.7*F1 + 0.3*IoU)', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Combined Metric')
    axes[1, 1].set_title('Combined Metric')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Weakly-Supervised Patch Transformer")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--baseline', type=str, default='checkpoints/patch_transformer_full/best.pth',
                        help='Path to baseline checkpoint')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for classification loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for localization loss')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases')
    
    args = parser.parse_args()
    
    train_weakly_supervised(
        config_path=args.config,
        baseline_checkpoint=args.baseline,
        alpha=args.alpha,
        beta=args.beta,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_wandb=args.wandb
    )
