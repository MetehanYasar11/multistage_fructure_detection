"""
Vision Transformer Binary Classifier for RCT Fracture Detection
Using auto-labeled dataset with SR+CLAHE preprocessing

Key improvements:
- Weighted loss for class imbalance
- ViT with attention mechanism
- Better generalization than simple CNN

Dataset: auto_labeled_crops_sr_clahe/
    - fractured: 486 samples
    - healthy: 1118 samples
    
Author: Master's Thesis Project
Date: December 17, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FractureDataset(Dataset):
    """Dataset for binary fracture classification"""
    def __init__(self, image_paths, labels, transform=None, image_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Resize if needed
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label, str(img_path)


class FractureBinaryClassifier(nn.Module):
    """Vision Transformer for binary fracture classification"""
    def __init__(self, model_name='vit_small_patch16_224', pretrained=True, dropout=0.3):
        super(FractureBinaryClassifier, self).__init__()
        
        self.model_name = model_name
        
        # Load pretrained ViT from timm
        if 'vit_tiny' in model_name:
            self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
            hidden_dim = 192
        elif 'vit_small' in model_name:
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
            hidden_dim = 384
        elif 'vit_base' in model_name:
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            hidden_dim = 768
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove original head
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            in_features = hidden_dim
        
        # Custom classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2)  # Binary classification
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def load_dataset(data_dir, val_split=0.15, test_split=0.15, random_state=42):
    """Load and split dataset"""
    print("\n" + "="*80)
    print("📂 Loading Dataset")
    print("="*80)
    
    data_path = Path(data_dir)
    
    # Load images and labels
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(['healthy', 'fractured']):
        class_dir = data_path / class_name
        class_images = sorted(list(class_dir.glob("*.jpg")))
        
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))
        
        print(f"  {class_name}: {len(class_images)} images (label={class_idx})")
    
    print(f"\nTotal images: {len(image_paths)}")
    
    # Calculate class weights for imbalanced dataset
    class_counts = np.bincount(labels)
    class_weights = len(labels) / (len(class_counts) * class_counts)
    
    print(f"\n⚖️  Class Distribution:")
    print(f"  Healthy (0): {class_counts[0]} ({class_counts[0]/len(labels)*100:.1f}%)")
    print(f"  Fractured (1): {class_counts[1]} ({class_counts[1]/len(labels)*100:.1f}%)")
    print(f"\n  Class Weights: {class_weights}")
    
    # Split: train vs (val+test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(val_split + test_split),
        random_state=random_state,
        stratify=labels
    )
    
    # Split: val vs test
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=temp_labels
    )
    
    print(f"\n📊 Split Statistics:")
    print(f"  Train: {len(train_paths)} images")
    print(f"    - Healthy: {sum(1 for l in train_labels if l == 0)}")
    print(f"    - Fractured: {sum(1 for l in train_labels if l == 1)}")
    print(f"  Val: {len(val_paths)} images")
    print(f"    - Healthy: {sum(1 for l in val_labels if l == 0)}")
    print(f"    - Fractured: {sum(1 for l in val_labels if l == 1)}")
    print(f"  Test: {len(test_paths)} images")
    print(f"    - Healthy: {sum(1 for l in test_labels if l == 0)}")
    print(f"    - Fractured: {sum(1 for l in test_labels if l == 1)}")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_weights


def get_transforms(augment=True, image_size=224):
    """Get data transforms"""
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, split='Val'):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'{split}')
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    loss = running_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return loss, accuracy, precision, recall, f1, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['Healthy', 'Fractured']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def train_model(
    data_dir="auto_labeled_crops_sr_clahe",
    model_name='vit_small_patch16_224',
    epochs=50,
    batch_size=32,
    lr=1e-4,
    image_size=224,
    device='cuda',
    output_dir='runs/vit_sr_clahe'
):
    """Main training function"""
    print("\n" + "="*80)
    print("🚀 Training Vision Transformer for Fracture Detection")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Dataset: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_weights = load_dataset(
        data_dir, val_split=0.15, test_split=0.15
    )
    
    # Get transforms
    train_transform, val_transform = get_transforms(augment=True, image_size=image_size)
    
    # Create datasets
    train_dataset = FractureDataset(train_paths, train_labels, train_transform, image_size)
    val_dataset = FractureDataset(val_paths, val_labels, val_transform, image_size)
    test_dataset = FractureDataset(test_paths, test_labels, val_transform, image_size)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    model = FractureBinaryClassifier(model_name=model_name, pretrained=True, dropout=0.3)
    model = model.to(device)
    
    # Weighted loss for class imbalance
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\n🏋️ Starting training...")
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = validate(model, val_loader, criterion, device, 'Val')
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        print(f"  Val   Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, output_path / 'best_model.pth')
            print(f"  ✅ Saved best model (Val Acc: {val_acc*100:.2f}%)")
    
    # Test on best model
    print(f"\n🔍 Testing best model...")
    checkpoint = torch.load(output_path / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device, 'Test'
    )
    
    # Save confusion matrix
    plot_confusion_matrix(test_labels, test_preds, output_path / 'confusion_matrix.png')
    
    # Save results
    results = {
        'model': model_name,
        'dataset': data_dir,
        'preprocessing': 'SR+CLAHE',
        'epochs': epochs,
        'best_val_acc': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'class_distribution': {
            'healthy': int(class_weights_tensor[0].item()),
            'fractured': int(class_weights_tensor[1].item())
        }
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history
    with open(output_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"\n📁 Results saved to: {output_path}")
    print("="*80)
    
    return model, history, results


if __name__ == "__main__":
    # Configuration
    config = {
        'data_dir': 'auto_labeled_crops_sr_clahe',
        'model_name': 'vit_small_patch16_224',  # ViT-Small
        'epochs': 50,
        'batch_size': 32,
        'lr': 1e-4,
        'image_size': 224,
        'device': 'cuda',
        'output_dir': 'runs/vit_sr_clahe_auto'
    }
    
    # Train
    model, history, results = train_model(**config)
