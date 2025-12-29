"""
FINAL STAGE 2 TRAINING
Train ViT-Small on expanded dataset (original + 15 new_data images)
"""

import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json

# Training configuration
CONFIG = {
    'model_name': 'vit_small_patch16_224',
    'num_classes': 2,
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Data paths
    'train_dir': 'auto_labeled_crops_sr_clahe_FINAL',  # Merged dataset
    'val_dir': 'manual_annotated_crops_sr_clahe',  # Validation (unchanged)
    
    # Output
    'output_dir': 'runs/FINAL_stage2_training',
    'checkpoint_path': 'detectors/FINAL_vit_classifier.pth',
    
    # Weighted loss for class imbalance
    # ImageFolder alphabetically: fractured=0, healthy=1
    'class_weights': [1.37, 0.73]  # [fractured, healthy] - more weight to minority class
}

def train():
    """Train final Stage 2 model"""
    
    print("="*80)
    print("FINAL STAGE 2 TRAINING")
    print("="*80)
    
    # Create output directory
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(CONFIG['train_dir'], transform=train_transform)
    val_dataset = datasets.ImageFolder(CONFIG['val_dir'], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = timm.create_model(CONFIG['model_name'], 
                             pretrained=True, 
                             num_classes=CONFIG['num_classes'])
    model = model.to(CONFIG['device'])
    
    # Loss and optimizer (weighted for class imbalance)
    class_weights = torch.tensor(CONFIG['class_weights']).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), 
                           lr=CONFIG['learning_rate'],
                           weight_decay=CONFIG['weight_decay'])
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG['checkpoint_path'])
            print(f"   ✅ Best model saved! Val Acc: {val_acc:.2f}%")
    
    # Save training history
    with open(f"{CONFIG['output_dir']}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved: {CONFIG['checkpoint_path']}")

if __name__ == '__main__':
    train()
