"""
Train YOLOv11n Classification with SR+CLAHE Preprocessing
Using auto-labeled dataset (1,604 crops)

Dataset: auto_labeled_crops_sr_clahe/
    - fractured/ (486 samples)
    - healthy/ (1118 samples)
"""
from ultralytics import YOLO
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def create_yolo_dataset_split(source_dir="auto_labeled_crops_sr_clahe", output_dir="rct_sr_clahe_dataset", val_split=0.15, test_split=0.15):
    """Create train/val/test split for YOLO classification"""
    print("\n" + "="*80)
    print("📂 Creating YOLO Dataset Split")
    print("="*80)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for class_name in ['fractured', 'healthy']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'fractured': {'train': 0, 'val': 0, 'test': 0},
        'healthy': {'train': 0, 'val': 0, 'test': 0}
    }
    
    # Process each class
    for class_name in ['fractured', 'healthy']:
        class_dir = source_path / class_name
        images = sorted(list(class_dir.glob("*.jpg")))
        
        print(f"\n{class_name.upper()}: {len(images)} images")
        
        if len(images) == 0:
            continue
        
        # Split: train vs (val+test)
        train_imgs, temp_imgs = train_test_split(
            images, 
            test_size=(val_split + test_split),
            random_state=42
        )
        
        # Split: val vs test
        val_ratio = val_split / (val_split + test_split)
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=(1 - val_ratio),
            random_state=42
        )
        
        # Copy files
        for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            dest_dir = output_path / split_name / class_name
            
            for img_path in split_imgs:
                shutil.copy2(img_path, dest_dir / img_path.name)
                stats[class_name][split_name] += 1
            
            print(f"  {split_name}: {len(split_imgs)} images")
    
    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    for split in ['train', 'val', 'test']:
        total = stats['fractured'][split] + stats['healthy'][split]
        frac_pct = stats['fractured'][split] / max(total, 1) * 100
        
        print(f"\n{split.upper()}: {total} images")
        print(f"  - Fractured: {stats['fractured'][split]} ({frac_pct:.1f}%)")
        print(f"  - Healthy: {stats['healthy'][split]} ({100-frac_pct:.1f}%)")
    
    print("\n" + "="*80)
    
    # Save stats
    stats_file = output_path / "split_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Dataset created: {output_path}")
    print(f"✅ Statistics saved: {stats_file}")
    
    return output_path

def train_yolo11n_sr_clahe(data_dir="rct_sr_clahe_dataset", epochs=100):
    """Train YOLOv11n with SR+CLAHE preprocessing"""
    print("\n" + "="*80)
    print("🚀 Training YOLOv11n Classifier with SR+CLAHE")
    print("="*80)
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model = YOLO('yolo11n-cls.pt')
    print("✅ Model loaded: YOLOv11n-cls")
    
    # Training configuration
    config = {
        'data': str(data_dir),
        'epochs': epochs,
        'batch': 16,
        'imgsz': 640,
        'patience': 20,
        'save': True,
        'device': device,
        'workers': 8,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'dropout': 0.2,
        'project': 'runs/sr_clahe_models',
        'name': 'yolo11n_sr_clahe_auto',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        'plots': True,
        'save_period': 10,
        
        # Augmentation
        'augment': True,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'flipud': 0.0,
    }
    
    print(f"\n📋 Configuration:")
    print(f"   Data: {data_dir}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch: {config['batch']}")
    print(f"   Image size: {config['imgsz']}")
    print(f"   Learning rate: {config['lr0']} → {config['lrf']*config['lr0']}")
    print(f"   Patience: {config['patience']}")
    print(f"   Dropout: {config['dropout']}")
    print("="*80)
    
    # Train
    print(f"\n🏋️ Starting training...")
    results = model.train(**config)
    
    # Validate on best model
    best_model_path = Path(config['project']) / config['name'] / 'weights' / 'best.pt'
    
    if best_model_path.exists():
        print(f"\n📊 Validating best model...")
        best_model = YOLO(str(best_model_path))
        
        # Validate on val set
        val_results = best_model.val(split='val')
        
        # Test on test set
        print(f"\n🔍 Testing on test set...")
        test_results = best_model.val(split='test')
        
        # Save results
        results_dict = {
            'model': 'YOLOv11n',
            'preprocessing': 'SR+CLAHE',
            'dataset': 'auto_labeled_crops',
            'total_samples': 1604,
            'epochs_trained': config['epochs'],
            'batch_size': config['batch'],
            'val_accuracy': float(val_results.top1),
            'test_accuracy': float(test_results.top1),
            'model_path': str(best_model_path)
        }
        
        output_file = "outputs/yolo11n_sr_clahe_auto_results.json"
        Path("outputs").mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n" + "="*80)
        print(f"✅ TRAINING COMPLETE")
        print(f"="*80)
        print(f"📊 Validation Accuracy: {val_results.top1:.2f}%")
        print(f"📊 Test Accuracy: {test_results.top1:.2f}%")
        print(f"💾 Best Model: {best_model_path}")
        print(f"📁 Results: {output_file}")
        print(f"="*80)
        
        return results, test_results.top1
    
    return results, None

def main():
    """Main execution"""
    
    # Step 1: Preprocess (if not done)
    sr_clahe_dir = Path("auto_labeled_crops_sr_clahe")
    if not sr_clahe_dir.exists():
        print("\n⚠️  SR+CLAHE preprocessed dataset not found!")
        print("   Please run: python preprocess_auto_labeled_sr_clahe.py")
        return
    
    print(f"✅ Found preprocessed dataset: {sr_clahe_dir}")
    
    # Step 2: Create dataset split
    print("\n📂 Creating dataset split...")
    dataset_path = create_yolo_dataset_split(
        source_dir=str(sr_clahe_dir),
        output_dir="rct_sr_clahe_dataset",
        val_split=0.15,
        test_split=0.15
    )
    
    # Step 3: Train model
    print("\n🚀 Starting training...")
    results, test_acc = train_yolo11n_sr_clahe(
        data_dir=str(dataset_path),
        epochs=100
    )
    
    if test_acc:
        print(f"\n🎉 Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
