"""
Train YOLOv11n Classification Model for RCT Fracture Detection

Dataset: auto_labeled_crops/
    - fractured/ (486 samples)
    - healthy/ (1118 samples)
    
Author: Master's Thesis Project
Date: December 17, 2025
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
from sklearn.model_selection import train_test_split
import json


def create_yolo_cls_dataset(source_dir="auto_labeled_crops", output_dir="rct_cls_dataset", val_split=0.2, test_split=0.1):
    """
    Create YOLO classification dataset structure
    
    YOLO Classification format:
    dataset/
        train/
            fractured/
                img1.jpg
                img2.jpg
            healthy/
                img3.jpg
                img4.jpg
        val/
            fractured/
                img5.jpg
            healthy/
                img6.jpg
        test/
            fractured/
                img7.jpg
            healthy/
                img8.jpg
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    print("="*80)
    print("CREATING YOLO CLASSIFICATION DATASET")
    print("="*80)
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Val split: {val_split*100}%")
    print(f"Test split: {test_split*100}%")
    print("="*80)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in ['fractured', 'healthy']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    stats = {
        'fractured': {'train': 0, 'val': 0, 'test': 0},
        'healthy': {'train': 0, 'val': 0, 'test': 0}
    }
    
    for class_name in ['fractured', 'healthy']:
        class_dir = source_path / class_name
        
        # Get all images
        images = sorted(list(class_dir.glob("*.jpg")))
        print(f"\n{class_name.upper()}: {len(images)} images")
        
        if len(images) == 0:
            print(f"  ⚠️ No images found in {class_dir}")
            continue
        
        # Split into train/val/test
        # First split: train vs (val+test)
        train_imgs, temp_imgs = train_test_split(
            images, 
            test_size=(val_split + test_split),
            random_state=42
        )
        
        # Second split: val vs test
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
                dest_path = dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)
                stats[class_name][split_name] += 1
            
            print(f"  {split_name}: {len(split_imgs)} images")
    
    # Save split info
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    for split in ['train', 'val', 'test']:
        total = stats['fractured'][split] + stats['healthy'][split]
        frac_pct = stats['fractured'][split] / max(total, 1) * 100
        healthy_pct = stats['healthy'][split] / max(total, 1) * 100
        
        print(f"\n{split.upper()} SET: {total} images")
        print(f"  - Fractured: {stats['fractured'][split]} ({frac_pct:.1f}%)")
        print(f"  - Healthy: {stats['healthy'][split]} ({healthy_pct:.1f}%)")
    
    total_all = sum(stats['fractured'].values()) + sum(stats['healthy'].values())
    print(f"\nTOTAL: {total_all} images")
    print("="*80)
    
    # Save statistics
    stats_path = output_path / "split_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✅ Statistics saved to: {stats_path}")
    
    return output_path


def train_yolo_classifier(
    data_dir="rct_cls_dataset",
    model_size="yolo11n-cls",
    epochs=100,
    imgsz=640,
    batch=16,
    save_dir="runs/rct_cls",
    device=0
):
    """
    Train YOLOv11 classification model
    
    Args:
        data_dir: Path to dataset (with train/val/test folders)
        model_size: Model size (yolo11n-cls, yolo11s-cls, yolo11m-cls)
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        save_dir: Directory to save results
        device: GPU device (0, 1, etc.) or 'cpu'
    """
    print("\n" + "="*80)
    print("TRAINING YOLO CLASSIFICATION MODEL")
    print("="*80)
    print(f"Model: {model_size}")
    print(f"Dataset: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Load model (YOLO will auto-download if not exists)
    model = YOLO(f"{model_size}.pt")
    
    # Train model
    results = model.train(
        data=str(Path(data_dir)),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='rct_fracture_cls',
        project=save_dir,
        device=device,
        
        # Augmentation settings
        augment=True,
        hsv_h=0.015,      # Hue augmentation
        hsv_s=0.7,        # Saturation augmentation
        hsv_v=0.4,        # Value augmentation
        degrees=10,       # Rotation
        translate=0.1,    # Translation
        scale=0.5,        # Scale
        flipud=0.0,       # No vertical flip for X-rays
        fliplr=0.5,       # Horizontal flip
        mosaic=0.0,       # No mosaic for classification
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,        # Initial learning rate
        lrf=0.01,         # Final learning rate (fraction of lr0)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        # Regularization
        dropout=0.2,      # Dropout rate
        
        # Other settings
        patience=20,      # Early stopping patience
        save=True,
        save_period=10,
        plots=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Best model saved at: {save_dir}/rct_fracture_cls/weights/best.pt")
    print("="*80)
    
    return results


def validate_model(model_path, data_dir="rct_cls_dataset", split='test'):
    """
    Validate model on test set
    
    Args:
        model_path: Path to trained model
        data_dir: Dataset directory
        split: Which split to validate on ('val' or 'test')
    """
    print("\n" + "="*80)
    print(f"VALIDATING ON {split.upper()} SET")
    print("="*80)
    
    # Load model
    model = YOLO(model_path)
    
    # Validate
    results = model.val(
        data=str(Path(data_dir)),
        split=split,
        batch=16,
        plots=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Top-1 Accuracy: {results.top1:.4f}")
    print(f"Top-5 Accuracy: {results.top5:.4f}")
    print("="*80)
    
    return results


def main():
    """Main execution"""
    
    # Step 1: Create YOLO classification dataset
    print("\n🔄 Step 1: Creating YOLO classification dataset...")
    dataset_path = create_yolo_cls_dataset(
        source_dir="auto_labeled_crops",
        output_dir="rct_cls_dataset",
        val_split=0.15,    # 15% validation
        test_split=0.15    # 15% test
    )
    
    # Step 2: Train model
    print("\n🚀 Step 2: Training YOLOv11n classification model...")
    results = train_yolo_classifier(
        data_dir=str(dataset_path),
        model_size="yolo11n-cls",  # Model name without .pt
        epochs=100,
        imgsz=640,
        batch=16,
        save_dir="runs/rct_cls",
        device=0  # Use GPU 0
    )
    
    # Step 3: Validate on test set
    print("\n📊 Step 3: Validating on test set...")
    best_model = "runs/rct_cls/rct_fracture_cls/weights/best.pt"
    test_results = validate_model(
        model_path=best_model,
        data_dir=str(dataset_path),
        split='test'
    )
    
    print("\n" + "="*80)
    print("✅ ALL STEPS COMPLETED!")
    print("="*80)
    print(f"📁 Dataset: {dataset_path}")
    print(f"🏆 Best model: {best_model}")
    print(f"📈 Test Accuracy: {test_results.top1:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
