"""
Train YOLOv11n classifier with SR+CLAHE preprocessing
Nano model - fast and efficient
"""
from ultralytics import YOLO
import torch
import json
from pathlib import Path

def train_yolo11n_sr_clahe():
    print("\n" + "="*70)
    print("🚀 Training YOLOv11n with SR+CLAHE")
    print("="*70)
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    # Load model
    model = YOLO('yolo11n-cls.pt')
    print("✅ Model loaded: YOLOv11n-cls")
    
    # Training configuration
    data_path = "manual_annotated_crops_sr_clahe"
    
    config = {
        'data': data_path,
        'epochs': 50,
        'batch': 16,  # Reduced from 32 for memory
        'imgsz': 640,
        'patience': 20,
        'save': True,
        'device': device,
        'workers': 4,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'project': 'runs/sr_clahe_models',
        'name': 'yolo11n_sr_clahe',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
    }
    
    print(f"\n📋 Configuration:")
    print(f"   Data: {data_path}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch: {config['batch']}")
    print(f"   Image size: {config['imgsz']}")
    print(f"   Learning rate: {config['lr0']}")
    print(f"   Patience: {config['patience']}")
    
    # Train
    print(f"\n🏋️ Starting training...")
    results = model.train(**config)
    
    # Validate
    print(f"\n📊 Validating...")
    val_results = model.val()
    
    # Test on best model
    best_model_path = Path(config['project']) / config['name'] / 'weights' / 'best.pt'
    if best_model_path.exists():
        print(f"\n🔍 Testing best model...")
        best_model = YOLO(str(best_model_path))
        test_results = best_model.val(split='test')
        
        # Save results
        results_dict = {
            'model': 'YOLOv11n',
            'preprocessing': 'SR+CLAHE',
            'epochs': config['epochs'],
            'batch_size': config['batch'],
            'test_accuracy': float(test_results.top1),
            'model_path': str(best_model_path)
        }
        
        output_file = "outputs/yolo11n_sr_clahe_results.json"
        Path("outputs").mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n" + "="*70)
        print(f"✅ TRAINING COMPLETE - YOLOv11n")
        print(f"="*70)
        print(f"📊 Test Accuracy: {test_results.top1:.2f}%")
        print(f"💾 Model: {best_model_path}")
        print(f"📁 Results: {output_file}")
        print(f"="*70)
    
    return results

if __name__ == "__main__":
    train_yolo11n_sr_clahe()
