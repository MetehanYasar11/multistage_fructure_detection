"""
Train YOLOv11 Ultra-Tiny model for Stage 2 fracture detection
Optimized for very small datasets (< 100 samples)

Dataset: 47 crops with fractures
- Train: 37 crops
- Val: 2 crops  
- Test: 8 crops

Model Architecture Experiments:
1. Ultra-Tiny: depth=0.20, width=0.15 (~500K-800K params)
2. Micro: depth=0.15, width=0.10 (~200K-400K params)
3. Custom: depth=0.25, width=0.20 (~1M params)
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def train_ultra_tiny_model(
    model_config='models/yolov11-ultratiny.yaml',
    data_yaml='stage2_fracture_dataset/data.yaml',
    epochs=300,
    imgsz=640,
    batch_size=4,  # Küçük batch daha iyi overfitting kontrolü
    patience=100,  # Early stopping
    save_dir='runs/stage2_ultratiny',
    depth_multiple=0.20,
    width_multiple=0.15,
    experiment_name='ultra_tiny'
):
    """
    Train ultra-tiny YOLO model for fracture detection
    
    Args:
        model_config: Path to custom model YAML
        data_yaml: Path to dataset configuration
        epochs: Maximum training epochs
        imgsz: Input image size
        batch_size: Training batch size (small for tiny datasets)
        patience: Early stopping patience
        save_dir: Directory to save results
        depth_multiple: Model depth scaling factor
        width_multiple: Model width scaling factor
        experiment_name: Name for this experiment
    """
    
    print(f"\n{'='*70}")
    print(f"Training YOLOv11 Ultra-Tiny Model")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Depth multiple: {depth_multiple}")
    print(f"  - Width multiple: {width_multiple}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Early stopping patience: {patience}")
    print(f"{'='*70}\n")
    
    # Load and modify model config
    config_path = Path(model_config)
    with open(config_path, 'r') as f:
        model_yaml = yaml.safe_load(f)
    
    # Update scaling factors
    if 'scales' in model_yaml:
        model_yaml['scales']['ultra-tiny'] = [depth_multiple, width_multiple, 512]
    
    # Save modified config
    temp_config = config_path.parent / f"{config_path.stem}_{experiment_name}.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(model_yaml, f)
    
    print(f"Using model config: {temp_config}")
    
    # Initialize model
    model = YOLO(temp_config, task='detect', scale='ultra-tiny')
    
    # Check estimated parameters
    try:
        model_info = model.model.info()
        print(f"\nModel Info:")
        print(f"  - Parameters: {model_info}")
    except:
        print("\nModel initialized (parameter count will be shown during training)")
    
    # Training hyperparameters optimized for small datasets
    training_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'patience': patience,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': True,  # Cache images for faster training
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'project': save_dir,
        'name': experiment_name,
        'exist_ok': True,
        'pretrained': False,  # No pretrained weights for ultra-tiny
        'optimizer': 'AdamW',  # Better for small datasets
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': True,  # Single class detection
        'rect': False,  # No rectangular training
        'cos_lr': True,  # Cosine LR scheduler
        'close_mosaic': 30,  # Disable mosaic augmentation in last 30 epochs
        'resume': False,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,  # No multi-scale for tiny models
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.1,  # Increased dropout for regularization
        'val': True,
        'split': 'val',
        'save_json': True,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 10,  # Max 10 fractures per crop
        'half': False,
        'dnn': False,
        'plots': True,
        'source': None,
        'vid_stride': 1,
        'stream_buffer': False,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'classes': None,
        'retina_masks': False,
        'embed': None,
        'show': False,
        'save_frames': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': None,
        
        # Hyperparameters for small datasets
        'lr0': 0.0005,  # Lower initial learning rate
        'lrf': 0.01,    # Lower final learning rate
        'momentum': 0.9,
        'weight_decay': 0.001,  # Increased weight decay
        'warmup_epochs': 5,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.1,  # Added label smoothing
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,  # Reduced rotation
        'translate': 0.1,
        'scale': 0.3,  # Reduced scale
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'bgr': 0.0,
        'mosaic': 0.5,  # Reduced mosaic probability
        'mixup': 0.1,   # Reduced mixup
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.2,  # Random erasing augmentation
        'crop_fraction': 1.0,
    }
    
    print(f"\nStarting training with {torch.cuda.device_count()} GPU(s)...")
    print(f"Device: {training_args['device']}")
    
    # Train the model
    results = model.train(**training_args)
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"{'='*70}")
    print(f"Results saved to: {save_dir}/{experiment_name}")
    print(f"Best model: {save_dir}/{experiment_name}/weights/best.pt")
    
    # Validate on test set
    print(f"\nValidating on test set...")
    metrics = model.val(data=data_yaml, split='test')
    
    print(f"\nTest Set Metrics:")
    print(f"  - mAP50: {metrics.box.map50:.3f}")
    print(f"  - mAP50-95: {metrics.box.map:.3f}")
    print(f"  - Precision: {metrics.box.mp:.3f}")
    print(f"  - Recall: {metrics.box.mr:.3f}")
    
    return model, results, metrics


def run_experiments():
    """
    Run multiple experiments with different model sizes
    """
    experiments = [
        {
            'name': 'ultra_tiny',
            'depth': 0.20,
            'width': 0.15,
            'description': 'Ultra-Tiny: ~500K-800K params (RECOMMENDED)'
        },
        {
            'name': 'micro',
            'depth': 0.15,
            'width': 0.10,
            'description': 'Micro: ~200K-400K params (Experimental)'
        },
        {
            'name': 'custom_balanced',
            'depth': 0.25,
            'width': 0.20,
            'description': 'Custom Balanced: ~1M params (Middle ground)'
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n{'#'*70}")
        print(f"# Experiment: {exp['description']}")
        print(f"{'#'*70}\n")
        
        model, train_results, metrics = train_ultra_tiny_model(
            depth_multiple=exp['depth'],
            width_multiple=exp['width'],
            experiment_name=exp['name'],
            epochs=300,
            batch_size=4,
            patience=100
        )
        
        results[exp['name']] = {
            'model': model,
            'train_results': train_results,
            'metrics': metrics,
            'depth': exp['depth'],
            'width': exp['width']
        }
    
    # Compare results
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Experiment':<20} {'Depth':<8} {'Width':<8} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print(f"{'-'*90}")
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name:<20} {result['depth']:<8.2f} {result['width']:<8.2f} "
              f"{metrics.box.map50:<10.3f} {metrics.box.map:<10.3f} "
              f"{metrics.box.mp:<10.3f} {metrics.box.mr:<10.3f}")
    
    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k]['metrics'].box.map50)
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Ultra-Tiny YOLOv11 for fracture detection')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'experiments'],
                      help='Training mode: single model or multiple experiments')
    parser.add_argument('--depth', type=float, default=0.20,
                      help='Depth multiple (default: 0.20)')
    parser.add_argument('--width', type=float, default=0.15,
                      help='Width multiple (default: 0.15)')
    parser.add_argument('--epochs', type=int, default=300,
                      help='Number of epochs (default: 300)')
    parser.add_argument('--batch', type=int, default=4,
                      help='Batch size (default: 4)')
    parser.add_argument('--patience', type=int, default=100,
                      help='Early stopping patience (default: 100)')
    parser.add_argument('--name', type=str, default='ultra_tiny',
                      help='Experiment name (default: ultra_tiny)')
    
    args = parser.parse_args()
    
    if args.mode == 'experiments':
        print("Running multiple experiments with different model sizes...")
        results = run_experiments()
    else:
        print(f"Training single model: depth={args.depth}, width={args.width}")
        model, results, metrics = train_ultra_tiny_model(
            depth_multiple=args.depth,
            width_multiple=args.width,
            epochs=args.epochs,
            batch_size=args.batch,
            patience=args.patience,
            experiment_name=args.name
        )
        
        print(f"\nTraining complete! Test your model:")
        print(f"  python visualize_stage2_detections.py --model runs/stage2_ultratiny/{args.name}/weights/best.pt")
