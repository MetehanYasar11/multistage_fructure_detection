"""
FINAL EVALUATION: Test Both Models on 55 Images
- Dataset_2021: 50 images (randomly sampled, stratified)
- new_data: 5 test images

Tests both ViT-Small and ViT-Tiny models
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Models
    'stage1_model': 'detectors/RCTdetector_v11x_v2.pt',
    'vit_small_checkpoint': 'detectors/FINAL_vit_classifier.pth',
    'vit_tiny_checkpoint': 'detectors/FINAL_vit_tiny_classifier.pth',
    
    # Test directories
    'dataset2021_fractured': r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Fractured',
    'dataset2021_healthy': r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Healthy',
    'newdata_test': 'new_data_test_images',
    
    # Sampling
    'n_fractured': 25,  # Sample 25 fractured
    'n_healthy': 25,    # Sample 25 healthy
    'random_seed': 42,
    
    # Detection parameters
    'conf_threshold': 0.3,
    'bbox_scale': 2.2,
    
    # Output
    'output_dir': 'outputs/FINAL_55image_evaluation'
}


# ============================================================================
# PREPROCESSING
# ============================================================================

def apply_sr_clahe(img, clip_limit=2.0, tile_size=16, sr_scale=4):
    """Apply SR+CLAHE preprocessing"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape[:2]
    sr_img = cv2.resize(gray, (w * sr_scale, h * sr_scale), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    sr_clahe = clahe.apply(sr_img)
    result = cv2.resize(sr_clahe, (w, h), interpolation=cv2.INTER_AREA)
    
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def get_vit_transform():
    """Get ViT preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_vit_model(checkpoint_path, model_size='small', device='cuda'):
    """Load ViT model"""
    if model_size == 'tiny':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
    else:  # small
        model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================================
# DETECTION & CLASSIFICATION
# ============================================================================

def scale_bbox(bbox, scale_factor):
    """Scale bbox around center"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    new_w = w * scale_factor
    new_h = h * scale_factor
    
    new_x1 = cx - new_w / 2
    new_y1 = cy - new_h / 2
    new_x2 = cx + new_w / 2
    new_y2 = cy + new_h / 2
    
    return [new_x1, new_y1, new_x2, new_y2]


def predict_crop(crop_img, vit_model, transform, device):
    """Predict fracture probability for crop"""
    # Apply SR+CLAHE
    crop_processed = apply_sr_clahe(crop_img)
    
    # Convert to PIL
    crop_pil = Image.fromarray(cv2.cvtColor(crop_processed, cv2.COLOR_BGR2RGB))
    
    # Transform and predict
    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = vit_model(crop_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    # ImageFolder: fractured=0, healthy=1
    fractured_prob = probs[0].item()
    healthy_prob = probs[1].item()
    
    return fractured_prob, healthy_prob


def evaluate_image(img_path, gt_label, stage1_model, vit_model, transform, device, conf, bbox_scale):
    """
    Evaluate single image
    Returns: dict with RCT detections and predictions
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # Stage 1: Detect RCTs
    results = stage1_model.predict(source=str(img_path), conf=conf, verbose=False)
    
    rct_detections = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        for box in boxes:
            cls_idx = int(box.cls[0].cpu().numpy())
            
            # Only process RCT (class 9)
            if cls_idx != 9:
                continue
            
            # Get bbox
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Scale bbox
            scaled = scale_bbox([x1, y1, x2, y2], bbox_scale)
            sx1, sy1, sx2, sy2 = map(int, scaled)
            
            # Clip to image bounds
            h, w = img.shape[:2]
            sx1 = max(0, sx1)
            sy1 = max(0, sy1)
            sx2 = min(w, sx2)
            sy2 = min(h, sy2)
            
            # Extract crop
            crop = img[sy1:sy2, sx1:sx2]
            
            if crop.size == 0:
                continue
            
            # Stage 2: Classify
            frac_prob, healthy_prob = predict_crop(crop, vit_model, transform, device)
            
            rct_detections.append({
                'bbox': [sx1, sy1, sx2, sy2],
                'fractured_prob': frac_prob,
                'healthy_prob': healthy_prob,
                'predicted_label': 'fractured' if frac_prob > 0.5 else 'healthy'
            })
    
    return {
        'image': img_path.name,
        'gt_label': gt_label,
        'num_rcts': len(rct_detections),
        'rct_detections': rct_detections,
        'image_prediction': 'fractured' if any(r['predicted_label'] == 'fractured' for r in rct_detections) else 'healthy'
    }


# ============================================================================
# DATASET SAMPLING
# ============================================================================

def sample_dataset2021(fractured_dir, healthy_dir, n_fractured, n_healthy, seed=42):
    """Sample stratified images from Dataset_2021"""
    random.seed(seed)
    
    fractured_images = list(Path(fractured_dir).glob('*.jpg'))
    healthy_images = list(Path(healthy_dir).glob('*.jpg'))
    
    # Sample
    sampled_fractured = random.sample(fractured_images, min(n_fractured, len(fractured_images)))
    sampled_healthy = random.sample(healthy_images, min(n_healthy, len(healthy_images)))
    
    # Create dataset with GT labels
    dataset = []
    for img in sampled_fractured:
        dataset.append({'path': img, 'gt_label': 'fractured', 'source': 'Dataset_2021'})
    for img in sampled_healthy:
        dataset.append({'path': img, 'gt_label': 'healthy', 'source': 'Dataset_2021'})
    
    return dataset


def load_newdata_test(test_dir):
    """Load new_data test images"""
    test_images = list(Path(test_dir).glob('*.jpg'))
    
    dataset = []
    for img in test_images:
        dataset.append({'path': img, 'gt_label': 'healthy', 'source': 'new_data'})  # All new_data test are healthy
    
    return dataset


# ============================================================================
# EVALUATION
# ============================================================================

def calculate_metrics(results):
    """Calculate evaluation metrics"""
    tp = fp = tn = fn = 0
    
    for result in results:
        gt = result['gt_label']
        pred = result['image_prediction']
        
        if gt == 'fractured' and pred == 'fractured':
            tp += 1
        elif gt == 'healthy' and pred == 'fractured':
            fp += 1
        elif gt == 'healthy' and pred == 'healthy':
            tn += 1
        elif gt == 'fractured' and pred == 'healthy':
            fn += 1
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_images': total,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def evaluate_model(model_name, model, stage1_model, dataset, device, conf, bbox_scale):
    """Evaluate a single model on dataset"""
    print(f"\n{'='*80}")
    print(f"🔬 Evaluating: {model_name}")
    print(f"{'='*80}")
    
    transform = get_vit_transform()
    results = []
    
    for item in tqdm(dataset, desc=f"Processing {model_name}"):
        result = evaluate_image(
            img_path=item['path'],
            gt_label=item['gt_label'],
            stage1_model=stage1_model,
            vit_model=model,
            transform=transform,
            device=device,
            conf=conf,
            bbox_scale=bbox_scale
        )
        
        if result:
            result['source'] = item['source']
            results.append(result)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print(f"\n📊 {model_name} Results:")
    print(f"   Total Images: {metrics['total_images']}")
    print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall: {metrics['recall']*100:.2f}%")
    print(f"   F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"\n   Confusion Matrix:")
    print(f"   TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    print(f"   FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'detailed_results': results
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("🎯 FINAL EVALUATION: 55 Test Images")
    print("="*80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(CONFIG['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample Dataset_2021
    print("\n📊 Sampling Dataset_2021...")
    dataset2021 = sample_dataset2021(
        CONFIG['dataset2021_fractured'],
        CONFIG['dataset2021_healthy'],
        CONFIG['n_fractured'],
        CONFIG['n_healthy'],
        CONFIG['random_seed']
    )
    print(f"   ✅ Sampled {len(dataset2021)} images:")
    print(f"      - Fractured: {sum(1 for d in dataset2021 if d['gt_label'] == 'fractured')}")
    print(f"      - Healthy: {sum(1 for d in dataset2021 if d['gt_label'] == 'healthy')}")
    
    # Load new_data test
    print("\n📊 Loading new_data test images...")
    newdata_test = load_newdata_test(CONFIG['newdata_test'])
    print(f"   ✅ Loaded {len(newdata_test)} test images")
    
    # Combine datasets
    full_dataset = dataset2021 + newdata_test
    print(f"\n📊 Total test set: {len(full_dataset)} images")
    
    # Load Stage 1 model
    print("\n📦 Loading Stage 1 RCT Detector...")
    stage1_model = YOLO(CONFIG['stage1_model'])
    print(f"   ✅ Loaded: {CONFIG['stage1_model']}")
    
    # Evaluate ViT-Small
    print("\n📦 Loading ViT-Small model...")
    vit_small = load_vit_model(CONFIG['vit_small_checkpoint'], 'small', device)
    print(f"   ✅ Loaded: {CONFIG['vit_small_checkpoint']}")
    
    small_results = evaluate_model(
        'ViT-Small',
        vit_small,
        stage1_model,
        full_dataset,
        device,
        CONFIG['conf_threshold'],
        CONFIG['bbox_scale']
    )
    
    # Evaluate ViT-Tiny
    print("\n📦 Loading ViT-Tiny model...")
    vit_tiny = load_vit_model(CONFIG['vit_tiny_checkpoint'], 'tiny', device)
    print(f"   ✅ Loaded: {CONFIG['vit_tiny_checkpoint']}")
    
    tiny_results = evaluate_model(
        'ViT-Tiny',
        vit_tiny,
        stage1_model,
        full_dataset,
        device,
        CONFIG['conf_threshold'],
        CONFIG['bbox_scale']
    )
    
    # Save results
    final_results = {
        'config': CONFIG,
        'dataset_info': {
            'total_images': len(full_dataset),
            'dataset2021_fractured': sum(1 for d in dataset2021 if d['gt_label'] == 'fractured'),
            'dataset2021_healthy': sum(1 for d in dataset2021 if d['gt_label'] == 'healthy'),
            'newdata_test': len(newdata_test)
        },
        'vit_small': small_results,
        'vit_tiny': tiny_results
    }
    
    output_file = output_path / 'evaluation_results_55images.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print comparison
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<20} {'ViT-Small':<15} {'ViT-Tiny':<15}")
    print("-" * 50)
    
    small_m = small_results['metrics']
    tiny_m = tiny_results['metrics']
    
    print(f"{'Accuracy':<20} {small_m['accuracy']*100:>6.2f}%       {tiny_m['accuracy']*100:>6.2f}%")
    print(f"{'Precision':<20} {small_m['precision']*100:>6.2f}%       {tiny_m['precision']*100:>6.2f}%")
    print(f"{'Recall':<20} {small_m['recall']*100:>6.2f}%       {tiny_m['recall']*100:>6.2f}%")
    print(f"{'F1-Score':<20} {small_m['f1_score']*100:>6.2f}%       {tiny_m['f1_score']*100:>6.2f}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
