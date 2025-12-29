"""
FINAL EVALUATION with RISK ZONE VISUALIZATION: 55 Test Images
- Dataset_2021: 50 images (25 fractured + 25 healthy)
- new_data: 5 test images (all healthy)

Generates risk zone visualizations for both ViT-Small and ViT-Tiny
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    'n_fractured': 25,
    'n_healthy': 25,
    'random_seed': 42,
    
    # Detection parameters
    'conf_threshold': 0.3,
    'bbox_scale': 2.2,
    
    # Risk zone thresholds
    'green_threshold': 0.80,  # healthy > 80% = GREEN
    'red_threshold': 0.80,    # fractured > 80% = RED
    
    # Output
    'output_base': 'outputs/FINAL_55images_riskzones_v2'  # New version with GT lines and transparent fills
}


# ============================================================================
# RISK ZONES
# ============================================================================

def get_risk_zone(fractured_prob, healthy_prob, green_th=0.80, red_th=0.80):
    """
    Determine risk zone based on probabilities
    🟢 GREEN: Healthy > 80% (very confident healthy)
    🟡 YELLOW: 20% < Fractured < 80% (uncertain)
    🔴 RED: Fractured > 80% (very confident fractured)
    """
    if healthy_prob > green_th:
        return 'green', '#00FF00', '🟢 SAFE'
    elif fractured_prob > red_th:
        return 'red', '#FF0000', '🔴 DANGER'
    else:
        return 'yellow', '#FFFF00', '🟡 WARNING'


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
# DETECTION & VISUALIZATION
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
    crop_processed = apply_sr_clahe(crop_img)
    crop_pil = Image.fromarray(cv2.cvtColor(crop_processed, cv2.COLOR_BGR2RGB))
    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = vit_model(crop_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    fractured_prob = probs[0].item()
    healthy_prob = probs[1].item()
    
    return fractured_prob, healthy_prob


def load_fracture_lines(img_path):
    """Load GT fracture lines from annotation file"""
    # Try to find annotation file (for new_data and Dataset_2021 if available)
    txt_path = img_path.with_suffix('.txt')
    
    if not txt_path.exists():
        return []
    
    lines = []
    try:
        with open(txt_path, 'r') as f:
            content = f.read().strip().split('\n')
        
        # Parse fracture lines (every 2 lines = 1 line segment)
        for i in range(0, len(content), 2):
            if i + 1 < len(content):
                line1 = content[i].strip().split()
                line2 = content[i+1].strip().split()
                
                if len(line1) >= 2 and len(line2) >= 2:
                    x1, y1 = float(line1[0]), float(line1[1])
                    x2, y2 = float(line2[0]), float(line2[1])
                    lines.append([(x1, y1), (x2, y2)])
    except:
        pass
    
    return lines


def visualize_image_with_riskzones(img_path, gt_label, stage1_model, vit_model, transform, device, 
                                   conf, bbox_scale, output_path, model_name):
    """
    Visualize single image with risk zones, GT fracture lines, and save
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load GT fracture lines
    gt_fracture_lines = load_fracture_lines(img_path)
    
    # Detect RCTs
    results = stage1_model.predict(source=str(img_path), conf=conf, verbose=False)
    
    rct_detections = []
    risk_zones = {'green': 0, 'yellow': 0, 'red': 0}
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        for box in boxes:
            cls_idx = int(box.cls[0].cpu().numpy())
            
            if cls_idx != 9:  # Only RCT
                continue
            
            # Get bbox
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Scale bbox
            scaled = scale_bbox([x1, y1, x2, y2], bbox_scale)
            sx1, sy1, sx2, sy2 = map(int, scaled)
            
            # Clip to bounds
            h, w = img.shape[:2]
            sx1 = max(0, sx1)
            sy1 = max(0, sy1)
            sx2 = min(w, sx2)
            sy2 = min(h, sy2)
            
            # Extract crop
            crop = img[sy1:sy2, sx1:sx2]
            if crop.size == 0:
                continue
            
            # Classify
            frac_prob, healthy_prob = predict_crop(crop, vit_model, transform, device)
            risk_zone, color, label = get_risk_zone(frac_prob, healthy_prob)
            
            risk_zones[risk_zone] += 1
            
            rct_detections.append({
                'bbox': [sx1, sy1, sx2, sy2],
                'fractured_prob': frac_prob,
                'healthy_prob': healthy_prob,
                'risk_zone': risk_zone,
                'risk_color': color
            })
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img_rgb)
    ax.axis('off')
    
    # Draw GT fracture lines FIRST (so they appear behind boxes)
    for line_start, line_end in gt_fracture_lines:
        ax.plot(
            [line_start[0], line_end[0]],
            [line_start[1], line_end[1]],
            color='cyan',
            linewidth=3,
            linestyle='--',
            label='GT Fracture Line' if line_start == gt_fracture_lines[0][0] else '',
            alpha=0.8,
            zorder=1  # Behind boxes
        )
    
    # Draw risk zones with TRANSPARENT fill
    for det in rct_detections:
        bbox = det['bbox']
        color = det['risk_color']
        frac_prob = det['fractured_prob']
        
        # Convert hex color to RGB for alpha
        if color == '#FF0000':  # Red
            fill_color = 'red'
        elif color == '#FFFF00':  # Yellow
            fill_color = 'yellow'
        else:  # Green
            fill_color = 'green'
        
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=4,
            edgecolor=color,
            facecolor=fill_color,
            alpha=0.2,  # Transparent fill
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add probability text
        ax.text(
            bbox[0], bbox[1] - 10,
            f'F:{frac_prob:.2f}',
            color=color,
            fontsize=12,
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
            zorder=3
        )
    
    # Add title with GT label and risk summary
    title = f"{img_path.name} | GT: {gt_label.upper()} | Model: {model_name}\n"
    title += f"RCTs: {len(rct_detections)} | "
    title += f"Green:{risk_zones['green']} Yellow:{risk_zones['yellow']} Red:{risk_zones['red']}"
    
    if gt_fracture_lines:
        title += f"\nGT Fracture Lines: {len(gt_fracture_lines)} (cyan dashed)"
    
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    
    # Add legend if fracture lines exist
    if gt_fracture_lines:
        ax.legend(loc='upper right', fontsize=12)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Determine image-level prediction
    image_pred = 'fractured' if any(d['risk_zone'] == 'red' for d in rct_detections) else 'healthy'
    
    return {
        'image': img_path.name,
        'gt_label': gt_label,
        'image_prediction': image_pred,
        'num_rcts': len(rct_detections),
        'risk_zones': risk_zones,
        'rct_detections': rct_detections
    }


# ============================================================================
# DATASET SAMPLING
# ============================================================================

def sample_dataset2021(fractured_dir, healthy_dir, n_fractured, n_healthy, seed=42):
    """Sample stratified images from Dataset_2021"""
    random.seed(seed)
    
    fractured_images = list(Path(fractured_dir).glob('*.jpg'))
    healthy_images = list(Path(healthy_dir).glob('*.jpg'))
    
    sampled_fractured = random.sample(fractured_images, min(n_fractured, len(fractured_images)))
    sampled_healthy = random.sample(healthy_images, min(n_healthy, len(healthy_images)))
    
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
        dataset.append({'path': img, 'gt_label': 'healthy', 'source': 'new_data'})
    
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


def evaluate_model_with_viz(model_name, model, stage1_model, dataset, device, conf, bbox_scale, output_dir):
    """Evaluate model and generate risk zone visualizations"""
    print(f"\n{'='*80}")
    print(f"🔬 Evaluating: {model_name}")
    print(f"{'='*80}")
    
    # Create output directory
    viz_dir = Path(output_dir) / model_name.lower().replace('-', '_')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    transform = get_vit_transform()
    results = []
    
    for item in tqdm(dataset, desc=f"Processing {model_name}"):
        output_path = viz_dir / f"{item['path'].stem}_riskzones.jpg"
        
        result = visualize_image_with_riskzones(
            img_path=item['path'],
            gt_label=item['gt_label'],
            stage1_model=stage1_model,
            vit_model=model,
            transform=transform,
            device=device,
            conf=conf,
            bbox_scale=bbox_scale,
            output_path=output_path,
            model_name=model_name
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
    print(f"\n✅ Visualizations saved to: {viz_dir}")
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'detailed_results': results,
        'viz_dir': str(viz_dir)
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("🎯 FINAL EVALUATION with RISK ZONE VISUALIZATION: 55 Test Images")
    print("="*80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(CONFIG['output_base'])
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
    
    small_results = evaluate_model_with_viz(
        'ViT-Small',
        vit_small,
        stage1_model,
        full_dataset,
        device,
        CONFIG['conf_threshold'],
        CONFIG['bbox_scale'],
        CONFIG['output_base']
    )
    
    # Evaluate ViT-Tiny
    print("\n📦 Loading ViT-Tiny model...")
    vit_tiny = load_vit_model(CONFIG['vit_tiny_checkpoint'], 'tiny', device)
    print(f"   ✅ Loaded: {CONFIG['vit_tiny_checkpoint']}")
    
    tiny_results = evaluate_model_with_viz(
        'ViT-Tiny',
        vit_tiny,
        stage1_model,
        full_dataset,
        device,
        CONFIG['conf_threshold'],
        CONFIG['bbox_scale'],
        CONFIG['output_base']
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
    
    output_file = output_path / 'evaluation_with_riskzones.json'
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
    print("🎨 Risk zone visualizations created for all 55 images!")
    print(f"   ViT-Small: {small_results['viz_dir']}")
    print(f"   ViT-Tiny: {tiny_results['viz_dir']}")
    print("="*80)


if __name__ == "__main__":
    main()
