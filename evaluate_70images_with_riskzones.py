"""
FINAL EVALUATION with RISK ZONE VISUALIZATION: 70 Test Images
- Dataset_2021: 50 images (25 fractured + 25 healthy)
- new_data: 20 test images (with crop-level GT)

Generates risk zone visualizations with ground truth fracture lines
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
    'vit_checkpoint': 'runs/vit_sr_clahe_auto/best_model.pth',
    
    # Test directories
    'dataset2021_fractured': r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Fractured',
    'dataset2021_healthy': r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Healthy',
    'newdata_test': r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\new_data\test',
    
    # Sampling
    'n_fractured': 25,
    'n_healthy': 25,
    'random_seed': 42,
    
    # Detection parameters
    'conf_threshold': 0.3,
    'bbox_scale': 2.2,
    
    # Risk zone thresholds (based on fractured crop ratio)
    'green_threshold': 0.0,   # 0% fractured = GREEN
    'yellow_max': 0.10,       # 1-10% fractured = YELLOW
    'red_threshold': 0.10,    # >10% fractured = RED
    
    # Output
    'output_base': 'outputs/FINAL_70images_riskzones'
}


# ============================================================================
# RISK ZONES
# ============================================================================

def get_risk_zone_from_ratio(fractured_ratio):
    """
    Determine risk zone based on fractured crop ratio
    🟢 GREEN: 0% fractured (all crops healthy)
    🟡 YELLOW: 1-10% fractured (uncertain)
    🔴 RED: >10% fractured (high risk)
    """
    if fractured_ratio == 0.0:
        return 'green', '#00FF00', '🟢 SAFE'
    elif fractured_ratio <= 0.10:
        return 'yellow', '#FFFF00', '🟡 WARNING'
    else:
        return 'red', '#FF0000', '🔴 DANGER'


def get_crop_risk_zone(fractured_prob, healthy_prob, green_th=0.80, red_th=0.80):
    """
    Determine crop-level risk zone based on probabilities
    """
    if healthy_prob > green_th:
        return 'green', '#00FF00'
    elif fractured_prob > red_th:
        return 'red', '#FF0000'
    else:
        return 'yellow', '#FFFF00'


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

class FractureBinaryClassifier(torch.nn.Module):
    """Vision Transformer for binary fracture classification"""
    def __init__(self, model_name='vit_small_patch16_224', dropout=0.3):
        super(FractureBinaryClassifier, self).__init__()
        
        if 'vit_small' in model_name:
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
        
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = torch.nn.Identity()
        else:
            in_features = 384
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(256, 2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def load_vit_model(checkpoint_path, device='cuda'):
    """Load ViT model"""
    model = FractureBinaryClassifier(model_name='vit_small_patch16_224', dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
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
    
    healthy_prob = probs[0].item()
    fractured_prob = probs[1].item()
    
    return fractured_prob, healthy_prob


def load_fracture_lines(img_path):
    """Load GT fracture lines from annotation file"""
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
                                   conf, bbox_scale, output_path):
    """
    Visualize single image with risk zones, GT fracture lines, and save
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Load GT fracture lines
    fracture_lines = load_fracture_lines(img_path)
    
    # Stage 1: Detect RCTs
    results_yolo = stage1_model.predict(img, conf=conf, verbose=False)
    
    if len(results_yolo) == 0 or len(results_yolo[0].boxes) == 0:
        # No RCTs detected
        rct_detections = []
        fractured_crops = 0
        total_crops = 0
        image_risk_zone = 'green'
        image_risk_color = '#00FF00'
    else:
        boxes = results_yolo[0].boxes
        fractured_crops = 0
        rct_detections = []
        
        # Process each crop - FILTER ONLY CLASS 9 (RCT)
        for box in boxes:
            cls = int(box.cls[0])
            if cls != 9:  # Skip non-RCT detections
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Scale bbox
            scaled_bbox = scale_bbox([x1, y1, x2, y2], bbox_scale)
            sx1, sy1, sx2, sy2 = map(int, scaled_bbox)
            sx1 = max(0, sx1)
            sy1 = max(0, sy1)
            sx2 = min(w, sx2)
            sy2 = min(h, sy2)
            
            # Crop
            crop = img[sy1:sy2, sx1:sx2]
            if crop.size == 0:
                continue
            
            # Stage 2: Classify
            fractured_prob, healthy_prob = predict_crop(crop, vit_model, transform, device)
            
            # Determine crop risk zone
            crop_risk_zone, crop_risk_color = get_crop_risk_zone(fractured_prob, healthy_prob)
            
            if fractured_prob > healthy_prob:
                fractured_crops += 1
            
            rct_detections.append({
                'bbox': [sx1, sy1, sx2, sy2],
                'fractured_prob': fractured_prob,
                'healthy_prob': healthy_prob,
                'risk_zone': crop_risk_zone,
                'risk_color': crop_risk_color
            })
        
        # Calculate total RCTs (class 9 only)
        total_crops = len(rct_detections)
        
        # Determine image-level risk zone
        fractured_ratio = fractured_crops / total_crops if total_crops > 0 else 0.0
        image_risk_zone, image_risk_color, _ = get_risk_zone_from_ratio(fractured_ratio)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Display image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    
    # Draw GT fracture lines (CYAN, thick, semi-transparent)
    for line in fracture_lines:
        (x1, y1), (x2, y2) = line
        ax.plot([x1, x2], [y1, y2], color='cyan', linewidth=4, alpha=0.7, 
               linestyle='-', label='GT Fracture' if line == fracture_lines[0] else '')
    
    # Draw RCT bboxes with risk zones (transparent fill + colored border)
    for det in rct_detections:
        x1, y1, x2, y2 = det['bbox']
        risk_color = det['risk_color']
        risk_zone = det['risk_zone']
        
        # Convert hex to RGB
        r = int(risk_color[1:3], 16) / 255.0
        g = int(risk_color[3:5], 16) / 255.0
        b = int(risk_color[5:7], 16) / 255.0
        
        # Draw filled rectangle with transparency
        rect_fill = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=0,
                                     edgecolor='none',
                                     facecolor=(r, g, b),
                                     alpha=0.25)
        ax.add_patch(rect_fill)
        
        # Draw border
        rect_border = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=3,
                                       edgecolor=risk_color,
                                       facecolor='none')
        ax.add_patch(rect_border)
        
        # Add probability text
        prob_text = f"{det['fractured_prob']:.2%}"
        ax.text(x1 + 5, y1 + 20, prob_text,
               fontsize=10, color='white', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=risk_color, alpha=0.8))
    
    # Title with risk zone and GT label
    title_parts = []
    title_parts.append(f"📊 {img_path.stem}")
    title_parts.append(f"GT: {gt_label.upper()}")
    title_parts.append(f"RCTs: {total_crops}")
    title_parts.append(f"Fractured: {fractured_crops}")
    if total_crops > 0:
        title_parts.append(f"Ratio: {fractured_crops/total_crops:.1%}")
    title_parts.append(f"Zone: {image_risk_zone.upper()}")
    
    ax.set_title(" | ".join(title_parts), fontsize=14, weight='bold', pad=10)
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor='#00FF00', alpha=0.3, edgecolor='#00FF00', linewidth=2, label='🟢 Green (Safe)'),
        patches.Patch(facecolor='#FFFF00', alpha=0.3, edgecolor='#FFFF00', linewidth=2, label='🟡 Yellow (Warning)'),
        patches.Patch(facecolor='#FF0000', alpha=0.3, edgecolor='#FF0000', linewidth=2, label='🔴 Red (Danger)'),
        plt.Line2D([0], [0], color='cyan', linewidth=4, alpha=0.7, label='GT Fracture Line')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Return results
    return {
        'image': img_path.name,
        'gt_label': gt_label,
        'num_rcts': total_crops,
        'fractured_crops': fractured_crops,
        'risk_zone': image_risk_zone,
        'rct_detections': rct_detections
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    print("="*80)
    print("🎯 70-IMAGE RISK ZONE VISUALIZATION")
    print("="*80)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load models
    print("\n📦 Loading models...")
    stage1_model = YOLO(CONFIG['stage1_model'])
    vit_model = load_vit_model(CONFIG['vit_checkpoint'], device)
    transform = get_vit_transform()
    print("✅ Models loaded")
    
    # Prepare test set
    print("\n📂 Preparing test set...")
    random.seed(CONFIG['random_seed'])
    
    test_images = []
    
    # Dataset_2021 fractured (25 images)
    fractured_dir = Path(CONFIG['dataset2021_fractured'])
    fractured_imgs = sorted(list(fractured_dir.glob('*.jpg')))
    sampled_fractured = random.sample(fractured_imgs, min(CONFIG['n_fractured'], len(fractured_imgs)))
    for img_path in sampled_fractured:
        test_images.append({'path': img_path, 'gt_label': 'fractured', 'source': 'Dataset_2021'})
    
    # Dataset_2021 healthy (25 images)
    healthy_dir = Path(CONFIG['dataset2021_healthy'])
    healthy_imgs = sorted(list(healthy_dir.glob('*.jpg')))
    sampled_healthy = random.sample(healthy_imgs, min(CONFIG['n_healthy'], len(healthy_imgs)))
    for img_path in sampled_healthy:
        test_images.append({'path': img_path, 'gt_label': 'healthy', 'source': 'Dataset_2021'})
    
    # new_data test (20 images)
    newdata_dir = Path(CONFIG['newdata_test'])
    for img_path in sorted(newdata_dir.glob('*.jpg')):
        # Check GT from TXT file
        txt_path = img_path.with_suffix('.txt')
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
                has_fractures = len(lines) > 0
        else:
            has_fractures = False
        
        gt_label = 'fractured' if has_fractures else 'healthy'
        test_images.append({'path': img_path, 'gt_label': gt_label, 'source': 'new_data'})
    
    print(f"✅ Total test images: {len(test_images)}")
    print(f"   - Dataset_2021 fractured: {sum(1 for t in test_images if t['source']=='Dataset_2021' and t['gt_label']=='fractured')}")
    print(f"   - Dataset_2021 healthy: {sum(1 for t in test_images if t['source']=='Dataset_2021' and t['gt_label']=='healthy')}")
    print(f"   - new_data: {sum(1 for t in test_images if t['source']=='new_data')}")
    
    # Create output directory
    output_dir = Path(CONFIG['output_base'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    print("\n🔬 Processing images...")
    results = []
    
    for test_item in tqdm(test_images, desc="Visualizing"):
        img_path = test_item['path']
        gt_label = test_item['gt_label']
        
        output_path = output_dir / f"{img_path.stem}_riskzones.jpg"
        
        result = visualize_image_with_riskzones(
            img_path, gt_label, stage1_model, vit_model, transform, device,
            CONFIG['conf_threshold'], CONFIG['bbox_scale'], output_path
        )
        
        if result:
            result['source'] = test_item['source']
            results.append(result)
    
    # Save results JSON
    output_json = output_dir / 'evaluation_with_riskzones.json'
    
    # Calculate metrics
    tp = sum(1 for r in results if r['gt_label'] == 'fractured' and r['risk_zone'] != 'green')
    tn = sum(1 for r in results if r['gt_label'] == 'healthy' and r['risk_zone'] == 'green')
    fp = sum(1 for r in results if r['gt_label'] == 'healthy' and r['risk_zone'] != 'green')
    fn = sum(1 for r in results if r['gt_label'] == 'fractured' and r['risk_zone'] == 'green')
    
    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    evaluation_data = {
        'config': CONFIG,
        'total_images': len(results),
        'metrics': {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'detailed_results': results
    }
    
    with open(output_json, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Images: {total}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\n✅ Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
