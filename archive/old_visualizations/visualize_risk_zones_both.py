"""
Comprehensive Risk Zone Evaluation on Both Fractured and Healthy Datasets
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json


class FractureBinaryClassifier(nn.Module):
    """Vision Transformer for binary fracture classification"""
    def __init__(self, model_name='vit_small_patch16_224', dropout=0.3):
        super(FractureBinaryClassifier, self).__init__()
        
        self.model_name = model_name
        
        if 'vit_tiny' in model_name:
            self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False)
            hidden_dim = 192
        elif 'vit_small' in model_name:
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
            hidden_dim = 384
        elif 'vit_base' in model_name:
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
            hidden_dim = 768
        
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            in_features = hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def load_vit_model(checkpoint_path, device='cuda'):
    """Load trained ViT model"""
    model = FractureBinaryClassifier(model_name='vit_small_patch16_224', dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def get_vit_transform():
    """Get ViT preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def apply_sr_clahe_preprocessing(img, clip_limit=2.0, tile_size=16, sr_scale=4):
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


def load_ground_truth_lines(gt_file):
    """Load fracture lines from ground truth file"""
    if not gt_file.exists():
        return []
    
    lines = []
    with open(gt_file, 'r') as f:
        content = f.read().strip().split('\n')
        
        i = 0
        while i < len(content):
            if i + 1 < len(content):
                line1 = content[i].strip().split()
                line2 = content[i+1].strip().split()
                
                if len(line1) >= 2 and len(line2) >= 2:
                    try:
                        x1 = float(line1[0])
                        y1 = float(line1[1])
                        x2 = float(line2[0])
                        y2 = float(line2[1])
                        lines.append([(x1, y1), (x2, y2)])
                    except:
                        pass
            i += 2
    
    return lines


def check_overlap_with_fracture(bbox, fracture_lines, margin_horizontal=0.05, margin_vertical=0.10):
    """Check if bbox overlaps with any fracture line"""
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    for line_start, line_end in fracture_lines:
        fx1, fy1 = line_start
        fx2, fy2 = line_end
        
        h_margin = bbox_width * margin_horizontal
        v_margin = bbox_height * margin_vertical
        
        frac_x1 = min(fx1, fx2) - h_margin
        frac_x2 = max(fx1, fx2) + h_margin
        frac_y1 = min(fy1, fy2) - v_margin
        frac_y2 = max(fy1, fy2) + v_margin
        
        if not (x2 < frac_x1 or x1 > frac_x2 or y2 < frac_y1 or y1 > frac_y2):
            return True
    
    return False


def scale_bbox(bbox, scale_factor):
    """Scale bbox around its center"""
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


def get_risk_zone(frac_prob, healthy_prob):
    """Determine risk zone based on probabilities"""
    if healthy_prob > 0.60:
        return 'green', '#00FF00', '🟢 SAFE'
    elif frac_prob > 0.60:
        return 'red', '#FF0000', '🔴 DANGER'
    else:
        return 'yellow', '#FFFF00', '🟡 WARNING'


def predict_crop_vit(crop_img, vit_model, transform, device):
    """Predict using ViT model"""
    crop_processed = apply_sr_clahe_preprocessing(crop_img)
    crop_pil = Image.fromarray(cv2.cvtColor(crop_processed, cv2.COLOR_BGR2RGB))
    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = vit_model(crop_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
    healthy_prob = probs[0].item()
    fractured_prob = probs[1].item()
    
    return healthy_prob, fractured_prob


def evaluate_dataset(stage1_model, vit_model, vit_transform, device, test_dir, dataset_name, max_images=50):
    """Evaluate on a dataset (Fractured or Healthy)"""
    
    print(f"\n{'='*80}")
    print(f"📊 EVALUATING: {dataset_name} Dataset")
    print(f"{'='*80}")
    
    test_path = Path(test_dir)
    test_images = sorted(list(test_path.glob('*.jpg')))[:max_images]
    
    stats = {
        'dataset': dataset_name,
        'total_images': len(test_images),
        'total_rcts': 0,
        'gt_fractured_rcts': 0,
        'gt_healthy_rcts': 0,
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'risk_zones': {'green': 0, 'yellow': 0, 'red': 0}
    }
    
    for img_path in tqdm(test_images, desc=f"Processing {dataset_name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        gt_file = img_path.with_suffix('.txt')
        gt_lines = load_ground_truth_lines(gt_file)
        
        # Stage 1: Detect RCTs
        results = stage1_model.predict(source=str(img_path), conf=0.3, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            continue
        
        boxes = results[0].boxes
        rct_detections = []
        
        for box in boxes:
            if int(box.cls[0]) == 9:  # RCT class
                bbox = box.xyxy[0].cpu().numpy()
                conf_score = float(box.conf[0])
                rct_detections.append({'bbox': bbox, 'conf': conf_score})
        
        if len(rct_detections) == 0:
            continue
        
        image_has_alarm = False
        
        for detection in rct_detections:
            bbox = detection['bbox']
            scaled_bbox = scale_bbox(bbox, 2.2)
            x1, y1, x2, y2 = [int(coord) for coord in scaled_bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Stage 2: Classify
            healthy_prob, fractured_prob = predict_crop_vit(crop, vit_model, vit_transform, device)
            zone_type, _, _ = get_risk_zone(fractured_prob, healthy_prob)
            stats['risk_zones'][zone_type] += 1
            
            # Ground truth
            gt_has_fracture = check_overlap_with_fracture(scaled_bbox, gt_lines)
            stats['total_rcts'] += 1
            
            if gt_has_fracture:
                stats['gt_fractured_rcts'] += 1
            else:
                stats['gt_healthy_rcts'] += 1
            
            if zone_type in ['yellow', 'red']:
                image_has_alarm = True
        
        # Evaluation
        has_gt_fracture = len(gt_lines) > 0
        
        if has_gt_fracture:
            if image_has_alarm:
                stats['true_positives'] += 1
            else:
                stats['false_negatives'] += 1
        else:
            if not image_has_alarm:
                stats['true_negatives'] += 1
            else:
                stats['false_positives'] += 1
    
    # Calculate metrics
    total_evaluated = stats['true_positives'] + stats['true_negatives'] + \
                     stats['false_positives'] + stats['false_negatives']
    
    if total_evaluated > 0:
        accuracy = (stats['true_positives'] + stats['true_negatives']) / total_evaluated
        
        if stats['true_positives'] + stats['false_positives'] > 0:
            precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
        else:
            precision = 0.0
        
        if stats['true_positives'] + stats['false_negatives'] > 0:
            recall = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
    else:
        accuracy = precision = recall = f1 = 0.0
    
    stats['metrics'] = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    # Print results
    print(f"\nTotal Images: {stats['total_images']}")
    print(f"Total RCTs: {stats['total_rcts']}")
    print(f"  - GT Fractured: {stats['gt_fractured_rcts']}")
    print(f"  - GT Healthy: {stats['gt_healthy_rcts']}")
    print()
    print(f"Risk Zones:")
    print(f"  🟢 Green: {stats['risk_zones']['green']} ({stats['risk_zones']['green']/max(stats['total_rcts'],1)*100:.1f}%)")
    print(f"  🟡 Yellow: {stats['risk_zones']['yellow']} ({stats['risk_zones']['yellow']/max(stats['total_rcts'],1)*100:.1f}%)")
    print(f"  🔴 Red: {stats['risk_zones']['red']} ({stats['risk_zones']['red']/max(stats['total_rcts'],1)*100:.1f}%)")
    print()
    print(f"Evaluation:")
    print(f"  TP: {stats['true_positives']} | TN: {stats['true_negatives']}")
    print(f"  FP: {stats['false_positives']} | FN: {stats['false_negatives']}")
    print()
    print(f"Metrics:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return stats


def main():
    print("=" * 80)
    print("🎯 COMPREHENSIVE RISK ZONE EVALUATION")
    print("=" * 80)
    print("Model: ViT-Small + SR+CLAHE")
    print("Test Accuracy: 78.26%")
    print("-" * 80)
    print("🟢 GREEN (Safe): Healthy > 60%")
    print("🟡 YELLOW (Warning): 40% ≤ Prob ≤ 60%")
    print("🔴 RED (Danger): Fractured > 60%")
    print("=" * 80)
    
    # Load models
    print("\n📦 Loading models...")
    STAGE1_MODEL = 'detectors/RCTdetector_v11x_v2.pt'
    VIT_CHECKPOINT = 'runs/vit_sr_clahe_auto/best_model.pth'
    
    stage1_model = YOLO(STAGE1_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model = load_vit_model(VIT_CHECKPOINT, device)
    vit_transform = get_vit_transform()
    
    print(f"✅ Stage 1: {STAGE1_MODEL}")
    print(f"✅ Stage 2: {VIT_CHECKPOINT}")
    print(f"✅ Device: {device}")
    
    # Evaluate on both datasets
    FRACTURED_DIR = r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Fractured'
    HEALTHY_DIR = r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Healthy'
    
    fractured_stats = evaluate_dataset(
        stage1_model, vit_model, vit_transform, device,
        FRACTURED_DIR, "FRACTURED", max_images=50
    )
    
    healthy_stats = evaluate_dataset(
        stage1_model, vit_model, vit_transform, device,
        HEALTHY_DIR, "HEALTHY", max_images=50
    )
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("📈 OVERALL STATISTICS")
    print(f"{'='*80}")
    
    overall_stats = {
        'fractured': fractured_stats,
        'healthy': healthy_stats,
        'combined': {
            'total_images': fractured_stats['total_images'] + healthy_stats['total_images'],
            'total_rcts': fractured_stats['total_rcts'] + healthy_stats['total_rcts'],
            'true_positives': fractured_stats['true_positives'] + healthy_stats['true_positives'],
            'true_negatives': fractured_stats['true_negatives'] + healthy_stats['true_negatives'],
            'false_positives': fractured_stats['false_positives'] + healthy_stats['false_positives'],
            'false_negatives': fractured_stats['false_negatives'] + healthy_stats['false_negatives']
        }
    }
    
    combined = overall_stats['combined']
    total = combined['true_positives'] + combined['true_negatives'] + \
            combined['false_positives'] + combined['false_negatives']
    
    if total > 0:
        acc = (combined['true_positives'] + combined['true_negatives']) / total
        if combined['true_positives'] + combined['false_positives'] > 0:
            prec = combined['true_positives'] / (combined['true_positives'] + combined['false_positives'])
        else:
            prec = 0.0
        if combined['true_positives'] + combined['false_negatives'] > 0:
            rec = combined['true_positives'] / (combined['true_positives'] + combined['false_negatives'])
        else:
            rec = 0.0
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
    else:
        acc = prec = rec = f1 = 0.0
    
    overall_stats['combined']['metrics'] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    
    print(f"\nCombined Results:")
    print(f"  Total Images: {combined['total_images']}")
    print(f"  Total RCTs: {combined['total_rcts']}")
    print(f"  TP: {combined['true_positives']} | TN: {combined['true_negatives']}")
    print(f"  FP: {combined['false_positives']} | FN: {combined['false_negatives']}")
    print()
    print(f"Overall Metrics:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("=" * 80)
    
    # Save results
    output_dir = Path('outputs/risk_zones_combined')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir}/evaluation_results.json")


if __name__ == "__main__":
    main()
