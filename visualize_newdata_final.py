"""
Risk Zone Visualization for NEW DATA Test Images with FINAL Model
Using FINAL ViT classifier trained on expanded dataset

Risk Zones:
🟢 GREEN (Safe): Healthy > 60% - No doctor review needed
🟡 YELLOW (Warning): 40% ≤ Prob ≤ 60% - Doctor should check  
🔴 RED (Danger): Fractured > 60% - ALARM! Must review

Test Set: 5 images from new_data (held-out test split)
Model: FINAL_vit_classifier.pth (trained on 1,573 crops)
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import json


class FractureBinaryClassifier(nn.Module):
    """Vision Transformer for binary fracture classification"""
    def __init__(self, model_name='vit_small_patch16_224', dropout=0.3):
        super(FractureBinaryClassifier, self).__init__()
        
        self.model_name = model_name
        
        # Load pretrained ViT
        if 'vit_tiny' in model_name:
            self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False)
            hidden_dim = 192
        elif 'vit_small' in model_name:
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
            hidden_dim = 384
        elif 'vit_base' in model_name:
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
            hidden_dim = 768
        
        # Remove original head
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            in_features = hidden_dim
        
        # Custom classification head
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
    """Load trained ViT model - FINAL model uses simple timm.create_model"""
    # FINAL model is just timm ViT without custom wrapper
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=2)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded FINAL model from: {checkpoint_path}")
    
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
    
    # Super-resolution
    sr_img = cv2.resize(gray, (w * sr_scale, h * sr_scale), interpolation=cv2.INTER_CUBIC)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    sr_clahe = clahe.apply(sr_img)
    
    # Resize back
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
    """
    IMPROVED risk zone thresholds:
    🟢 GREEN: Healthy > 80% (very confident healthy)
    🟡 YELLOW: 20% < Fractured < 80% (uncertain)
    🔴 RED: Fractured > 80% (very confident fractured)
    """
    if healthy_prob > 0.80:
        return 'green', '#00FF00', '🟢 SAFE'
    elif frac_prob > 0.80:
        return 'red', '#FF0000', '🔴 DANGER'
    else:
        return 'yellow', '#FFFF00', '🟡 WARNING'


def predict_crop_vit(crop_img, vit_model, transform, device):
    """
    Predict using ViT model
    Returns: (healthy_prob, fractured_prob)
    """
    # Apply SR+CLAHE
    crop_processed = apply_sr_clahe_preprocessing(crop_img)
    
    # Convert to PIL for transforms
    crop_pil = Image.fromarray(cv2.cvtColor(crop_processed, cv2.COLOR_BGR2RGB))
    
    # Apply transform
    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = vit_model(crop_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
    healthy_prob = probs[0].item()
    fractured_prob = probs[1].item()
    
    return healthy_prob, fractured_prob


def visualize_risk_zones_vit(
    stage1_model_path, 
    vit_checkpoint_path,
    test_dir, 
    output_dir,
    confidence=0.3,
    bbox_scale=2.2,
    max_images=20
):
    """
    Visualize predictions with risk zones using ViT classifier
    """
    
    print("=" * 80)
    print("🎯 IMPROVED RISK ZONE VISUALIZATION (ViT + SR+CLAHE)")
    print("=" * 80)
    print("Model: Vision Transformer Small")
    print("Test Accuracy: 78.26%")
    print("Preprocessing: SR+CLAHE")
    print("-" * 80)
    print("🟢 GREEN (Safe): Healthy > 80% - No review needed")
    print("🟡 YELLOW (Warning): 20% < Fractured < 80% - Doctor should check")
    print("🔴 RED (Danger): Fractured > 80% - ALARM! Must review")
    print("=" * 80)
    
    # Load models
    print("\n📦 Loading models...")
    stage1_model = YOLO(stage1_model_path)
    print(f"✅ Stage 1 (RCT Detector): {stage1_model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model = load_vit_model(vit_checkpoint_path, device)
    vit_transform = get_vit_transform()
    print(f"✅ Stage 2 (ViT Classifier): {vit_checkpoint_path}")
    print(f"✅ Device: {device}")
    
    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test directory
    test_path = Path(test_dir)
    test_images = sorted(list(test_path.glob('*.jpg')))[:max_images]
    
    print(f"\n📂 Processing {len(test_images)} test images...")
    
    # Statistics
    stats = {
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
    
    # Process each image
    for img_idx, img_path in enumerate(tqdm(test_images, desc="Processing")):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Load ground truth
        gt_file = img_path.with_suffix('.txt')
        gt_lines = load_ground_truth_lines(gt_file)
        
        # Stage 1: Detect RCTs
        results = stage1_model.predict(source=str(img_path), conf=confidence, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            continue
        
        # Filter RCT detections (class 9)
        boxes = results[0].boxes
        rct_detections = []
        
        for box in boxes:
            if int(box.cls[0]) == 9:  # RCT class
                bbox = box.xyxy[0].cpu().numpy()
                conf_score = float(box.conf[0])
                rct_detections.append({'bbox': bbox, 'conf': conf_score})
        
        if len(rct_detections) == 0:
            continue
        
        # Create visualization
        vis_img = img.copy()
        
        # Draw ground truth fracture lines (thin red)
        for line_start, line_end in gt_lines:
            pt1 = (int(line_start[0]), int(line_start[1]))
            pt2 = (int(line_end[0]), int(line_end[1]))
            cv2.line(vis_img, pt1, pt2, (0, 0, 255), 2)
        
        # Process each RCT
        image_has_alarm = False
        
        for rct_idx, detection in enumerate(rct_detections):
            bbox = detection['bbox']
            
            # Scale bbox
            scaled_bbox = scale_bbox(bbox, bbox_scale)
            x1, y1, x2, y2 = [int(coord) for coord in scaled_bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Extract crop
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Stage 2: Classify with ViT
            healthy_prob, fractured_prob = predict_crop_vit(crop, vit_model, vit_transform, device)
            
            # Determine risk zone
            zone_type, zone_color, zone_label = get_risk_zone(fractured_prob, healthy_prob)
            stats['risk_zones'][zone_type] += 1
            
            # Check ground truth
            gt_has_fracture = check_overlap_with_fracture(scaled_bbox, gt_lines)
            stats['total_rcts'] += 1
            
            if gt_has_fracture:
                stats['gt_fractured_rcts'] += 1
            else:
                stats['gt_healthy_rcts'] += 1
            
            # Draw bbox with risk zone color
            color_bgr = {
                'green': (0, 255, 0),
                'yellow': (0, 255, 255),
                'red': (0, 0, 255)
            }[zone_type]
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color_bgr, 3)
            
            # Add label
            label = f"{zone_label} H:{healthy_prob:.2f} F:{fractured_prob:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color_bgr, -1)
            cv2.putText(vis_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Check if alarm zone
            if zone_type in ['yellow', 'red']:
                image_has_alarm = True
        
        # Evaluation: Check if prediction matches GT
        has_gt_fracture = len(gt_lines) > 0
        
        if has_gt_fracture:
            # GT Fractured: Should have at least one yellow/red zone
            if image_has_alarm:
                stats['true_positives'] += 1
            else:
                stats['false_negatives'] += 1
        else:
            # GT Healthy: Should have all green zones
            if not image_has_alarm:
                stats['true_negatives'] += 1
            else:
                stats['false_positives'] += 1
        
        # Save visualization
        output_file = output_path / f"{img_path.stem}_risk_zones.jpg"
        cv2.imwrite(str(output_file), vis_img)
    
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
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Images: {stats['total_images']}")
    print(f"Total RCTs Detected: {stats['total_rcts']}")
    print(f"  - GT Fractured: {stats['gt_fractured_rcts']}")
    print(f"  - GT Healthy: {stats['gt_healthy_rcts']}")
    print()
    print(f"Risk Zone Distribution:")
    print(f"  🟢 Green (Safe): {stats['risk_zones']['green']} ({stats['risk_zones']['green']/max(stats['total_rcts'],1)*100:.1f}%)")
    print(f"  🟡 Yellow (Warning): {stats['risk_zones']['yellow']} ({stats['risk_zones']['yellow']/max(stats['total_rcts'],1)*100:.1f}%)")
    print(f"  🔴 Red (Danger): {stats['risk_zones']['red']} ({stats['risk_zones']['red']/max(stats['total_rcts'],1)*100:.1f}%)")
    print()
    print(f"Image-Level Evaluation:")
    print(f"  True Positives (Detected fractures): {stats['true_positives']}")
    print(f"  True Negatives (Correctly identified healthy): {stats['true_negatives']}")
    print(f"  False Positives (False alarms): {stats['false_positives']}")
    print(f"  False Negatives (Missed fractures): {stats['false_negatives']}")
    print()
    print(f"📈 Metrics:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("=" * 80)
    
    # Save statistics
    stats['metrics'] = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    stats_file = output_path / 'evaluation_results.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"✅ Statistics saved to: {stats_file}")
    
    return stats


if __name__ == "__main__":
    # Configuration
    STAGE1_MODEL = 'detectors/RCTdetector_v11x_v2.pt'
    VIT_CHECKPOINT = 'detectors/FINAL_vit_classifier.pth'  # FINAL MODEL
    TEST_DIR = 'new_data_test_images'  # 5 test images from new_data
    OUTPUT_DIR = 'outputs/FINAL_newdata_risk_zones'
    
    # Run visualization with FINAL model on new_data test images
    visualize_risk_zones_vit(
        stage1_model_path=STAGE1_MODEL,
        vit_checkpoint_path=VIT_CHECKPOINT,
        test_dir=TEST_DIR,
        output_dir=OUTPUT_DIR,
        confidence=0.3,  # RCT detection confidence
        bbox_scale=2.2,  # Bbox expansion
        max_images=5  # Only 5 test images
    )
