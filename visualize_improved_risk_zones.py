"""
IMPROVED Risk Zone Visualization with Transparency and Smart Thresholds

New Risk Zone Logic:
🟢 GREEN (Safe): Healthy > 80% - Very confident healthy
🟡 YELLOW (Warning): 20% < Fractured < 80% - Uncertain region  
🔴 RED (Danger): Fractured > 80% - Very confident fractured

Visual Features:
- Transparent colored boxes (alpha=0.3)
- Clear risk indicators
- Smart overlap detection
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
    def __init__(self, model_name='vit_small_patch16_224', dropout=0.3):
        super(FractureBinaryClassifier, self).__init__()
        
        if 'vit_small' in model_name:
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
        
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            in_features = 384
        
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
    model = FractureBinaryClassifier(model_name='vit_small_patch16_224', dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def get_vit_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def apply_sr_clahe_preprocessing(img, clip_limit=2.0, tile_size=16, sr_scale=4):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    h, w = gray.shape
    new_h, new_w = h * sr_scale, w * sr_scale
    sr_img = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(sr_img)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr


def load_ground_truth_lines(gt_file):
    """Load fracture lines from ground truth TXT file"""
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


def get_improved_risk_zone(healthy_prob, fractured_prob):
    """
    Improved risk zone classification
    
    🟢 GREEN: H > 80% (very confident healthy)
    🟡 YELLOW: 20% < F < 80% (uncertain)
    🔴 RED: F > 80% (very confident fractured)
    """
    if healthy_prob > 0.80:
        return 'green', (0, 255, 0), '🟢 SAFE'
    elif fractured_prob > 0.80:
        return 'red', (0, 0, 255), '🔴 DANGER'
    else:
        return 'yellow', (0, 255, 255), '🟡 WARNING'


def draw_transparent_box(img, x1, y1, x2, y2, color, alpha=0.3, thickness=3):
    """
    Draw transparent colored box with thick border
    """
    # Create overlay for transparency
    overlay = img.copy()
    
    # Draw filled rectangle on overlay
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Blend with original (transparency)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw thick border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    return img


def predict_crop_vit(crop_img, vit_model, transform, device):
    crop_processed = apply_sr_clahe_preprocessing(crop_img)
    crop_pil = Image.fromarray(cv2.cvtColor(crop_processed, cv2.COLOR_BGR2RGB))
    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = vit_model(crop_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
    healthy_prob = probs[0].item()
    fractured_prob = probs[1].item()
    
    return healthy_prob, fractured_prob


def visualize_improved_risk_zones(
    stage1_model_path,
    vit_checkpoint_path,
    test_dir,
    output_dir,
    confidence=0.5,
    bbox_scale=2.2,
    max_images=20
):
    """
    Visualize with improved transparent risk zones
    """
    
    print("=" * 80)
    print("🎯 IMPROVED RISK ZONE VISUALIZATION")
    print("=" * 80)
    print("New Thresholds:")
    print("  🟢 GREEN (Safe): Healthy > 80%")
    print("  🟡 YELLOW (Warning): 20% < Fractured < 80%")
    print("  🔴 RED (Danger): Fractured > 80%")
    print()
    print("Visual: Transparent boxes (alpha=0.3) + thick borders")
    print("=" * 80)
    print()
    
    # Load models
    print("📦 Loading models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    stage1_model = YOLO(stage1_model_path)
    vit_model = load_vit_model(vit_checkpoint_path, device)
    transform = get_vit_transform()
    
    print(f"✅ Stage 1: {stage1_model_path}")
    print(f"✅ Stage 2: {vit_checkpoint_path}")
    print(f"✅ Device: {device}")
    print()
    
    # Get test images
    test_path = Path(test_dir)
    image_files = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"📂 Processing {len(image_files)} images...")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'total_images': 0,
        'total_crops': 0,
        'risk_zones': {'green': 0, 'yellow': 0, 'red': 0}
    }
    
    all_predictions = []
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        stats['total_images'] += 1
        vis_img = img.copy()
        
        # Load ground truth lines
        gt_file = img_path.with_suffix('.txt')
        gt_lines = load_ground_truth_lines(gt_file)
        
        # Draw GT fracture lines (BLUE)
        for line_start, line_end in gt_lines:
            pt1 = (int(line_start[0]), int(line_start[1]))
            pt2 = (int(line_end[0]), int(line_end[1]))
            cv2.line(vis_img, pt1, pt2, (255, 0, 0), 3)  # Blue line, thickness 3
        
        # Stage 1: Detect RCTs
        results = stage1_model.predict(source=str(img_path), conf=confidence, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            continue
        
        boxes = results[0].boxes
        
        # Filter RCT detections (class 9)
        rct_detections = []
        for box in boxes:
            if int(box.cls[0]) == 9:  # RCT class
                bbox = box.xyxy[0].cpu().numpy()
                conf_score = float(box.conf[0])
                rct_detections.append({'bbox': bbox, 'conf': conf_score})
        
        if len(rct_detections) == 0:
            continue
        
        image_predictions = []
        
        # Process each detected RCT
        for detection in rct_detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            
            # Expand bbox
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            new_w = int(w * bbox_scale)
            new_h = int(h * bbox_scale)
            x1_exp = max(0, cx - new_w // 2)
            y1_exp = max(0, cy - new_h // 2)
            x2_exp = min(img.shape[1], cx + new_w // 2)
            y2_exp = min(img.shape[0], cy + new_h // 2)
            
            # Crop
            crop = img[y1_exp:y2_exp, x1_exp:x2_exp]
            if crop.size == 0:
                continue
            
            # Stage 2: Classify
            healthy_prob, fractured_prob = predict_crop_vit(
                crop, vit_model, transform, device
            )
            
            # Determine risk zone (IMPROVED)
            zone, color, label = get_improved_risk_zone(healthy_prob, fractured_prob)
            
            stats['total_crops'] += 1
            stats['risk_zones'][zone] += 1
            
            # Draw TRANSPARENT box
            vis_img = draw_transparent_box(
                vis_img, x1_exp, y1_exp, x2_exp, y2_exp, 
                color, alpha=0.3, thickness=4
            )
            
            # Add label with probabilities
            label_text = f"{label} H:{healthy_prob:.2f} F:{fractured_prob:.2f}"
            
            # Label background
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                vis_img, 
                (x1_exp, y1_exp - text_h - 10),
                (x1_exp + text_w + 10, y1_exp),
                (0, 0, 0), -1
            )
            
            # Label text
            cv2.putText(
                vis_img, label_text,
                (x1_exp + 5, y1_exp - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )
            
            # Store prediction
            image_predictions.append({
                'bbox': [x1_exp, y1_exp, x2_exp, y2_exp],
                'healthy_prob': float(healthy_prob),
                'fractured_prob': float(fractured_prob),
                'risk_zone': zone
            })
        
        all_predictions.append({
            'image': img_path.name,
            'predictions': image_predictions
        })
        
        # Add legend
        legend_y = 30
        legends = [
            ('🟢 GREEN: H > 80%', (0, 255, 0)),
            ('🟡 YELLOW: 20% < F < 80%', (0, 255, 255)),
            ('🔴 RED: F > 80%', (0, 0, 255))
        ]
        
        for text, color in legends:
            cv2.rectangle(vis_img, (10, legend_y - 20), (300, legend_y + 10), (0, 0, 0), -1)
            cv2.putText(vis_img, text, (20, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            legend_y += 40
        
        # Save
        output_file = output_path / f"{img_path.stem}_improved_risk_zones.jpg"
        cv2.imwrite(str(output_file), vis_img)
    
    # Print results
    print()
    print("=" * 80)
    print("📊 VISUALIZATION RESULTS")
    print("=" * 80)
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Crops: {stats['total_crops']}")
    print()
    print("Risk Zone Distribution:")
    total = stats['total_crops']
    print(f"  🟢 GREEN (Safe):    {stats['risk_zones']['green']:3d} ({stats['risk_zones']['green']/max(total,1)*100:.1f}%)")
    print(f"  🟡 YELLOW (Warning): {stats['risk_zones']['yellow']:3d} ({stats['risk_zones']['yellow']/max(total,1)*100:.1f}%)")
    print(f"  🔴 RED (Danger):     {stats['risk_zones']['red']:3d} ({stats['risk_zones']['red']/max(total,1)*100:.1f}%)")
    print("=" * 80)
    
    # Save stats
    results = {
        'stats': stats,
        'predictions': all_predictions
    }
    
    stats_file = output_path / 'improved_risk_zones_results.json'
    with open(stats_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Visualizations saved to: {output_path}")
    print(f"✅ Statistics saved to: {stats_file}")
    
    return stats


if __name__ == "__main__":
    # Configuration
    STAGE1_MODEL = 'detectors/RCTdetector_v11x_v2.pt'
    VIT_CHECKPOINT = 'runs/vit_sr_clahe_auto/best_model.pth'
    TEST_DIR = r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\new_data\test'
    OUTPUT_DIR = 'outputs/improved_risk_zones'
    
    # Run visualization with improved confidence threshold
    visualize_improved_risk_zones(
        stage1_model_path=STAGE1_MODEL,
        vit_checkpoint_path=VIT_CHECKPOINT,
        test_dir=TEST_DIR,
        output_dir=OUTPUT_DIR,
        confidence=0.5,  # Increased from 0.3 to reduce false detections
        bbox_scale=2.2,
        max_images=20
    )
