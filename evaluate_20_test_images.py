"""
20 Test Image Crop-Level Evaluation with TXT Ground Truth

TXT Format: Each line = center coordinates of ONE fractured RCT
Example:
1923.0000       952.0000
1946.0000       985.0000

This means 2 fractured RCTs in the image.

Evaluation:
- Stage 1 detects ALL RCTs (fractured + healthy)
- Stage 2 predicts each crop
- Match predictions with GT fractured centers (within distance threshold)
- TP: Fractured crop correctly predicted as fractured
- FP: Healthy crop incorrectly predicted as fractured
- FN: Fractured crop incorrectly predicted as healthy
- TN: Healthy crop correctly predicted as healthy
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
import math


class FractureBinaryClassifier(nn.Module):
    """Vision Transformer for binary fracture classification"""
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


def load_txt_ground_truth(txt_path):
    """
    Load fractured RCT center coordinates from TXT file
    Returns: list of (cx, cy) tuples
    """
    fractured_centers = []
    
    if not Path(txt_path).exists():
        return fractured_centers
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                cx = float(parts[0])
                cy = float(parts[1])
                fractured_centers.append((cx, cy))
    
    return fractured_centers


def bbox_center(x1, y1, x2, y2):
    """Get bbox center"""
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(p1, p2):
    """Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def is_fractured_crop(crop_center, fractured_centers, threshold=100):
    """
    Check if crop is fractured by checking distance to any GT fractured center
    threshold: max distance in pixels (default 100)
    """
    for fc in fractured_centers:
        if distance(crop_center, fc) < threshold:
            return True
    return False


def predict_crop_vit(crop_img, vit_model, transform, device):
    crop_processed = apply_sr_clahe_preprocessing(crop_img)
    crop_pil = Image.fromarray(cv2.cvtColor(crop_processed, cv2.COLOR_BGR2RGB))
    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = vit_model(crop_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
    healthy_prob = probs[0].item()
    fractured_prob = probs[1].item()
    predicted_label = 1 if fractured_prob > healthy_prob else 0
    
    return healthy_prob, fractured_prob, predicted_label


def evaluate_20_test_images(
    stage1_model_path,
    vit_checkpoint_path,
    test_dir,
    output_file,
    confidence=0.3,
    bbox_scale=2.2,
    distance_threshold=100
):
    """
    Evaluate full pipeline on 20 test images with TXT ground truth
    """
    
    print("=" * 80)
    print("🎯 20 TEST IMAGE CROP-LEVEL EVALUATION")
    print("=" * 80)
    print("Stage 1: YOLOv11x_v2 RCT Detector")
    print("Stage 2: ViT-Small + SR+CLAHE Classifier")
    print("Ground Truth: TXT files with fractured RCT centers")
    print(f"Distance Threshold: {distance_threshold}px")
    print("=" * 80)
    print()
    
    # Load models
    print("📦 Loading models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    stage1_model = YOLO(stage1_model_path)
    print(f"✅ Stage 1: {stage1_model_path}")
    
    vit_model = load_vit_model(vit_checkpoint_path, device)
    transform = get_vit_transform()
    print(f"✅ Stage 2: {vit_checkpoint_path}")
    print(f"✅ Device: {device}")
    print()
    
    # Get test images
    test_path = Path(test_dir)
    image_files = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
    
    print(f"📂 Found {len(image_files)} test images")
    print()
    
    # Statistics
    crop_results = []
    stats = {
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'total_crops': 0,
        'gt_fractured': 0,
        'gt_healthy': 0,
        'total_gt_fractured_centers': 0
    }
    
    # Process each image
    print("🔬 Processing images...")
    for img_path in tqdm(image_files, desc="Evaluating"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_name = img_path.stem
        
        # Load GT fractured centers from TXT
        txt_path = img_path.with_suffix('.txt')
        fractured_centers = load_txt_ground_truth(txt_path)
        stats['total_gt_fractured_centers'] += len(fractured_centers)
        
        # Stage 1: Detect RCTs
        results = stage1_model.predict(img, conf=confidence, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            # If no RCTs detected but GT has fractures -> all FN
            if len(fractured_centers) > 0:
                stats['false_negatives'] += len(fractured_centers)
                stats['gt_fractured'] += len(fractured_centers)
            continue
        
        boxes = results[0].boxes
        
        # Process each detected RCT
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
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
            
            # Stage 2: Classify crop
            healthy_prob, fractured_prob, pred_label = predict_crop_vit(
                crop, vit_model, transform, device
            )
            
            # Determine GT label based on distance to fractured centers
            crop_center = bbox_center(x1_exp, y1_exp, x2_exp, y2_exp)
            gt_label = 1 if is_fractured_crop(crop_center, fractured_centers, distance_threshold) else 0
            
            # Update statistics
            stats['total_crops'] += 1
            
            if gt_label == 1:
                stats['gt_fractured'] += 1
                if pred_label == 1:
                    stats['true_positives'] += 1
                    correct = True
                else:
                    stats['false_negatives'] += 1
                    correct = False
            else:
                stats['gt_healthy'] += 1
                if pred_label == 0:
                    stats['true_negatives'] += 1
                    correct = True
                else:
                    stats['false_positives'] += 1
                    correct = False
            
            # Store result
            crop_results.append({
                'image': img_path.name,
                'bbox': [x1_exp, y1_exp, x2_exp, y2_exp],
                'crop_center': list(crop_center),
                'gt_label': int(gt_label),
                'pred_label': int(pred_label),
                'healthy_prob': float(healthy_prob),
                'fractured_prob': float(fractured_prob),
                'correct': correct,
                'nearest_fractured_distance': min([distance(crop_center, fc) for fc in fractured_centers]) if fractured_centers else None
            })
    
    print()
    
    # Calculate metrics
    tp = stats['true_positives']
    tn = stats['true_negatives']
    fp = stats['false_positives']
    fn = stats['false_negatives']
    
    total = tp + tn + fp + fn
    
    if total > 0:
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        accuracy = precision = recall = specificity = f1 = 0.0
    
    # Print results
    print("=" * 80)
    print("📊 CROP-LEVEL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Images: {len(image_files)}")
    print(f"Total GT Fractured Centers (from TXT): {stats['total_gt_fractured_centers']}")
    print(f"Total Crops Detected by Stage 1: {total}")
    print(f"  - GT Fractured (within {distance_threshold}px): {stats['gt_fractured']}")
    print(f"  - GT Healthy: {stats['gt_healthy']}")
    print()
    print("Confusion Matrix:")
    print(f"  ┌─────────────┬──────────┬──────────┐")
    print(f"  │             │ Pred 0   │ Pred 1   │")
    print(f"  ├─────────────┼──────────┼──────────┤")
    print(f"  │ GT 0 (H)    │ {tn:3d} (TN)│ {fp:3d} (FP)│")
    print(f"  │ GT 1 (F)    │ {fn:3d} (FN)│ {tp:3d} (TP)│")
    print(f"  └─────────────┴──────────┴──────────┘")
    print()
    print("📈 Metrics:")
    print(f"  Accuracy:    {accuracy*100:.2f}%  [{tp+tn}/{total}]")
    print(f"  Precision:   {precision*100:.2f}%  [{tp}/{tp+fp}]")
    print(f"  Recall:      {recall*100:.2f}%  [{tp}/{tp+fn}]")
    print(f"  Specificity: {specificity*100:.2f}%  [{tn}/{tn+fp}]")
    print(f"  F1 Score:    {f1:.4f}")
    print("=" * 80)
    
    # Save results
    results = {
        'test_images': len(image_files),
        'total_gt_fractured_centers': stats['total_gt_fractured_centers'],
        'total_crops': total,
        'gt_fractured_crops': stats['gt_fractured'],
        'gt_healthy_crops': stats['gt_healthy'],
        'distance_threshold': distance_threshold,
        'confusion_matrix': {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1)
        },
        'crop_predictions': crop_results
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    # Configuration
    STAGE1_MODEL = 'detectors/RCTdetector_v11x_v2.pt'
    VIT_CHECKPOINT = 'runs/vit_sr_clahe_auto/best_model.pth'
    TEST_DIR = r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\new_data\test'
    OUTPUT_FILE = 'outputs/20_test_images_crop_level_evaluation.json'
    
    # Run evaluation
    evaluate_20_test_images(
        stage1_model_path=STAGE1_MODEL,
        vit_checkpoint_path=VIT_CHECKPOINT,
        test_dir=TEST_DIR,
        output_file=OUTPUT_FILE,
        confidence=0.3,
        bbox_scale=2.2,
        distance_threshold=100  # 100 pixels distance threshold
    )
