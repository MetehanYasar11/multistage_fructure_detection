"""
Full Pipeline Crop-Level Evaluation with Ground Truth Matching

This script evaluates the full two-stage pipeline at CROP LEVEL:
1. Stage 1 (YOLOv11x_v2) detects RCTs in panoramic X-rays
2. Stage 2 (ViT-Small + SR+CLAHE) classifies each crop
3. Compare predictions with crop-level GT labels

Metrics:
- TP: Fractured crop correctly predicted as fractured
- TN: Healthy crop correctly predicted as healthy  
- FP: Healthy crop incorrectly predicted as fractured
- FN: Fractured crop incorrectly predicted as healthy

Test Set: 20 fractured panoramic images from Dataset_2021
Expected: ~62 total crops (mix of fractured and healthy RCTs)
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
        gray = img
    
    # Super-resolution (bicubic upscaling)
    h, w = gray.shape
    new_h, new_w = h * sr_scale, w * sr_scale
    sr_img = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(sr_img)
    
    # Convert back to BGR for consistency
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr


def load_ground_truth_annotations(gt_file):
    """
    Load ground truth annotations for crops
    Expected format: JSON with image-level annotations
    """
    if not Path(gt_file).exists():
        print(f"⚠️  GT file not found: {gt_file}")
        return {}
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    return gt_data


def predict_crop_vit(crop_img, vit_model, transform, device):
    """
    Predict using ViT model
    Returns: (healthy_prob, fractured_prob, predicted_label)
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
    predicted_label = 1 if fractured_prob > healthy_prob else 0
    
    return healthy_prob, fractured_prob, predicted_label


def evaluate_pipeline_crop_level(
    stage1_model_path,
    vit_checkpoint_path,
    test_images_dir,
    gt_annotations_file,
    output_file,
    confidence=0.3,
    bbox_scale=2.2
):
    """
    Evaluate full pipeline at crop level with GT matching
    """
    
    print("=" * 80)
    print("🎯 FULL PIPELINE CROP-LEVEL EVALUATION")
    print("=" * 80)
    print("Stage 1: YOLOv11x_v2 RCT Detector")
    print("Stage 2: ViT-Small + SR+CLAHE Classifier")
    print("Evaluation: Crop-level GT matching")
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
    
    # Load ground truth
    print(f"📋 Loading ground truth: {gt_annotations_file}")
    gt_data = load_ground_truth_annotations(gt_annotations_file)
    
    # Get annotations list (support both 'annotations' and 'all_crops' keys)
    annotations = gt_data.get('annotations', [])
    if len(annotations) == 0:
        annotations = gt_data.get('all_crops', [])
    
    print(f"✅ GT loaded for {len(annotations)} crops")
    print()
    
    # Get test images
    test_path = Path(test_images_dir)
    image_files = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {test_images_dir}")
        return
    
    print(f"📂 Found {len(image_files)} test images")
    print()
    
    # Evaluation statistics
    crop_results = []
    stats = {
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'total_crops': 0,
        'gt_fractured': 0,
        'gt_healthy': 0
    }
    
    # Create GT lookup dictionary
    gt_lookup = {}
    for ann in annotations:
        image_name = ann.get('image', '')
        bbox = tuple(ann.get('bbox', []))
        gt_label = ann.get('gt_label', 0)
        key = (image_name, bbox)
        gt_lookup[key] = gt_label
    
    # Process each image
    print("🔬 Processing images...")
    for img_path in tqdm(image_files, desc="Evaluating"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_name = img_path.name
        
        # Stage 1: Detect RCTs
        results = stage1_model.predict(
            img,
            conf=confidence,
            verbose=False
        )
        
        if len(results) == 0 or len(results[0].boxes) == 0:
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
            
            # Look up GT label
            bbox_tuple = (x1_exp, y1_exp, x2_exp, y2_exp)
            
            # Try to find matching GT (allow some tolerance for bbox coordinates)
            gt_label = None
            for (gt_img, gt_bbox), gt_lbl in gt_lookup.items():
                if gt_img == img_name:
                    # Check if bboxes are close enough (within 20 pixels)
                    if (abs(gt_bbox[0] - x1_exp) < 20 and 
                        abs(gt_bbox[1] - y1_exp) < 20 and
                        abs(gt_bbox[2] - x2_exp) < 20 and
                        abs(gt_bbox[3] - y2_exp) < 20):
                        gt_label = gt_lbl
                        break
            
            # If no GT found, skip this crop
            if gt_label is None:
                continue
            
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
                'image': img_name,
                'bbox': [x1_exp, y1_exp, x2_exp, y2_exp],
                'gt_label': int(gt_label),
                'pred_label': int(pred_label),
                'healthy_prob': float(healthy_prob),
                'fractured_prob': float(fractured_prob),
                'correct': correct
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
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        
        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
    else:
        accuracy = precision = recall = specificity = f1 = 0.0
    
    # Print results
    print("=" * 80)
    print("📊 CROP-LEVEL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Crops Evaluated: {total}")
    print(f"  - GT Fractured: {stats['gt_fractured']}")
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
        'total_crops': total,
        'gt_fractured_crops': stats['gt_fractured'],
        'gt_healthy_crops': stats['gt_healthy'],
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
    GT_FILE = 'outputs/risk_zones_vit/stage2_gt_evaluation/stage2_evaluation_results_gt.json'
    OUTPUT_FILE = 'outputs/full_pipeline_crop_level_evaluation.json'
    
    # Run evaluation
    evaluate_pipeline_crop_level(
        stage1_model_path=STAGE1_MODEL,
        vit_checkpoint_path=VIT_CHECKPOINT,
        test_images_dir=TEST_DIR,
        gt_annotations_file=GT_FILE,
        output_file=OUTPUT_FILE,
        confidence=0.3,
        bbox_scale=2.2
    )
