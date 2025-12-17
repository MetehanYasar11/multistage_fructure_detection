"""
Stage 2 Precise Evaluation with GT Fracture Lines
===================================================

Ground Truth Logic:
- Fractured Crop: Fracture line INTERSECTS with expanded bbox
- Healthy Crop: NO fracture line intersection with expanded bbox

This is the EXACT same logic used in auto-labeling!
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class FractureBinaryClassifier(nn.Module):
    """Vision Transformer for binary fracture classification"""
    def __init__(self, model_name='vit_small_patch16_224', dropout=0.3):
        super(FractureBinaryClassifier, self).__init__()
        
        if 'vit_small' in model_name:
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
            hidden_dim = 384
        elif 'vit_tiny' in model_name:
            self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False)
            hidden_dim = 192
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


def line_intersects_bbox(line_start, line_end, bbox):
    """
    Check if a line intersects with a bounding box using Liang-Barsky algorithm
    EXACT SAME LOGIC AS AUTO-LABELING!
    """
    x1, y1, x2, y2 = bbox
    lx1, ly1 = line_start
    lx2, ly2 = line_end
    
    dx = lx2 - lx1
    dy = ly2 - ly1
    
    p = [-dx, dx, -dy, dy]
    q = [lx1 - x1, x2 - lx1, ly1 - y1, y2 - ly1]
    
    u1 = 0.0
    u2 = 1.0
    
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return False
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > u1:
                    u1 = t
            else:
                if t < u2:
                    u2 = t
    
    if u1 > u2:
        return False
    
    return True


def check_crop_has_fracture(bbox, fracture_lines):
    """
    Check if crop contains fracture line
    Returns True if ANY fracture line intersects the bbox
    """
    for line_start, line_end in fracture_lines:
        if line_intersects_bbox(line_start, line_end, bbox):
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
    predicted_label = 1 if fractured_prob > healthy_prob else 0
    
    return predicted_label, healthy_prob, fractured_prob


def evaluate_stage2_with_gt(
    stage1_model_path,
    vit_checkpoint_path,
    test_dir,
    output_dir,
    confidence=0.3,
    bbox_scale=2.2,
    max_images=50
):
    """
    Evaluate Stage 2 with GT fracture lines
    """
    
    print("=" * 80)
    print("🎯 STAGE 2 EVALUATION WITH GT FRACTURE LINES")
    print("=" * 80)
    print("Model: ViT-Small + SR+CLAHE")
    print("GT Logic: Fracture line INTERSECTION with expanded bbox")
    print("=" * 80)
    
    # Load models
    print("\n📦 Loading models...")
    stage1_model = YOLO(stage1_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model = load_vit_model(vit_checkpoint_path, device)
    vit_transform = get_vit_transform()
    
    print(f"✅ Stage 1: {stage1_model_path}")
    print(f"✅ Stage 2: {vit_checkpoint_path}")
    print(f"✅ Device: {device}")
    
    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test directory
    test_path = Path(test_dir)
    test_images = sorted(list(test_path.glob('*.jpg')))[:max_images]
    
    print(f"\n📂 Processing {len(test_images)} test images...")
    
    # Collect all predictions and ground truths
    all_gt_labels = []
    all_pred_labels = []
    all_crops_data = []
    
    total_crops = 0
    
    for img_idx, img_path in enumerate(tqdm(test_images, desc="Evaluating")):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Load ground truth fracture lines
        gt_file = img_path.with_suffix('.txt')
        gt_lines = load_ground_truth_lines(gt_file)
        
        # Stage 1: Detect RCTs
        results = stage1_model.predict(source=str(img_path), conf=confidence, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            continue
        
        boxes = results[0].boxes
        
        for box in boxes:
            if int(box.cls[0]) == 9:  # RCT class
                bbox = box.xyxy[0].cpu().numpy()
                conf_score = float(box.conf[0])
                
                # Scale bbox (same as training)
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
                
                # GT Label: Check if crop contains fracture line
                gt_has_fracture = check_crop_has_fracture(scaled_bbox, gt_lines)
                gt_label = 1 if gt_has_fracture else 0
                
                # Stage 2: Predict with ViT
                pred_label, healthy_prob, fractured_prob = predict_crop_vit(
                    crop, vit_model, vit_transform, device
                )
                
                # Store results
                all_gt_labels.append(gt_label)
                all_pred_labels.append(pred_label)
                
                all_crops_data.append({
                    'image': img_path.name,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'gt_label': int(gt_label),
                    'pred_label': int(pred_label),
                    'healthy_prob': float(healthy_prob),
                    'fractured_prob': float(fractured_prob),
                    'correct': bool(gt_label == pred_label)
                })
                
                total_crops += 1
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("📊 RESULTS")
    print(f"{'='*80}")
    print(f"Total Crops Evaluated: {total_crops}")
    print(f"  - GT Healthy: {all_gt_labels.count(0)}")
    print(f"  - GT Fractured: {all_gt_labels.count(1)}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_gt_labels, all_pred_labels)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"  True Negatives (Healthy → Healthy): {tn}")
    print(f"  False Positives (Healthy → Fractured): {fp}")
    print(f"  False Negatives (Fractured → Healthy): {fn}")
    print(f"  True Positives (Fractured → Fractured): {tp}")
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📈 Metrics:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"{'='*80}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['Healthy', 'Fractured'],
                yticklabels=['Healthy', 'Fractured'],
                cbar_kws={'label': 'Percentage'})
    
    plt.title(f'Stage 2 Confusion Matrix (ViT + SR+CLAHE)\nAccuracy: {accuracy*100:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth (GT Fracture Lines)', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    
    # Add counts in each cell
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percentage = cm_normalized[i, j]
            plt.text(j + 0.5, i + 0.7, f'n={count}', 
                    ha='center', va='center', fontsize=10, color='darkred')
    
    plt.tight_layout()
    cm_path = output_path / 'stage2_confusion_matrix_gt.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved: {cm_path}")
    plt.close()
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(all_gt_labels, all_pred_labels, 
                                target_names=['Healthy', 'Fractured'],
                                digits=4))
    
    # Save results
    results = {
        'total_crops': total_crops,
        'gt_distribution': {
            'healthy': int(all_gt_labels.count(0)),
            'fractured': int(all_gt_labels.count(1))
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1)
        },
        'all_crops': all_crops_data
    }
    
    results_file = output_path / 'stage2_evaluation_results_gt.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved: {results_file}")
    
    # Save some example visualizations
    print(f"\n🎨 Creating example visualizations...")
    
    # Find some correct and incorrect predictions
    correct_healthy = [d for d in all_crops_data if d['gt_label'] == 0 and d['correct']]
    correct_fractured = [d for d in all_crops_data if d['gt_label'] == 1 and d['correct']]
    incorrect_healthy = [d for d in all_crops_data if d['gt_label'] == 0 and not d['correct']]
    incorrect_fractured = [d for d in all_crops_data if d['gt_label'] == 1 and not d['correct']]
    
    print(f"  ✅ Correct Healthy: {len(correct_healthy)}")
    print(f"  ✅ Correct Fractured: {len(correct_fractured)}")
    print(f"  ❌ False Positives (Healthy→Fractured): {len(incorrect_healthy)}")
    print(f"  ❌ False Negatives (Fractured→Healthy): {len(incorrect_fractured)}")
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Stage 2 Evaluation Summary', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix (counts)
    ax = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Healthy', 'Fractured'],
                yticklabels=['Healthy', 'Fractured'])
    ax.set_title('Confusion Matrix (Counts)')
    ax.set_ylabel('Ground Truth')
    ax.set_xlabel('Predicted')
    
    # 2. Metrics Bar Chart
    ax = axes[0, 1]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1']
    metrics_values = [accuracy, precision, recall, specificity, f1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax.legend()
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    
    # 3. GT Distribution
    ax = axes[1, 0]
    gt_counts = [all_gt_labels.count(0), all_gt_labels.count(1)]
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax.pie(gt_counts, labels=['Healthy', 'Fractured'],
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title(f'Ground Truth Distribution\n(Total: {total_crops} crops)')
    
    # 4. Prediction Distribution
    ax = axes[1, 1]
    pred_counts = [all_pred_labels.count(0), all_pred_labels.count(1)]
    wedges, texts, autotexts = ax.pie(pred_counts, labels=['Healthy', 'Fractured'],
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title(f'Prediction Distribution\n(Total: {total_crops} crops)')
    
    plt.tight_layout()
    summary_path = output_path / 'stage2_evaluation_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"✅ Summary visualization saved: {summary_path}")
    plt.close()
    
    return results


if __name__ == "__main__":
    # Configuration
    STAGE1_MODEL = 'detectors/RCTdetector_v11x_v2.pt'
    VIT_CHECKPOINT = 'runs/vit_sr_clahe_auto/best_model.pth'
    TEST_DIR = r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Fractured'
    OUTPUT_DIR = 'outputs/risk_zones_vit/stage2_gt_evaluation'
    
    # Run evaluation
    results = evaluate_stage2_with_gt(
        stage1_model_path=STAGE1_MODEL,
        vit_checkpoint_path=VIT_CHECKPOINT,
        test_dir=TEST_DIR,
        output_dir=OUTPUT_DIR,
        confidence=0.3,
        bbox_scale=2.2,
        max_images=50
    )
