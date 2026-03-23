"""
70 Image Combined Test - Risk Zone Visualization with Ground Truth
==================================================================

Creates transparent risk zone visualization showing:
- Green Zone: Auto-triaged healthy (0% fractured crops)
- Yellow Zone: Uncertain (1-10% fractured crops)  
- Red Zone: High risk (>10% fractured crops)

With ground truth labels and confusion matrix overlay.
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
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns


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
    """Apply Super-Resolution + CLAHE preprocessing"""
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


def predict_crop_vit(crop_img, vit_model, transform, device):
    """Predict single crop using ViT with SR+CLAHE preprocessing"""
    crop_processed = apply_sr_clahe_preprocessing(crop_img)
    crop_pil = Image.fromarray(cv2.cvtColor(crop_processed, cv2.COLOR_BGR2RGB))
    crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = vit_model(crop_tensor)
        probs = torch.softmax(outputs, dim=1)
        healthy_prob = probs[0, 0].item()
        fractured_prob = probs[0, 1].item()
        pred_class = torch.argmax(probs, dim=1).item()
    
    return healthy_prob, fractured_prob, pred_class


def load_test_images_combined(dataset_2021_fractured_dir, dataset_2021_healthy_dir, new_data_test_dir):
    """Load combined test set"""
    test_images = []
    
    # Load Dataset_2021 fractured (25 images)
    fractured_2021_path = Path(dataset_2021_fractured_dir)
    fractured_files = sorted(fractured_2021_path.glob('*.jpg'))[:25]
    for img_path in fractured_files:
        test_images.append({
            'path': img_path,
            'gt_label': 'fractured',
            'source': 'Dataset_2021'
        })
    
    # Load Dataset_2021 healthy (25 images)
    healthy_2021_path = Path(dataset_2021_healthy_dir)
    healthy_files = sorted(healthy_2021_path.glob('*.jpg'))[:25]
    for img_path in healthy_files:
        test_images.append({
            'path': img_path,
            'gt_label': 'healthy',
            'source': 'Dataset_2021'
        })
    
    # Load new_data test set (20 images)
    new_data_path = Path(new_data_test_dir)
    for img_path in sorted(new_data_path.glob('*.jpg')):
        txt_path = img_path.with_suffix('.txt')
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
                has_fractures = len(lines) > 0
        else:
            has_fractures = False
        
        gt_label = 'fractured' if has_fractures else 'healthy'
        test_images.append({
            'path': img_path,
            'gt_label': gt_label,
            'source': 'new_data'
        })
    
    return test_images


def assign_risk_zone(fractured_ratio, total_crops):
    """Assign risk zone based on fractured crop ratio"""
    if fractured_ratio == 0.0:
        return 'green'  # Auto-triaged healthy
    elif fractured_ratio <= 0.10:
        return 'yellow'  # Uncertain, needs review
    else:
        return 'red'  # High risk


def main():
    # Configuration
    stage1_model_path = "detectors/RCTdetector_v11x_v2.pt"
    stage2_model_path = "runs/vit_sr_clahe_auto/best_model.pth"
    dataset_2021_fractured_dir = r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Fractured"
    dataset_2021_healthy_dir = r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset\Healthy"
    new_data_test_dir = r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\new_data\test"
    confidence = 0.3
    bbox_scale = 2.2
    output_dir = Path("outputs/risk_zone_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Load models
    print("📦 Loading Stage 1 (YOLOv11x)...")
    stage1_model = YOLO(stage1_model_path)
    
    print("📦 Loading Stage 2 (ViT-Small)...")
    vit_model = load_vit_model(stage2_model_path, device)
    transform = get_vit_transform()
    
    # Load test images
    print("\n📂 Loading combined test set...")
    test_images = load_test_images_combined(
        dataset_2021_fractured_dir, 
        dataset_2021_healthy_dir,
        new_data_test_dir
    )
    
    print(f"\n✅ Total: {len(test_images)} images")
    fractured_count = sum(1 for t in test_images if t['gt_label'] == 'fractured')
    healthy_count = sum(1 for t in test_images if t['gt_label'] == 'healthy')
    print(f"  - Fractured: {fractured_count}")
    print(f"  - Healthy: {healthy_count}")
    
    # Process images and collect risk zone data
    results = []
    risk_zone_stats = {
        'green': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'images': []},
        'yellow': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'images': []},
        'red': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'images': []}
    }
    
    print("\n🔬 Processing images...")
    for test_item in tqdm(test_images, desc="Evaluating"):
        img_path = test_item['path']
        gt_label = test_item['gt_label']
        source = test_item['source']
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Stage 1: Detect RCTs
        results_yolo = stage1_model.predict(img, conf=confidence, verbose=False)
        
        if len(results_yolo) == 0 or len(results_yolo[0].boxes) == 0:
            # No RCTs detected
            pred_label = 'healthy'
            risk_zone = 'green'
            fractured_ratio = 0.0
            total_crops = 0
            fractured_crops = 0
        else:
            boxes = results_yolo[0].boxes
            total_crops = len(boxes)
            fractured_crops = 0
            
            # Process each crop
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
                
                # Stage 2: Classify
                healthy_prob, fractured_prob, crop_pred = predict_crop_vit(
                    crop, vit_model, transform, device
                )
                
                if crop_pred == 1:  # Fractured
                    fractured_crops += 1
            
            # Calculate ratio and assign risk zone
            fractured_ratio = fractured_crops / total_crops if total_crops > 0 else 0.0
            risk_zone = assign_risk_zone(fractured_ratio, total_crops)
            
            # Image-level prediction based on risk zone
            if risk_zone == 'green':
                pred_label = 'healthy'
            else:
                pred_label = 'fractured'
        
        # Determine outcome
        gt_binary = 1 if gt_label == 'fractured' else 0
        pred_binary = 1 if pred_label == 'fractured' else 0
        
        if gt_binary == 1 and pred_binary == 1:
            outcome = 'TP'
        elif gt_binary == 0 and pred_binary == 0:
            outcome = 'TN'
        elif gt_binary == 0 and pred_binary == 1:
            outcome = 'FP'
        else:  # gt_binary == 1 and pred_binary == 0
            outcome = 'FN'
        
        # Store result
        result = {
            'image_name': img_path.stem,
            'gt_label': gt_label,
            'pred_label': pred_label,
            'risk_zone': risk_zone,
            'fractured_ratio': fractured_ratio,
            'total_crops': total_crops,
            'fractured_crops': fractured_crops,
            'outcome': outcome,
            'source': source
        }
        results.append(result)
        
        # Update risk zone statistics
        risk_zone_stats[risk_zone][outcome] += 1
        risk_zone_stats[risk_zone]['images'].append(result)
    
    # Create visualization
    print("\n🎨 Creating risk zone visualization...")
    create_risk_zone_visualization(results, risk_zone_stats, output_dir)
    
    # Save detailed results
    output_file = output_dir / 'risk_zone_detailed_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    print("\n✅ Risk zone visualization complete!")


def create_risk_zone_visualization(results, risk_zone_stats, output_dir):
    """Create comprehensive risk zone visualization"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Colors
    zone_colors = {
        'green': '#4CAF50',
        'yellow': '#FFC107',
        'red': '#F44336'
    }
    
    # Subplot 1: Risk Zone Distribution with GT Labels
    ax1 = fig.add_subplot(gs[0, 0])
    
    zone_counts = {'green': 0, 'yellow': 0, 'red': 0}
    zone_gt_fractured = {'green': 0, 'yellow': 0, 'red': 0}
    zone_gt_healthy = {'green': 0, 'yellow': 0, 'red': 0}
    
    for result in results:
        zone = result['risk_zone']
        zone_counts[zone] += 1
        if result['gt_label'] == 'fractured':
            zone_gt_fractured[zone] += 1
        else:
            zone_gt_healthy[zone] += 1
    
    zones = ['Green\n(0%)', 'Yellow\n(1-10%)', 'Red\n(>10%)']
    x_pos = np.arange(len(zones))
    width = 0.35
    
    # Stacked bars showing GT distribution
    fractured_counts = [zone_gt_fractured['green'], zone_gt_fractured['yellow'], zone_gt_fractured['red']]
    healthy_counts = [zone_gt_healthy['green'], zone_gt_healthy['yellow'], zone_gt_healthy['red']]
    
    bars1 = ax1.bar(x_pos, fractured_counts, width, label='GT: Fractured', 
                    color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos, healthy_counts, width, bottom=fractured_counts,
                    label='GT: Healthy', color='#4ECDC4', alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    
    # Add count labels
    for i, (frac, heal) in enumerate(zip(fractured_counts, healthy_counts)):
        total = frac + heal
        # Fractured label
        if frac > 0:
            ax1.text(i, frac/2, str(frac), ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white')
        # Healthy label
        if heal > 0:
            ax1.text(i, frac + heal/2, str(heal), ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
        # Total label
        ax1.text(i, total + 1, f'n={total}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Risk Zone (Fractured Crop Ratio)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Images', fontsize=14, fontweight='bold')
    ax1.set_title('Risk Zone Distribution with Ground Truth Labels', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(zones, fontsize=12)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(zone_counts.values()) + 5)
    
    # Subplot 2: Confusion Matrix per Risk Zone
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create confusion data for each zone
    confusion_data = []
    zone_labels = []
    
    for zone in ['green', 'yellow', 'red']:
        tp = risk_zone_stats[zone]['TP']
        fp = risk_zone_stats[zone]['FP']
        tn = risk_zone_stats[zone]['TN']
        fn = risk_zone_stats[zone]['FN']
        
        confusion_data.append([tp, fn, fp, tn])
        
        zone_name = f"{zone.capitalize()}\n(n={tp+fp+tn+fn})"
        zone_labels.append(zone_name)
    
    confusion_array = np.array(confusion_data).T
    
    # Plot grouped confusion matrix
    x = np.arange(len(zone_labels))
    width = 0.2
    
    bars_tp = ax2.bar(x - 1.5*width, confusion_array[0], width, label='TP', 
                     color='#2E7D32', alpha=0.8, edgecolor='black')
    bars_fn = ax2.bar(x - 0.5*width, confusion_array[1], width, label='FN',
                     color='#C62828', alpha=0.8, edgecolor='black')
    bars_fp = ax2.bar(x + 0.5*width, confusion_array[2], width, label='FP',
                     color='#F57C00', alpha=0.8, edgecolor='black')
    bars_tn = ax2.bar(x + 1.5*width, confusion_array[3], width, label='TN',
                     color='#1565C0', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars_tp, bars_fn, bars_fp, bars_tn]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{int(height)}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')
    
    ax2.set_xlabel('Risk Zone', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Cases', fontsize=14, fontweight='bold')
    ax2.set_title('Confusion Matrix Breakdown by Risk Zone',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(zone_labels, fontsize=11)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9, ncol=2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Subplot 3: Performance Metrics per Risk Zone
    ax3 = fig.add_subplot(gs[0, 2])
    
    metrics_data = []
    for zone in ['green', 'yellow', 'red']:
        tp = risk_zone_stats[zone]['TP']
        fp = risk_zone_stats[zone]['FP']
        tn = risk_zone_stats[zone]['TN']
        fn = risk_zone_stats[zone]['FN']
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics_data.append([accuracy * 100, precision * 100, recall * 100])
    
    metrics_array = np.array(metrics_data).T
    
    x = np.arange(len(zone_labels))
    width = 0.25
    
    bars_acc = ax3.bar(x - width, metrics_array[0], width, label='Accuracy',
                      color='#7E57C2', alpha=0.8, edgecolor='black')
    bars_prec = ax3.bar(x, metrics_array[1], width, label='Precision',
                       color='#26A69A', alpha=0.8, edgecolor='black')
    bars_rec = ax3.bar(x + width, metrics_array[2], width, label='Recall',
                      color='#EF5350', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars_acc, bars_prec, bars_rec]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Risk Zone', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax3.set_title('Performance Metrics by Risk Zone',
                  fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(zone_labels, fontsize=11)
    ax3.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, 110)
    
    # Subplot 4: Fractured Ratio Distribution
    ax4 = fig.add_subplot(gs[1, :])
    
    # Separate by GT label
    fractured_ratios_gt_frac = [r['fractured_ratio'] * 100 for r in results if r['gt_label'] == 'fractured']
    fractured_ratios_gt_heal = [r['fractured_ratio'] * 100 for r in results if r['gt_label'] == 'healthy']
    
    # Create bins
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    
    # Plot histograms
    ax4.hist(fractured_ratios_gt_frac, bins=bins, alpha=0.7, label='GT: Fractured',
            color='#FF6B6B', edgecolor='black', linewidth=1.5)
    ax4.hist(fractured_ratios_gt_heal, bins=bins, alpha=0.7, label='GT: Healthy',
            color='#4ECDC4', edgecolor='black', linewidth=1.5)
    
    # Add risk zone boundaries
    ax4.axvline(x=0, color='#4CAF50', linewidth=3, linestyle='--', 
               label='Green Zone (0%)', alpha=0.8)
    ax4.axvspan(0, 10, alpha=0.15, color='#FFC107', label='Yellow Zone (1-10%)')
    ax4.axvspan(10, 100, alpha=0.15, color='#F44336', label='Red Zone (>10%)')
    
    ax4.set_xlabel('Fractured Crop Ratio (%)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Images', fontsize=14, fontweight='bold')
    ax4.set_title('Distribution of Fractured Crop Ratios with Ground Truth Labels',
                  fontsize=16, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add summary text
    summary_text = f"""
    Total Images: {len(results)}
    Green Zone: {zone_counts['green']} ({zone_counts['green']/len(results)*100:.1f}%)
    Yellow Zone: {zone_counts['yellow']} ({zone_counts['yellow']/len(results)*100:.1f}%)
    Red Zone: {zone_counts['red']} ({zone_counts['red']/len(results)*100:.1f}%)
    """
    
    fig.text(0.02, 0.02, summary_text.strip(), fontsize=11, 
            family='monospace', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('70-Image Combined Test: Risk Zone Evaluation with Ground Truth',
                fontsize=20, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = output_dir / 'risk_zone_evaluation_with_GT.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Risk zone visualization saved: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
