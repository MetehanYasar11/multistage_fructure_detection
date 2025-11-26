"""
Full End-to-End Pipeline: Stage 1 RCT Detection + Stage 2 Fracture Classification
Tests the complete system on real validation data and calculates actual performance metrics.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import timm
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Paths
STAGE1_MODEL = "detectors/RCTdetector_v11x.pt"
STAGE2_MODEL = "runs/vit_classifier/best_model.pt"
DATASET_ROOT = r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset"
SPLIT_FILE = "vision_transformer/outputs/splits/train_val_test_split.json"
STAGE2_FRACTURE_DATASET = "stage2_fracture_dataset"
OUTPUT_DIR = "runs/full_pipeline_validation"

# Stage 1 hyperparameters (same as training)
SCALE_FACTOR = 3.0
CONF_THRESHOLD = 0.15

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


class FractureBinaryClassifier(torch.nn.Module):
    """Custom ViT-based binary classifier (same as train_vit_classifier.py)"""
    
    def __init__(self):
        super().__init__()
        # Load pretrained ViT backbone (pretrained=False for loading checkpoint)
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
        
        # Custom classification head (must match training exactly)
        hidden_dim = self.backbone.num_features  # 192 for vit_tiny
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),          # Index 0
            torch.nn.Linear(hidden_dim, 256),  # Index 1
            torch.nn.ReLU(),                   # Index 2
            torch.nn.Dropout(0.3),             # Index 3
            torch.nn.Linear(256, 2)            # Index 4
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class Stage2Classifier:
    """Vision Transformer binary classifier for fracture detection"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Load model architecture (same as training)
        self.model = FractureBinaryClassifier()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Transform (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """
        Predict if crop has fracture
        Args:
            image: PIL Image or numpy array
        Returns:
            has_fracture (bool), confidence (float)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Transform and predict
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        has_fracture = (pred_class == 1)
        return has_fracture, confidence


def load_validation_images():
    """Load validation split image paths"""
    with open(SPLIT_FILE, 'r') as f:
        split_data = json.load(f)
    
    # Get validation split indices
    val_indices = split_data['val']
    
    # Get all image names from Fractured and Healthy directories
    fractured_dir = Path(DATASET_ROOT) / "Fractured"
    healthy_dir = Path(DATASET_ROOT) / "Healthy"
    
    all_images = []
    # Collect from Fractured
    for f in sorted(fractured_dir.glob("*.jpg")):
        all_images.append(('Fractured', f.stem))
    for f in sorted(fractured_dir.glob("*.png")):
        all_images.append(('Fractured', f.stem))
    
    # Collect from Healthy
    for f in sorted(healthy_dir.glob("*.jpg")):
        all_images.append(('Healthy', f.stem))
    for f in sorted(healthy_dir.glob("*.png")):
        all_images.append(('Healthy', f.stem))
    
    # Get validation images by index
    val_images = [all_images[idx] for idx in val_indices]
    
    print(f"✓ Loaded {len(val_images)} validation images")
    print(f"  Fractured: {sum(1 for cls, _ in val_images if cls == 'Fractured')}")
    print(f"  Healthy: {sum(1 for cls, _ in val_images if cls == 'Healthy')}")
    return val_images


def load_fracture_ground_truth():
    """
    Load ground truth for fractured teeth from Stage 2 dataset
    Returns: dict mapping image_name -> list of crop indices with fractures
    """
    gt_data = {}
    
    if not os.path.exists(STAGE2_FRACTURE_DATASET):
        print("⚠ Stage 2 fracture dataset not found, will use image-level labels")
        # Use class labels: Fractured images have fractures, Healthy don't
        # But we don't know which crops have fractures without manual annotation
        return gt_data
    
    # Read all crop filenames (format: XXXX_cropYY.png or XXXX_cropYY.jpg)
    for crop_file in os.listdir(STAGE2_FRACTURE_DATASET):
        if crop_file.endswith(('.png', '.jpg')):
            # Parse filename: 0001_crop02.png -> image: 0001, crop_idx: 2
            parts = crop_file.replace('.png', '').replace('.jpg', '').split('_crop')
            if len(parts) == 2:
                image_name = parts[0]
                crop_idx = int(parts[1])
                
                if image_name not in gt_data:
                    gt_data[image_name] = []
                gt_data[image_name].append(crop_idx)
    
    print(f"✓ Loaded ground truth for {len(gt_data)} images with fractures")
    if len(gt_data) == 0:
        print("⚠ No crop-level annotations found. Will use image-level evaluation.")
    return gt_data


def stage1_detect_rcts(image_path, model, scale_factor=3.0, conf=0.15):
    """
    Stage 1: Detect RCT teeth using YOLOv11x
    Returns: list of crops (numpy arrays) and their bboxes
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return [], []
    
    h, w = img.shape[:2]
    
    # Upscale for better detection
    img_scaled = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
    
    # Detect
    results = model(img_scaled, conf=conf, verbose=False)
    
    crops = []
    bboxes = []
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Scale back to original coordinates
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)
            
            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Extract crop
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                bboxes.append([x1, y1, x2, y2])
    
    return crops, bboxes


def stage2_classify_fractures(crops, classifier):
    """
    Stage 2: Classify each crop for fracture presence
    Returns: list of (has_fracture, confidence) tuples
    """
    predictions = []
    
    for crop in crops:
        has_fracture, confidence = classifier.predict(crop)
        predictions.append((has_fracture, confidence))
    
    return predictions


def evaluate_single_image(image_info, stage1_model, stage2_classifier, gt_data):
    """
    Evaluate pipeline on a single image
    Args:
        image_info: tuple of (class_name, image_stem) e.g. ('Fractured', '0001')
    Returns: detailed results dict
    """
    class_name, image_stem = image_info
    
    # Try both .jpg and .png extensions
    image_path_jpg = os.path.join(DATASET_ROOT, class_name, f"{image_stem}.jpg")
    image_path_png = os.path.join(DATASET_ROOT, class_name, f"{image_stem}.png")
    image_path = image_path_jpg if os.path.exists(image_path_jpg) else image_path_png
    
    # Stage 1: Detect RCTs
    crops, bboxes = stage1_detect_rcts(image_path, stage1_model, SCALE_FACTOR, CONF_THRESHOLD)
    
    # Stage 2: Classify fractures
    predictions = stage2_classify_fractures(crops, stage2_classifier)
    
    # Ground truth: image-level label
    gt_has_fracture = (class_name == 'Fractured')
    
    # Get crop-level ground truth if available
    gt_fracture_crops = gt_data.get(image_stem, [])
    
    # Analyze results
    # Image-level: Does the pipeline detect ANY fracture in this image?
    predicted_has_fracture = any(has_frac for has_frac, _ in predictions)
    
    # Image-level classification
    if gt_has_fracture and predicted_has_fracture:
        # True Positive: Fractured image, found fracture
        image_level_result = 'TP'
    elif gt_has_fracture and not predicted_has_fracture:
        # False Negative: Fractured image, but didn't find fracture
        image_level_result = 'FN'
    elif not gt_has_fracture and predicted_has_fracture:
        # False Positive: Healthy image, but predicted fracture
        image_level_result = 'FP'
    else:
        # True Negative: Healthy image, correctly predicted no fracture
        image_level_result = 'TN'
    
    results = {
        'image_name': image_stem,
        'class': class_name,
        'num_rcts_detected': len(crops),
        'num_fractures_predicted': sum(1 for has_frac, _ in predictions if has_frac),
        'gt_has_fracture': gt_has_fracture,
        'predicted_has_fracture': predicted_has_fracture,
        'image_level_result': image_level_result,
        'predictions': [],
        'gt_fracture_crops': gt_fracture_crops  # Empty if no crop-level annotation
    }
    
    # Store each crop prediction
    for crop_idx, (has_fracture, confidence) in enumerate(predictions):
        pred_result = {
            'crop_idx': crop_idx,
            'predicted_fracture': has_fracture,
            'confidence': confidence
        }
        results['predictions'].append(pred_result)
    
    return results


def visualize_pipeline_result(image_info, crops, predictions, bboxes, gt_data, save_path):
    """
    Visualize pipeline results: show detected RCTs and classification results
    Args:
        image_info: tuple of (class_name, image_stem)
    """
    class_name, image_stem = image_info
    
    # Try both extensions
    image_path_jpg = os.path.join(DATASET_ROOT, class_name, f"{image_stem}.jpg")
    image_path_png = os.path.join(DATASET_ROOT, class_name, f"{image_stem}.png")
    image_path = image_path_jpg if os.path.exists(image_path_jpg) else image_path_png
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    gt_fracture_crops = gt_data.get(image_stem, [])
    
    # Draw bboxes on image
    img_drawn = img_rgb.copy()
    for crop_idx, (bbox, (has_fracture, confidence)) in enumerate(zip(bboxes, predictions)):
        x1, y1, x2, y2 = bbox
        
        # Determine color based on prediction and ground truth
        if has_fracture:
            if crop_idx in gt_fracture_crops:
                color = (0, 255, 0)  # Green: True Positive
                label = f"TP: {confidence:.2f}"
            else:
                color = (255, 165, 0)  # Orange: False Positive
                label = f"FP: {confidence:.2f}"
        else:
            if crop_idx in gt_fracture_crops:
                color = (255, 0, 0)  # Red: False Negative (missed)
                label = "FN"
            else:
                color = (128, 128, 128)  # Gray: True Negative
                label = f"TN: {confidence:.2f}"
        
        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img_drawn, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)
    
    # Create figure
    num_crops = len(crops)
    if num_crops == 0:
        # Just show the image
        plt.figure(figsize=(10, 8))
        plt.imshow(img_drawn)
        plt.title(f"{class_name}/{image_stem} - No RCTs detected")
        plt.axis('off')
    else:
        # Show image + crops
        cols = min(4, num_crops + 1)
        rows = (num_crops + cols) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # First subplot: full image
        axes[0, 0].imshow(img_drawn)
        axes[0, 0].set_title(f"{class_name}/{image_stem}\n{len(crops)} RCTs, {len(gt_fracture_crops)} GT fractures")
        axes[0, 0].axis('off')
        
        # Rest: individual crops
        for idx, (crop, (has_fracture, confidence)) in enumerate(zip(crops, predictions)):
            row = (idx + 1) // cols
            col = (idx + 1) % cols
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(crop_rgb)
            
            # Determine status
            if has_fracture:
                if idx in gt_fracture_crops:
                    status = "✓ TP"
                    color = 'green'
                else:
                    status = "✗ FP"
                    color = 'orange'
            else:
                if idx in gt_fracture_crops:
                    status = "✗ FN"
                    color = 'red'
                else:
                    status = "✓ TN"
                    color = 'gray'
            
            axes[row, col].set_title(f"Crop {idx}\n{status} ({confidence:.2f})", 
                                      color=color, fontweight='bold')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for idx in range(num_crops + 1, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_full_pipeline():
    """
    Run complete pipeline on validation set
    """
    print("=" * 80)
    print("FULL END-TO-END PIPELINE EVALUATION")
    print("=" * 80)
    print(f"Stage 1 Model: {STAGE1_MODEL}")
    print(f"Stage 2 Model: {STAGE2_MODEL}")
    print(f"Scale Factor: {SCALE_FACTOR}x")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print()
    
    # Load models
    print("Loading models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    stage1_model = YOLO(STAGE1_MODEL)
    stage2_classifier = Stage2Classifier(STAGE2_MODEL, device=device)
    print("✓ Models loaded\n")
    
    # Load validation data
    val_images = load_validation_images()
    gt_data = load_fracture_ground_truth()
    
    # Statistics
    total_images = len(val_images)
    images_with_fractures = len([img for img in val_images if img in gt_data])
    
    print(f"Validation Set Statistics:")
    print(f"  Total images: {total_images}")
    print(f"  Images with fractures (GT): {images_with_fractures}")
    print(f"  Images without fractures: {total_images - images_with_fractures}")
    print()
    
    # Run pipeline
    print("Running pipeline on validation set...")
    print()
    
    all_results = []
    # Image-level metrics
    image_tp = 0
    image_fp = 0
    image_fn = 0
    image_tn = 0
    total_rcts_detected = 0
    total_fractures_predicted = 0
    
    # Create visualization directory
    viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    for image_info in tqdm(val_images, desc="Processing images"):
        # Evaluate
        result = evaluate_single_image(image_info, stage1_model, stage2_classifier, gt_data)
        all_results.append(result)
        
        # Accumulate image-level stats
        if result['image_level_result'] == 'TP':
            image_tp += 1
        elif result['image_level_result'] == 'FP':
            image_fp += 1
        elif result['image_level_result'] == 'FN':
            image_fn += 1
        elif result['image_level_result'] == 'TN':
            image_tn += 1
        
        total_rcts_detected += result['num_rcts_detected']
        total_fractures_predicted += result['num_fractures_predicted']
        
        # Visualize interesting cases (fractured images or false positives/negatives)
        if result['gt_has_fracture'] or result['image_level_result'] in ['FP', 'FN']:
            class_name, image_stem = image_info
            image_path_jpg = os.path.join(DATASET_ROOT, class_name, f"{image_stem}.jpg")
            image_path_png = os.path.join(DATASET_ROOT, class_name, f"{image_stem}.png")
            image_path = image_path_jpg if os.path.exists(image_path_jpg) else image_path_png
            
            crops, bboxes = stage1_detect_rcts(image_path, stage1_model, SCALE_FACTOR, CONF_THRESHOLD)
            predictions = stage2_classify_fractures(crops, stage2_classifier)
            
            viz_path = os.path.join(viz_dir, f"{class_name}_{image_stem}_result.png")
            visualize_pipeline_result(image_info, crops, predictions, bboxes, gt_data, viz_path)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("FULL PIPELINE PERFORMANCE - IMAGE LEVEL")
    print("=" * 80)
    print()
    
    # Ground truth statistics
    total_fracture_images_gt = sum(1 for r in all_results if r['gt_has_fracture'])
    total_healthy_images_gt = sum(1 for r in all_results if not r['gt_has_fracture'])
    
    print("VALIDATION SET:")
    print(f"  Total images: {total_images}")
    print(f"  Fractured images (GT): {total_fracture_images_gt}")
    print(f"  Healthy images (GT): {total_healthy_images_gt}")
    print()
    
    # Stage 1 statistics
    images_with_rct = sum(1 for r in all_results if r['num_rcts_detected'] > 0)
    print("STAGE 1 - RCT DETECTION:")
    print(f"  Images with RCT detected: {images_with_rct}/{total_images} ({images_with_rct/total_images*100:.1f}%)")
    print(f"  Total RCT crops extracted: {total_rcts_detected}")
    print(f"  Avg RCTs per image: {total_rcts_detected/total_images:.2f}")
    print()
    
    # Stage 2 statistics
    images_with_fracture_pred = sum(1 for r in all_results if r['predicted_has_fracture'])
    print("STAGE 2 - FRACTURE CLASSIFICATION:")
    print(f"  Images predicted with fracture: {images_with_fracture_pred}/{total_images}")
    print(f"  Total fracture predictions: {total_fractures_predicted}")
    print()
    
    # Image-level confusion matrix
    print("IMAGE-LEVEL CONFUSION MATRIX:")
    print(f"  True Positives (TP): {image_tp} - Fractured images correctly identified")
    print(f"  False Positives (FP): {image_fp} - Healthy images incorrectly flagged")
    print(f"  False Negatives (FN): {image_fn} - Fractured images missed")
    print(f"  True Negatives (TN): {image_tn} - Healthy images correctly identified")
    print()
    
    # Calculate precision, recall, F1
    precision = image_tp / (image_tp + image_fp) if (image_tp + image_fp) > 0 else 0
    recall = image_tp / (image_tp + image_fn) if (image_tp + image_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (image_tp + image_tn) / (image_tp + image_fp + image_fn + image_tn) if (image_tp + image_fp + image_fn + image_tn) > 0 else 0
    specificity = image_tn / (image_tn + image_fp) if (image_tn + image_fp) > 0 else 0
    
    print("PIPELINE CLASSIFICATION METRICS (IMAGE-LEVEL):")
    print(f"  ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ✅ Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  ✅ Recall (Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
    print(f"  ✅ Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  ✅ F1 Score: {f1:.4f} ({f1*100:.2f}%)")
    print()
    
    # Clinical interpretation
    print("CLINICAL INTERPRETATION:")
    print(f"  Out of {total_fracture_images_gt} fractured images:")
    print(f"    → Correctly detected: {image_tp} ({recall*100:.1f}%)")
    print(f"    → Missed: {image_fn} ({(image_fn/total_fracture_images_gt*100) if total_fracture_images_gt > 0 else 0:.1f}%)")
    print()
    print(f"  Out of {total_healthy_images_gt} healthy images:")
    print(f"    → Correctly identified: {image_tn} ({specificity*100:.1f}%)")
    print(f"    → False alarms: {image_fp} ({(image_fp/total_healthy_images_gt*100) if total_healthy_images_gt > 0 else 0:.1f}%)")
    print()
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stage1_model': STAGE1_MODEL,
        'stage2_model': STAGE2_MODEL,
        'scale_factor': SCALE_FACTOR,
        'conf_threshold': CONF_THRESHOLD,
        'validation_set': {
            'total_images': total_images,
            'images_with_fractures': images_with_fractures,
            'images_without_fractures': total_images - images_with_fractures
        },
        'stage1_metrics': {
            'total_rcts_detected': total_rcts_detected,
            'images_with_rct': sum(1 for r in all_results if r['num_rcts_detected'] > 0)
        },
        'stage2_metrics': {
            'total_fractures_predicted': total_fractures_predicted,
            'images_with_fracture_predicted': sum(1 for r in all_results if r['predicted_has_fracture'])
        },
        'image_level_metrics': {
            'true_positives': image_tp,
            'false_positives': image_fp,
            'false_negatives': image_fn,
            'true_negatives': image_tn
        },
        'classification_metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy)
        },
        'detailed_results': all_results
    }
    
    results_path = os.path.join(OUTPUT_DIR, 'pipeline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Visualizations saved to: {viz_dir}")
    print()
    
    # Create summary visualization
    create_summary_plots(results_summary, OUTPUT_DIR)
    
    return results_summary


def create_summary_plots(results, output_dir):
    """Create summary visualization plots"""
    
    # 1. Confusion Matrix (Image-Level)
    tp = results['image_level_metrics']['true_positives']
    fp = results['image_level_metrics']['false_positives']
    fn = results['image_level_metrics']['false_negatives']
    tn = results['image_level_metrics']['true_negatives']
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Fractured'],
                yticklabels=['Healthy', 'Fractured'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Full Pipeline (Image Level)', fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Metrics Bar Chart
    metrics = results['classification_metrics']
    metric_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    metric_values = [metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['accuracy']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12)
    plt.title('Full Pipeline Performance Metrics', fontsize=14, fontweight='bold')
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Summary plots created")


if __name__ == "__main__":
    results = run_full_pipeline()
    
    print("\n" + "=" * 80)
    print("PIPELINE EVALUATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"📊 Precision: {results['classification_metrics']['precision']:.2%}")
    print(f"📊 Recall: {results['classification_metrics']['recall']:.2%}")
    print(f"📊 F1 Score: {results['classification_metrics']['f1_score']:.2%}")
    print(f"📊 Accuracy: {results['classification_metrics']['accuracy']:.2%}")
    print()
    print(f"Results saved to: {OUTPUT_DIR}")
