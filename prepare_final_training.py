"""
FINAL TRAINING: Add 15 new_data images to training set and retrain Stage 2
Then evaluate on 55 test images (50 from Dataset_2021 + 5 remaining from new_data)

Strategy:
1. Find new_data images (20 images from professors)
2. Split: 15 for training, 5 for testing
3. Extract crops from 15 new_data images using Stage 1 detector
4. Add to existing training set
5. Retrain Stage 2 ViT classifier
6. Evaluate on 55 test images (Dataset_2021 50 + new_data 5)
"""

import os
import shutil
import json
import random
from pathlib import Path
import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def find_new_data_location():
    """Find new_data folder location"""
    
    print("🔍 Searching for new_data location...")
    
    possible_locations = [
        Path(r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\new_data\test'),
        Path(r'C:\Users\maspe\OneDrive\Masaüstü\masterthesis\new_data'),
        Path('/ultralytics/data/new_data'),
        Path('/workspace/data/new_data'),
        Path('/workspace/new_data'),
        Path('data/new_data'),
        Path('../data/new_data'),
        Path('new_data'),
    ]
    
    for location in possible_locations:
        if location.exists():
            print(f"   ✅ Found: {location}")
            return location
    
    print("   ❌ new_data not found in common locations")
    print("   Please specify the path manually")
    return None

def get_new_data_images(new_data_path):
    """Get all images from new_data folder"""
    
    print(f"\n📂 Loading images from: {new_data_path}")
    
    # Only get JPG files (not duplicates from multiple txt files)
    images = list(new_data_path.glob('*.jpg'))
    
    print(f"   Found {len(images)} images")
    
    # Also get ground truth labels
    gt_labels = {}
    for img_path in images:
        # Find all txt files for this image
        base_name = img_path.stem  # e.g., '962144_1'
        txt_files = list(new_data_path.glob(f'{base_name}*.txt'))
        
        # If any txt has '1', image is fractured
        is_fractured = False
        for txt_file in txt_files:
            try:
                label = txt_file.read_text().strip()
                if label == '1':
                    is_fractured = True
                    break
            except:
                pass
        
        gt_labels[img_path.name] = 'fractured' if is_fractured else 'healthy'
    
    print(f"   📊 Ground truth: {sum(1 for v in gt_labels.values() if v == 'fractured')} fractured, "
          f"{sum(1 for v in gt_labels.values() if v == 'healthy')} healthy")
    
    # Save ground truth
    with open('new_data_ground_truth.json', 'w') as f:
        json.dump(gt_labels, f, indent=2)
    print(f"   ✅ Ground truth saved: new_data_ground_truth.json")
    
    return sorted(images)

def split_new_data_images(images):
    """Split new_data into 15 train + 5 test"""
    
    print("\n✂️ Splitting new_data images...")
    
    # Shuffle for random split
    images_shuffled = images.copy()
    random.shuffle(images_shuffled)
    
    train_images = images_shuffled[:15]
    test_images = images_shuffled[15:20]
    
    print(f"   📚 Training: {len(train_images)} images")
    print(f"   📊 Testing: {len(test_images)} images")
    
    # Save split info
    split_info = {
        'train': [str(img.name) for img in train_images],
        'test': [str(img.name) for img in test_images]
    }
    
    with open('new_data_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"   ✅ Split saved to: new_data_split.json")
    
    return train_images, test_images


def load_fracture_lines(annotation_path):
    """
    Load fracture line coordinates from annotation file
    
    Format: Every 2 lines = 1 fracture line
        x1 y1  ← line start
        x2 y2  ← line end
    
    Args:
        annotation_path: Path to annotation .txt file
        
    Returns:
        List of fracture lines: [(x1, y1, x2, y2), ...]
    """
    if not annotation_path.exists():
        return []
    
    lines = []
    with open(annotation_path, 'r') as f:
        content = f.read().strip().split('\n')
        
        # Each line has 2 points (4 coordinates)
        # Every 2 lines = 1 fracture line
        for i in range(0, len(content), 2):
            if i + 1 < len(content):
                try:
                    point1 = [float(x) for x in content[i].split()]
                    point2 = [float(x) for x in content[i + 1].split()]
                    
                    if len(point1) >= 2 and len(point2) >= 2:
                        x1, y1 = point1[0], point1[1]
                        x2, y2 = point2[0], point2[1]
                        
                        lines.append((x1, y1, x2, y2))
                except (ValueError, IndexError) as e:
                    print(f"      ⚠️  Warning: Failed to parse line {i//2+1}: {e}")
                    continue
    
    return lines


def line_intersects_bbox(line, bbox):
    """
    Check if a fracture line intersects with a bbox
    
    Args:
        line: (x1, y1, x2, y2) - fracture line coordinates
        bbox: (x1, y1, x2, y2) - bbox coordinates
        
    Returns:
        True if line intersects bbox
    """
    lx1, ly1, lx2, ly2 = line
    bx1, by1, bx2, by2 = bbox
    
    # Check if line endpoints are inside bbox
    if (bx1 <= lx1 <= bx2 and by1 <= ly1 <= by2) or \
       (bx1 <= lx2 <= bx2 and by1 <= ly2 <= by2):
        return True
    
    # Check if line crosses bbox boundaries (simplified)
    # If line endpoints are on opposite sides of bbox, it likely crosses
    line_crosses_x = (lx1 < bx1 and lx2 > bx2) or (lx1 > bx2 and lx2 < bx1)
    line_crosses_y = (ly1 < by1 and ly2 > by2) or (ly1 > by2 and ly2 < by1)
    
    if line_crosses_x or line_crosses_y:
        return True
    
    return False


def extract_crops_from_images(images, new_data_path, detector_path, output_dir, label_suffix=""):
    """
    Extract RCT crops from images using Stage 1 detector
    AND label them based on fracture line detection
    
    Args:
        images: List of image paths
        new_data_path: Path to new_data directory (for annotation files)
        detector_path: Path to YOLOv11x detector
        output_dir: Output directory for crops
        label_suffix: Suffix for crop labels (e.g., "_new_train", "_new_test")
    """
    
    print(f"\n🔍 Extracting crops from {len(images)} images...")
    print(f"   Detector: {detector_path}")
    print(f"   Output: {output_dir}")
    
    # Load detector
    detector = YOLO(str(detector_path))
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detection config
    conf_threshold = 0.3
    bbox_scale = 2.2
    
    crops_info = []
    total_crops = 0
    fractured_crops = 0
    healthy_crops = 0
    
    for img_path in images:
        print(f"\n   Processing: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"      ⚠️ Failed to load image")
            continue
        
        # Load fracture lines (if annotation exists)
        annotation_path = img_path.with_suffix('.txt')
        fracture_lines = load_fracture_lines(annotation_path)
        
        if fracture_lines:
            print(f"      📍 Found {len(fracture_lines)} fracture lines")
        
        # Run detection
        results = detector.predict(
            source=img,
            conf=conf_threshold,
            verbose=False
        )
        
        # Extract crops
        image_crops = 0
        
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get class index
                cls_idx = int(box.cls[0].cpu().numpy())
                
                # ONLY extract RCT crops (class 9)
                if cls_idx != 9:
                    continue
                
                # Get bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Calculate center and expand bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = (x2 - x1) * bbox_scale
                h = (y2 - y1) * bbox_scale
                
                # New bbox coordinates
                x1_new = max(0, int(cx - w/2))
                y1_new = max(0, int(cy - h/2))
                x2_new = min(img.shape[1], int(cx + w/2))
                y2_new = min(img.shape[0], int(cy + h/2))
                
                # Check if any fracture line intersects this bbox
                has_fracture = False
                if fracture_lines:
                    for line in fracture_lines:
                        if line_intersects_bbox(line, (x1_new, y1_new, x2_new, y2_new)):
                            has_fracture = True
                            break
                
                # Extract crop
                crop = img[y1_new:y2_new, x1_new:x2_new]
                
                if crop.size == 0:
                    continue
                
                # Save crop
                crop_name = f"{img_path.stem}_crop{i:02d}{label_suffix}.jpg"
                crop_path = output_dir / crop_name
                cv2.imwrite(str(crop_path), crop)
                
                # Determine label
                label = 'fractured' if has_fracture else 'healthy'
                
                crops_info.append({
                    'image': img_path.name,
                    'crop': crop_name,
                    'bbox': [int(x1_new), int(y1_new), int(x2_new), int(y2_new)],
                    'confidence': float(conf),
                    'label': label,
                    'has_fracture_line': has_fracture
                })
                
                image_crops += 1
                total_crops += 1
                
                if has_fracture:
                    fractured_crops += 1
                else:
                    healthy_crops += 1
        
        print(f"      ✅ Extracted {image_crops} crops")
    
    print(f"\n   📊 Total crops extracted: {total_crops}")
    print(f"      🔴 Fractured: {fractured_crops}")
    print(f"      🟢 Healthy: {healthy_crops}")
    
    print(f"\n   📊 Total crops extracted: {total_crops}")
    
    # Save crops info
    info_path = output_dir / 'crops_info.json'
    with open(info_path, 'w') as f:
        json.dump(crops_info, f, indent=2)
    
    print(f"   ✅ Crops info saved: {info_path}")
    
    return crops_info

def apply_sr_clahe_preprocessing(input_dir, output_dir):
    """Apply SR+CLAHE preprocessing to crops"""
    
    print(f"\n🎨 Applying SR+CLAHE preprocessing...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    crops = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    for crop_path in crops:
        # Load image
        img = cv2.imread(str(crop_path))
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Super-resolution (4x bicubic)
        h, w = img.shape
        img_sr = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        img_clahe = clahe.apply(img_sr)
        
        # Save
        output_path = output_dir / crop_path.name
        cv2.imwrite(str(output_path), img_clahe)
    
    print(f"   ✅ Preprocessed {len(crops)} crops")


def split_crops_by_class(crops_info, train_ratio=0.75, seed=42):
    """
    Split crops into train/validation sets, stratified by class
    
    Args:
        crops_info: List of crop metadata with 'label' field
        train_ratio: Ratio of training samples (default 0.75)
        seed: Random seed for reproducibility
        
    Returns:
        train_crops, val_crops (lists of crop names)
    """
    print(f"\n✂️  Splitting crops by class ({int(train_ratio*100)}% train / {int((1-train_ratio)*100)}% val)...")
    
    # Separate crops by class
    fractured_crops = [c for c in crops_info if c.get('label') == 'fractured']
    healthy_crops = [c for c in crops_info if c.get('label') == 'healthy']
    
    print(f"   Total crops: {len(crops_info)}")
    print(f"   - Fractured: {len(fractured_crops)}")
    print(f"   - Healthy: {len(healthy_crops)}")
    
    # Split each class separately
    random.seed(seed)
    
    # Shuffle fractured
    random.shuffle(fractured_crops)
    n_train_fractured = int(len(fractured_crops) * train_ratio)
    train_fractured = fractured_crops[:n_train_fractured]
    val_fractured = fractured_crops[n_train_fractured:]
    
    # Shuffle healthy
    random.shuffle(healthy_crops)
    n_train_healthy = int(len(healthy_crops) * train_ratio)
    train_healthy = healthy_crops[:n_train_healthy]
    val_healthy = healthy_crops[n_train_healthy:]
    
    # Combine
    train_crops = train_fractured + train_healthy
    val_crops = val_fractured + val_healthy
    
    print(f"\n   📊 Training set: {len(train_crops)} crops")
    print(f"      - Fractured: {len(train_fractured)}")
    print(f"      - Healthy: {len(train_healthy)}")
    
    print(f"\n   📊 Validation set: {len(val_crops)} crops")
    print(f"      - Fractured: {len(val_fractured)}")
    print(f"      - Healthy: {len(val_healthy)}")
    
    # Save split info
    split_info = {
        'train': {
            'total': len(train_crops),
            'fractured': [c['crop'] for c in train_fractured],
            'healthy': [c['crop'] for c in train_healthy]
        },
        'val': {
            'total': len(val_crops),
            'fractured': [c['crop'] for c in val_fractured],
            'healthy': [c['crop'] for c in val_healthy]
        }
    }
    
    with open('new_data_crops_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n   ✅ Split info saved: new_data_crops_split.json")
    
    return train_crops, val_crops


def merge_training_sets(original_train_dir, new_crops_dir, output_dir):
    """Merge original training crops with new crops from new_data"""
    
    print(f"\n🔀 Merging training sets...")
    print(f"   Original: {original_train_dir}")
    print(f"   New crops: {new_crops_dir}")
    print(f"   Output: {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy original training crops
    original_crops = list(Path(original_train_dir).glob('*/*'))
    for crop in original_crops:
        if crop.is_file():
            class_dir = output_dir / crop.parent.name
            class_dir.mkdir(exist_ok=True)
            shutil.copy(crop, class_dir / crop.name)
    
    print(f"   ✅ Copied {len(original_crops)} original crops")
    
    # Copy new crops (need manual labeling or auto-labeling)
    new_crops = list(Path(new_crops_dir).glob('*.jpg'))
    print(f"   ⚠️ New crops: {len(new_crops)} (need labeling)")
    print(f"   💡 Use manual annotation or Stage 2 predictions for labeling")
    
    return output_dir

def create_final_training_script():
    """Create training script for final Stage 2 model"""
    
    script_content = '''"""
FINAL STAGE 2 TRAINING
Train ViT-Small on expanded dataset (original + 15 new_data images)
"""

import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json

# Training configuration
CONFIG = {
    'model_name': 'vit_small_patch16_224',
    'num_classes': 2,
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Data paths
    'train_dir': 'auto_labeled_crops_sr_clahe_FINAL',  # Merged dataset
    'val_dir': 'manual_annotated_crops_sr_clahe',  # Validation (unchanged)
    
    # Output
    'output_dir': 'runs/FINAL_stage2_training',
    'checkpoint_path': 'detectors/FINAL_vit_classifier.pth',
    
    # Weighted loss for class imbalance
    'class_weights': [0.73, 1.37]  # [healthy, fractured]
}

def train():
    """Train final Stage 2 model"""
    
    print("="*80)
    print("FINAL STAGE 2 TRAINING")
    print("="*80)
    
    # Create output directory
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(CONFIG['train_dir'], transform=train_transform)
    val_dataset = datasets.ImageFolder(CONFIG['val_dir'], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = timm.create_model(CONFIG['model_name'], 
                             pretrained=True, 
                             num_classes=CONFIG['num_classes'])
    model = model.to(CONFIG['device'])
    
    # Loss and optimizer (weighted for class imbalance)
    class_weights = torch.tensor(CONFIG['class_weights']).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), 
                           lr=CONFIG['learning_rate'],
                           weight_decay=CONFIG['weight_decay'])
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG['checkpoint_path'])
            print(f"   ✅ Best model saved! Val Acc: {val_acc:.2f}%")
    
    # Save training history
    with open(f"{CONFIG['output_dir']}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\\n✅ Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved: {CONFIG['checkpoint_path']}")

if __name__ == '__main__':
    train()
'''
    
    with open('train_final_stage2.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("\n✅ Created: train_final_stage2.py")

def create_final_evaluation_script():
    """Create evaluation script for 55 test images (50 + 5)"""
    
    script_content = '''"""
FINAL EVALUATION: Test on 55 images (Dataset_2021: 50 + new_data: 5)
"""

import torch
import timm
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
from ultralytics import YOLO

# Configuration
CONFIG = {
    'stage1_detector': 'detectors/RCTdetector_v11x_v2.pt',
    'stage2_classifier': 'detectors/FINAL_vit_classifier.pth',
    'test_images_dataset2021': 'path/to/Dataset_2021/test',  # 50 images
    'test_images_newdata': 'new_data_test_images',  # 5 images
    'output_dir': 'runs/FINAL_evaluation_55_images',
    
    # Detection config
    'conf_threshold': 0.3,
    'bbox_scale': 2.2,
    
    # Classification config
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def evaluate_55_images():
    """Evaluate pipeline on 55 test images"""
    
    print("="*80)
    print("FINAL EVALUATION: 55 Test Images (50 + 5)")
    print("="*80)
    
    # Load models
    print("\\n📦 Loading models...")
    detector = YOLO(CONFIG['stage1_detector'])
    
    classifier = timm.create_model('vit_small_patch16_224', num_classes=2)
    classifier.load_state_dict(torch.load(CONFIG['stage2_classifier']))
    classifier = classifier.to(CONFIG['device'])
    classifier.eval()
    
    # Get test images
    dataset2021_images = list(Path(CONFIG['test_images_dataset2021']).glob('*.jpg'))
    newdata_images = list(Path(CONFIG['test_images_newdata']).glob('*.jpg'))
    
    all_test_images = dataset2021_images + newdata_images
    
    print(f"\\n📊 Test images:")
    print(f"   Dataset_2021: {len(dataset2021_images)}")
    print(f"   new_data: {len(newdata_images)}")
    print(f"   TOTAL: {len(all_test_images)}")
    
    # Run evaluation
    results = []
    
    for img_path in all_test_images:
        print(f"\\n   Processing: {img_path.name}")
        
        # Stage 1: Detection
        img = cv2.imread(str(img_path))
        detections = detector.predict(source=img, conf=CONFIG['conf_threshold'], verbose=False)
        
        # Extract and classify crops
        fractured_count = 0
        
        for result in detections:
            boxes = result.boxes
            
            for box in boxes:
                # Extract crop (with bbox expansion)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = (x2 - x1) * CONFIG['bbox_scale']
                h = (y2 - y1) * CONFIG['bbox_scale']
                
                x1_new = max(0, int(cx - w/2))
                y1_new = max(0, int(cy - h/2))
                x2_new = min(img.shape[1], int(cx + w/2))
                y2_new = min(img.shape[0], int(cy + h/2))
                
                crop = img[y1_new:y2_new, x1_new:x2_new]
                
                # Preprocess crop (SR+CLAHE)
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_sr = cv2.resize(crop_gray, (crop_gray.shape[1]*4, crop_gray.shape[0]*4), 
                                    interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
                crop_clahe = clahe.apply(crop_sr)
                
                # Convert to tensor
                crop_pil = Image.fromarray(crop_clahe).convert('RGB')
                crop_tensor = torch.from_numpy(np.array(crop_pil)).permute(2, 0, 1).float() / 255.0
                crop_tensor = torch.nn.functional.interpolate(
                    crop_tensor.unsqueeze(0), size=(224, 224), mode='bilinear'
                )
                crop_tensor = (crop_tensor - 0.5) / 0.5
                crop_tensor = crop_tensor.to(CONFIG['device'])
                
                # Stage 2: Classification
                with torch.no_grad():
                    output = classifier(crop_tensor)
                    prob = torch.softmax(output, dim=1)[0]
                    prediction = prob[1].item()  # Fractured probability
                
                if prediction > 0.5:
                    fractured_count += 1
        
        # Image-level prediction
        image_prediction = "fractured" if fractured_count > 0 else "healthy"
        
        results.append({
            'image': img_path.name,
            'source': 'Dataset_2021' if img_path in dataset2021_images else 'new_data',
            'prediction': image_prediction,
            'fractured_crops': fractured_count
        })
        
        print(f"      Prediction: {image_prediction} ({fractured_count} fractured crops)")
    
    # Save results
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    with open(f"{CONFIG['output_dir']}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n✅ Evaluation completed!")
    print(f"   Results saved: {CONFIG['output_dir']}/results.json")
    print(f"\\n💡 Now add ground truth labels and calculate metrics!")

if __name__ == '__main__':
    evaluate_55_images()
'''
    
    with open('evaluate_final_55_images.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Created: evaluate_final_55_images.py")

def main():
    """Main workflow"""
    
    print("="*80)
    print("🎓 FINAL TRAINING PREPARATION")
    print("Add 15 new_data images to training, test on 55 images (50 + 5)")
    print("="*80)
    
    # Step 1: Find new_data location
    new_data_path = find_new_data_location()
    
    if new_data_path is None:
        print("\n❌ ERROR: Cannot find new_data folder")
        print("💡 Please set the path manually in the script")
        return
    
    # Step 2: Get and split images
    images = get_new_data_images(new_data_path)
    
    if len(images) != 20:
        print(f"\n⚠️ WARNING: Expected 20 images, found {len(images)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    train_images, test_images = split_new_data_images(images)
    
    # Step 3: Extract crops from training images
    detector_path = Path('detectors/RCTdetector_v11x_v2.pt')
    
    if not detector_path.exists():
        print(f"\n❌ ERROR: Detector not found: {detector_path}")
        return
    
    crops_output_dir = Path('new_data_crops_train')
    extract_crops_from_images(
        train_images,
        new_data_path,  # Pass new_data_path for annotation files
        detector_path, 
        crops_output_dir,
        label_suffix="_new_train"
    )
    
    # Step 4: Apply SR+CLAHE preprocessing
    preprocessed_dir = Path('new_data_crops_train_sr_clahe')
    apply_sr_clahe_preprocessing(crops_output_dir, preprocessed_dir)
    
    # Step 5: Split crops by class (75% train / 25% val)
    crops_info_path = crops_output_dir / 'crops_info.json'
    with open(crops_info_path, 'r') as f:
        crops_info = json.load(f)
    
    train_crops, val_crops = split_crops_by_class(crops_info, train_ratio=0.75, seed=42)
    
    # Step 6: Copy test images to separate folder
    test_images_dir = Path('new_data_test_images')
    test_images_dir.mkdir(exist_ok=True)
    
    for img in test_images:
        shutil.copy(img, test_images_dir / img.name)
    
    print(f"\n✅ Test images copied to: {test_images_dir}")
    
    # Step 7: Create training script
    create_final_training_script()
    create_final_evaluation_script()
    
    print("\n" + "="*80)
    print("✅ PREPARATION COMPLETED!")
    print("="*80)
    print("\n📋 NEXT STEPS:")
    print("   1. Run auto-labeling and merging:")
    print("      python auto_label_and_merge.py")
    print("      - Will use 75% of new crops for training")
    print("      - Will use 25% of new crops for validation")
    print()
    print("   2. Train final Stage 2 model:")
    print("      python train_final_stage2.py")
    print()
    print("   3. Evaluate on 55 test images:")
    print("      python evaluate_final_55_images.py")
    print()
    print("💡 Files created:")
    print("   - new_data_split.json (image-level train/test split)")
    print("   - new_data_crops_split.json (crop-level train/val split)")
    print("   - train_final_stage2.py (training script)")
    print("   - evaluate_final_55_images.py (evaluation script)")

if __name__ == '__main__':
    main()
