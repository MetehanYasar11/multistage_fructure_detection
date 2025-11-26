"""
Stage 2 Dataset Creation

Stage 1 RCT detection sonuçlarını kullanarak Stage 2 için dataset oluşturur:
1. RCT bbox'larını 3x scale ile genişlet
2. Confidence threshold = 0.15
3. Sadece içinde ground truth fracture line olan bbox'ları kaydet
4. Crop'ları ve etiketleri YOLO formatında kaydet

Author: Master's Thesis Project  
Date: November 25, 2025
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from tqdm import tqdm
import shutil

# Monkey patch ultralytics
import ultralytics.nn.tasks as tasks_module

def patched_torch_safe_load(file):
    """Patched version that uses weights_only=False"""
    try:
        ckpt = torch.load(file, map_location="cpu", weights_only=False)
        return ckpt, file
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

tasks_module.torch_safe_load = patched_torch_safe_load


class Stage2DatasetCreator:
    """Creates Stage 2 dataset from Stage 1 detections"""
    
    def __init__(
        self,
        detector_path="detectors/RCTdetector_v11x.pt",
        dataset_root=r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset",
        split_file="vision_transformer/outputs/splits/train_val_test_split.json",
        output_dir="stage2_fracture_dataset",
        conf_threshold=0.15,
        scale_factor=3.0
    ):
        self.detector_path = Path(detector_path)
        self.dataset_root = Path(dataset_root)
        self.split_file = Path(split_file)
        self.output_dir = Path(output_dir)
        self.conf_threshold = conf_threshold
        self.scale_factor = scale_factor
        
        # Load YOLO model
        print(f"Loading YOLO detector from {self.detector_path}...")
        self.model = YOLO(str(self.detector_path))
        
        # Load splits
        with open(self.split_file, 'r') as f:
            self.splits = json.load(f)
        
        # RCT class index
        self.rct_class_idx = 9
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'images_with_rct': 0,
            'total_crops': 0,
            'crops_with_fracture': 0,
            'by_split': {}
        }
    
    def load_ground_truth_lines(self, idx):
        """Load ground truth fracture lines"""
        annotation_path = self.dataset_root / "Fractured" / f"{idx:04d}.txt"
        
        if not annotation_path.exists():
            return []
        
        lines = []
        with open(annotation_path, 'r') as f:
            content = f.read().strip().split('\n')
            
            # Each fracture = 2 lines (2 points)
            for i in range(0, len(content), 2):
                if i + 1 < len(content):
                    try:
                        point1 = [float(x) for x in content[i].split()]
                        point2 = [float(x) for x in content[i + 1].split()]
                        
                        x1, y1 = point1[0], point1[1]
                        x2, y2 = point2[0], point2[1]
                        
                        lines.append((x1, y1, x2, y2))
                    except:
                        continue
        
        return lines
    
    def detect_rct_expanded(self, image_path):
        """Detect RCT teeth with expanded bboxes"""
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=0.45,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            # Get image size for clipping
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                
                if cls == self.rct_class_idx:
                    conf = float(boxes.conf[i])
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    # Expand bbox by scale_factor
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Calculate center
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Scale width and height
                    new_width = width * self.scale_factor
                    new_height = height * self.scale_factor
                    
                    # Calculate new bbox
                    x1_new = cx - new_width / 2
                    y1_new = cy - new_height / 2
                    x2_new = cx + new_width / 2
                    y2_new = cy + new_height / 2
                    
                    # Clip to image bounds
                    x1_new = max(0, x1_new)
                    y1_new = max(0, y1_new)
                    x2_new = min(img_width, x2_new)
                    y2_new = min(img_height, y2_new)
                    
                    detections.append({
                        'bbox': [x1_new, y1_new, x2_new, y2_new],
                        'conf': conf
                    })
        
        return detections
    
    def bbox_contains_line(self, bbox, line):
        """Check if bbox contains any part of the fracture line"""
        x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox
        x1_line, y1_line, x2_line, y2_line = line
        
        # Check if either endpoint is inside bbox
        point1_inside = (x1_bbox <= x1_line <= x2_bbox and y1_bbox <= y1_line <= y2_bbox)
        point2_inside = (x1_bbox <= x2_line <= x2_bbox and y1_bbox <= y2_line <= y2_bbox)
        
        # Check if line intersects bbox (rough approximation)
        # For now, we consider it a match if ANY endpoint is inside
        return point1_inside or point2_inside
    
    def convert_to_yolo_format(self, line, bbox, crop_width, crop_height):
        """Convert fracture line to YOLO bbox format relative to crop"""
        x1_line, y1_line, x2_line, y2_line = line
        x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox
        
        # Convert line coordinates to crop space
        x1_crop = x1_line - x1_bbox
        y1_crop = y1_line - y1_bbox
        x2_crop = x2_line - x1_bbox
        y2_crop = y2_line - y1_bbox
        
        # Get bounding box around the line
        x_min = min(x1_crop, x2_crop)
        y_min = min(y1_crop, y2_crop)
        x_max = max(x1_crop, x2_crop)
        y_max = max(y1_crop, y2_crop)
        
        # Add small padding to make bbox around line
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(crop_width, x_max + padding)
        y_max = min(crop_height, y_max + padding)
        
        # Convert to YOLO format (center_x, center_y, width, height) normalized
        center_x = ((x_min + x_max) / 2) / crop_width
        center_y = ((y_min + y_max) / 2) / crop_height
        width = (x_max - x_min) / crop_width
        height = (y_max - y_min) / crop_height
        
        return center_x, center_y, width, height
    
    def process_image(self, idx, split_name):
        """Process single image: detect RCT, crop, save if has fracture"""
        image_path = self.dataset_root / "Fractured" / f"{idx:04d}.png"
        
        if not image_path.exists():
            return 0
        
        # Load image
        image = Image.open(image_path)
        
        # Detect RCT
        detections = self.detect_rct_expanded(image_path)
        
        if len(detections) == 0:
            return 0
        
        # Load ground truth
        gt_lines = self.load_ground_truth_lines(idx)
        
        if len(gt_lines) == 0:
            return 0
        
        # Process each detection
        crops_saved = 0
        
        for det_idx, detection in enumerate(detections):
            bbox = detection['bbox']
            conf = detection['conf']
            
            # Check which ground truth lines are inside this bbox
            lines_inside = []
            for line in gt_lines:
                if self.bbox_contains_line(bbox, line):
                    lines_inside.append(line)
            
            # Only save crop if it contains fracture
            if len(lines_inside) > 0:
                # Crop image
                x1, y1, x2, y2 = [int(x) for x in bbox]
                crop = image.crop((x1, y1, x2, y2))
                crop_width, crop_height = crop.size
                
                # Save crop
                output_images_dir = self.output_dir / split_name / "images"
                output_labels_dir = self.output_dir / split_name / "labels"
                output_images_dir.mkdir(parents=True, exist_ok=True)
                output_labels_dir.mkdir(parents=True, exist_ok=True)
                
                crop_filename = f"{idx:04d}_crop{det_idx:02d}.png"
                crop_path = output_images_dir / crop_filename
                crop.save(crop_path)
                
                # Save labels in YOLO format
                label_filename = f"{idx:04d}_crop{det_idx:02d}.txt"
                label_path = output_labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    for line in lines_inside:
                        # Class 0 = fracture
                        cx, cy, w, h = self.convert_to_yolo_format(line, bbox, crop_width, crop_height)
                        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                
                crops_saved += 1
                self.stats['crops_with_fracture'] += 1
        
        self.stats['total_crops'] += len(detections)
        
        return crops_saved
    
    def create_dataset(self):
        """Create complete Stage 2 dataset"""
        print(f"\n{'='*60}")
        print(f"Stage 2 Dataset Creation")
        print(f"{'='*60}")
        print(f"Confidence Threshold: {self.conf_threshold}")
        print(f"Scale Factor: {self.scale_factor}x")
        print(f"Output Directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Create output directory
        if self.output_dir.exists():
            print(f"⚠️  Output directory exists. Removing...")
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True)
        
        # Process each split
        for split_name in ['train', 'val', 'test']:
            print(f"\n📁 Processing {split_name.upper()} split...")
            
            indices = self.splits.get(split_name, [])
            
            # Filter only fractured images
            fractured_dir = self.dataset_root / "Fractured"
            fractured_files = {int(f.stem) for f in fractured_dir.glob("*.png")}
            fractured_indices = [idx for idx in indices if idx in fractured_files]
            
            print(f"   Total images: {len(indices)}")
            print(f"   Fractured images: {len(fractured_indices)}")
            
            split_stats = {
                'images_with_rct': 0,
                'crops_saved': 0
            }
            
            for idx in tqdm(fractured_indices, desc=f"   {split_name}"):
                crops = self.process_image(idx, split_name)
                if crops > 0:
                    split_stats['images_with_rct'] += 1
                    split_stats['crops_saved'] += crops
            
            self.stats['by_split'][split_name] = split_stats
            self.stats['total_images'] += len(fractured_indices)
            self.stats['images_with_rct'] += split_stats['images_with_rct']
            
            print(f"   ✅ Images with RCT: {split_stats['images_with_rct']}")
            print(f"   ✅ Crops with fracture: {split_stats['crops_saved']}")
        
        # Create data.yaml for YOLO training
        self._create_data_yaml()
        
        # Save statistics
        self._save_statistics()
        
        # Print final summary
        self._print_summary()
    
    def _create_data_yaml(self):
        """Create data.yaml for YOLO training"""
        yaml_path = self.output_dir / "data.yaml"
        
        yaml_content = f"""# Stage 2 Fracture Detection Dataset
# Created: November 25, 2025

# Paths
path: {self.output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: 1  # number of classes
names: ['fracture']  # class names

# Training settings (recommended)
# epochs: 100
# batch: 16
# imgsz: 640
# conf: 0.25
# iou: 0.45
"""
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ Created data.yaml at {yaml_path}")
    
    def _save_statistics(self):
        """Save statistics to JSON"""
        stats_path = self.output_dir / "dataset_statistics.json"
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✅ Saved statistics to {stats_path}")
    
    def _print_summary(self):
        """Print final summary"""
        print(f"\n{'='*60}")
        print(f"📊 STAGE 2 DATASET CREATION SUMMARY")
        print(f"{'='*60}")
        print(f"\n📁 Overall Statistics:")
        print(f"   Total fractured images: {self.stats['total_images']}")
        print(f"   Images with RCT detected: {self.stats['images_with_rct']}")
        print(f"   Total RCT crops: {self.stats['total_crops']}")
        print(f"   Crops with fracture: {self.stats['crops_with_fracture']}")
        
        if self.stats['total_crops'] > 0:
            fracture_rate = (self.stats['crops_with_fracture'] / self.stats['total_crops']) * 100
            print(f"   Fracture rate: {fracture_rate:.1f}%")
        
        print(f"\n📊 By Split:")
        for split_name, split_stats in self.stats['by_split'].items():
            print(f"   {split_name.upper()}:")
            print(f"      Images with RCT: {split_stats['images_with_rct']}")
            print(f"      Crops saved: {split_stats['crops_saved']}")
        
        print(f"\n{'='*60}")
        print(f"✅ Dataset ready for Stage 2 training!")
        print(f"📍 Location: {self.output_dir.absolute()}")
        print(f"🚀 Train with: yolo detect train data={self.output_dir / 'data.yaml'}")
        print(f"{'='*60}\n")


def main():
    creator = Stage2DatasetCreator(
        conf_threshold=0.15,
        scale_factor=3.0
    )
    creator.create_dataset()


if __name__ == "__main__":
    main()
