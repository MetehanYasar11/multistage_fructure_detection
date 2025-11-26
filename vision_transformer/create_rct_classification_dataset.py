"""
RCT Classification Dataset Creator

Bu script:
1. YOLOv11x RCT detector modelini kullanarak dişleri tespit eder
2. Her tespit edilen diş için:
   - Diş bölgesini crop eder
   - Original image'deki fracture line coordinates kontrol eder
   - Bbox içinde fracture line varsa → 'fractured' sınıfı
   - Bbox içinde fracture line yoksa → 'healthy' sınıfı
3. Yeni dataset: RCT_classification_dataset/

Author: Master's Thesis Project
Date: November 23, 2025
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional
import shutil

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Monkey patch ultralytics to use weights_only=False
import ultralytics.nn.tasks as tasks_module
_original_torch_safe_load = tasks_module.torch_safe_load

def patched_torch_safe_load(file):
    """Patched version that uses weights_only=False"""
    import torch
    try:
        ckpt = torch.load(file, map_location="cpu", weights_only=False)
        return ckpt, file
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

tasks_module.torch_safe_load = patched_torch_safe_load


class RCTDatasetCreator:
    """RCT Classification Dataset Creator"""
    
    def __init__(
        self,
        detector_path: str,
        source_dataset: str,
        output_dataset: str,
        rct_class_name: str = "Root Canal Treatment",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize RCT Dataset Creator
        
        Args:
            detector_path: Path to YOLOv11x RCT detector model (.pt file)
            source_dataset: Path to original dataset (Fractured/Healthy folders)
            output_dataset: Path to output RCT classification dataset
            rct_class_name: Name of RCT class in YOLO model
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.detector_path = Path(detector_path)
        self.source_dataset = Path(source_dataset)
        self.output_dataset = Path(output_dataset)
        self.rct_class_name = rct_class_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO detector
        print(f"Loading YOLOv11x detector from: {self.detector_path}")
        self.model = YOLO(str(self.detector_path))
        
        # Get RCT class index
        self.rct_class_idx = self._get_rct_class_index()
        print(f"RCT class '{self.rct_class_name}' has index: {self.rct_class_idx}")
        
        # Create output directories
        self._create_output_dirs()
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_rct_detected': 0,
            'fractured_teeth': 0,
            'healthy_teeth': 0,
            'skipped_no_detection': 0
        }
    
    def _get_rct_class_index(self) -> int:
        """Get class index for Root Canal Treatment"""
        class_names = self.model.names  # Dict: {0: 'class1', 1: 'class2', ...}
        
        for idx, name in class_names.items():
            if name == self.rct_class_name:
                return idx
        
        raise ValueError(
            f"Class '{self.rct_class_name}' not found in model. "
            f"Available classes: {list(class_names.values())}"
        )
    
    def _create_output_dirs(self):
        """Create output directory structure"""
        # Main directories
        self.fractured_dir = self.output_dataset / "fractured"
        self.healthy_dir = self.output_dataset / "healthy"
        
        self.fractured_dir.mkdir(parents=True, exist_ok=True)
        self.healthy_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata directory
        self.metadata_dir = self.output_dataset / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        print(f"Output dataset will be created at: {self.output_dataset}")
    
    def _load_fracture_lines(self, annotation_path: Path) -> List[Tuple[float, float, float, float]]:
        """
        Load fracture line coordinates from annotation file
        
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
            for i in range(0, len(content), 2):
                if i + 1 < len(content):
                    point1 = [float(x) for x in content[i].split()]
                    point2 = [float(x) for x in content[i + 1].split()]
                    
                    x1, y1 = point1[0], point1[1]
                    x2, y2 = point2[0], point2[1]
                    
                    lines.append((x1, y1, x2, y2))
        
        return lines
    
    def _line_intersects_bbox(
        self,
        line: Tuple[float, float, float, float],
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """
        Check if fracture line intersects with bounding box
        
        Args:
            line: Fracture line (x1, y1, x2, y2)
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            True if line intersects or is inside bbox
        """
        x1, y1, x2, y2 = line
        x_min, y_min, x_max, y_max = bbox
        
        # Check if either endpoint is inside bbox
        point1_inside = (x_min <= x1 <= x_max) and (y_min <= y1 <= y_max)
        point2_inside = (x_min <= x2 <= x_max) and (y_min <= y2 <= y_max)
        
        if point1_inside or point2_inside:
            return True
        
        # Check if line crosses bbox boundaries (simplified)
        # For dental X-rays, checking endpoints is usually sufficient
        # since fracture lines are small relative to tooth size
        
        return False
    
    def process_image(
        self,
        image_path: Path,
        annotation_path: Optional[Path] = None,
        is_fractured_image: bool = False
    ) -> int:
        """
        Process single image: detect RCT teeth and crop them
        
        Args:
            image_path: Path to image file
            annotation_path: Path to annotation file (for fractured images)
            is_fractured_image: Whether this is from Fractured folder
            
        Returns:
            Number of teeth detected and saved
        """
        # Load image
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        # Load fracture lines (if fractured image)
        fracture_lines = []
        if is_fractured_image and annotation_path and annotation_path.exists():
            fracture_lines = self._load_fracture_lines(annotation_path)
        
        # Run YOLO detection
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        if len(results) == 0 or results[0].boxes is None:
            return 0
        
        # Filter for RCT detections
        boxes = results[0].boxes
        rct_detections = []
        
        for box in boxes:
            cls = int(box.cls[0])
            if cls == self.rct_class_idx:
                # Get bounding box in xyxy format
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                rct_detections.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'confidence': conf
                })
        
        if len(rct_detections) == 0:
            return 0
        
        # Process each detected RCT tooth
        num_saved = 0
        image_stem = image_path.stem
        
        for idx, detection in enumerate(rct_detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Check if fracture line is inside this bbox
            has_fracture = False
            if is_fractured_image:
                for line in fracture_lines:
                    if self._line_intersects_bbox(line, bbox):
                        has_fracture = True
                        break
            
            # Crop tooth region
            x_min, y_min, x_max, y_max = bbox
            
            # Add small padding
            padding = 20
            x_min = max(0, int(x_min) - padding)
            y_min = max(0, int(y_min) - padding)
            x_max = min(img_width, int(x_max) + padding)
            y_max = min(img_height, int(y_max) + padding)
            
            cropped_tooth = image.crop((x_min, y_min, x_max, y_max))
            
            # Determine output directory
            if has_fracture:
                output_dir = self.fractured_dir
                self.stats['fractured_teeth'] += 1
            else:
                output_dir = self.healthy_dir
                self.stats['healthy_teeth'] += 1
            
            # Save cropped tooth
            output_filename = f"{image_stem}_tooth{idx:02d}.jpg"
            output_path = output_dir / output_filename
            cropped_tooth.save(output_path, quality=95)
            
            # Save metadata
            metadata = {
                'original_image': str(image_path.name),
                'tooth_index': idx,
                'bbox': {
                    'x_min': float(x_min),
                    'y_min': float(y_min),
                    'x_max': float(x_max),
                    'y_max': float(y_max)
                },
                'confidence': float(confidence),
                'has_fracture': has_fracture,
                'fracture_lines_in_bbox': []
            }
            
            # Add fracture line info if applicable
            if has_fracture:
                for line in fracture_lines:
                    if self._line_intersects_bbox(line, bbox):
                        metadata['fracture_lines_in_bbox'].append({
                            'x1': float(line[0]),
                            'y1': float(line[1]),
                            'x2': float(line[2]),
                            'y2': float(line[3])
                        })
            
            metadata_path = self.metadata_dir / f"{output_filename}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            num_saved += 1
        
        return num_saved
    
    def process_dataset(self):
        """Process entire dataset"""
        print("\n" + "="*80)
        print("PROCESSING FRACTURED IMAGES")
        print("="*80)
        
        # Process Fractured images
        fractured_dir = self.source_dataset / "Fractured"
        if fractured_dir.exists():
            fractured_images = sorted(list(fractured_dir.glob("*.jpg")))
            
            for img_path in tqdm(fractured_images, desc="Fractured"):
                annotation_path = img_path.with_suffix('.txt')
                num_detected = self.process_image(
                    img_path,
                    annotation_path,
                    is_fractured_image=True
                )
                
                self.stats['total_images'] += 1
                if num_detected > 0:
                    self.stats['total_rct_detected'] += num_detected
                else:
                    self.stats['skipped_no_detection'] += 1
        
        print("\n" + "="*80)
        print("PROCESSING HEALTHY IMAGES")
        print("="*80)
        
        # Process Healthy images
        healthy_dir = self.source_dataset / "Healthy"
        if healthy_dir.exists():
            healthy_images = sorted(list(healthy_dir.glob("*.jpg")))
            
            for img_path in tqdm(healthy_images, desc="Healthy"):
                num_detected = self.process_image(
                    img_path,
                    annotation_path=None,
                    is_fractured_image=False
                )
                
                self.stats['total_images'] += 1
                if num_detected > 0:
                    self.stats['total_rct_detected'] += num_detected
                else:
                    self.stats['skipped_no_detection'] += 1
    
    def save_statistics(self):
        """Save dataset statistics"""
        print("\n" + "="*80)
        print("DATASET CREATION COMPLETED!")
        print("="*80)
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"Total RCT teeth detected: {self.stats['total_rct_detected']}")
        print(f"  - Fractured teeth: {self.stats['fractured_teeth']}")
        print(f"  - Healthy teeth: {self.stats['healthy_teeth']}")
        print(f"Images skipped (no RCT detected): {self.stats['skipped_no_detection']}")
        print("="*80)
        
        # Save to JSON
        stats_path = self.output_dataset / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\nStatistics saved to: {stats_path}")
        print(f"Dataset location: {self.output_dataset}")


def main():
    """Main execution"""
    # Paths
    DETECTOR_PATH = "detectors/RCTdetector_v11x.pt"
    SOURCE_DATASET = r"c:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset"
    OUTPUT_DATASET = "RCT_classification_dataset"
    
    # Create dataset
    creator = RCTDatasetCreator(
        detector_path=DETECTOR_PATH,
        source_dataset=SOURCE_DATASET,
        output_dataset=OUTPUT_DATASET,
        rct_class_name="Root Canal Treatment",
        conf_threshold=0.25,  # Lower threshold to catch more RCT teeth
        iou_threshold=0.45
    )
    
    # Process all images
    creator.process_dataset()
    
    # Save statistics
    creator.save_statistics()


if __name__ == "__main__":
    main()
