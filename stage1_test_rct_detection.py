"""
Stage 1: Test RCT Detection on Training Set

Bu script YOLOv11x RCT detector'ın eğitim veriseti üzerinde 
RCT (Root Canal Treatment) sınıfını ne doğrulukta tespit ettiğini kontrol eder.

Author: Master's Thesis Project  
Date: November 25, 2025
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Monkey patch ultralytics to use weights_only=False
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


class RCTDetectionTester:
    """Test RCT detection performance"""
    
    def __init__(
        self,
        detector_path: str = "detectors/RCTdetector_v11x.pt",
        dataset_root: str = r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset",
        split_file: str = "vision_transformer/outputs/splits/train_val_test_split.json",
        rct_class_name: str = "Root Canal Treatment",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize RCT Detection Tester
        
        Args:
            detector_path: Path to YOLOv11x RCT detector model
            dataset_root: Root directory of dataset
            split_file: Path to train/val/test split JSON
            rct_class_name: Name of RCT class in YOLO model
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.detector_path = Path(detector_path)
        self.dataset_root = Path(dataset_root)
        self.split_file = Path(split_file)
        self.rct_class_name = rct_class_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO detector
        print(f"Loading YOLOv11x detector from: {self.detector_path}")
        if not self.detector_path.exists():
            raise FileNotFoundError(f"Detector not found: {self.detector_path}")
        
        self.model = YOLO(str(self.detector_path))
        
        # Get RCT class index
        self.rct_class_idx = self._get_rct_class_index()
        print(f"✓ RCT class '{self.rct_class_name}' has index: {self.rct_class_idx}")
        
        # Load dataset split
        self.train_indices = self._load_train_split()
        print(f"✓ Loaded {len(self.train_indices)} training images")
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'images_with_rct': 0,
            'images_without_rct': 0,
            'total_rct_detections': 0,
            'fractured_with_rct': 0,
            'fractured_without_rct': 0,
            'healthy_with_rct': 0,
            'healthy_without_rct': 0,
            'detection_details': []
        }
    
    def _get_rct_class_index(self) -> int:
        """Get class index for Root Canal Treatment"""
        class_names = self.model.names  # Dict: {0: 'class1', 1: 'class2', ...}
        
        print(f"\nAvailable classes in model:")
        for idx, name in class_names.items():
            print(f"  [{idx}] {name}")
        
        for idx, name in class_names.items():
            if name == self.rct_class_name:
                return idx
        
        raise ValueError(
            f"Class '{self.rct_class_name}' not found in model. "
            f"Available classes: {list(class_names.values())}"
        )
    
    def _load_train_split(self) -> List[int]:
        """Load training split indices - or get all files if split doesn't match"""
        # Get all fractured files
        fractured_dir = self.dataset_root / "Fractured"
        if fractured_dir.exists():
            files = list(fractured_dir.glob("*.png"))
            # Extract indices from filenames (0002.png -> 2)
            indices = [int(f.stem) for f in files]
            print(f"Found {len(indices)} fractured images in dataset")
            return indices
        
        # Fallback to split file
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        
        with open(self.split_file, 'r') as f:
            splits = json.load(f)
        
        return splits['train']
    
    def _get_image_path(self, idx: int) -> Tuple[Path, str]:
        """
        Get image path and label for given index
        
        Returns:
            (image_path, label) where label is 'fractured' or 'healthy'
        """
        # Format index with 4 digits (0001, 0002, etc.)
        filename = f"{idx:04d}.png"
        
        # Try fractured first
        fractured_path = self.dataset_root / "Fractured" / filename
        if fractured_path.exists():
            return fractured_path, 'fractured'
        
        # Try healthy
        healthy_path = self.dataset_root / "Healthy" / filename
        if healthy_path.exists():
            return healthy_path, 'healthy'
        
        raise FileNotFoundError(f"Image {filename} not found in Fractured or Healthy folders")
    
    def detect_rct(self, image_path: Path) -> List[Dict]:
        """
        Detect RCT teeth in image
        
        Returns:
            List of detections: [{'bbox': [x1, y1, x2, y2], 'conf': float}, ...]
        """
        # Run YOLO inference
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                
                # Only keep RCT class
                if cls == self.rct_class_idx:
                    conf = float(boxes.conf[i])
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    detections.append({
                        'bbox': bbox.tolist(),
                        'conf': conf
                    })
        
        return detections
    
    def test_dataset(self):
        """Test RCT detection on training dataset"""
        print("\n" + "="*80)
        print("TESTING RCT DETECTION ON TRAINING SET")
        print("="*80)
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
        print(f"Total images: {len(self.train_indices)}")
        print()
        
        # Process each image
        for idx in tqdm(self.train_indices, desc="Testing RCT detection"):
            try:
                image_path, label = self._get_image_path(idx)
            except FileNotFoundError as e:
                print(f"\n⚠️  {e}")
                continue
            
            # Detect RCT
            detections = self.detect_rct(image_path)
            
            # Update statistics
            self.stats['total_images'] += 1
            
            if len(detections) > 0:
                self.stats['images_with_rct'] += 1
                self.stats['total_rct_detections'] += len(detections)
                
                if label == 'fractured':
                    self.stats['fractured_with_rct'] += 1
                else:
                    self.stats['healthy_with_rct'] += 1
            else:
                self.stats['images_without_rct'] += 1
                
                if label == 'fractured':
                    self.stats['fractured_without_rct'] += 1
                else:
                    self.stats['healthy_without_rct'] += 1
            
            # Store detail
            self.stats['detection_details'].append({
                'idx': idx,
                'label': label,
                'num_detections': len(detections),
                'detections': detections
            })
        
        # Print results
        self._print_results()
        
        # Save results
        self._save_results()
    
    def _print_results(self):
        """Print detection statistics"""
        stats = self.stats
        
        print("\n" + "="*80)
        print("RCT DETECTION RESULTS")
        print("="*80)
        
        if stats['total_images'] == 0:
            print("\n❌ No images were tested!")
            return
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total images tested: {stats['total_images']}")
        print(f"  Images WITH RCT detected: {stats['images_with_rct']} ({stats['images_with_rct']/stats['total_images']*100:.1f}%)")
        print(f"  Images WITHOUT RCT detected: {stats['images_without_rct']} ({stats['images_without_rct']/stats['total_images']*100:.1f}%)")
        print(f"  Total RCT detections: {stats['total_rct_detections']}")
        print(f"  Avg detections per image (with RCT): {stats['total_rct_detections']/max(1, stats['images_with_rct']):.2f}")
        
        print(f"\n🦷 Fractured Images:")
        total_fractured = stats['fractured_with_rct'] + stats['fractured_without_rct']
        print(f"  Total fractured: {total_fractured}")
        print(f"  With RCT detected: {stats['fractured_with_rct']} ({stats['fractured_with_rct']/max(1, total_fractured)*100:.1f}%)")
        print(f"  Without RCT detected: {stats['fractured_without_rct']} ({stats['fractured_without_rct']/max(1, total_fractured)*100:.1f}%)")
        
        print(f"\n✅ Healthy Images:")
        total_healthy = stats['healthy_with_rct'] + stats['healthy_without_rct']
        print(f"  Total healthy: {total_healthy}")
        print(f"  With RCT detected: {stats['healthy_with_rct']} ({stats['healthy_with_rct']/max(1, total_healthy)*100:.1f}%)")
        print(f"  Without RCT detected: {stats['healthy_without_rct']} ({stats['healthy_without_rct']/max(1, total_healthy)*100:.1f}%)")
        
        print(f"\n🔍 Detection Distribution:")
        detection_counts = {}
        for detail in stats['detection_details']:
            count = detail['num_detections']
            detection_counts[count] = detection_counts.get(count, 0) + 1
        
        for count in sorted(detection_counts.keys()):
            print(f"  {count} detections: {detection_counts[count]} images ({detection_counts[count]/stats['total_images']*100:.1f}%)")
        
        # Key metric for Stage 1
        print(f"\n🎯 KEY METRIC FOR STAGE 1:")
        rct_detection_rate = stats['images_with_rct'] / stats['total_images'] * 100
        print(f"  RCT Detection Rate: {rct_detection_rate:.1f}%")
        
        if rct_detection_rate >= 80:
            print(f"  ✅ EXCELLENT! Stage 1 is ready for Stage 2")
        elif rct_detection_rate >= 60:
            print(f"  ⚠️  ACCEPTABLE. Stage 1 might need some tuning")
        else:
            print(f"  ❌ POOR. Stage 1 needs significant improvement")
    
    def _save_results(self):
        """Save results to JSON file"""
        output_file = Path("outputs/stage1_rct_detection_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Main function"""
    # Create tester
    tester = RCTDetectionTester(
        detector_path="detectors/RCTdetector_v11x.pt",
        dataset_root=r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset",
        split_file="vision_transformer/outputs/splits/train_val_test_split.json",
        rct_class_name="Root Canal Treatment",
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Test dataset
    tester.test_dataset()


if __name__ == "__main__":
    main()
