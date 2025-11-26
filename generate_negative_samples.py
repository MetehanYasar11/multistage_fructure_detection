"""
Generate Negative Samples for Binary Classification
Extract RCT crops from non-fractured panoramic X-ray images

This script creates a balanced dataset:
- Positive samples: RCT crops with fractures (already have 47)
- Negative samples: RCT crops without fractures (need to generate)

Author: Dental AI Team
Date: November 2025
"""

import cv2
import numpy as np
from pathlib import Path
import json
from ultralytics import YOLO
from tqdm import tqdm
import shutil


def generate_negative_samples(
    fractured_dir='C:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset/Fractured',
    non_fractured_dir='C:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset/Non_fractured',
    split_file='vision_transformer/outputs/splits/train_val_test_split.json',
    rct_detector_path='detectors/RCTdetector_v11x.pt',
    output_dir='stage2_negative_samples',
    target_class=9,  # RCT class
    conf_threshold=0.15,
    scale_factor=3.0,
    max_samples=100,
    crop_size=(640, 640)
):
    """
    Generate negative samples (RCT crops without fractures)
    
    Strategy:
    1. Use RCT detector on non-fractured panoramic images
    2. Extract crops with same preprocessing as positive samples
    3. Save to output directory
    
    Args:
        fractured_dir: Directory with fractured panoramic images
        non_fractured_dir: Directory with non-fractured panoramic images
        split_file: JSON file with train/val/test split
        rct_detector_path: Path to trained RCT detector
        output_dir: Output directory for negative samples
        target_class: RCT class ID
        conf_threshold: Confidence threshold for RCT detection
        scale_factor: Bbox scaling factor (same as positive samples)
        max_samples: Maximum negative samples to generate
        crop_size: Size of output crops
    """
    
    print("="*70)
    print("GENERATING NEGATIVE SAMPLES (RCT crops without fractures)")
    print("="*70)
    
    # Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load split file
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    # Get non-fractured images from test split (most reliable)
    non_frac_images = []
    
    # Check which dataset has non-fractured images
    non_frac_path = Path(non_fractured_dir)
    if non_frac_path.exists():
        print(f"\nUsing non-fractured images from: {non_frac_path}")
        non_frac_images = list(non_frac_path.glob('*.jpg')) + list(non_frac_path.glob('*.png'))
        print(f"Found {len(non_frac_images)} non-fractured images")
    else:
        print(f"\nWARNING: Non-fractured directory not found: {non_frac_path}")
        print("Trying to use fractured images from different split...")
        
        # Use fractured test images that are different from training
        frac_path = Path(fractured_dir)
        if frac_path.exists():
            all_frac = list(frac_path.glob('*.jpg')) + list(frac_path.glob('*.png'))
            # Split file contains indices, not paths - use train split indices
            train_indices = set(splits.get('train', []))
            # Use images NOT in training split (val + test)
            non_frac_images = [img for idx, img in enumerate(all_frac) if idx not in train_indices]
            print(f"Using {len(non_frac_images)} fractured images from val/test split as pseudo-negatives")
    
    if len(non_frac_images) == 0:
        print("\nERROR: No images found for negative sample generation!")
        print("Please check your dataset paths.")
        return
    
    # Limit number of images to process
    if len(non_frac_images) > max_samples // 2:  # Assume ~2 RCT per image
        non_frac_images = non_frac_images[:max_samples // 2]
        print(f"Limited to {len(non_frac_images)} images to avoid overgeneration")
    
    # Load RCT detector
    print(f"\nLoading RCT detector: {rct_detector_path}")
    model = YOLO(rct_detector_path)
    
    # Process images
    print(f"\nProcessing images to extract RCT crops...")
    negative_samples = []
    crop_count = 0
    
    for img_path in tqdm(non_frac_images, desc="Extracting RCT crops"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        height, width = image.shape[:2]
        
        # Run RCT detection
        results = model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False,
            classes=[target_class]
        )
        
        # Process detections
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for idx, box in enumerate(boxes):
                # Get bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Calculate center and dimensions
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                # Scale bbox
                scaled_w = w * scale_factor
                scaled_h = h * scale_factor
                
                # Calculate scaled bbox
                scaled_x1 = int(cx - scaled_w / 2)
                scaled_y1 = int(cy - scaled_h / 2)
                scaled_x2 = int(cx + scaled_w / 2)
                scaled_y2 = int(cy + scaled_h / 2)
                
                # Clip to image boundaries
                scaled_x1 = max(0, scaled_x1)
                scaled_y1 = max(0, scaled_y1)
                scaled_x2 = min(width, scaled_x2)
                scaled_y2 = min(height, scaled_y2)
                
                # Crop image
                crop = image[scaled_y1:scaled_y2, scaled_x1:scaled_x2]
                
                if crop.size == 0:
                    continue
                
                # Resize to standard size
                crop_resized = cv2.resize(crop, crop_size, interpolation=cv2.INTER_LANCZOS4)
                
                # Save crop
                crop_filename = f"neg_{img_path.stem}_rct{idx}_{crop_count:04d}.jpg"
                crop_path = output_path / crop_filename
                cv2.imwrite(str(crop_path), crop_resized)
                
                negative_samples.append({
                    'crop_path': str(crop_path),
                    'source_image': str(img_path),
                    'bbox_original': [int(x1), int(y1), int(x2), int(y2)],
                    'bbox_scaled': [scaled_x1, scaled_y1, scaled_x2, scaled_y2],
                    'confidence': float(conf),
                    'scale_factor': scale_factor
                })
                
                crop_count += 1
                
                # Stop if we have enough samples
                if crop_count >= max_samples:
                    break
        
        if crop_count >= max_samples:
            break
    
    print(f"\n{'='*70}")
    print(f"NEGATIVE SAMPLE GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Generated {crop_count} negative samples (RCT crops without fractures)")
    print(f"Saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'total_samples': crop_count,
        'source_images': len(non_frac_images),
        'rct_detector': str(rct_detector_path),
        'conf_threshold': conf_threshold,
        'scale_factor': scale_factor,
        'crop_size': crop_size,
        'samples': negative_samples
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {output_path / 'metadata.json'}")
    
    return crop_count, output_path


def check_and_balance_dataset(positive_dir='stage2_fracture_dataset', negative_dir='stage2_negative_samples'):
    """
    Check if we have balanced positive and negative samples
    """
    pos_path = Path(positive_dir)
    neg_path = Path(negative_dir)
    
    # Count positive samples
    pos_count = 0
    for split in ['train', 'val', 'test']:
        img_dir = pos_path / split / 'images'
        if img_dir.exists():
            pos_count += len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    
    # Count negative samples
    neg_count = 0
    if neg_path.exists():
        neg_count = len(list(neg_path.glob('*.jpg')) + list(neg_path.glob('*.png')))
    
    print(f"\n{'='*70}")
    print(f"DATASET BALANCE CHECK")
    print(f"{'='*70}")
    print(f"Positive samples (with fractures): {pos_count}")
    print(f"Negative samples (no fractures): {neg_count}")
    print(f"Balance ratio: {neg_count/max(pos_count, 1):.2f}:1")
    
    if neg_count < pos_count:
        print(f"\nWARNING: Need {pos_count - neg_count} more negative samples!")
        print(f"Recommendation: Generate at least {pos_count} negative samples")
        return False
    elif neg_count > pos_count * 1.5:
        print(f"\nINFO: You have more negative samples than needed")
        print(f"This is OK, but you can limit to {pos_count} for balance")
        return True
    else:
        print(f"\n✓ Dataset is balanced!")
        return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate negative samples for binary classification')
    parser.add_argument('--non_fractured_dir', type=str,
                      default='C:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset/Non_fractured',
                      help='Directory with non-fractured panoramic images')
    parser.add_argument('--fractured_dir', type=str,
                      default='C:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset/Fractured',
                      help='Directory with fractured panoramic images (fallback)')
    parser.add_argument('--split_file', type=str,
                      default='vision_transformer/outputs/splits/train_val_test_split.json',
                      help='Path to split JSON file')
    parser.add_argument('--rct_detector', type=str,
                      default='detectors/RCTdetector_v11x.pt',
                      help='Path to RCT detector model')
    parser.add_argument('--output_dir', type=str,
                      default='stage2_negative_samples',
                      help='Output directory for negative samples')
    parser.add_argument('--max_samples', type=int, default=100,
                      help='Maximum number of negative samples to generate')
    parser.add_argument('--conf', type=float, default=0.15,
                      help='RCT detection confidence threshold')
    parser.add_argument('--scale', type=float, default=3.0,
                      help='Bbox scaling factor')
    parser.add_argument('--check_only', action='store_true',
                      help='Only check dataset balance without generating')
    
    args = parser.parse_args()
    
    if args.check_only:
        check_and_balance_dataset()
    else:
        # Generate negative samples
        count, output_dir = generate_negative_samples(
            fractured_dir=args.fractured_dir,
            non_fractured_dir=args.non_fractured_dir,
            split_file=args.split_file,
            rct_detector_path=args.rct_detector,
            output_dir=args.output_dir,
            conf_threshold=args.conf,
            scale_factor=args.scale,
            max_samples=args.max_samples
        )
        
        # Check balance
        print()
        check_and_balance_dataset(negative_dir=output_dir)
        
        print(f"\n{'='*70}")
        print(f"NEXT STEPS:")
        print(f"{'='*70}")
        print(f"1. Review generated negative samples in: {output_dir}")
        print(f"2. Train ViT classifier:")
        print(f"   python train_vit_classifier.py --negative_dir {output_dir}")
        print(f"{'='*70}")
