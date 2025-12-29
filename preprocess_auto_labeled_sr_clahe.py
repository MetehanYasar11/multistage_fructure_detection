"""
Apply SR + CLAHE preprocessing to auto_labeled_crops
Creates: auto_labeled_crops_sr_clahe/
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def apply_clahe(img, clip_limit=2.0, tile_size=16):
    """Apply CLAHE preprocessing"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)

def apply_super_resolution_bicubic(img, scale=4):
    """Apply bicubic 4x upscaling"""
    h, w = img.shape[:2]
    enhanced = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    return enhanced

def preprocess_sr_clahe(img):
    """Apply SR then CLAHE"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Step 1: Super-resolution (4x upscale)
    sr_img = apply_super_resolution_bicubic(img_gray, scale=4)
    
    # Step 2: CLAHE on SR image
    sr_clahe = apply_clahe(sr_img, clip_limit=2.0, tile_size=16)
    
    # Resize back to original size for consistent training
    h, w = img_gray.shape[:2]
    result = cv2.resize(sr_clahe, (w, h), interpolation=cv2.INTER_AREA)
    
    # Convert back to BGR for saving
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result_bgr

def preprocess_dataset(input_dir, output_dir):
    """Preprocess all images with SR+CLAHE"""
    print("\n" + "="*80)
    print("🎨 Preprocessing Auto-Labeled Dataset: SR + CLAHE")
    print("="*80)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directories
    for class_name in ['fractured', 'healthy']:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    
    # Process each class
    for class_name in ['fractured', 'healthy']:
        input_class_dir = input_dir / class_name
        output_class_dir = output_dir / class_name
        
        # Get all images
        images = list(input_class_dir.glob("*.jpg")) + list(input_class_dir.glob("*.png"))
        
        print(f"\n📂 Processing {class_name}: {len(images)} images")
        
        for img_path in tqdm(images, desc=f"  {class_name}"):
            # Load image
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"⚠️  Could not load: {img_path.name}")
                continue
            
            # Apply SR + CLAHE
            processed = preprocess_sr_clahe(img)
            
            # Save
            output_path = output_class_dir / img_path.name
            cv2.imwrite(str(output_path), processed)
            total_processed += 1
    
    # Copy metadata if exists
    for meta_file in ['dataset_statistics.json', 'README.md']:
        if (input_dir / meta_file).exists():
            shutil.copy(input_dir / meta_file, output_dir / meta_file)
    
    print(f"\n" + "="*80)
    print(f"✅ Preprocessing Complete!")
    print(f"="*80)
    print(f"Total processed: {total_processed} images")
    print(f"Output directory: {output_dir}")
    print(f"="*80)

if __name__ == "__main__":
    # Configuration
    input_dir = "auto_labeled_crops"
    output_dir = "auto_labeled_crops_sr_clahe"
    
    # Preprocess
    preprocess_dataset(input_dir, output_dir)
