import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def apply_clahe(img, clip_limit=2.0, tile_size=16):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)

def apply_super_resolution_bicubic(img, scale=4):
    h, w = img.shape[:2]
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def preprocess_sr_clahe(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    sr_img = apply_super_resolution_bicubic(img_gray, scale=4)
    sr_clahe = apply_clahe(sr_img, clip_limit=2.0, tile_size=16)
    
    h, w = img_gray.shape[:2]
    result = cv2.resize(sr_clahe, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def preprocess_segmentation_dataset(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    
    images = list((input_dir / "images").glob("*.png")) + list((input_dir / "images").glob("*.jpg"))
    for img_path in tqdm(images, desc="Preprocessing Images (SR+CLAHE)"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        processed = preprocess_sr_clahe(img)
        cv2.imwrite(str(output_dir / "images" / img_path.name), processed)
        
        # Copy the mask identically since images are resized back to original shape
        mask_path = input_dir / "masks" / img_path.name
        if mask_path.exists():
            shutil.copy(mask_path, output_dir / "masks" / img_path.name)

if __name__ == "__main__":
    preprocess_segmentation_dataset("auto_labeled_segmentation", "auto_labeled_segmentation_sr_clahe")