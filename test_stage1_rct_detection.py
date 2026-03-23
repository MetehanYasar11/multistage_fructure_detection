"""
Test Script: Stage 1 RCT Detection on Okan Dataset
Shows RCT detection results with bounding boxes on a single panoramic X-ray image
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import random

# Paths
STAGE1_MODEL = "detectors/RCTdetector_v11x.pt"
OKAN_DATASET = "okandataset_final"
OUTPUT_DIR = "outputs/stage1_test_results"

def test_stage1_detection():
    """Test Stage 1 RCT detector on a random Okan dataset image"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Stage 1 model
    print(f"Loading Stage 1 RCT detector: {STAGE1_MODEL}")
    model = YOLO(STAGE1_MODEL)
    
    # Get a random image from Okan dataset
    okan_images = list(Path(OKAN_DATASET).rglob("*.jpg")) + \
                  list(Path(OKAN_DATASET).rglob("*.png"))
    
    if not okan_images:
        print(f"ERROR: No images found in {OKAN_DATASET}")
        return
    
    # Select random image
    test_image = random.choice(okan_images)
    print(f"\nTest image: {test_image.name}")
    print(f"Full path: {test_image}")
    
    # Run detection
    print("\nRunning RCT detection...")
    results = model.predict(
        source=str(test_image),
        conf=0.25,  # Confidence threshold
        iou=0.45,   # NMS IoU threshold
        save=False,
        verbose=True
    )
    
    # Get detection results
    result = results[0]
    boxes = result.boxes
    
    print(f"\n{'='*60}")
    print(f"DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {test_image.name}")
    print(f"Image size: {result.orig_shape}")
    print(f"Number of RCT detections: {len(boxes)}")
    
    # Load original image
    img = cv2.imread(str(test_image))
    img_with_boxes = img.copy()
    
    # Draw bounding boxes
    rct_count = 0
    if len(boxes) > 0:
        print(f"\n{'Detect #':<10} {'Confidence':<12} {'BBox (x1,y1,x2,y2)':<30} {'Class':<10}")
        print("-" * 70)
        
        for idx, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls]
            
            # Only process RCT detections
            if "Root Canal Treatment" not in class_name and "RCT" not in class_name:
                continue
            
            rct_count += 1
            print(f"{rct_count:<10} {conf:.4f}      ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}){'':<8} {class_name}")
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0),  # Green color
                         3)  # Thickness
            
            # Add label
            label = f"RCT {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img_with_boxes,
                         (int(x1), int(y1) - label_h - 10),
                         (int(x1) + label_w, int(y1)),
                         (0, 255, 0),
                         -1)
            cv2.putText(img_with_boxes, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 0, 0), 2)
    else:
        print("No RCT detections found!")
    
    # Update result message
    print(f"\nFiltered to RCT only: {rct_count} detections")
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, f"stage1_result_{test_image.stem}.jpg")
    cv2.imwrite(output_path, img_with_boxes)
    
    # Also save original for comparison
    original_path = os.path.join(OUTPUT_DIR, f"stage1_original_{test_image.stem}.jpg")
    cv2.imwrite(original_path, img)
    
    print(f"\n{'='*60}")
    print(f"OUTPUT FILES")
    print(f"{'='*60}")
    print(f"Original image: {original_path}")
    print(f"Detection result: {output_path}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return output_path, rct_count

if __name__ == "__main__":
    print("="*60)
    print("Stage 1 RCT Detection Test")
    print("="*60)
    
    output_file, num_detections = test_stage1_detection()
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
    print(f"✓ Detection completed with {num_detections} RCT(s) found")
    print(f"✓ Result image saved")
