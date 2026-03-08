import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

import ultralytics.nn.tasks as tasks_module

def patched_torch_safe_load(file):
    import torch
    try:
        ckpt = torch.load(file, map_location="cpu", weights_only=False)
        return ckpt, file
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

tasks_module.torch_safe_load = patched_torch_safe_load

def line_intersection_bbox(line, bbox):
    """Clip line to bounding box, returning intersection segment if any."""
    x1, y1, x2, y2 = line
    x_min, y_min, x_max, y_max = bbox

    # Cohen-Sutherland clipping algorithm
    INSIDE = 0; LEFT = 1; RIGHT = 2; BOTTOM = 4; TOP = 8
    def compute_out_code(x, y):
        code = INSIDE
        if x < x_min: code |= LEFT
        elif x > x_max: code |= RIGHT
        if y < y_min: code |= BOTTOM
        elif y > y_max: code |= TOP
        return code

    outcode1 = compute_out_code(x1, y1)
    outcode2 = compute_out_code(x2, y2)
    accept = False

    while True:
        if not (outcode1 | outcode2):
            accept = True
            break
        elif outcode1 & outcode2:
            break
        else:
            outcode_out = outcode1 if outcode1 else outcode2
            if outcode_out & TOP:
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif outcode_out & BOTTOM:
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif outcode_out & RIGHT:
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif outcode_out & LEFT:
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min
                
            if outcode_out == outcode1:
                x1, y1 = x, y
                outcode1 = compute_out_code(x1, y1)
            else:
                x2, y2 = x, y
                outcode2 = compute_out_code(x2, y2)

    if accept:
        return (x1, y1, x2, y2)
    return None

def process_image_for_segmentation(img_path, txt_path, detector, output_dir, bbox_scale=2.2):
    img = cv2.imread(str(img_path))
    if img is None: return

    # load lines
    lines = []
    if txt_path and txt_path.exists():
        with open(txt_path, 'r') as f:
            content = f.read().strip().split('\n')
            for i in range(0, len(content), 2):
                if i + 1 < len(content):
                    try:
                        p1 = [float(x) for x in content[i].split()]
                        p2 = [float(x) for x in content[i + 1].split()]
                        lines.append((p1[0], p1[1], p2[0], p2[1]))
                    except:
                        pass
    
    results = detector(img, verbose=False, conf=0.3, iou=0.45, classes=[9])
    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        
        for i, (box, c, cl) in enumerate(zip(boxes, conf, cls)):
            if c < 0.3: continue # confidence threshold
            if int(cl) != 9: continue # ONLY Root Canal Treatment!
            
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            c_x = x_min + w/2
            c_y = y_min + h/2
            
            new_w = w * bbox_scale
            new_h = h * bbox_scale
            
            nx_min = max(0, int(c_x - new_w/2))
            ny_min = max(0, int(c_y - new_h/2))
            nx_max = min(img.shape[1], int(c_x + new_w/2))
            ny_max = min(img.shape[0], int(c_y + new_h/2))
            
            crop_img = img[ny_min:ny_max, nx_min:nx_max]
            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0: continue
            
            # create mask
            mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
            crop_box = (nx_min, ny_min, nx_max, ny_max)
            
            has_fracture = False
            for line in lines:
                clipped = line_intersection_bbox(line, crop_box)
                if clipped:
                    has_fracture = True
                    cx1, cy1, cx2, cy2 = clipped
                    # shift to crop coordinates
                    cx1 = int(cx1 - nx_min)
                    cy1 = int(cy1 - ny_min)
                    cx2 = int(cx2 - nx_min)
                    cy2 = int(cy2 - ny_min)
                    # draw line with thickness 3
                    cv2.line(mask, (cx1, cy1), (cx2, cy2), 255, thickness=3)
            
            label_str = "fractured" if has_fracture else "healthy"
            base_name = f"{img_path.stem}_crop{i}_{label_str}"
            
            cv2.imwrite(str(output_dir / "images" / f"{base_name}.png"), crop_img)
            cv2.imwrite(str(output_dir / "masks" / f"{base_name}.png"), mask)

def main():
    source_dir = Path("okandataset_final/Dataset")
    output_dir = Path("auto_labeled_segmentation")
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    
    detector = YOLO("detectors/RCTdetector_v11x_v2.pt")
    
    all_imgs = list((source_dir / "Fractured").glob("*.jpg")) + list((source_dir / "Fractured").glob("*.png"))
    all_imgs += list((source_dir / "Healthy").glob("*.jpg")) + list((source_dir / "Healthy").glob("*.png"))
    
    for img_path in tqdm(all_imgs, desc="Generating Segmentation Dataset"):
        txt_path = img_path.with_suffix(".txt")
        process_image_for_segmentation(img_path, txt_path, detector, output_dir)

if __name__ == "__main__":
    main()