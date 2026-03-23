"""
3-Class Segmentation Dataset Generator  (Liang-Barsky clipping)
================================================================
Creates segmentation crops with 3-class masks:
  - 0   = Background
  - 128 = RCT canal filling  (from Healthy/*.txt)
  - 255 = Vertical root fracture (from Fractured/*.txt)

GT lines are clipped to the crop rectangle with the Liang-Barsky
algorithm so that every annotation line that intersects a crop is
correctly retained.  Crops with NO GT line inside are **skipped**
(no all-black masks enter the dataset).

Usage:
    python create_auto_seg_crops_3class.py
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ── Patch for safe loading ──────────────────────────────────────────────────
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


# ── Liang-Barsky Line Clipping ──────────────────────────────────────────────
def liang_barsky_clip(line, bbox):
    """Clip a line segment to a rectangular bounding box.

    Uses the Liang-Barsky algorithm which is more efficient and
    numerically robust than Cohen-Sutherland for axis-aligned
    rectangles.

    Args:
        line: (x1, y1, x2, y2) – endpoints of the segment.
        bbox: (x_min, y_min, x_max, y_max) – clipping rectangle.

    Returns:
        (cx1, cy1, cx2, cy2) clipped segment, or None if the line
        is entirely outside the rectangle.
    """
    x1, y1, x2, y2 = line
    x_min, y_min, x_max, y_max = bbox

    dx = x2 - x1
    dy = y2 - y1

    # p, q arrays for the four edges
    p = [-dx, dx, -dy, dy]
    q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]

    t0 = 0.0
    t1 = 1.0

    for pi, qi in zip(p, q):
        if pi == 0:
            # Line is parallel to this edge
            if qi < 0:
                return None          # outside and parallel → reject
            # else: inside this edge → continue
        else:
            t = qi / pi
            if pi < 0:              # entering edge
                t0 = max(t0, t)
            else:                    # leaving edge
                t1 = min(t1, t)
            if t0 > t1:
                return None          # fully clipped away

    cx1 = x1 + t0 * dx
    cy1 = y1 + t0 * dy
    cx2 = x1 + t1 * dx
    cy2 = y1 + t1 * dy
    return (cx1, cy1, cx2, cy2)


def parse_lines_from_txt(txt_path):
    """Parse line coordinates from annotation txt file.
    
    Format: pairs of lines, each pair = one line segment
        x1 y1
        x2 y2
    """
    lines = []
    if txt_path is None or not txt_path.exists():
        return lines
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
    return lines


def process_image_for_segmentation(img_path, txt_path, detector, output_dir,
                                    is_fractured, bbox_scale=2.2):
    """Process a single image: detect RCT regions, create 3-class masks.

    Crops whose GT annotation lines do NOT intersect the crop rectangle
    are **skipped entirely** so no all-black masks enter the dataset.

    Args:
        img_path: Path to the panoramic radiograph
        txt_path: Path to annotation txt (fracture lines or canal lines)
        detector: YOLO detector model
        output_dir: Output directory
        is_fractured: True if image is from Fractured folder
        bbox_scale: Bounding box scale factor for cropping

    Returns:
        (n_fractured, n_healthy, n_skipped) crop counts
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0, 0

    # Parse annotation lines
    lines = parse_lines_from_txt(txt_path)

    # Determine mask value based on source folder
    line_value = 255 if is_fractured else 128

    results = detector(img, verbose=False, conf=0.3, iou=0.45, classes=[9])

    n_fractured = 0
    n_healthy = 0
    n_skipped = 0

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy()
        conf_vals = res.boxes.conf.cpu().numpy()
        cls_vals = res.boxes.cls.cpu().numpy()

        for i, (box, c, cl) in enumerate(zip(boxes, conf_vals, cls_vals)):
            if c < 0.3:
                continue
            if int(cl) != 9:
                continue  # ONLY Root Canal Treatment

            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            c_x = x_min + w / 2
            c_y = y_min + h / 2

            new_w = w * bbox_scale
            new_h = h * bbox_scale

            nx_min = max(0, int(c_x - new_w / 2))
            ny_min = max(0, int(c_y - new_h / 2))
            nx_max = min(img.shape[1], int(c_x + new_w / 2))
            ny_max = min(img.shape[0], int(c_y + new_h / 2))

            crop_img = img[ny_min:ny_max, nx_min:nx_max]
            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue

            # Create 3-class mask (uint8)
            mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
            crop_box = (nx_min, ny_min, nx_max, ny_max)

            has_line_in_crop = False
            for line in lines:
                clipped = liang_barsky_clip(line, crop_box)
                if clipped:
                    has_line_in_crop = True
                    cx1, cy1, cx2, cy2 = clipped
                    # Shift to crop-local coordinates
                    cx1 = int(round(cx1 - nx_min))
                    cy1 = int(round(cy1 - ny_min))
                    cx2 = int(round(cx2 - nx_min))
                    cy2 = int(round(cy2 - ny_min))
                    # Draw with appropriate class value
                    cv2.line(mask, (cx1, cy1), (cx2, cy2), int(line_value),
                             thickness=3)

            # ── Skip crops with no GT inside ────────────────────────────
            # A crop with no annotation line → all-black mask → useless
            # for training and will confuse the model.
            if not has_line_in_crop:
                n_skipped += 1
                continue

            # ── Label from source folder ────────────────────────────────
            if is_fractured:
                label_str = "fractured"
                n_fractured += 1
            else:
                label_str = "healthy"
                n_healthy += 1

            base_name = f"{img_path.stem}_crop{i}_{label_str}"

            cv2.imwrite(str(output_dir / "images" / f"{base_name}.png"),
                        crop_img)
            cv2.imwrite(str(output_dir / "masks" / f"{base_name}.png"), mask)

    return n_fractured, n_healthy, n_skipped


def main():
    source_dir = Path("okandataset_final/Dataset")
    output_dir = Path("auto_labeled_segmentation_3class")
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    detector = YOLO("detectors/RCTdetector_v11x_v2.pt")

    # ── Collect images with folder info ──────────────────────────────────
    fractured_imgs = (
        list((source_dir / "Fractured").glob("*.jpg"))
        + list((source_dir / "Fractured").glob("*.png"))
    )
    healthy_imgs = (
        list((source_dir / "Healthy").glob("*.jpg"))
        + list((source_dir / "Healthy").glob("*.png"))
    )

    total_fractured = 0
    total_healthy = 0
    total_skipped = 0

    print(f"Source: {len(fractured_imgs)} fractured + {len(healthy_imgs)} healthy images")
    print(f"Output: {output_dir}")
    print()

    # ── Process Fractured images ─────────────────────────────────────────
    print("Processing Fractured images (lines → fracture class = 255)...")
    for img_path in tqdm(fractured_imgs, desc="Fractured"):
        txt_path = img_path.with_suffix(".txt")
        nf, nh, ns = process_image_for_segmentation(
            img_path, txt_path, detector, output_dir,
            is_fractured=True
        )
        total_fractured += nf
        total_healthy += nh
        total_skipped += ns

    # ── Process Healthy images ───────────────────────────────────────────
    print("\nProcessing Healthy images (lines → canal class = 128)...")
    for img_path in tqdm(healthy_imgs, desc="Healthy"):
        txt_path = img_path.with_suffix(".txt")
        nf, nh, ns = process_image_for_segmentation(
            img_path, txt_path, detector, output_dir,
            is_fractured=False
        )
        total_fractured += nf
        total_healthy += nh
        total_skipped += ns

    # ── Summary ──────────────────────────────────────────────────────────
    total = total_fractured + total_healthy
    print(f"\n{'='*60}")
    print(f"3-Class Segmentation Dataset Generated  (Liang-Barsky)")
    print(f"{'='*60}")
    print(f"Total crops kept:  {total}")
    print(f"  Fractured:       {total_fractured}  (mask has 255 = fracture)")
    print(f"  Healthy (canal): {total_healthy}  (mask has 128 = canal)")
    print(f"  Skipped (no GT): {total_skipped}  (line outside crop → dropped)")
    print(f"Output dir:        {output_dir}")
    print(f"Mask classes:      0=background, 128=canal, 255=fracture")

    # ── Verify mask distribution ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Mask Verification")
    print(f"{'='*60}")
    mask_dir = output_dir / "masks"
    masks_with_255 = 0
    masks_with_128 = 0
    masks_all_black = 0
    for mp in mask_dir.glob("*.png"):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        unique = set(np.unique(m))
        if 255 in unique:
            masks_with_255 += 1
        if 128 in unique:
            masks_with_128 += 1
        if unique == {0}:
            masks_all_black += 1

    print(f"Masks with fracture (255): {masks_with_255}")
    print(f"Masks with canal   (128): {masks_with_128}")
    print(f"Masks all-black    (  0): {masks_all_black}")


if __name__ == "__main__":
    main()
