"""
U2-Net v3 — 3-Class Segmentation Evaluation
=============================================
Evaluates the 3-class model on the validation set and saves:
  - Per-class metrics (IoU, Dice, Precision, Recall)
  - Confusion matrix
  - Visual predictions (image | GT mask | predicted mask)

Classes:
    0 = Background (black)
    1 = Canal filling (blue)
    2 = Fracture (red)
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import model and dataset from training script
from train_u2net_v3 import (
    U2Net, VRFSeg3ClassDataset,
    NUM_CLASSES, CLASS_NAMES, compute_metrics
)

# Colour map for visualisation
#   class 0 (bg)       → black
#   class 1 (canal)    → blue   (BGR: 255, 128, 0)
#   class 2 (fracture) → red    (BGR: 0, 0, 255)
CMAP = np.array([
    [0,   0,   0],    # background
    [255, 128, 0],    # canal  — blue in BGR
    [0,   0,   255],  # fracture — red in BGR
], dtype=np.uint8)


def mask_to_colour(mask_2d):
    """Convert class-index mask [H,W] to BGR colour image [H,W,3]."""
    h, w = mask_2d.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        colour[mask_2d == c] = CMAP[c]
    return colour


def evaluate():
    IMG_SIZE    = 256
    DILATE_K    = 5
    SPLIT_RATIO = 0.2
    RANDOM_SEED = 42
    MAX_VIS     = 40       # maximum number of visual samples to save

    dataset_dir = Path("auto_labeled_segmentation_3class_sr_clahe")
    model_path  = Path("runs/u2net_v3_3class/best_u2net_v3.pth")
    output_dir  = Path("runs/u2net_v3_3class/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualisations"
    vis_dir.mkdir(exist_ok=True)

    all_images = sorted((dataset_dir / "images").glob("*.png"))
    _, val_imgs = train_test_split(
        all_images, test_size=SPLIT_RATIO, random_state=RANDOM_SEED
    )

    val_ds = VRFSeg3ClassDataset(val_imgs, dataset_dir / "masks",
                                  img_size=IMG_SIZE, dilate_kernel=DILATE_K)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = U2Net(in_ch=1, out_ch=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device,
                                      weights_only=True))
    model.eval()

    # ── Accumulators ──
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    all_metrics = {c: {'iou': [], 'dice': [], 'precision': [], 'recall': []}
                   for c in CLASS_NAMES}

    # Per-category counters for image-level classification
    # "Does this image contain fracture pixels?"
    img_tp = 0  # GT has fracture, pred has fracture
    img_fp = 0  # GT no fracture, pred has fracture
    img_fn = 0  # GT has fracture, pred no fracture
    img_tn = 0  # GT no fracture, pred no fracture

    print(f"Evaluating {len(val_imgs)} validation images...")
    print(f"Model: {model_path}")
    print()

    vis_count = 0

    with torch.no_grad():
        for idx in tqdm(range(len(val_ds)), desc="Evaluating"):
            img_t, mask_t = val_ds[idx]
            img_t  = img_t.unsqueeze(0).to(device)    # [1, 1, H, W]
            mask_t = mask_t.unsqueeze(0).to(device)    # [1, H, W]

            logits = model(img_t)                      # [1, C, H, W]
            pred   = logits.argmax(dim=1)              # [1, H, W]

            # ── Per-sample metrics ──
            m = compute_metrics(logits, mask_t)
            for cname, mdict in m.items():
                for k, v in mdict.items():
                    all_metrics[cname][k].append(v)

            # ── Confusion matrix ──
            p_np = pred.cpu().numpy().flatten()
            t_np = mask_t.cpu().numpy().flatten()
            for gt_c in range(NUM_CLASSES):
                for pr_c in range(NUM_CLASSES):
                    confusion[gt_c, pr_c] += ((t_np == gt_c) & (p_np == pr_c)).sum()

            # ── Image-level fracture detection ──
            gt_has_frac   = (mask_t == 2).any().item()
            pred_has_frac = (pred == 2).any().item()
            if gt_has_frac and pred_has_frac:
                img_tp += 1
            elif not gt_has_frac and pred_has_frac:
                img_fp += 1
            elif gt_has_frac and not pred_has_frac:
                img_fn += 1
            else:
                img_tn += 1

            # ── Visualisation ──
            if vis_count < MAX_VIS:
                # Original image (grayscale → BGR for stacking)
                img_np = (img_t[0, 0].cpu().numpy() * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

                gt_colour   = mask_to_colour(mask_t[0].cpu().numpy().astype(np.int64))
                pred_colour = mask_to_colour(pred[0].cpu().numpy().astype(np.int64))

                # Stack: image | GT | prediction
                canvas = np.hstack([img_bgr, gt_colour, pred_colour])

                # Add labels
                h = canvas.shape[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas, "Image", (10, 20), font, 0.5,
                            (255,255,255), 1)
                cv2.putText(canvas, "GT", (IMG_SIZE+10, 20), font, 0.5,
                            (255,255,255), 1)
                cv2.putText(canvas, "Pred", (2*IMG_SIZE+10, 20), font, 0.5,
                            (255,255,255), 1)

                fname = val_imgs[idx].stem
                cv2.imwrite(str(vis_dir / f"{fname}.png"), canvas)
                vis_count += 1

    # ── Aggregate metrics ──
    print(f"\n{'='*70}")
    print(f"  U2-Net v3 — 3-Class Evaluation Results")
    print(f"{'='*70}")

    results = {}
    for cname in CLASS_NAMES:
        avg = {k: np.mean(v) if v else 0.0
               for k, v in all_metrics[cname].items()}
        results[cname] = avg
        print(f"\n  [{cname.upper()}]")
        print(f"    IoU:       {avg['iou']:.4f}")
        print(f"    Dice:      {avg['dice']:.4f}")
        print(f"    Precision: {avg['precision']:.4f}")
        print(f"    Recall:    {avg['recall']:.4f}")

    # mIoU / mDice (all classes)
    miou  = np.mean([results[c]['iou']  for c in CLASS_NAMES])
    mdice = np.mean([results[c]['dice'] for c in CLASS_NAMES])
    # mIoU / mDice (foreground only: canal + fracture)
    fg_miou  = np.mean([results[c]['iou']  for c in ['canal', 'fracture']])
    fg_mdice = np.mean([results[c]['dice'] for c in ['canal', 'fracture']])

    print(f"\n  {'─'*40}")
    print(f"  mIoU  (all 3):       {miou:.4f}")
    print(f"  mDice (all 3):       {mdice:.4f}")
    print(f"  mIoU  (fg only):     {fg_miou:.4f}")
    print(f"  mDice (fg only):     {fg_mdice:.4f}")

    # ── Image-level fracture detection ──
    img_prec = img_tp / (img_tp + img_fp + 1e-6)
    img_rec  = img_tp / (img_tp + img_fn + 1e-6)
    img_f1   = 2 * img_prec * img_rec / (img_prec + img_rec + 1e-6)
    img_acc  = (img_tp + img_tn) / (img_tp + img_tn + img_fp + img_fn + 1e-6)

    print(f"\n  {'─'*40}")
    print(f"  Image-Level Fracture Detection:")
    print(f"    TP: {img_tp}  FP: {img_fp}  FN: {img_fn}  TN: {img_tn}")
    print(f"    Accuracy:  {img_acc:.4f}")
    print(f"    Precision: {img_prec:.4f}")
    print(f"    Recall:    {img_rec:.4f}")
    print(f"    F1:        {img_f1:.4f}")

    # ── Confusion matrix ──
    print(f"\n  {'─'*40}")
    print(f"  Pixel-Level Confusion Matrix:")
    print(f"  {'':>12s} {'Pred_BG':>12s} {'Pred_Canal':>12s} {'Pred_Frac':>12s}")
    for i, cname in enumerate(CLASS_NAMES):
        row = ''.join(f"{confusion[i,j]:>12d}" for j in range(NUM_CLASSES))
        print(f"  {'GT_'+cname:>12s}{row}")

    print(f"\n{'='*70}")

    # ── Save results ──
    save_data = {
        'per_class': results,
        'mIoU_all': miou, 'mDice_all': mdice,
        'mIoU_fg': fg_miou, 'mDice_fg': fg_mdice,
        'image_level': {
            'tp': img_tp, 'fp': img_fp, 'fn': img_fn, 'tn': img_tn,
            'accuracy': img_acc, 'precision': img_prec,
            'recall': img_rec, 'f1': img_f1
        },
        'confusion_matrix': confusion.tolist(),
        'n_val_images': len(val_imgs)
    }

    with open(str(output_dir / "eval_metrics.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to: {output_dir}")
    print(f"  Visualisations:   {vis_dir} ({vis_count} images)")


if __name__ == "__main__":
    evaluate()
