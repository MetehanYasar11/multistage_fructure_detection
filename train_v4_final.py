"""
Final Training — U2-Net v4
===========================
Uses the EXACT same code from run_ablation.py (which ran 50 epochs NaN-free)
but with exp6_all_combined config at 200 epochs + 20-epoch early stopping.

Config: raw data + dilation 11 + combo loss (FTL + 0.5*clDice) + augmentation
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

NUM_CLASSES = 3
CLASS_NAMES = ["background", "canal", "fracture"]

from train_u2net_v3 import U2Net, ConvBNReLU

# ═══════════════════════════════════════════════════════════════
#  DATASET — identical to run_ablation.py AblationDataset
# ═══════════════════════════════════════════════════════════════

class AblationDataset(Dataset):
    """3-class dataset with configurable augmentation and dilation.
       EXACT COPY from run_ablation.py to ensure same behavior."""

    def __init__(self, image_paths, mask_dir, img_size=256,
                 dilate_kernel=5, augment=False):
        self.image_paths = image_paths
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.dilate_kernel = dilate_kernel
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def _augment(self, img, mask):
        """Apply augmentations to both image and mask consistently."""
        # Horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        # Vertical flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        # Random rotation ±15°
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT)
        # Brightness / contrast jitter
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-15, 15)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
        # Gaussian noise
        if random.random() > 0.3:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # Dilate thin annotations
        if self.dilate_kernel > 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8)
            frac = (mask == 255).astype(np.uint8)
            canal = (mask == 128).astype(np.uint8)
            if frac.any(): frac = cv2.dilate(frac, kernel, iterations=1)
            if canal.any(): canal = cv2.dilate(canal, kernel, iterations=1)
            class_mask = np.zeros_like(mask, dtype=np.int64)
            class_mask[canal > 0] = 1
            class_mask[frac > 0] = 2
        else:
            class_mask = np.zeros_like(mask, dtype=np.int64)
            class_mask[mask == 128] = 1
            class_mask[mask == 255] = 2

        if self.augment:
            img, class_mask = self._augment(img, class_mask.astype(np.uint8))
            class_mask = class_mask.astype(np.int64)

        # Min-max normalization (same as run_ablation.py)
        img = img.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(class_mask)
        return img_t, mask_t


# ═══════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS — identical to run_ablation.py
# ═══════════════════════════════════════════════════════════════

def focal_tversky_loss(logits, target, alpha=0.3, beta=0.7, gamma=0.75,
                       num_classes=NUM_CLASSES, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    tversky_per_class = []
    for c in range(num_classes):
        p = probs[:, c].contiguous().view(-1)
        t = target_oh[:, c].contiguous().view(-1)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        tversky_per_class.append(tversky)

    mean_tversky = torch.stack(tversky_per_class).mean()
    return (1.0 - mean_tversky) ** gamma


def soft_skeletonize(x, iters=3):
    for _ in range(iters):
        min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        contour = F.relu(x - min_pool)
        x = F.relu(x - contour)
    return x


def cl_dice_loss(logits, target, num_classes=NUM_CLASSES, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    cl_dice_per_class = []
    for c in range(1, num_classes):
        p = probs[:, c:c+1]
        t = target_oh[:, c:c+1]

        skel_p = soft_skeletonize(p)
        skel_t = soft_skeletonize(t)

        tprec = ((skel_p * t).sum() + smooth) / (skel_p.sum() + smooth)
        tsens = ((skel_t * p).sum() + smooth) / (skel_t.sum() + smooth)

        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + smooth)
        cl_dice_per_class.append(cl_dice)

    return 1.0 - torch.stack(cl_dice_per_class).mean()


def ds_combo(outputs, target, class_weights=None):
    """Focal Tversky + clDice combined, deep supervision — from run_ablation.py"""
    total = 0.0
    for o in outputs:
        total += focal_tversky_loss(o, target)
        total += 0.5 * cl_dice_loss(o, target)
    return total


# ═══════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(pred_logits, target):
    pred = pred_logits.argmax(dim=1)
    metrics = {}
    for c in range(NUM_CLASSES):
        p = (pred == c).float().view(-1)
        t = (target == c).float().view(-1)
        tp = (p * t).sum().item()
        fp = p.sum().item() - tp
        fn = t.sum().item() - tp
        union = tp + fp + fn
        metrics[CLASS_NAMES[c]] = {
            'iou': tp / (union + 1e-6),
            'dice': (2 * tp) / (2 * tp + fp + fn + 1e-6),
            'precision': tp / (tp + fp + 1e-6),
            'recall': tp / (tp + fn + 1e-6),
        }
    return metrics


def compute_class_weights(image_paths, mask_dir, img_size, dilate_kernel, device):
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)

    for img_path in image_paths:
        mask_path = Path(mask_dir) / img_path.name
        if not mask_path.exists():
            continue
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        frac = (m == 255).astype(np.uint8)
        canal = (m == 128).astype(np.uint8)
        if frac.any(): frac = cv2.dilate(frac, kernel, iterations=1)
        if canal.any(): canal = cv2.dilate(canal, kernel, iterations=1)
        cm = np.zeros_like(m, dtype=np.int64)
        cm[canal > 0] = 1
        cm[frac > 0] = 2

        for c in range(NUM_CLASSES):
            pixel_counts[c] += (cm == c).sum()

    total = pixel_counts.sum()
    weights = total / (NUM_CLASSES * pixel_counts + 1e-6)
    weights = weights / weights.min()
    weights = np.clip(weights, 1.0, 50.0)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    IMG_SIZE = 256
    BATCH_SIZE = 4
    LR = 0.0002
    NUM_EPOCHS = 200
    EARLY_STOP_PATIENCE = 20
    DILATE_KERNEL = 11
    SPLIT_RATIO = 0.2
    RANDOM_SEED = 42

    RUN_DIR = Path("runs/u2net_v4_final")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    DATASET_DIR = Path("auto_labeled_segmentation_3class")  # raw, no CLAHE

    all_images = sorted((DATASET_DIR / "images").glob("*.png"))
    print(f"  Total images: {len(all_images)}")

    train_imgs, val_imgs = train_test_split(
        all_images, test_size=SPLIT_RATIO, random_state=RANDOM_SEED
    )
    mask_dir = DATASET_DIR / "masks"

    train_ds = AblationDataset(train_imgs, mask_dir, IMG_SIZE,
                                DILATE_KERNEL, augment=True)
    val_ds = AblationDataset(val_imgs, mask_dir, IMG_SIZE,
                              DILATE_KERNEL, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = U2Net(in_ch=1, out_ch=NUM_CLASSES).to(device)

    class_weights = compute_class_weights(train_imgs, mask_dir, IMG_SIZE,
                                           DILATE_KERNEL, device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    n_frac = sum(1 for p in all_images if 'fractured' in p.name)
    n_heal = sum(1 for p in all_images if 'healthy' in p.name)

    config = {
        'exp_name': 'u2net_v4_final',
        'description': 'exp6 config exact (raw+dil11+combo+aug) 200ep early_stop=20',
        'dataset_dir': str(DATASET_DIR),
        'dilate_kernel': DILATE_KERNEL,
        'loss_fn': 'combo (FTL + 0.5*clDice) — deep supervision',
        'augment': True,
        'num_epochs': NUM_EPOCHS,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'batch_size': BATCH_SIZE, 'lr': LR, 'img_size': IMG_SIZE,
        'train_size': len(train_imgs), 'val_size': len(val_imgs),
        'n_fractured': n_frac, 'n_healthy': n_heal,
        'class_weights': class_weights.cpu().tolist(),
    }
    with open(str(RUN_DIR / "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  U2-Net v4 Final Training (exp6 config, 200 epochs)")
    print(f"{'='*70}")
    print(f"  Data:       {DATASET_DIR.name} ({len(train_imgs)} train / {len(val_imgs)} val)")
    print(f"  Frac/Heal:  {n_frac}/{n_heal}")
    print(f"  Dilation:   {DILATE_KERNEL}×{DILATE_KERNEL}")
    print(f"  Loss:       Combo (Focal Tversky + 0.5×clDice) — deep supervision")
    print(f"  Augment:    True (flip/rotate/brightness/noise)")
    print(f"  Weights:    bg={class_weights[0]:.1f} canal={class_weights[1]:.1f} frac={class_weights[2]:.1f}")
    print(f"  Epochs:     {NUM_EPOCHS} (early stop patience={EARLY_STOP_PATIENCE})")
    print(f"  LR:         {LR} → ReduceOnPlateau(patience=5, factor=0.5)")
    print(f"  Grad clip:  max_norm=1.0")
    print(f"{'='*70}")

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'val_dice_frac': [], 'val_dice_canal': [], 'val_dice_bg': [],
        'lr': []
    }

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = ds_combo(outputs, masks)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        all_metrics = {c: {'iou': [], 'dice': [], 'precision': [], 'recall': []}
                       for c in CLASS_NAMES}
        img_tp = img_fp = img_fn = img_tn = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)

                vloss = focal_tversky_loss(logits, masks) + 0.5 * cl_dice_loss(logits, masks)
                if torch.isfinite(vloss):
                    val_loss += vloss.item()
                else:
                    val_loss += 10.0

                m = compute_metrics(logits, masks)
                for cname, md in m.items():
                    for k, v in md.items():
                        all_metrics[cname][k].append(v)

                pred = logits.argmax(dim=1)
                for j in range(imgs.size(0)):
                    gt_frac = (masks[j] == 2).any().item()
                    pr_frac = (pred[j] == 2).any().item()
                    if gt_frac and pr_frac: img_tp += 1
                    elif not gt_frac and pr_frac: img_fp += 1
                    elif gt_frac and not pr_frac: img_fn += 1
                    else: img_tn += 1

        val_loss /= len(val_loader)
        avg = {c: {k: np.mean(v) for k, v in all_metrics[c].items()}
               for c in CLASS_NAMES}

        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_dice_frac'].append(float(avg['fracture']['dice']))
        history['val_dice_canal'].append(float(avg['canal']['dice']))
        history['val_dice_bg'].append(float(avg['background']['dice']))
        history['lr'].append(float(lr_now))

        improved = val_loss < best_val_loss

        if (epoch + 1) % 10 == 0 or epoch == 0 or improved:
            elapsed_now = (time.time() - start_time) / 60
            print(f"  E{epoch+1:4d}/{NUM_EPOCHS} │ "
                  f"TrL:{train_loss:.3f} VL:{val_loss:.3f} │ "
                  f"Dice[f]:{avg['fracture']['dice']:.4f} "
                  f"[c]:{avg['canal']['dice']:.4f} "
                  f"[bg]:{avg['background']['dice']:.4f} │ "
                  f"LR:{lr_now:.6f} │ {elapsed_now:.1f}m"
                  f"{' ★' if improved else ''}")

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), str(RUN_DIR / "best_model.pth"))
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  ⛔ Early stopping at epoch {epoch+1} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            print(f"     Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}")
            break

    total_time = time.time() - start_time

    # ═══════════════════════════════════════════════════════════
    #  FINAL EVALUATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Loading best model (epoch {best_epoch}) for final evaluation...")
    print(f"{'='*70}")

    model.load_state_dict(torch.load(str(RUN_DIR / "best_model.pth"),
                                      map_location=device, weights_only=True))
    model.eval()

    final_metrics = {c: {'iou': [], 'dice': [], 'precision': [], 'recall': []}
                     for c in CLASS_NAMES}
    img_tp = img_fp = img_fn = img_tn = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            m = compute_metrics(logits, masks)
            for cname, md in m.items():
                for k, v in md.items():
                    final_metrics[cname][k].append(v)

            pred = logits.argmax(dim=1)
            for j in range(imgs.size(0)):
                gt_frac = (masks[j] == 2).any().item()
                pr_frac = (pred[j] == 2).any().item()
                if gt_frac and pr_frac: img_tp += 1
                elif not gt_frac and pr_frac: img_fp += 1
                elif gt_frac and not pr_frac: img_fn += 1
                else: img_tn += 1

    final_avg = {c: {k: float(np.mean(v)) for k, v in final_metrics[c].items()}
                 for c in CLASS_NAMES}

    img_prec = img_tp / (img_tp + img_fp + 1e-6)
    img_rec = img_tp / (img_tp + img_fn + 1e-6)
    img_f1 = 2 * img_prec * img_rec / (img_prec + img_rec + 1e-6)
    img_acc = (img_tp + img_tn) / (img_tp + img_tn + img_fp + img_fn + 1e-6)

    results = {
        'exp_name': 'u2net_v4_final',
        'best_val_loss': float(best_val_loss),
        'best_epoch': best_epoch,
        'total_epochs_run': epoch + 1,
        'early_stopped': no_improve >= EARLY_STOP_PATIENCE,
        'elapsed_min': total_time / 60,
        'per_class': final_avg,
        'mIoU_fg': float(np.mean([final_avg[c]['iou'] for c in ['canal', 'fracture']])),
        'mDice_fg': float(np.mean([final_avg[c]['dice'] for c in ['canal', 'fracture']])),
        'image_level': {
            'tp': img_tp, 'fp': img_fp, 'fn': img_fn, 'tn': img_tn,
            'accuracy': float(img_acc), 'precision': float(img_prec),
            'recall': float(img_rec), 'f1': float(img_f1)
        }
    }

    with open(str(RUN_DIR / "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(str(RUN_DIR / "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), str(RUN_DIR / "last_model.pth"))

    print(f"\n{'#'*70}")
    print(f"  U2-Net v4 FINAL RESULTS")
    print(f"{'#'*70}")
    print(f"  Best epoch:     {best_epoch} / {epoch+1} run")
    print(f"  Early stopped:  {no_improve >= EARLY_STOP_PATIENCE}")
    print(f"  Val loss:       {best_val_loss:.4f}")
    print(f"  ────────────────────────────────────")
    print(f"  Dice[fracture]: {final_avg['fracture']['dice']:.4f}")
    print(f"  Dice[canal]:    {final_avg['canal']['dice']:.4f}")
    print(f"  Dice[bg]:       {final_avg['background']['dice']:.4f}")
    print(f"  mDice(fg):      {results['mDice_fg']:.4f}")
    print(f"  mIoU(fg):       {results['mIoU_fg']:.4f}")
    print(f"  ────────────────────────────────────")
    print(f"  Image-level:")
    print(f"    TP={img_tp}  FP={img_fp}  FN={img_fn}  TN={img_tn}")
    print(f"    Precision: {img_prec:.4f}")
    print(f"    Recall:    {img_rec:.4f}")
    print(f"    F1:        {img_f1:.4f}")
    print(f"    Accuracy:  {img_acc:.4f}")
    print(f"  ────────────────────────────────────")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Model saved: {RUN_DIR / 'best_model.pth'}")
    print(f"  Results:     {RUN_DIR / 'results.json'}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
