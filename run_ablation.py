"""
Ablation Study — 6 Experiments × 50 Epochs
============================================
Each experiment tests ONE change vs the baseline (v3 training):
  Baseline: SR+CLAHE data, dilation=5, CE+Dice loss, no augmentation

  Exp 1: No CLAHE (raw data)         — İnönü 2025 found CLAHE hurts
  Exp 2: Larger dilation 11×11       — increase foreground from 0.56% → ~2%
  Exp 3: Focal Tversky Loss          — better for extreme imbalance
  Exp 4: Data Augmentation           — flip/rotate/brightness/elastic
  Exp 5: clDice Loss                 — topology-preserving for thin lines
  Exp 6: All combined                — no CLAHE + dil11 + FTL + aug + clDice

Results saved to: runs/ablation/<exp_name>/
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
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt
import random

NUM_CLASSES = 3
CLASS_NAMES = ["background", "canal", "fracture"]

# ═══════════════════════════════════════════════════════════════
#  Import model architecture from v3
# ═══════════════════════════════════════════════════════════════
from train_u2net_v3 import U2Net, ConvBNReLU

# ═══════════════════════════════════════════════════════════════
#  DATASET with augmentation support
# ═══════════════════════════════════════════════════════════════

class AblationDataset(Dataset):
    """3-class dataset with configurable augmentation and dilation."""

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
        # Horizontal flip (50%)
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # Vertical flip (30%)
        if random.random() > 0.7:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        # Random rotation ±15°
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)

        # Brightness/contrast jitter
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-15, 15)     # brightness
            img = np.clip(alpha * img.astype(np.float32) + beta,
                          0, 255).astype(np.uint8)

        # Gaussian noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise,
                          0, 255).astype(np.uint8)

        return img, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))

        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # Augmentation (before dilation, on raw pixels)
        if self.augment:
            img, mask = self._augment(img, mask)

        # Dilation per class
        if self.dilate_kernel > 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel),
                             np.uint8)
            frac = (mask == 255).astype(np.uint8)
            canal = (mask == 128).astype(np.uint8)
            if frac.any():
                frac = cv2.dilate(frac, kernel, iterations=1)
            if canal.any():
                canal = cv2.dilate(canal, kernel, iterations=1)
            class_mask = np.zeros_like(mask, dtype=np.int64)
            class_mask[canal > 0] = 1
            class_mask[frac > 0] = 2
        else:
            class_mask = np.zeros_like(mask, dtype=np.int64)
            class_mask[mask == 128] = 1
            class_mask[mask == 255] = 2

        # Min-max normalization
        img = img.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = np.zeros_like(img)

        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(class_mask)
        return img_t, mask_t


# ═══════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def multiclass_dice_loss(logits, target, num_classes=NUM_CLASSES, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dice_per_class = []
    for c in range(num_classes):
        p = probs[:, c].contiguous().view(-1)
        t = target_oh[:, c].contiguous().view(-1)
        inter = (p * t).sum()
        dice = (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)
        dice_per_class.append(dice)
    return 1.0 - torch.stack(dice_per_class).mean()


def ce_dice_loss(logits, target, class_weights=None):
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = multiclass_dice_loss(logits, target)
    return ce + dice


def focal_tversky_loss(logits, target, alpha=0.3, beta=0.7, gamma=0.75,
                       num_classes=NUM_CLASSES, smooth=1.0):
    """
    Focal Tversky Loss (Abraham & Khan, 2019).
    alpha < beta → penalise FN more than FP (boost recall).
    gamma < 1    → focus on hard examples.
    """
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
    """Differentiable soft skeletonization for clDice."""
    for _ in range(iters):
        # Erosion via min-pool, then subtract to get boundary
        min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        contour = F.relu(x - min_pool)
        x = F.relu(x - contour)
    return x


def cl_dice_loss(logits, target, num_classes=NUM_CLASSES, smooth=1.0):
    """
    Centerline Dice (clDice) — topology-preserving loss for thin structures.
    Computes Dice on skeletonized predictions and targets.
    """
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    cl_dice_per_class = []
    for c in range(1, num_classes):  # skip background
        p = probs[:, c:c+1]
        t = target_oh[:, c:c+1]

        skel_p = soft_skeletonize(p)
        skel_t = soft_skeletonize(t)

        # Tprec: how much of pred skeleton is covered by GT
        tprec = ((skel_p * t).sum() + smooth) / (skel_p.sum() + smooth)
        # Tsens: how much of GT skeleton is covered by pred
        tsens = ((skel_t * p).sum() + smooth) / (skel_t.sum() + smooth)

        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + smooth)
        cl_dice_per_class.append(cl_dice)

    return 1.0 - torch.stack(cl_dice_per_class).mean()


# ── Deep supervision wrappers ──

def ds_ce_dice(outputs, target, class_weights=None):
    return sum(ce_dice_loss(o, target, class_weights) for o in outputs)


def ds_focal_tversky(outputs, target, class_weights=None):
    return sum(focal_tversky_loss(o, target) for o in outputs)


def ds_cldice(outputs, target, class_weights=None):
    """CE + clDice combined, deep supervision."""
    total = 0.0
    for o in outputs:
        total += F.cross_entropy(o, target, weight=class_weights)
        total += cl_dice_loss(o, target)
    return total


def ds_combo(outputs, target, class_weights=None):
    """Focal Tversky + clDice combined, deep supervision."""
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


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════

def compute_class_weights(image_paths, mask_dir, img_size, dilate_kernel, device):
    """Compute inverse frequency class weights from training masks."""
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8) if dilate_kernel > 0 else None

    for img_path in image_paths:
        mask_path = Path(mask_dir) / img_path.name
        if not mask_path.exists():
            continue
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        if kernel is not None:
            frac = (m == 255).astype(np.uint8)
            canal = (m == 128).astype(np.uint8)
            if frac.any(): frac = cv2.dilate(frac, kernel, iterations=1)
            if canal.any(): canal = cv2.dilate(canal, kernel, iterations=1)
            cm = np.zeros_like(m, dtype=np.int64)
            cm[canal > 0] = 1
            cm[frac > 0] = 2
        else:
            cm = np.zeros_like(m, dtype=np.int64)
            cm[m == 128] = 1
            cm[m == 255] = 2

        for c in range(NUM_CLASSES):
            pixel_counts[c] += (cm == c).sum()

    total = pixel_counts.sum()
    weights = total / (NUM_CLASSES * pixel_counts + 1e-6)
    weights = weights / weights.min()
    weights = np.clip(weights, 1.0, 50.0)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def run_experiment(exp_name, dataset_dir, dilate_kernel, loss_fn_name,
                   augment, num_epochs=50):
    """Run a single ablation experiment."""

    IMG_SIZE = 256
    BATCH_SIZE = 4
    LR = 0.0002
    SPLIT_RATIO = 0.2
    RANDOM_SEED = 42

    run_dir = Path(f"runs/ablation/{exp_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(dataset_dir)
    all_images = sorted((dataset_dir / "images").glob("*.png"))

    if len(all_images) == 0:
        print(f"  ❌ No images in {dataset_dir}!")
        return None

    train_imgs, val_imgs = train_test_split(
        all_images, test_size=SPLIT_RATIO, random_state=RANDOM_SEED
    )

    mask_dir = dataset_dir / "masks"

    train_ds = AblationDataset(train_imgs, mask_dir, IMG_SIZE,
                                dilate_kernel, augment=augment)
    val_ds = AblationDataset(val_imgs, mask_dir, IMG_SIZE,
                              dilate_kernel, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U2Net(in_ch=1, out_ch=NUM_CLASSES).to(device)

    class_weights = compute_class_weights(train_imgs, mask_dir, IMG_SIZE,
                                           dilate_kernel, device)

    # Select loss function
    if loss_fn_name == "ce_dice":
        loss_fn = lambda outputs, target: ds_ce_dice(outputs, target, class_weights)
    elif loss_fn_name == "focal_tversky":
        loss_fn = lambda outputs, target: ds_focal_tversky(outputs, target, class_weights)
    elif loss_fn_name == "cldice":
        loss_fn = lambda outputs, target: ds_cldice(outputs, target, class_weights)
    elif loss_fn_name == "combo":
        loss_fn = lambda outputs, target: ds_combo(outputs, target, class_weights)
    else:
        loss_fn = lambda outputs, target: ds_ce_dice(outputs, target, class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [],
               'val_dice_frac': [], 'val_dice_canal': [], 'val_dice_bg': []}

    # ── Config printout ──
    config = {
        'exp_name': exp_name,
        'dataset_dir': str(dataset_dir),
        'dilate_kernel': dilate_kernel,
        'loss_fn': loss_fn_name,
        'augment': augment,
        'num_epochs': num_epochs,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'train_size': len(train_imgs),
        'val_size': len(val_imgs),
        'class_weights': class_weights.cpu().tolist(),
    }
    with open(str(run_dir / "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    n_frac = sum(1 for p in all_images if 'fractured' in p.name)
    n_heal = sum(1 for p in all_images if 'healthy' in p.name)

    print(f"\n{'='*60}")
    print(f"  EXP: {exp_name}")
    print(f"{'='*60}")
    print(f"  Data:     {dataset_dir.name} ({len(train_imgs)} train / {len(val_imgs)} val)")
    print(f"  Frac/Heal: {n_frac}/{n_heal}")
    print(f"  Dilation: {dilate_kernel}x{dilate_kernel}")
    print(f"  Loss:     {loss_fn_name}")
    print(f"  Augment:  {augment}")
    print(f"  Weights:  bg={class_weights[0]:.1f} canal={class_weights[1]:.1f} frac={class_weights[2]:.1f}")
    print(f"  Epochs:   {num_epochs}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
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

                # Use same loss for validation comparison
                if loss_fn_name == "ce_dice":
                    vloss = ce_dice_loss(logits, masks, class_weights)
                elif loss_fn_name == "focal_tversky":
                    vloss = focal_tversky_loss(logits, masks)
                elif loss_fn_name == "cldice":
                    vloss = F.cross_entropy(logits, masks, weight=class_weights) + cl_dice_loss(logits, masks)
                elif loss_fn_name == "combo":
                    vloss = focal_tversky_loss(logits, masks) + 0.5 * cl_dice_loss(logits, masks)
                else:
                    vloss = ce_dice_loss(logits, masks, class_weights)
                val_loss += vloss.item()

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

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice_frac'].append(avg['fracture']['dice'])
        history['val_dice_canal'].append(avg['canal']['dice'])
        history['val_dice_bg'].append(avg['background']['dice'])

        lr_now = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0 or val_loss < best_val_loss:
            print(f"  E{epoch+1:3d} │ TrL:{train_loss:.3f} VL:{val_loss:.3f} │ "
                  f"Dice[f]:{avg['fracture']['dice']:.4f} [c]:{avg['canal']['dice']:.4f} "
                  f"[bg]:{avg['background']['dice']:.4f} │ LR:{lr_now:.6f}"
                  f"{' ★' if val_loss < best_val_loss else ''}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(run_dir / "best_model.pth"))

    elapsed = time.time() - start_time

    # ── Final evaluation on best model ──
    model.load_state_dict(torch.load(str(run_dir / "best_model.pth"),
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
        'exp_name': exp_name,
        'best_val_loss': best_val_loss,
        'best_epoch': int(np.argmin(history['val_loss'])) + 1,
        'elapsed_min': elapsed / 60,
        'per_class': final_avg,
        'mIoU_fg': float(np.mean([final_avg[c]['iou'] for c in ['canal', 'fracture']])),
        'mDice_fg': float(np.mean([final_avg[c]['dice'] for c in ['canal', 'fracture']])),
        'image_level': {
            'tp': img_tp, 'fp': img_fp, 'fn': img_fn, 'tn': img_tn,
            'accuracy': img_acc, 'precision': img_prec,
            'recall': img_rec, 'f1': img_f1
        }
    }

    with open(str(run_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(str(run_dir / "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ── {exp_name} RESULTS ──")
    print(f"  Best epoch: {results['best_epoch']} | Val loss: {best_val_loss:.4f}")
    print(f"  Dice[frac]: {final_avg['fracture']['dice']:.4f}  "
          f"Dice[canal]: {final_avg['canal']['dice']:.4f}")
    print(f"  mIoU(fg): {results['mIoU_fg']:.4f}  mDice(fg): {results['mDice_fg']:.4f}")
    print(f"  Img-level → Prec:{img_prec:.3f} Rec:{img_rec:.3f} F1:{img_f1:.3f} Acc:{img_acc:.3f}")
    print(f"  Time: {elapsed/60:.1f} min")

    return results


# ═══════════════════════════════════════════════════════════════
#  MAIN — Run all 6 experiments
# ═══════════════════════════════════════════════════════════════

def main():
    EPOCHS = 50

    RAW_DIR = "auto_labeled_segmentation_3class"
    CLAHE_DIR = "auto_labeled_segmentation_3class_sr_clahe"

    experiments = [
        # (name, dataset_dir, dilate_kernel, loss_fn, augment)
        ("exp1_no_clahe",       RAW_DIR,   5,  "ce_dice",        False),
        ("exp2_dilation11",     CLAHE_DIR, 11, "ce_dice",        False),
        ("exp3_focal_tversky",  CLAHE_DIR, 5,  "focal_tversky",  False),
        ("exp4_augmentation",   CLAHE_DIR, 5,  "ce_dice",        True),
        ("exp5_cldice",         CLAHE_DIR, 5,  "cldice",         False),
        ("exp6_all_combined",   RAW_DIR,   11, "combo",          True),
    ]

    all_results = []

    print(f"\n{'#'*70}")
    print(f"  ABLATION STUDY — {len(experiments)} experiments × {EPOCHS} epochs")
    print(f"{'#'*70}")

    for i, (name, ddir, dk, loss, aug) in enumerate(experiments, 1):
        print(f"\n{'#'*70}")
        print(f"  [{i}/{len(experiments)}] {name}")
        print(f"{'#'*70}")

        result = run_experiment(name, ddir, dk, loss, aug, EPOCHS)
        if result:
            all_results.append(result)

    # ── Summary table ──
    print(f"\n\n{'#'*70}")
    print(f"  ABLATION STUDY — SUMMARY")
    print(f"{'#'*70}")
    print(f"\n{'Experiment':<25s} {'BestE':>5s} {'ValLoss':>8s} "
          f"{'D[frac]':>8s} {'D[canal]':>8s} {'mDice_fg':>8s} "
          f"{'ImgF1':>8s} {'ImgRec':>8s} {'Time':>6s}")
    print(f"{'-'*90}")

    for r in all_results:
        print(f"{r['exp_name']:<25s} "
              f"{r['best_epoch']:>5d} "
              f"{r['best_val_loss']:>8.4f} "
              f"{r['per_class']['fracture']['dice']:>8.4f} "
              f"{r['per_class']['canal']['dice']:>8.4f} "
              f"{r['mDice_fg']:>8.4f} "
              f"{r['image_level']['f1']:>8.4f} "
              f"{r['image_level']['recall']:>8.4f} "
              f"{r['elapsed_min']:>5.1f}m")

    # Save summary
    with open("runs/ablation/summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Full results saved to: runs/ablation/")
    print(f"  Summary: runs/ablation/summary.json")


if __name__ == "__main__":
    main()
