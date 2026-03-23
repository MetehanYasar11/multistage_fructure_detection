"""
U2-Net Full — 3-Class Segmentation for VRF Detection
=====================================================
Classes:
    0 = Background
    1 = RCT canal filling  (healthy structure — do NOT segment as fracture)
    2 = Vertical root fracture  (target pathology)

Architecture identical to train_u2net_v2.py (paper-aligned Full U2-Net)
but adapted for multi-class output:
    - out_ch = 3
    - CrossEntropyLoss + multi-class Dice loss
    - Mask encoding: pixel values 0/128/255 → class indices 0/1/2
    - Deep supervision on all 7 outputs

Reference:
  İnönü N, et al. "Deep learning-based detection of separated root canal
  instruments in panoramic radiographs using a U2-Net architecture."
  Diagnostics. 2025;15(14):1744.
"""

import os
import json
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

NUM_CLASSES = 3
CLASS_NAMES = ["background", "canal", "fracture"]

# ═══════════════════════════════════════════════════════════════
#  BUILDING BLOCKS  (identical to v2)
# ═══════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3,
                              padding=1 * dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def _upsample_like(src, target):
    return F.interpolate(src, size=target.shape[2:],
                         mode='bilinear', align_corners=False)


# ═══════════════════════════════════════════════════════════════
#  RSU BLOCKS
# ═══════════════════════════════════════════════════════════════

class RSU7(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.enc1 = ConvBNReLU(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc2 = ConvBNReLU(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc3 = ConvBNReLU(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc4 = ConvBNReLU(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc5 = ConvBNReLU(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc6 = ConvBNReLU(mid_ch, mid_ch)
        self.bot  = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        self.dec6 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec5 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec4 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec3 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec2 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec1 = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = self.enc1(x0);  x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2));  x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4));  x6 = self.enc6(self.pool5(x5))
        x7 = self.bot(x6)
        d6 = self.dec6(torch.cat([x7, x6], 1))
        d5 = self.dec5(torch.cat([_upsample_like(d6, x5), x5], 1))
        d4 = self.dec4(torch.cat([_upsample_like(d5, x4), x4], 1))
        d3 = self.dec3(torch.cat([_upsample_like(d4, x3), x3], 1))
        d2 = self.dec2(torch.cat([_upsample_like(d3, x2), x2], 1))
        d1 = self.dec1(torch.cat([_upsample_like(d2, x1), x1], 1))
        return x0 + d1


class RSU6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.enc1 = ConvBNReLU(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc2 = ConvBNReLU(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc3 = ConvBNReLU(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc4 = ConvBNReLU(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc5 = ConvBNReLU(mid_ch, mid_ch)
        self.bot  = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        self.dec5 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec4 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec3 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec2 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec1 = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = self.enc1(x0);  x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2));  x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))
        x6 = self.bot(x5)
        d5 = self.dec5(torch.cat([x6, x5], 1))
        d4 = self.dec4(torch.cat([_upsample_like(d5, x4), x4], 1))
        d3 = self.dec3(torch.cat([_upsample_like(d4, x3), x3], 1))
        d2 = self.dec2(torch.cat([_upsample_like(d3, x2), x2], 1))
        d1 = self.dec1(torch.cat([_upsample_like(d2, x1), x1], 1))
        return x0 + d1


class RSU5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.enc1 = ConvBNReLU(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc2 = ConvBNReLU(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc3 = ConvBNReLU(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc4 = ConvBNReLU(mid_ch, mid_ch)
        self.bot  = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        self.dec4 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec3 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec2 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec1 = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = self.enc1(x0);  x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2));  x4 = self.enc4(self.pool3(x3))
        x5 = self.bot(x4)
        d4 = self.dec4(torch.cat([x5, x4], 1))
        d3 = self.dec3(torch.cat([_upsample_like(d4, x3), x3], 1))
        d2 = self.dec2(torch.cat([_upsample_like(d3, x2), x2], 1))
        d1 = self.dec1(torch.cat([_upsample_like(d2, x1), x1], 1))
        return x0 + d1


class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.enc1 = ConvBNReLU(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc2 = ConvBNReLU(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.enc3 = ConvBNReLU(mid_ch, mid_ch)
        self.bot  = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        self.dec3 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec2 = ConvBNReLU(mid_ch * 2, mid_ch)
        self.dec1 = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = self.enc1(x0);  x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.bot(x3)
        d3 = self.dec3(torch.cat([x4, x3], 1))
        d2 = self.dec2(torch.cat([_upsample_like(d3, x2), x2], 1))
        d1 = self.dec1(torch.cat([_upsample_like(d2, x1), x1], 1))
        return x0 + d1


class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.enc1 = ConvBNReLU(out_ch, mid_ch, dirate=1)
        self.enc2 = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        self.enc3 = ConvBNReLU(mid_ch, mid_ch, dirate=4)
        self.bot  = ConvBNReLU(mid_ch, mid_ch, dirate=8)
        self.dec3 = ConvBNReLU(mid_ch * 2, mid_ch, dirate=4)
        self.dec2 = ConvBNReLU(mid_ch * 2, mid_ch, dirate=2)
        self.dec1 = ConvBNReLU(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = self.enc1(x0);  x2 = self.enc2(x1);  x3 = self.enc3(x2)
        x4 = self.bot(x3)
        d3 = self.dec3(torch.cat([x4, x3], 1))
        d2 = self.dec2(torch.cat([d3, x2], 1))
        d1 = self.dec1(torch.cat([d2, x1], 1))
        return x0 + d1


# ═══════════════════════════════════════════════════════════════
#  FULL U2-NET  (3-class output)
# ═══════════════════════════════════════════════════════════════

class U2Net(nn.Module):
    """
    Full U2-Net: 6 encoder + 5 decoder stages.
    Reduced channel config (~11M params) to prevent overfitting
    on our 770-crop dataset.
    """

    def __init__(self, in_ch=1, out_ch=NUM_CLASSES):
        super().__init__()
        # ── Encoder ──
        self.stage1  = RSU7(in_ch, 16, 32)
        self.pool12  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2  = RSU6(32, 16, 64)
        self.pool23  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3  = RSU5(64, 32, 128)
        self.pool34  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4  = RSU4(128, 64, 256)
        self.pool45  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5  = RSU4F(256, 128, 256)
        self.pool56  = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6  = RSU4F(256, 128, 256)

        # ── Decoder ──
        self.stage5d = RSU4F(512, 128, 256)
        self.stage4d = RSU4(512, 64, 128)
        self.stage3d = RSU5(256, 32, 64)
        self.stage2d = RSU6(128, 16, 32)
        self.stage1d = RSU7(64, 8, 32)

        # ── Side outputs (deep supervision) — each outputs NUM_CLASSES channels
        self.side1 = nn.Conv2d(32,  out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(32,  out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64,  out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(256, out_ch, 3, padding=1)

        # ── Fuse: 6 * out_ch → out_ch ──
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx1 = self.stage1(x);             hx = self.pool12(hx1)
        hx2 = self.stage2(hx);            hx = self.pool23(hx2)
        hx3 = self.stage3(hx);            hx = self.pool34(hx3)
        hx4 = self.stage4(hx);            hx = self.pool45(hx4)
        hx5 = self.stage5(hx);            hx = self.pool56(hx5)
        hx6 = self.stage6(hx)

        hx5d = self.stage5d(torch.cat([_upsample_like(hx6, hx5), hx5], 1))
        hx4d = self.stage4d(torch.cat([_upsample_like(hx5d, hx4), hx4], 1))
        hx3d = self.stage3d(torch.cat([_upsample_like(hx4d, hx3), hx3], 1))
        hx2d = self.stage2d(torch.cat([_upsample_like(hx3d, hx2), hx2], 1))
        hx1d = self.stage1d(torch.cat([_upsample_like(hx2d, hx1), hx1], 1))

        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6),  d1)

        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))

        if self.training:
            return d0, d1, d2, d3, d4, d5, d6   # all logits [B, C, H, W]
        else:
            return d0                             # fused logits


# ═══════════════════════════════════════════════════════════════
#  3-CLASS DATASET
# ═══════════════════════════════════════════════════════════════

class VRFSeg3ClassDataset(Dataset):
    """
    Loads SR+CLAHE preprocessed crops and 3-class masks.

    Mask pixel mapping:
        0   → class 0 (background)
        128 → class 1 (canal filling)
        255 → class 2 (fracture)

    Returns:
        img:  [1, H, W]  float32  (grayscale, min-max normalised)
        mask: [H, W]     int64    (class indices 0/1/2)
    """

    def __init__(self, image_paths, mask_dir, img_size=256,
                 dilate_kernel=5):
        self.image_paths = image_paths
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.dilate_kernel = dilate_kernel

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # ── Image: grayscale + min-max norm ──
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = np.zeros_like(img)

        # ── Mask: 3-class ──
        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # ── Morphological dilation on annotation lines ──
        # Dilate each class separately to avoid class bleeding
        if self.dilate_kernel > 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel),
                             np.uint8)
            # Fracture lines (255)
            frac_mask = (mask == 255).astype(np.uint8)
            if frac_mask.any():
                frac_mask = cv2.dilate(frac_mask, kernel, iterations=1)

            # Canal lines (128)
            canal_mask = (mask == 128).astype(np.uint8)
            if canal_mask.any():
                canal_mask = cv2.dilate(canal_mask, kernel, iterations=1)

            # Rebuild class index mask (fracture takes priority over canal)
            class_mask = np.zeros_like(mask, dtype=np.int64)
            class_mask[canal_mask > 0] = 1
            class_mask[frac_mask > 0]  = 2  # fracture overwrites canal
        else:
            class_mask = np.zeros_like(mask, dtype=np.int64)
            class_mask[mask == 128] = 1
            class_mask[mask == 255] = 2

        img_t  = torch.from_numpy(img).unsqueeze(0)          # [1, H, W]
        mask_t = torch.from_numpy(class_mask)                 # [H, W] int64

        return img_t, mask_t


# ═══════════════════════════════════════════════════════════════
#  LOSS — CE + Multi-class Dice
# ═══════════════════════════════════════════════════════════════

def multiclass_dice_loss(logits, target, num_classes=NUM_CLASSES, smooth=1.0):
    """
    Compute 1 - mean(per-class Dice) using soft probabilities.
    
    Args:
        logits: [B, C, H, W] raw network output
        target: [B, H, W] int64 class indices
    """
    probs = F.softmax(logits, dim=1)                          # [B, C, H, W]
    target_oh = F.one_hot(target, num_classes)                 # [B, H, W, C]
    target_oh = target_oh.permute(0, 3, 1, 2).float()        # [B, C, H, W]

    dice_per_class = []
    for c in range(num_classes):
        p = probs[:, c].contiguous().view(-1)
        t = target_oh[:, c].contiguous().view(-1)
        inter = (p * t).sum()
        dice = (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)
        dice_per_class.append(dice)

    return 1.0 - torch.stack(dice_per_class).mean()


def ce_dice_loss(logits, target, class_weights=None):
    """
    CrossEntropy + Multi-class Dice combined loss.
    
    CE provides stable per-pixel gradients.
    Dice handles severe class imbalance (lines are <1% of pixels).
    """
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = multiclass_dice_loss(logits, target)
    return ce + dice


def deep_supervision_loss(outputs, target, class_weights=None):
    """Sum loss over fused + 6 side outputs (deep supervision)."""
    total = 0.0
    for out in outputs:
        total += ce_dice_loss(out, target, class_weights)
    return total


# ═══════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(pred_logits, target, num_classes=NUM_CLASSES):
    """
    Compute per-class IoU and Dice from logits.
    
    Returns dict: {class_name: {iou, dice, precision, recall}}
    """
    pred = pred_logits.argmax(dim=1)  # [B, H, W]
    metrics = {}
    for c in range(num_classes):
        p = (pred == c).float().view(-1)
        t = (target == c).float().view(-1)
        inter = (p * t).sum().item()
        union = p.sum().item() + t.sum().item() - inter
        tp = inter
        fp = p.sum().item() - tp
        fn = t.sum().item() - tp

        iou   = tp / (union + 1e-6)
        dice  = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        prec  = tp / (tp + fp + 1e-6)
        rec   = tp / (tp + fn + 1e-6)

        metrics[CLASS_NAMES[c]] = {
            'iou': iou, 'dice': dice,
            'precision': prec, 'recall': rec
        }
    return metrics


# ═══════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════

def train():
    # ── Config ──
    IMG_SIZE     = 256
    BATCH_SIZE   = 4
    LR           = 0.0002
    NUM_EPOCHS   = 30
    DILATE_K     = 5
    SPLIT_RATIO  = 0.2
    RANDOM_SEED  = 42

    dataset_dir = Path("auto_labeled_segmentation_3class_sr_clahe")
    all_images  = sorted((dataset_dir / "images").glob("*.png"))

    if len(all_images) == 0:
        print("❌ No images found!")
        return

    train_imgs, val_imgs = train_test_split(
        all_images, test_size=SPLIT_RATIO, random_state=RANDOM_SEED
    )

    train_ds = VRFSeg3ClassDataset(train_imgs, dataset_dir / "masks",
                                    img_size=IMG_SIZE, dilate_kernel=DILATE_K)
    val_ds   = VRFSeg3ClassDataset(val_imgs,   dataset_dir / "masks",
                                    img_size=IMG_SIZE, dilate_kernel=DILATE_K)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = U2Net(in_ch=1, out_ch=NUM_CLASSES).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 U2-Net Full (3-class) | Params: {total_params:,} | "
          f"Trainable: {train_params:,}")

    # ── Compute class weights from training set ──
    print("Computing class weights from training masks...")
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for img_path in tqdm(train_imgs, desc="Scanning masks", leave=False):
        mask_path = dataset_dir / "masks" / img_path.name
        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (IMG_SIZE, IMG_SIZE),
                           interpolation=cv2.INTER_NEAREST)
            # Apply same dilation as dataset
            if DILATE_K > 0:
                kernel = np.ones((DILATE_K, DILATE_K), np.uint8)
                frac = (m == 255).astype(np.uint8)
                canal = (m == 128).astype(np.uint8)
                if frac.any():
                    frac = cv2.dilate(frac, kernel, iterations=1)
                if canal.any():
                    canal = cv2.dilate(canal, kernel, iterations=1)
                cm = np.zeros_like(m, dtype=np.int64)
                cm[canal > 0] = 1
                cm[frac > 0]  = 2
            else:
                cm = np.zeros_like(m, dtype=np.int64)
                cm[m == 128] = 1
                cm[m == 255] = 2

            for c in range(NUM_CLASSES):
                pixel_counts[c] += (cm == c).sum()

    total_px = pixel_counts.sum()
    # Inverse frequency weighting: w_c = total / (C * count_c)
    class_weights = total_px / (NUM_CLASSES * pixel_counts + 1e-6)
    # Normalise so min weight = 1.0
    class_weights = class_weights / class_weights.min()
    # Cap max weight to prevent gradient explosion
    class_weights = np.clip(class_weights, 1.0, 50.0)

    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"  Pixel counts: bg={pixel_counts[0]:.0f}  canal={pixel_counts[1]:.0f}  "
          f"fracture={pixel_counts[2]:.0f}")
    print(f"  Class weights: bg={class_weights[0]:.2f}  canal={class_weights[1]:.2f}  "
          f"fracture={class_weights[2]:.2f}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    run_dir = Path("runs/u2net_v3_3class")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [],
               'val_dice_fracture': [], 'val_dice_canal': []}

    n_frac = sum(1 for p in all_images if 'fractured' in p.name)
    n_heal = sum(1 for p in all_images if 'healthy' in p.name)

    print(f"\n{'='*60}")
    print(f"  U2-Net v3 — 3-Class Segmentation Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Images      : {len(train_imgs)} train / {len(val_imgs)} val")
    print(f"  Fractured   : {n_frac}")
    print(f"  Healthy     : {n_heal}")
    print(f"  Classes     : {CLASS_NAMES}")
    print(f"  Input       : {IMG_SIZE}x{IMG_SIZE} grayscale")
    print(f"  Batch       : {BATCH_SIZE}")
    print(f"  LR          : {LR}")
    print(f"  Epochs      : {NUM_EPOCHS}")
    print(f"  Loss        : CE + Multi-class Dice (deep supervision)")
    print(f"  Grad Clip   : max_norm=1.0")
    print(f"  Dilation    : {DILATE_K}x{DILATE_K}")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        for imgs, masks in pbar:
            imgs  = imgs.to(device)           # [B, 1, H, W]
            masks = masks.to(device)           # [B, H, W] int64

            optimizer.zero_grad()
            outputs = model(imgs)              # (d0, d1, ..., d6)
            loss = deep_supervision_loss(outputs, masks, class_weights_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        all_metrics = {c: {'iou': [], 'dice': [], 'precision': [], 'recall': []}
                       for c in CLASS_NAMES}

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)           # fused logits [B, C, H, W]
                loss = ce_dice_loss(logits, masks, class_weights_t)
                val_loss += loss.item()

                batch_metrics = compute_metrics(logits, masks)
                for cname, mdict in batch_metrics.items():
                    for k, v in mdict.items():
                        all_metrics[cname][k].append(v)

        val_loss /= len(val_loader)

        # Average metrics
        avg = {}
        for cname in CLASS_NAMES:
            avg[cname] = {k: np.mean(v) if v else 0.0
                          for k, v in all_metrics[cname].items()}

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice_fracture'].append(avg['fracture']['dice'])
        history['val_dice_canal'].append(avg['canal']['dice'])

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d} │ TrLoss: {train_loss:.4f} │ "
              f"ValLoss: {val_loss:.4f} │ "
              f"Dice[frac]: {avg['fracture']['dice']:.4f}  "
              f"Dice[canal]: {avg['canal']['dice']:.4f}  "
              f"Dice[bg]: {avg['background']['dice']:.4f} │ "
              f"LR: {lr_now:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       str(run_dir / "best_u2net_v3.pth"))
            print(f"  💾 Best model saved! (val_loss={val_loss:.4f})")

    # ── Save history ──
    with open(str(run_dir / "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Model: {run_dir / 'best_u2net_v3.pth'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
