"""
U2-Net Full Architecture for VRF (Vertical Root Fracture) Segmentation
======================================================================
Faithfully adapted from:
  İnönü N, Aksoy U, Kırmızı D, Aksoy S, Akkaya N, Orhan K.
  "Deep learning-based detection of separated root canal instruments
   in panoramic radiographs using a U2-Net architecture."
  Diagnostics. 2025;15(14):1744.

Key decisions aligned with the paper:
  - Full U2-Net (not Lite): 6 encoder stages + 5 decoder stages
  - RSU-7, RSU-6, RSU-5, RSU-4, RSU-4F blocks (varying depth)
  - Weighted Dice Loss (inverse frequency weighting)
  - AdamW optimizer, lr = 0.0002
  - Deep supervision (loss on every side output + fused output)
  - Grayscale input (single channel) — paper uses single channel

Adapted to our pipeline:
  - Input: RCT tooth crops (variable size, resized to 256x256)
  - Mask: binary (white line = fracture, black = background)
  - Morphological dilation on GT lines (our lines are 1px thin)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ═══════════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU, with optional dilation."""
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3,
                              padding=1 * dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def _upsample_like(src, target):
    """Bilinear upsample *src* to match the spatial size of *target*."""
    return F.interpolate(src, size=target.shape[2:],
                         mode='bilinear', align_corners=False)


# ═══════════════════════════════════════════════════════════════
#  RSU BLOCKS  (Residual U-blocks of varying depth)
#  - The "micro U-Net inside U-Net" that gives U2-Net its power
# ═══════════════════════════════════════════════════════════════

class RSU7(nn.Module):
    """RSU-7: 7-level residual U-block (deepest, highest resolution)."""
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

        x1 = self.enc1(x0)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))
        x6 = self.enc6(self.pool5(x5))

        x7 = self.bot(x6)

        d6 = self.dec6(torch.cat([x7, x6], dim=1))
        d5 = self.dec5(torch.cat([_upsample_like(d6, x5), x5], dim=1))
        d4 = self.dec4(torch.cat([_upsample_like(d5, x4), x4], dim=1))
        d3 = self.dec3(torch.cat([_upsample_like(d4, x3), x3], dim=1))
        d2 = self.dec2(torch.cat([_upsample_like(d3, x2), x2], dim=1))
        d1 = self.dec1(torch.cat([_upsample_like(d2, x1), x1], dim=1))

        return x0 + d1  # residual connection


class RSU6(nn.Module):
    """RSU-6: 6-level residual U-block."""
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

        x1 = self.enc1(x0)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))

        x6 = self.bot(x5)

        d5 = self.dec5(torch.cat([x6, x5], dim=1))
        d4 = self.dec4(torch.cat([_upsample_like(d5, x4), x4], dim=1))
        d3 = self.dec3(torch.cat([_upsample_like(d4, x3), x3], dim=1))
        d2 = self.dec2(torch.cat([_upsample_like(d3, x2), x2], dim=1))
        d1 = self.dec1(torch.cat([_upsample_like(d2, x1), x1], dim=1))

        return x0 + d1


class RSU5(nn.Module):
    """RSU-5: 5-level residual U-block."""
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

        x1 = self.enc1(x0)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        x5 = self.bot(x4)

        d4 = self.dec4(torch.cat([x5, x4], dim=1))
        d3 = self.dec3(torch.cat([_upsample_like(d4, x3), x3], dim=1))
        d2 = self.dec2(torch.cat([_upsample_like(d3, x2), x2], dim=1))
        d1 = self.dec1(torch.cat([_upsample_like(d2, x1), x1], dim=1))

        return x0 + d1


class RSU4(nn.Module):
    """RSU-4: 4-level residual U-block."""
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

        x1 = self.enc1(x0)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        x4 = self.bot(x3)

        d3 = self.dec3(torch.cat([x4, x3], dim=1))
        d2 = self.dec2(torch.cat([_upsample_like(d3, x2), x2], dim=1))
        d1 = self.dec1(torch.cat([_upsample_like(d2, x1), x1], dim=1))

        return x0 + d1


class RSU4F(nn.Module):
    """RSU-4F: 4-level residual U-block with DILATED convolutions
    instead of pooling. Used at the bottleneck where spatial size
    is already small — avoids further downsampling."""
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

        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x4 = self.bot(x3)

        d3 = self.dec3(torch.cat([x4, x3], dim=1))
        d2 = self.dec2(torch.cat([d3, x2], dim=1))
        d1 = self.dec1(torch.cat([d2, x1], dim=1))

        return x0 + d1


# ═══════════════════════════════════════════════════════════════
#  FULL U2-NET
# ═══════════════════════════════════════════════════════════════

class U2Net(nn.Module):
    """
    Full U2-Net with 6 encoder stages + 5 decoder stages.
    
    Encoder:  RSU-7 → RSU-6 → RSU-5 → RSU-4 → RSU-4F → RSU-4F
    Decoder:  RSU-4F → RSU-4 → RSU-5 → RSU-6 → RSU-7
    
    Each stage produces a side output; all are fused for the
    final prediction (deep supervision).
    
    Paper channel config (original U2-Net):
      En1: RSU7(in, 32, 64)   En2: RSU6(64, 32, 128)
      En3: RSU5(128, 64, 256) En4: RSU4(256, 128, 512)
      En5: RSU4F(512, 256, 512) En6: RSU4F(512, 256, 512)
    
    We use a slightly reduced config to fit our GPU memory
    while keeping the full 6+5 structure:
      En1: RSU7(in, 32, 64)   En2: RSU6(64, 32, 128)
      En3: RSU5(128, 64, 256) En4: RSU4(256, 128, 512)
      En5: RSU4F(512, 256, 512) En6: RSU4F(512, 256, 512)
    """

    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        # Channel config: reduced from original to prevent overfitting
        # on our smaller crop dataset (1633 crops vs paper's 191 full OPGs).
        # Structure is identical: 6 encoder + 5 decoder with RSU-7/6/5/4/4F.

        # ── Encoder ──
        self.stage1 = RSU7(in_ch, 16, 32)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(32, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(128, 64, 256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(256, 128, 256)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(256, 128, 256)

        # ── Decoder ──
        self.stage5d = RSU4F(512, 128, 256)
        self.stage4d = RSU4(512, 64, 128)
        self.stage3d = RSU5(256, 32, 64)
        self.stage2d = RSU6(128, 16, 32)
        self.stage1d = RSU7(64, 8, 32)

        # ── Side outputs (deep supervision) ──
        self.side1 = nn.Conv2d(32,  out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(32,  out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64,  out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(256, out_ch, 3, padding=1)

        # ── Fuse all side outputs ──
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        # ── Encoder ──
        hx1  = self.stage1(x)
        hx   = self.pool12(hx1)

        hx2  = self.stage2(hx)
        hx   = self.pool23(hx2)

        hx3  = self.stage3(hx)
        hx   = self.pool34(hx3)

        hx4  = self.stage4(hx)
        hx   = self.pool45(hx4)

        hx5  = self.stage5(hx)
        hx   = self.pool56(hx5)

        hx6  = self.stage6(hx)

        # ── Decoder ──
        hx5d = self.stage5d(torch.cat([_upsample_like(hx6, hx5), hx5], dim=1))
        hx4d = self.stage4d(torch.cat([_upsample_like(hx5d, hx4), hx4], dim=1))
        hx3d = self.stage3d(torch.cat([_upsample_like(hx4d, hx3), hx3], dim=1))
        hx2d = self.stage2d(torch.cat([_upsample_like(hx3d, hx2), hx2], dim=1))
        hx1d = self.stage1d(torch.cat([_upsample_like(hx2d, hx1), hx1], dim=1))

        # ── Side outputs ──
        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6),  d1)

        # ── Fused output ──
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1))

        if self.training:
            return d0, d1, d2, d3, d4, d5, d6
        else:
            return d0  # return logits; apply sigmoid externally


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

class VRFSegDataset(Dataset):
    """
    Loads crop images and corresponding binary masks.
    
    Paper approach:
      - Grayscale input (single channel)
      - Per-image min-max normalization
      - No augmentation (paper found it didn't help)
    
    Our adaptation:
      - Morphological dilation on GT lines (ours are 1px thin,
        paper had polygon annotations with natural width)
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

        # ── Load image as grayscale (paper: single channel) ──
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # ── Per-image min-max normalization (paper method) ──
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # ── Load mask ──
        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # ── Dilate thin fracture lines ──
        if mask.max() > 0 and self.dilate_kernel > 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel),
                             np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        mask = (mask > 127).astype(np.float32)

        # ── To tensors: [1, H, W] ──
        img_t  = torch.from_numpy(img).unsqueeze(0)      # [1, H, W]
        mask_t = torch.from_numpy(mask).unsqueeze(0)      # [1, H, W]

        return img_t, mask_t


# ═══════════════════════════════════════════════════════════════
#  LOSS — Weighted Dice Loss (paper method)
# ═══════════════════════════════════════════════════════════════

def weighted_dice_loss(pred_logits, target):
    """
    Weighted Dice loss from the paper:
      'class weights inversely proportional to class frequencies'
    
    For binary segmentation:
      w_fg = total_pixels / (2 * fg_pixels)
      w_bg = total_pixels / (2 * bg_pixels)
    
    Dice_class = 2 * |pred ∩ gt| / (|pred| + |gt|)
    Weighted_Dice = Σ(w_c * Dice_c) / Σ(w_c)
    Loss = 1 - Weighted_Dice
    """
    pred = torch.sigmoid(pred_logits)
    smooth = 1.0

    # Flatten to (B, pixels)
    pred_flat = pred.view(pred.size(0), -1)
    tgt_flat  = target.view(target.size(0), -1)

    total_pixels = tgt_flat.size(1)

    # Per-sample foreground / background pixel counts
    fg_pixels = tgt_flat.sum(dim=1).clamp(min=1)          # [B]
    bg_pixels = (total_pixels - fg_pixels).clamp(min=1)    # [B]

    # Inverse frequency weights
    w_fg = total_pixels / (2.0 * fg_pixels)   # [B]
    w_bg = total_pixels / (2.0 * bg_pixels)   # [B]

    # Foreground Dice
    inter_fg = (pred_flat * tgt_flat).sum(dim=1)
    dice_fg  = (2.0 * inter_fg + smooth) / \
               (pred_flat.sum(dim=1) + tgt_flat.sum(dim=1) + smooth)

    # Background Dice
    pred_bg  = 1.0 - pred_flat
    tgt_bg   = 1.0 - tgt_flat
    inter_bg = (pred_bg * tgt_bg).sum(dim=1)
    dice_bg  = (2.0 * inter_bg + smooth) / \
               (pred_bg.sum(dim=1) + tgt_bg.sum(dim=1) + smooth)

    # Weighted average
    weighted_dice = (w_fg * dice_fg + w_bg * dice_bg) / (w_fg + w_bg)

    return 1.0 - weighted_dice.mean()


def bce_weighted_dice_loss(pred_logits, target):
    """
    BCE + Weighted Dice combined loss.
    
    Pure Dice loss can be numerically unstable when foreground is
    extremely sparse (our case: ~1% of pixels). Adding BCE provides
    a stable per-pixel gradient signal that prevents mode collapse.
    
    The paper (Table 2) tested multiple loss variants — this combination
    is the most stable for thin-line segmentation.
    """
    # 1) Weighted BCE: compute pos_weight from the batch itself
    tgt_flat = target.view(target.size(0), -1)
    num_pos = tgt_flat.sum().clamp(min=1)
    num_neg = (tgt_flat.numel() - num_pos).clamp(min=1)
    pos_w = (num_neg / num_pos).clamp(max=50.0)  # cap to prevent explosion

    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target,
        pos_weight=pos_w.expand_as(pred_logits)
    )

    # 2) Weighted Dice
    dice = weighted_dice_loss(pred_logits, target)

    return bce + dice


def deep_supervision_loss(outputs, target):
    """
    Sum loss over the fused output + all 6 side outputs.
    Paper: 'end-to-end training' with supervision at every level.
    """
    total = 0.0
    for out in outputs:
        total += bce_weighted_dice_loss(out, target)
    return total


# ═══════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════

def train():
    # ── Config (paper-aligned) ──
    IMG_SIZE     = 256        # Paper: 512x1024 (full OPG). Ours: crops → 256x256 is sufficient
    BATCH_SIZE   = 4          # Paper: 4
    LR           = 0.0002     # Paper: 0.0002
    NUM_EPOCHS   = 30         # Paper: 500. We start with 30 as a trial run
    DILATE_K     = 5          # Our lines are 1px; paper had polygon masks
    SPLIT_RATIO  = 0.2        # Paper: 60/20/20. We use 80/20 for now
    RANDOM_SEED  = 42

    dataset_dir = Path("auto_labeled_segmentation_sr_clahe")
    all_images  = sorted((dataset_dir / "images").glob("*.png"))

    if len(all_images) == 0:
        print("❌ No images found!")
        return

    train_imgs, val_imgs = train_test_split(
        all_images, test_size=SPLIT_RATIO, random_state=RANDOM_SEED
    )

    train_ds = VRFSegDataset(train_imgs, dataset_dir / "masks",
                              img_size=IMG_SIZE, dilate_kernel=DILATE_K)
    val_ds   = VRFSegDataset(val_imgs, dataset_dir / "masks",
                              img_size=IMG_SIZE, dilate_kernel=DILATE_K)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = U2Net(in_ch=1, out_ch=1).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 U2-Net Full | Total params: {total_params:,} | "
          f"Trainable: {train_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    os.makedirs("runs/u2net_v2", exist_ok=True)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n{'='*60}")
    print(f"  U2-Net Full Training — Paper-Aligned Configuration")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Images      : {len(train_imgs)} train / {len(val_imgs)} val")
    print(f"  Fractured   : {sum(1 for p in all_images if 'fractured' in p.name)}")
    print(f"  Healthy     : {sum(1 for p in all_images if 'healthy' in p.name)}")
    print(f"  Input size  : {IMG_SIZE}x{IMG_SIZE} (grayscale)")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  LR          : {LR}")
    print(f"  Epochs      : {NUM_EPOCHS}")
    print(f"  Loss        : BCE + Weighted Dice (stabilized)")
    print(f"  Grad Clip   : max_norm=1.0")
    print(f"  Dilation    : {DILATE_K}x{DILATE_K}")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)          # (d0, d1, d2, d3, d4, d5, d6)
            loss = deep_supervision_loss(outputs, masks)
            loss.backward()

            # Gradient clipping — prevents Dice loss gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_dice_scores = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                logits = model(imgs)  # raw logits in eval mode
                loss = bce_weighted_dice_loss(logits, masks)
                val_loss += loss.item()

                # Compute Dice for monitoring
                pred = torch.sigmoid(logits)
                pred_bin = (pred > 0.5).float()
                for j in range(imgs.size(0)):
                    p = pred_bin[j].view(-1)
                    t = masks[j].view(-1)
                    if t.sum() > 0:
                        inter = (p * t).sum()
                        dice = (2 * inter) / (p.sum() + t.sum() + 1e-6)
                        val_dice_scores.append(dice.item())

        val_loss /= len(val_loader)
        mean_dice = np.mean(val_dice_scores) if val_dice_scores else 0.0

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d} │ Train Loss: {train_loss:.4f} │ "
              f"Val Loss: {val_loss:.4f} │ Val Dice: {mean_dice:.4f} │ "
              f"LR: {lr_now:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "runs/u2net_v2/best_u2net_v2.pth")
            print(f"  💾 Best model saved! (val_loss={val_loss:.4f})")

    # ── Save training history ──
    import json
    with open("runs/u2net_v2/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Model: runs/u2net_v2/best_u2net_v2.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
