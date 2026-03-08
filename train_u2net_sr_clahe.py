import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn.functional as F

# ==========================================
# U2-Net Architecture (Salient Object Detection)
# Referenced from İnönü N, vd., Diagnostics 2025.
# ==========================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def upsample_like(src, tgt):
    return F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=False)

class RSU4(nn.Module):
    """Residual U-block: Micro-UNet inside the main UNet for extremely fine details"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.c_in = ConvBNReLU(in_ch, out_ch)
        
        self.c1 = ConvBNReLU(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.c2 = ConvBNReLU(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2)
        self.c3 = ConvBNReLU(mid_ch, mid_ch)
        
        # Dilated convolution at the bottom to increase receptive field without losing resolution
        self.c4 = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        
        self.c3_up = ConvBNReLU(mid_ch*2, mid_ch)
        self.c2_up = ConvBNReLU(mid_ch*2, mid_ch)
        self.c1_up = ConvBNReLU(mid_ch*2, out_ch)

    def forward(self, x):
        x0 = self.c_in(x)
        
        x1 = self.c1(x0)
        x2 = self.c2(self.pool1(x1))
        x3 = self.c3(self.pool2(x2))
        x4 = self.c4(x3)
        
        x3_up = self.c3_up(torch.cat([x4, x3], dim=1))
        x2_up = self.c2_up(torch.cat([upsample_like(x3_up, x2), x2], dim=1))
        x1_up = self.c1_up(torch.cat([upsample_like(x2_up, x1), x1], dim=1))
        
        # Residual Connection
        return x0 + x1_up

class U2Net_Lite(nn.Module):
    """U2-Net Lite adapted for very fine VRF segmentation"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        
        # Encoder
        self.stage1 = RSU4(in_ch, 16, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.stage2 = RSU4(64, 16, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.stage3 = RSU4(64, 16, 64)
        
        # Decoder
        self.stage2_up = RSU4(128, 16, 64)
        self.stage1_up = RSU4(128, 16, 64)
        
        # Side Outputs (Deep Supervision)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        
        # Fused Output
        self.outconv = nn.Conv2d(3, out_ch, 1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool1(hx1))
        hx3 = self.stage3(self.pool2(hx2))
        
        hx2_up = self.stage2_up(torch.cat([upsample_like(hx3, hx2), hx2], dim=1))
        hx1_up = self.stage1_up(torch.cat([upsample_like(hx2_up, hx1), hx1], dim=1))
        
        # Generate multi-level probability maps
        d1 = self.side1(hx1_up)
        d2 = upsample_like(self.side2(hx2_up), d1)
        d3 = upsample_like(self.side3(hx3), d1)
        
        # Fuse maps
        d_out = self.outconv(torch.cat([d1, d2, d3], dim=1))
        
        if self.training:
            # U2Net trains by summing the loss from all multi-level outputs
            return d_out, d1, d2, d3
        else:
            return d_out

# ==========================================
# Dataset (With target Dilation Morphological logic)
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_dir, img_size=(224, 224)):
        self.image_paths = image_paths
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        
        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # SOLUTION #2: Dilation (Thicken the root fracture line so the network doesn't drop it in max-pools)
        if mask.max() > 0:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
        mask = (mask > 127).astype(np.float32)
        
        img_tensor = self.transform(img)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return img_tensor, mask_tensor

# ==========================================
# Specialized Loss for High Class Imbalance
# ==========================================
def focal_bce_dice_loss(pred, target, pos_weight=50.0):
    """
    Combines Weighted BCE (forcing the model to care about the rare white pixels)
    with Dice loss to evaluate the structure shape.
    """
    weight = torch.tensor(pos_weight).to(pred.device)
    bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight)
    
    pred_sig = torch.sigmoid(pred)
    smooth = 1.0
    intersection = (pred_sig * target).sum(dim=(2,3))
    dice = 1 - (2.0 * intersection + smooth) / (pred_sig.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
    
    return bce + dice.mean()

def u2net_loss(outputs, target):
    """Deep Supervision Loss: sums the loss over fused map and all intermediate maps"""
    loss = 0.0
    # The paper forces the network to learn the fracture on multiple abstraction levels
    for out in outputs:
        loss += focal_bce_dice_loss(out, target, pos_weight=50.0)
    return loss

def train_model():
    dataset_dir = Path("auto_labeled_segmentation_sr_clahe")
    all_images = list((dataset_dir / "images").glob("*.png"))
    
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
    train_ds = SegmentationDataset(train_imgs, dataset_dir / "masks")
    val_ds = SegmentationDataset(val_imgs, dataset_dir / "masks")
    
    # Num workers 0 to prevent windows multiprocessing crash
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = U2Net_Lite().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Learning rate scheduler will drop LR when plateauing to fine-tune the lines
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    num_epochs = 15
    best_val_loss = float('inf')
    
    os.makedirs("runs/u2net_segmentation", exist_ok=True)
    
    print(f"Training U2-Net Lite on {device} with {len(train_imgs)} images...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass outputs 4 layers in train mode
            outputs = model(imgs) 
            
            loss = u2net_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                # Forward pass outputs 1 layer in eval mode
                outputs = model(imgs) 
                
                loss = focal_bce_dice_loss(outputs, masks, pos_weight=50.0)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "runs/u2net_segmentation/best_u2net.pth")
            print("💾 Saved best U2-Net model!")

if __name__ == "__main__":
    train_model()