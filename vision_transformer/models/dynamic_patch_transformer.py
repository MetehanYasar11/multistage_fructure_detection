"""
Dynamic Patch Transformer for Variable-Sized RCT Crops

Key Features:
- Minimum patch size: 10x10
- Target: ~300 patches per image (like successful baseline)
- Adaptive grid based on input size
- Maintains baseline architecture success

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class DynamicPatchTransformer(nn.Module):
    """
    Patch-based Transformer with dynamic patch grid calculation
    
    Maintains same architecture as baseline but adapts to variable input sizes
    while keeping patch density similar to successful baseline (300 patches for 1400x2800)
    """
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        min_patch_size=10,
        target_patch_density=300 / (1400 * 2800)  # patches per pixel from baseline
    ):
        super().__init__()
        
        self.d_model = d_model
        self.min_patch_size = min_patch_size
        self.target_patch_density = target_patch_density
        
        print(f"\nDynamic Patch Transformer Configuration:")
        print(f"  d_model: {d_model}")
        print(f"  nhead: {nhead}")
        print(f"  num_layers: {num_layers}")
        print(f"  min_patch_size: {min_patch_size}x{min_patch_size}")
        print(f"  target_patch_density: {target_patch_density:.2e} patches/pixel")
        
        # Patch CNN Encoder (same as baseline)
        self.patch_cnn = nn.Sequential(
            # Layer 1: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /2
            
            # Layer 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /4
            
            # Layer 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /8
            
            # Layer 4: 256 -> d_model
            nn.Conv2d(256, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> (B, d_model, 1, 1)
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_model))  # Max 1000 patches
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, 1)
        )
        
        # Stats for logging
        self.last_patch_grid = None
        self.last_patch_size = None
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    def compute_dynamic_patches(self, height: int, width: int) -> Tuple[int, int, int]:
        """
        Compute optimal patch grid and size for given image dimensions
        
        Strategy:
        1. Calculate target number of patches based on image area and baseline density
        2. Find patch size that gives closest to target patches while >= min_patch_size
        3. Ensure patch count is reasonable (between 50-500)
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            (patch_size, num_patches_h, num_patches_w)
        """
        area = height * width
        target_patches = int(area * self.target_patch_density)
        
        # Clamp to reasonable range
        target_patches = max(50, min(500, target_patches))
        
        # Find optimal patch size
        # Total patches = (H / patch_size) * (W / patch_size)
        # patch_size = sqrt(H * W / target_patches)
        optimal_patch_size = math.sqrt(area / target_patches)
        
        # Round to get integer patch size, ensure >= min
        patch_size = max(self.min_patch_size, int(optimal_patch_size))
        
        # Calculate actual grid
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        
        # Ensure at least 1 patch in each dimension
        num_patches_h = max(1, num_patches_h)
        num_patches_w = max(1, num_patches_w)
        
        # Adjust patch size to use full image
        actual_patch_h = height // num_patches_h
        actual_patch_w = width // num_patches_w
        
        # Use square patches (average of h and w)
        final_patch_size = (actual_patch_h + actual_patch_w) // 2
        final_patch_size = max(self.min_patch_size, final_patch_size)
        
        return final_patch_size, num_patches_h, num_patches_w
    
    def extract_patches(self, x: torch.Tensor, patch_size: int, num_h: int, num_w: int) -> torch.Tensor:
        """
        Extract patches from image tensor
        
        Args:
            x: (B, C, H, W)
            patch_size: Size of each patch
            num_h: Number of patches vertically
            num_w: Number of patches horizontally
            
        Returns:
            patches: (B, num_h * num_w, C, patch_size, patch_size)
        """
        B, C, H, W = x.shape
        
        # Calculate actual patch dimensions
        patch_h = H // num_h
        patch_w = W // num_w
        
        # Extract using unfold
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        # Shape: (B, C, num_h, num_w, patch_h, patch_w)
        
        # Rearrange to (B, num_h * num_w, C, patch_h, patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, num_h * num_w, C, patch_h, patch_w)
        
        # Resize patches to uniform size if needed
        if patch_h != patch_size or patch_w != patch_size:
            B_p, N, C_p, _, _ = patches.shape
            patches = patches.view(B_p * N, C_p, patch_h, patch_w)
            patches = nn.functional.interpolate(
                patches,
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False
            )
            patches = patches.view(B_p, N, C_p, patch_size, patch_size)
        
        return patches
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic patching
        
        Args:
            x: (B, C, H, W) - Variable H, W
            
        Returns:
            logits: (B,) - Binary classification logits
        """
        B, C, H, W = x.shape
        
        # Compute dynamic patch configuration
        patch_size, num_h, num_w = self.compute_dynamic_patches(H, W)
        num_patches = num_h * num_w
        
        # Store for logging
        self.last_patch_grid = (num_h, num_w)
        self.last_patch_size = patch_size
        
        # Extract patches
        patches = self.extract_patches(x, patch_size, num_h, num_w)
        # Shape: (B, num_patches, C, patch_size, patch_size)
        
        # Encode each patch with CNN
        B_p, N, C_p, P_h, P_w = patches.shape
        patches_flat = patches.view(B_p * N, C_p, P_h, P_w)
        
        patch_features = self.patch_cnn(patches_flat)  # (B*N, d_model, 1, 1)
        patch_features = patch_features.view(B_p, N, self.d_model)  # (B, N, d_model)
        
        # Add positional encoding
        patch_features = patch_features + self.pos_embedding[:, :N, :]
        
        # Transformer encoding
        encoded = self.transformer_encoder(patch_features)  # (B, N, d_model)
        
        # Global pooling
        global_features = encoded.mean(dim=1)  # (B, d_model)
        
        # Classification
        logits = self.classifier(global_features)  # (B, 1)
        logits = logits.squeeze(1)  # (B,)
        
        return logits
    
    def get_last_patch_info(self) -> dict:
        """Get information about last forward pass patches"""
        if self.last_patch_grid is None:
            return {}
        
        num_h, num_w = self.last_patch_grid
        return {
            'patch_size': self.last_patch_size,
            'grid_h': num_h,
            'grid_w': num_w,
            'total_patches': num_h * num_w
        }


def test_dynamic_patching():
    """Test dynamic patch computation with different image sizes"""
    print("="*80)
    print("TESTING DYNAMIC PATCH TRANSFORMER")
    print("="*80)
    
    model = DynamicPatchTransformer(
        d_model=512,
        nhead=8,
        num_layers=6,
        min_patch_size=10,
        target_patch_density=300 / (1400 * 2800)
    )
    model.eval()
    
    # Test with different RCT crop sizes
    test_sizes = [
        (224, 224, "Square crop"),
        (300, 200, "Horizontal crop"),
        (200, 300, "Vertical crop"),
        (400, 150, "Wide crop"),
        (150, 400, "Tall crop"),
        (1400, 2800, "Full panoramic (baseline)")
    ]
    
    print("\n" + "="*80)
    print("PATCH CONFIGURATION FOR DIFFERENT IMAGE SIZES")
    print("="*80)
    
    for height, width, desc in test_sizes:
        # Create dummy input
        x = torch.randn(2, 3, height, width)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Get patch info
        info = model.get_last_patch_info()
        
        print(f"\n{desc}: {height}x{width}")
        print(f"  Patch size: {info['patch_size']}x{info['patch_size']}")
        print(f"  Grid: {info['grid_h']}x{info['grid_w']}")
        print(f"  Total patches: {info['total_patches']}")
        print(f"  Patch density: {info['total_patches']/(height*width):.2e} patches/pixel")
        print(f"  Output shape: {output.shape}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    test_dynamic_patching()
