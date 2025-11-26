"""
Attention-Guided Patch Transformer for Fracture Localization

Architecture:
1. Patch Transformer (baseline) → Classification + Patch attention scores
2. Attention Map → Which patches contain fractures
3. Bbox Generation → Convert high-attention patches to bboxes
4. RCT Intersection → Match with RCT detector results

Key Innovation:
- Patch-level attention weights reveal WHERE fracture is located
- No need for pixel-level annotations
- Weakly-supervised localization from image-level labels

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class PatchCNNEncoder(nn.Module):
    """CNN to encode each patch"""
    
    def __init__(self, d_model=512):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 100x100 -> 50x50
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 50x50 -> 25x25
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 25x25 -> 12x12
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 12x12 -> 6x6
            nn.Conv2d(256, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> (B, d_model, 1, 1)
        )
    
    def forward(self, x):
        return self.encoder(x).squeeze(-1).squeeze(-1)  # (B, d_model)


class AttentionGuidedPatchTransformer(nn.Module):
    """
    Patch Transformer with Attention-Based Localization
    
    Outputs:
    - Classification logit
    - Patch attention scores (for localization)
    - Attention map visualization
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (1400, 2800),
        patch_size: int = 100,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Calculate grid
        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        print(f"\nAttention-Guided Patch Transformer:")
        print(f"  Image size: {image_size[0]}x{image_size[1]}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Grid: {self.num_patches_h}x{self.num_patches_w} = {self.num_patches} patches")
        print(f"  d_model: {d_model}, heads: {nhead}, layers: {num_layers}")
        
        # Patch encoder
        self.patch_encoder = PatchCNNEncoder(d_model)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Attention head for localization
        self.attention_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)  # Score per patch
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from image
        
        Args:
            x: (B, C, H, W)
            
        Returns:
            patches: (B, num_patches, C, patch_size, patch_size)
        """
        B, C, H, W = x.shape
        
        # Unfold to patches
        patches = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, nh, W, ps)
        patches = patches.unfold(3, self.patch_size, self.patch_size)  # (B, C, nh, nw, ps, ps)
        
        # Rearrange to (B, nh*nw, C, ps, ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, self.num_patches, C, self.patch_size, self.patch_size)
        
        return patches
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: (B, C, H, W) input image
            return_attention: If True, return patch attention scores
            
        Returns:
            logits: (B, num_classes) classification logits
            attention_map: (B, num_patches_h, num_patches_w) if return_attention=True
        """
        B = x.size(0)
        
        # Extract patches
        patches = self.extract_patches(x)  # (B, N, C, ps, ps)
        
        # Encode patches
        B_p, N, C, ps_h, ps_w = patches.shape
        patches_flat = patches.view(B_p * N, C, ps_h, ps_w)
        
        patch_features = self.patch_encoder(patches_flat)  # (B*N, d_model)
        patch_features = patch_features.view(B_p, N, self.d_model)  # (B, N, d_model)
        
        # Add positional encoding
        patch_features = patch_features + self.pos_embedding
        
        # Transformer encoding
        encoded = self.transformer(patch_features)  # (B, N, d_model)
        
        # Classification: global average pooling
        global_features = encoded.mean(dim=1)  # (B, d_model)
        logits = self.classifier(global_features)  # (B, num_classes)
        logits = logits.squeeze(1)  # (B,)
        
        # Attention scores for localization
        if return_attention:
            attention_scores = self.attention_head(encoded)  # (B, N, 1)
            attention_scores = attention_scores.squeeze(-1)  # (B, N)
            
            # Reshape to 2D grid
            attention_map = attention_scores.view(B, self.num_patches_h, self.num_patches_w)
            
            # Normalize with softmax
            attention_map = F.softmax(attention_map.view(B, -1), dim=1)
            attention_map = attention_map.view(B, self.num_patches_h, self.num_patches_w)
            
            return logits, attention_map
        
        return logits, None
    
    def get_fracture_bboxes(
        self,
        attention_map: torch.Tensor,
        threshold: float = 0.5,
        min_cluster_size: int = 2
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Convert attention map to bounding boxes
        
        Args:
            attention_map: (B, nh, nw) attention scores
            threshold: Attention threshold (percentile-based)
            min_cluster_size: Minimum connected patches to form bbox
            
        Returns:
            bboxes: List of bbox lists per batch
                    Each bbox: (x1, y1, x2, y2) in pixel coordinates
        """
        B, nh, nw = attention_map.shape
        attention_map_np = attention_map.detach().cpu().numpy()
        
        all_bboxes = []
        
        for b in range(B):
            attn = attention_map_np[b]
            
            # Dynamic threshold: top X% of attention
            threshold_value = np.percentile(attn, threshold * 100)
            binary_mask = (attn > threshold_value).astype(np.uint8)
            
            # Find connected components
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary_mask)
            
            bboxes = []
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if component.sum() < min_cluster_size:
                    continue
                
                # Get bounding box
                rows, cols = np.where(component)
                r_min, r_max = rows.min(), rows.max()
                c_min, c_max = cols.min(), cols.max()
                
                # Convert patch coordinates to pixel coordinates
                x1 = c_min * self.patch_size
                y1 = r_min * self.patch_size
                x2 = (c_max + 1) * self.patch_size
                y2 = (r_max + 1) * self.patch_size
                
                bboxes.append((x1, y1, x2, y2))
            
            all_bboxes.append(bboxes)
        
        return all_bboxes
    
    def visualize_attention(
        self,
        image: torch.Tensor,
        attention_map: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention map overlaid on image
        
        Args:
            image: (C, H, W) input image
            attention_map: (nh, nw) attention scores
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        import cv2
        
        # Convert image to numpy
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Upsample attention map to image size
        attn_np = attention_map.cpu().numpy()
        attn_resized = cv2.resize(
            attn_np,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create heatmap
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_np)
        plt.imshow(attn_resized, cmap='jet', alpha=0.5)
        plt.title("Attention Map (Red=High)")
        plt.colorbar()
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def test_attention_model():
    """Test attention-guided model"""
    print("="*80)
    print("TESTING ATTENTION-GUIDED PATCH TRANSFORMER")
    print("="*80)
    
    # Create model
    model = AttentionGuidedPatchTransformer(
        image_size=(1400, 2800),
        patch_size=100,
        d_model=512,
        nhead=8,
        num_layers=6
    )
    model.eval()
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 1400, 2800)
    
    with torch.no_grad():
        # Classification only
        logits, _ = model(x, return_attention=False)
        print(f"Logits shape: {logits.shape}")
        
        # With attention
        logits, attention_map = model(x, return_attention=True)
        print(f"Attention map shape: {attention_map.shape}")
        print(f"Attention range: [{attention_map.min():.4f}, {attention_map.max():.4f}]")
        
        # Get bboxes
        bboxes = model.get_fracture_bboxes(attention_map, threshold=0.7)
        print(f"\nDetected bboxes (sample 0): {len(bboxes[0])} regions")
        for i, bbox in enumerate(bboxes[0][:3]):  # Show first 3
            print(f"  Region {i+1}: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    test_attention_model()
