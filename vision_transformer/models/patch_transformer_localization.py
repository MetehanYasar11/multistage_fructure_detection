"""
Patch-Based Transformer with Spatial Localization

Enhanced version that performs BOTH:
1. Global Classification: "Is there a fracture in the image?"
2. Patch-Level Localization: "Which patches contain the fracture?"

Key Improvements over baseline:
- Separate prediction heads for each patch (no global pooling until final)
- Multi-task loss: Global BCE + Patch-level localization loss
- Weakly-supervised: Uses only image-level labels
- Spatial heatmap: Visualize which patches are "fracture-positive"

Architecture:
    Image (1400×2800) → 392 Patches (100×100)
    ↓
    CNN Encoder (ResNet18) → 392 × 512D features
    ↓
    Positional Encoding (2D) → Spatial awareness
    ↓
    Transformer Encoder (6 layers) → Contextual features
    ↓
    ┌─────────────────┬─────────────────┐
    │  Patch Head     │  Global Head    │
    │  (392 logits)   │  (1 logit)      │
    └─────────────────┴─────────────────┘
    ↓                 ↓
    Patch Probs      Final Prediction
    (Heatmap)        (Classification)

Author: Master's Thesis Project
Date: October 28, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict
import timm


class PatchExtractor(nn.Module):
    """Extract non-overlapping patches from panoramic X-ray images."""
    
    def __init__(self, patch_size: int = 100):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, num_patches, C, patch_size, patch_size)
            num_patches_h: Vertical patch count
            num_patches_w: Horizontal patch count
        """
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        H_crop = num_patches_h * patch_size
        W_crop = num_patches_w * patch_size
        x = x[:, :, :H_crop, :W_crop]
        
        x = x.view(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        
        num_patches = num_patches_h * num_patches_w
        patches = x.reshape(B, num_patches, C, patch_size, patch_size)
        
        return patches, num_patches_h, num_patches_w


class PatchCNNEncoder(nn.Module):
    """CNN encoder for individual patches."""
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        feature_dim: int = 512,
        patch_size: int = 100
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        backbone_dim = self.backbone.num_features
        
        if backbone_dim != feature_dim:
            self.projection = nn.Linear(backbone_dim, feature_dim)
        else:
            self.projection = nn.Identity()
        
        print(f"Patch CNN Encoder: {backbone} -> {feature_dim}D features")
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, num_patches, C, patch_size, patch_size)
        Returns:
            features: (B, num_patches, feature_dim)
        """
        B, num_patches, C, H, W = patches.shape
        patches_flat = patches.view(B * num_patches, C, H, W)
        features = self.backbone(patches_flat)
        features = self.projection(features)
        features = features.view(B, num_patches, self.feature_dim)
        return features


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for patch grid."""
    
    def __init__(self, feature_dim: int, max_h: int = 50, max_w: int = 50):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        pe_h = torch.zeros(max_h, feature_dim // 2)
        pe_w = torch.zeros(max_w, feature_dim // 2)
        
        position_h = torch.arange(0, max_h).unsqueeze(1).float()
        position_w = torch.arange(0, max_w).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, feature_dim // 2, 2).float() * 
                            (-math.log(10000.0) / (feature_dim // 2)))
        
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
        
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
    
    def forward(
        self,
        x: torch.Tensor,
        num_patches_h: int,
        num_patches_w: int
    ) -> torch.Tensor:
        """Add 2D positional encoding."""
        B, num_patches, _ = x.shape
        
        pos_enc = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                pe = torch.cat([self.pe_h[i], self.pe_w[j]], dim=0)
                pos_enc.append(pe)
        
        pos_enc = torch.stack(pos_enc, dim=0).to(x.device)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)
        
        return x + pos_enc


class TransformerEncoder(nn.Module):
    """Transformer encoder to model spatial relationships between patches."""
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        print(f"Transformer: {num_layers} layers, {num_heads} heads, {feature_dim}D")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, feature_dim)
        Returns:
            encoded: (B, num_patches, feature_dim)
        """
        return self.transformer(x)


class PatchTransformerWithLocalization(nn.Module):
    """
    Enhanced Patch Transformer with Spatial Localization.
    
    Key Difference from Baseline:
    - Baseline: Transformer → Global Pooling → 1 logit
    - This: Transformer → Patch Head (392 logits) + Global Head (1 logit)
    
    Benefits:
    - Spatial heatmap showing fracture locations
    - Multi-task learning improves global classification
    - Interpretable for radiologists
    
    Args:
        image_size: (H, W) tuple for input image size
        patch_size: Size of each patch (100×100 default)
        cnn_backbone: CNN backbone for patch encoding
        feature_dim: Feature dimension for transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        use_global_head: Whether to use separate global classification head
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (1400, 2800),
        patch_size: int = 100,
        cnn_backbone: str = 'resnet18',
        feature_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_global_head: bool = True
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.use_global_head = use_global_head
        
        H, W = image_size
        self.num_patches_h = H // patch_size
        self.num_patches_w = W // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        print("="*80)
        print("PATCH TRANSFORMER WITH SPATIAL LOCALIZATION")
        print("="*80)
        print(f"Image size: {H}x{W}")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Grid: {self.num_patches_h}x{self.num_patches_w} = {self.num_patches} patches")
        print(f"Feature dim: {feature_dim}")
        print(f"Mode: Multi-task (Patch + Global)" if use_global_head else "Patch-only")
        
        # 1. Patch extraction
        self.patch_extractor = PatchExtractor(patch_size=patch_size)
        
        # 2. CNN encoder for patches
        self.patch_encoder = PatchCNNEncoder(
            backbone=cnn_backbone,
            pretrained=True,
            feature_dim=feature_dim,
            patch_size=patch_size
        )
        
        # 3. Positional encoding
        self.pos_encoding = PositionalEncoding2D(
            feature_dim=feature_dim,
            max_h=self.num_patches_h + 10,
            max_w=self.num_patches_w + 10
        )
        
        # 4. Transformer encoder
        self.transformer = TransformerEncoder(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=4,
            dropout=dropout
        )
        
        # 5. Prediction heads
        # Patch-level head: Predicts fracture probability for EACH patch
        self.patch_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1)  # Binary logit per patch
        )
        
        # Global head: Predicts overall image-level classification
        if use_global_head:
            # Use a global token (like BERT's [CLS])
            self.global_token = nn.Parameter(torch.randn(1, 1, feature_dim))
            
            self.global_head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, 1)  # Binary logit for image
            )
        
        print(f"Patch Head: {feature_dim} -> {feature_dim//2} -> 1 (per patch)")
        if use_global_head:
            print(f"Global Head: {feature_dim} -> {feature_dim//2} -> 1 (global)")
        print("="*80)
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*80)
    
    def _initialize_weights(self):
        """
        Initialize weights for better training.
        - Linear layers: Xavier/Kaiming initialization
        - Final prediction layers: Small weights for stable start
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        # Initialize final prediction layers with smaller weights
        for head in [self.patch_head, self.global_head if self.use_global_head else None]:
            if head is not None:
                # Last linear layer should have small weights
                if isinstance(head[-1], nn.Linear):
                    nn.init.trunc_normal_(head[-1].weight, std=0.001)
                    if head[-1].bias is not None:
                        # Small negative bias for class imbalance (more healthy than fractured)
                        nn.init.constant_(head[-1].bias, -0.01)
    
    def forward(
        self,
        x: torch.Tensor,
        return_patch_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task predictions.
        
        Args:
            x: Input images (B, C, H, W)
            return_patch_logits: If True, return raw patch logits (for visualization)
            
        Returns:
            Dictionary containing:
                - 'global_logits': (B, 1) - Final image-level classification
                - 'patch_logits': (B, num_patches, 1) - Per-patch predictions
                - 'patch_probs': (B, num_patches) - Patch probabilities (sigmoid)
                - 'num_patches_h': Vertical patch count
                - 'num_patches_w': Horizontal patch count
        """
        B = x.shape[0]
        
        # 1. Extract patches
        patches, num_patches_h, num_patches_w = self.patch_extractor(x)
        # patches: (B, num_patches, C, patch_size, patch_size)
        
        # 2. Encode patches with CNN
        patch_features = self.patch_encoder(patches)
        # patch_features: (B, num_patches, feature_dim)
        
        # 3. Add positional encoding
        patch_features = self.pos_encoding(patch_features, num_patches_h, num_patches_w)
        # patch_features: (B, num_patches, feature_dim)
        
        # 4. Add global token if using global head
        if self.use_global_head:
            global_token = self.global_token.expand(B, -1, -1)  # (B, 1, feature_dim)
            transformer_input = torch.cat([global_token, patch_features], dim=1)
            # transformer_input: (B, 1 + num_patches, feature_dim)
        else:
            transformer_input = patch_features
        
        # 5. Transformer encoding
        transformer_output = self.transformer(transformer_input)
        # transformer_output: (B, 1 + num_patches, feature_dim) or (B, num_patches, feature_dim)
        
        # 6. Split global token and patch tokens
        if self.use_global_head:
            global_features = transformer_output[:, 0, :]  # (B, feature_dim)
            patch_features_encoded = transformer_output[:, 1:, :]  # (B, num_patches, feature_dim)
        else:
            patch_features_encoded = transformer_output
        
        # 7. Patch-level predictions
        patch_logits = self.patch_head(patch_features_encoded)  # (B, num_patches, 1)
        patch_logits = patch_logits.squeeze(-1)  # (B, num_patches)
        
        # 8. Global prediction
        if self.use_global_head:
            # Use global token for classification
            global_logits = self.global_head(global_features)  # (B, 1)
        else:
            # Aggregate patch predictions (max pooling like baseline)
            global_logits = patch_logits.max(dim=1, keepdim=True)[0]  # (B, 1)
        
        # 9. Convert patch logits to probabilities
        patch_probs = torch.sigmoid(patch_logits)  # (B, num_patches)
        
        # Return comprehensive output
        output = {
            'global_logits': global_logits,  # (B, 1) - for BCE loss
            'patch_logits': patch_logits,    # (B, num_patches) - for patch loss
            'patch_probs': patch_probs,      # (B, num_patches) - for visualization
            'num_patches_h': num_patches_h,
            'num_patches_w': num_patches_w
        }
        
        return output
    
    def get_patch_heatmap(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate spatial heatmap showing fracture locations.
        
        Args:
            x: Input images (B, C, H, W)
            threshold: Probability threshold for "fracture" patch
            
        Returns:
            heatmap: (B, num_patches_h, num_patches_w) - 2D heatmap
            patch_probs: (B, num_patches) - Flat probabilities
        """
        with torch.no_grad():
            output = self.forward(x, return_patch_logits=True)
            patch_probs = output['patch_probs']  # (B, num_patches)
            nh, nw = output['num_patches_h'], output['num_patches_w']
            
            # Reshape to 2D grid
            heatmap = patch_probs.view(-1, nh, nw)  # (B, nh, nw)
            
        return heatmap, patch_probs


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING PATCH TRANSFORMER WITH LOCALIZATION")
    print("="*80 + "\n")
    
    # Create model
    model = PatchTransformerWithLocalization(
        image_size=(1400, 2800),
        patch_size=100,
        cnn_backbone='resnet18',
        feature_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        use_global_head=True
    )
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 1400, 2800)
    
    print("\n📥 Input shape:", x.shape)
    
    # Forward pass
    output = model(x)
    
    print("\n📤 Output shapes:")
    print(f"  - global_logits: {output['global_logits'].shape}")
    print(f"  - patch_logits: {output['patch_logits'].shape}")
    print(f"  - patch_probs: {output['patch_probs'].shape}")
    print(f"  - Grid: {output['num_patches_h']}×{output['num_patches_w']}")
    
    # Get heatmap
    heatmap, patch_probs = model.get_patch_heatmap(x)
    print(f"\n🗺️ Heatmap shape: {heatmap.shape}")
    print(f"  - Patch probs range: [{patch_probs.min():.3f}, {patch_probs.max():.3f}]")
    
    # Check variance
    for i in range(batch_size):
        probs = patch_probs[i].numpy()
        print(f"\n📊 Image {i} patch statistics:")
        print(f"  - Mean: {probs.mean():.4f}")
        print(f"  - Std:  {probs.std():.4f}")
        print(f"  - Min:  {probs.min():.4f}")
        print(f"  - Max:  {probs.max():.4f}")
    
    print("\n✅ Model test completed!")
    print("="*80)
