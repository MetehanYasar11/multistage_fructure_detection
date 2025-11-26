"""
Weakly-Supervised Patch Transformer with Attention Head

Extends the baseline Patch Transformer with an attention head for localization.
Multi-task learning: Classification (image-level) + Localization (attention map)

Key Features:
- Dual outputs: class logits + attention heatmap
- Attention head learns fracture location from bbox supervision
- Preserves classification performance while adding localization

Architecture:
    Input (1400x2800) 
    → Patch Encoder (ResNet18) 
    → Transformer (6 layers)
    → Dual Heads:
        ├── Classification Head: [B, 2]
        └── Attention Head: [B, 1, 14, 28]

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PatchCNNEncoder(nn.Module):
    """
    CNN encoder for individual patches using timm backbone.
    Compatible with baseline patch_transformer.py
    """
    def __init__(
        self, 
        backbone: str = 'resnet18',
        pretrained: bool = True,
        feature_dim: int = 512
    ):
        super().__init__()
        import timm
        
        self.feature_dim = feature_dim
        
        # Load pretrained backbone (without classifier)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        backbone_dim = self.backbone.num_features
        
        # Project to target feature dimension if needed
        if backbone_dim != feature_dim:
            self.projection = nn.Linear(backbone_dim, feature_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] - batch of patches
        Returns:
            [B, feature_dim] - patch embeddings
        """
        features = self.backbone(x)  # [B, backbone_dim]
        projected = self.projection(features)  # [B, feature_dim]
        return projected


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for patch grid.
    Cloned from original patch_transformer.py
    """
    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        
        # Create 2D positional encoding
        pe = torch.zeros(height, width, d_model)
        
        # Height encoding
        y_pos = torch.arange(0, height).unsqueeze(1).float()
        x_pos = torch.arange(0, width).unsqueeze(0).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Encode height
        pe[:, :, 0::2] = torch.sin(y_pos * div_term).unsqueeze(1).expand(-1, width, -1)
        pe[:, :, 1::2] = torch.cos(y_pos * div_term).unsqueeze(1).expand(-1, width, -1)
        
        # Flatten to sequence
        pe = pe.view(-1, d_model)  # [H*W, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, d_model] where N = H*W
        Returns:
            [B, N, d_model] with positional encoding added
        """
        return x + self.pe.unsqueeze(0)


class WeaklySupervisedPatchTransformer(nn.Module):
    """
    Patch Transformer with Attention Head for Weakly-Supervised Localization.
    
    Multi-task learning:
    1. Classification: Fractured vs Healthy (image-level supervision)
    2. Localization: Attention map (bbox supervision for fractured images)
    
    Attributes:
        image_size: (H, W) of input image
        patch_size: Size of each square patch
        feature_dim: Dimension of patch embeddings
        num_classes: Number of output classes (2: healthy/fractured)
        attention_resolution: (H, W) of output attention map
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (1400, 2800),
        patch_size: int = 100,
        feature_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        num_classes: int = 2,
        attention_channels: int = 256
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Calculate patch grid dimensions
        self.grid_h = image_size[0] // patch_size
        self.grid_w = image_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        print("="*70)
        print("WEAKLY-SUPERVISED PATCH TRANSFORMER")
        print("="*70)
        print(f"Image size: {image_size[0]}x{image_size[1]}")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Grid: {self.grid_h}x{self.grid_w} = {self.num_patches} patches")
        print(f"Feature dim: {feature_dim}")
        
        # 1. Patch Encoder (CNN)
        self.patch_encoder = PatchCNNEncoder(
            backbone='resnet18',
            pretrained=True,
            feature_dim=feature_dim
        )
        print(f"Patch CNN Encoder: resnet18 -> {feature_dim}D features")
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding2D(
            d_model=feature_dim,
            height=self.grid_h,
            width=self.grid_w
        )
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        print(f"Transformer: {num_layers} layers, {num_heads} heads, {feature_dim}D")
        
        # 4. Classification Head (same as original)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        print(f"Classification Head: {feature_dim} -> {num_classes}")
        
        # 5. NEW: Attention Head for Localization
        # Takes spatial features [B, C, H, W] and outputs attention map [B, 1, H, W]
        # Deeper architecture with residual-like connections
        self.attention_head = nn.Sequential(
            # First block
            nn.Conv2d(feature_dim, attention_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(attention_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Second block
            nn.Conv2d(attention_channels, attention_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(attention_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Third block
            nn.Conv2d(attention_channels, attention_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(attention_channels // 2),
            nn.ReLU(),
            
            # Output - NO sigmoid here, we'll use BCE with logits
            nn.Conv2d(attention_channels // 2, 1, kernel_size=1)
        )
        print(f"Attention Head: {feature_dim} -> 1 channel (size: {self.grid_h}x{self.grid_w})")
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size (FP32): ~{total_params * 4 / 1e6:.2f} MB")
        print("="*70)
    
    def _init_weights(self):
        """Initialize weights for attention head with SMALL final layer"""
        for i, m in enumerate(self.attention_head.modules()):
            if isinstance(m, nn.Conv2d):
                # Use smaller initialization for final layer (layer index 16)
                if i == 16:  # Final 1x1 conv
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)  # VERY small weights
                    if m.bias is not None:
                        nn.init.constant_(m.bias, -2.0)  # Negative bias = low initial attention
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from input image.
        
        Args:
            x: [B, 3, H, W] - input images
            
        Returns:
            [B*num_patches, 3, patch_size, patch_size] - extracted patches
        """
        B, C, H, W = x.shape
        
        # Unfold to extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size) \
                   .unfold(3, self.patch_size, self.patch_size)
        # [B, 3, grid_h, grid_w, patch_size, patch_size]
        
        # Reshape to batch of patches
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # [B, grid_h, grid_w, 3, patch_size, patch_size]
        
        patches = patches.view(B * self.num_patches, C, self.patch_size, self.patch_size)
        # [B*num_patches, 3, patch_size, patch_size]
        
        return patches
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with dual outputs.
        
        Args:
            x: [B, 3, H, W] - input images
            return_attention: Whether to compute attention map
            
        Returns:
            Tuple of:
                - logits: [B, num_classes] - classification logits
                - attention_map: [B, 1, grid_h, grid_w] - attention heatmap (if return_attention=True)
        """
        B = x.size(0)
        
        # 1. Extract patches
        patches = self.extract_patches(x)  # [B*N, 3, patch_size, patch_size]
        
        # 2. Encode patches
        patch_embeddings = self.patch_encoder(patches)  # [B*N, feature_dim]
        
        # 3. Reshape to batch of sequences
        patch_embeddings = patch_embeddings.view(B, self.num_patches, self.feature_dim)
        # [B, N, feature_dim]
        
        # 4. Add positional encoding
        patch_embeddings = self.pos_encoder(patch_embeddings)
        
        # 5. Transformer encoding
        transformer_out = self.transformer(patch_embeddings)  # [B, N, feature_dim]
        
        # 6. Reshape for spatial operations
        # [B, N, feature_dim] -> [B, feature_dim, grid_h, grid_w]
        spatial_features = transformer_out.view(B, self.grid_h, self.grid_w, self.feature_dim)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 7. Attention Head - compute attention weights first
        attention_map = self.attention_head(spatial_features)  # [B, 1, grid_h, grid_w]
        
        # 8. Classification Head - USE ATTENTION-WEIGHTED POOLING
        # Instead of max pooling (which loses spatial info), use attention weights
        # to aggregate features from important regions
        if not return_attention:
            # During inference without attention, fall back to max pooling
            pooled_features = transformer_out.max(dim=1)[0]  # [B, feature_dim]
        else:
            # Attention-weighted pooling: weight each patch by its attention score
            attention_weights = torch.sigmoid(attention_map)  # [B, 1, grid_h, grid_w]
            attention_weights = attention_weights.view(B, 1, -1)  # [B, 1, N]
            
            # Normalize attention weights
            attention_weights = attention_weights / (attention_weights.sum(dim=2, keepdim=True) + 1e-8)
            
            # Weighted sum of patch features
            # transformer_out: [B, N, feature_dim]
            # attention_weights: [B, 1, N]
            pooled_features = torch.bmm(attention_weights, transformer_out).squeeze(1)  # [B, feature_dim]
        
        logits = self.classifier(pooled_features)  # [B, num_classes]
        
        if return_attention:
            return logits, attention_map
        else:
            return logits, None
    
    def get_patch_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-patch predictions (for visualization/debugging).
        
        Args:
            x: [B, 3, H, W] - input images
            
        Returns:
            [B, grid_h, grid_w] - per-patch classification scores
        """
        B = x.size(0)
        
        # Extract and encode patches
        patches = self.extract_patches(x)
        patch_embeddings = self.patch_encoder(patches)
        patch_embeddings = patch_embeddings.view(B, self.num_patches, self.feature_dim)
        
        # Add positional encoding
        patch_embeddings = self.pos_encoder(patch_embeddings)
        
        # Transformer
        transformer_out = self.transformer(patch_embeddings)
        
        # Classify each patch independently
        patch_logits = self.classifier(transformer_out)  # [B, N, num_classes]
        patch_probs = F.softmax(patch_logits, dim=-1)[:, :, 1]  # [B, N] - fractured prob
        
        # Reshape to grid
        patch_probs = patch_probs.view(B, self.grid_h, self.grid_w)
        
        return patch_probs
    
    def load_from_baseline(self, baseline_checkpoint_path: str):
        """
        Load weights from baseline Patch Transformer (without attention head).
        
        This allows transfer learning from the pre-trained classification model.
        Only the attention head weights will be randomly initialized.
        
        Args:
            baseline_checkpoint_path: Path to baseline model checkpoint
        """
        print(f"\n[*] Loading baseline weights from: {baseline_checkpoint_path}")
        
        checkpoint = torch.load(baseline_checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            baseline_state = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            val_f1 = checkpoint.get('best_val_f1', 'unknown')
            print(f"    Baseline trained for {epoch} epochs, Val F1: {val_f1}")
        else:
            baseline_state = checkpoint
        
        # Load matching weights (patch_encoder, transformer, classifier)
        current_state = self.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for key, value in baseline_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                current_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        
        self.load_state_dict(current_state)
        
        print(f"    [OK] Loaded {len(loaded_keys)} parameter tensors")
        print(f"    [SKIP] Skipped {len(skipped_keys)} parameter tensors (attention head - will be trained from scratch)")
        print(f"    [*] Ready for weakly-supervised training!")


def test_model():
    """Test model creation and forward pass"""
    print("\n" + "="*70)
    print("TESTING WEAKLY-SUPERVISED PATCH TRANSFORMER")
    print("="*70)
    
    # Create model
    model = WeaklySupervisedPatchTransformer(
        image_size=(1400, 2800),
        patch_size=100,
        feature_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 1400, 2800)
    print(f"\nInput shape: {x.shape}")
    
    # Forward with attention
    logits, attention = model(x, return_attention=True)
    print(f"Classification logits: {logits.shape}")
    print(f"Attention map: {attention.shape}")
    
    # Get patch predictions
    patch_preds = model.get_patch_predictions(x)
    print(f"Patch predictions: {patch_preds.shape}")
    
    print("\n[OK] Model test passed!")
    print("="*70)


if __name__ == "__main__":
    test_model()
