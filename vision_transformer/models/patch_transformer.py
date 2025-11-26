"""
Patch-Based Transformer Classifier for Dental X-Ray Fracture Detection

This model processes panoramic X-rays using a patch-based approach inspired by
Vision Transformers (ViT), specifically designed for detecting small fractured
instruments in large panoramic images.

Architecture:
    1. Patch Extraction: Split image into 100×100 patches
    2. CNN Feature Extraction: Each patch → CNN → feature vector
    3. Transformer Encoder: Model spatial relationships between patches
    4. Patch Classification: Each patch → binary logit (has fracture or not)
    5. Global Aggregation: Combine patch predictions → final classification

Key Design Choices:
    - 100×100 patches: Small enough to focus on local details (fractured instruments)
    - CNN per patch: Extract visual features efficiently
    - Transformer: Capture long-range dependencies across panoramic image
    - Patch-wise logits: Interpretable (which patches have fractures)
    - Max pooling aggregation: If ANY patch has fracture → image is fractured

Author: Master's Thesis Project
Date: October 28, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import timm


class PatchExtractor(nn.Module):
    """
    Extract non-overlapping patches from panoramic X-ray images.
    
    For a 1400×2800 image with 100×100 patches:
        - Horizontal patches: 2800 / 100 = 28
        - Vertical patches: 1400 / 100 = 14
        - Total patches: 28 × 14 = 392 patches
    """
    
    def __init__(self, patch_size: int = 100):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Extract patches from input images.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            patches: (B, num_patches, C, patch_size, patch_size)
            num_patches_h: Number of patches vertically
            num_patches_w: Number of patches horizontally
        """
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        # Calculate number of patches
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # Crop to fit patches perfectly (discard remainder pixels)
        H_crop = num_patches_h * patch_size
        W_crop = num_patches_w * patch_size
        x = x[:, :, :H_crop, :W_crop]
        
        # Reshape to patches
        # (B, C, H, W) → (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        x = x.view(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        
        # Permute to group patches
        # (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        # → (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        
        # Flatten spatial patches
        # (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        # → (B, num_patches_h * num_patches_w, C, patch_size, patch_size)
        num_patches = num_patches_h * num_patches_w
        patches = x.reshape(B, num_patches, C, patch_size, patch_size)
        
        return patches, num_patches_h, num_patches_w


class PatchCNNEncoder(nn.Module):
    """
    CNN encoder for individual patches.
    
    Extracts features from each 100×100 patch using a lightweight CNN.
    Can use pretrained models (e.g., ResNet, EfficientNet) or custom CNN.
    """
    
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
        
        # Load pretrained backbone (without classifier)
        if backbone.startswith('resnet'):
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool='avg'
            )
            backbone_dim = self.backbone.num_features
        elif backbone.startswith('efficientnet'):
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool='avg'
            )
            backbone_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Project to target feature dimension if needed
        if backbone_dim != feature_dim:
            self.projection = nn.Linear(backbone_dim, feature_dim)
        else:
            self.projection = nn.Identity()
        
        print(f"Patch CNN Encoder: {backbone} -> {feature_dim}D features")
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Extract features from patches.
        
        Args:
            patches: (B, num_patches, C, patch_size, patch_size)
            
        Returns:
            features: (B, num_patches, feature_dim)
        """
        B, num_patches, C, H, W = patches.shape
        
        # Flatten batch and patches: (B * num_patches, C, H, W)
        patches_flat = patches.view(B * num_patches, C, H, W)
        
        # Extract features: (B * num_patches, backbone_dim)
        features = self.backbone(patches_flat)
        
        # Project to target dimension: (B * num_patches, feature_dim)
        features = self.projection(features)
        
        # Reshape back: (B, num_patches, feature_dim)
        features = features.view(B, num_patches, self.feature_dim)
        
        return features


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for patch grid.
    
    Adds spatial position information to patch features, preserving
    the 2D structure of the panoramic X-ray.
    """
    
    def __init__(self, feature_dim: int, max_h: int = 50, max_w: int = 50):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Create 2D positional encodings
        pe_h = torch.zeros(max_h, feature_dim // 2)
        pe_w = torch.zeros(max_w, feature_dim // 2)
        
        position_h = torch.arange(0, max_h).unsqueeze(1).float()
        position_w = torch.arange(0, max_w).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, feature_dim // 2, 2).float() * 
                            (-math.log(10000.0) / (feature_dim // 2)))
        
        # Sine and cosine for height
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        
        # Sine and cosine for width
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
        
        # Register as buffer (not trainable parameter)
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
    
    def forward(
        self,
        x: torch.Tensor,
        num_patches_h: int,
        num_patches_w: int
    ) -> torch.Tensor:
        """
        Add 2D positional encoding.
        
        Args:
            x: (B, num_patches, feature_dim)
            num_patches_h: Vertical patch count
            num_patches_w: Horizontal patch count
            
        Returns:
            x_pos: (B, num_patches, feature_dim) with positional encoding
        """
        B, num_patches, _ = x.shape
        
        # Create position encodings for current grid
        pos_enc = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Concatenate height and width encodings
                pe = torch.cat([self.pe_h[i], self.pe_w[j]], dim=0)
                pos_enc.append(pe)
        
        pos_enc = torch.stack(pos_enc, dim=0).to(x.device)  # (num_patches, feature_dim)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, num_patches, feature_dim)
        
        return x + pos_enc


class TransformerEncoder(nn.Module):
    """
    Transformer encoder to model spatial relationships between patches.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        print(f"Transformer: {num_layers} layers, {num_heads} heads, {feature_dim}D")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer encoding.
        
        Args:
            x: (B, num_patches, feature_dim)
            
        Returns:
            encoded: (B, num_patches, feature_dim)
        """
        return self.transformer(x)


class PatchTransformerClassifier(nn.Module):
    """
    Patch-based Transformer for dental fracture detection.
    
    Complete pipeline:
        Image → Patches → CNN Features → Positional Encoding 
        → Transformer → Patch Logits → Global Aggregation → Final Prediction
    
    Args:
        image_size: (H, W) tuple for input image size
        patch_size: Size of each patch (100×100 default)
        cnn_backbone: CNN backbone for patch encoding
        feature_dim: Feature dimension for transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        aggregation: How to combine patch predictions ('max', 'mean', 'attention')
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
        aggregation: str = 'max'
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.aggregation = aggregation
        
        H, W = image_size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w
        
        print("="*70)
        print("PATCH TRANSFORMER CLASSIFIER")
        print("="*70)
        print(f"Image size: {H}x{W}")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Grid: {num_patches_h}x{num_patches_w} = {num_patches} patches")
        print(f"Feature dim: {feature_dim}")
        print(f"Aggregation: {aggregation}")
        
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
            max_h=num_patches_h + 10,  # Extra buffer
            max_w=num_patches_w + 10
        )
        
        # 4. Transformer encoder
        self.transformer = TransformerEncoder(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 5. Patch-wise classification head
        self.patch_classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1)  # Binary logit per patch
        )
        
        # 6. Attention-based aggregation (if using)
        if aggregation == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size (FP32): ~{total_params * 4 / 1024**2:.2f} MB")
        print("="*70)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            logits: Binary classification logits (B, 1)
        """
        B = x.size(0)
        
        # 1. Extract patches: (B, num_patches, C, patch_size, patch_size)
        patches, num_patches_h, num_patches_w = self.patch_extractor(x)
        
        # 2. CNN encode patches: (B, num_patches, feature_dim)
        patch_features = self.patch_encoder(patches)
        
        # 3. Add positional encoding: (B, num_patches, feature_dim)
        patch_features = self.pos_encoding(patch_features, num_patches_h, num_patches_w)
        
        # 4. Transformer encoding: (B, num_patches, feature_dim)
        encoded_features = self.transformer(patch_features)
        
        # 5. Patch-wise classification: (B, num_patches, 1)
        patch_logits = self.patch_classifier(encoded_features)
        
        # 6. Aggregate patch predictions
        if self.aggregation == 'max':
            # Max pooling: If ANY patch predicts fracture → image has fracture
            global_logit = patch_logits.max(dim=1)[0]  # (B, 1)
        
        elif self.aggregation == 'mean':
            # Mean pooling: Average confidence across patches
            global_logit = patch_logits.mean(dim=1)  # (B, 1)
        
        elif self.aggregation == 'attention':
            # Attention-weighted aggregation
            attention_scores = self.attention_weights(encoded_features)  # (B, num_patches, 1)
            attention_weights = F.softmax(attention_scores, dim=1)  # (B, num_patches, 1)
            global_logit = (patch_logits * attention_weights).sum(dim=1)  # (B, 1)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return global_logit
    
    def get_patch_predictions(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Get patch-wise predictions for visualization.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            patch_logits: (B, num_patches, 1)
            num_patches_h: Vertical patch count
            num_patches_w: Horizontal patch count
        """
        # Extract and encode patches
        patches, num_patches_h, num_patches_w = self.patch_extractor(x)
        patch_features = self.patch_encoder(patches)
        patch_features = self.pos_encoding(patch_features, num_patches_h, num_patches_w)
        encoded_features = self.transformer(patch_features)
        
        # Get patch logits
        patch_logits = self.patch_classifier(encoded_features)
        
        return patch_logits, num_patches_h, num_patches_w


# Factory function
def create_patch_transformer(
    image_size: Tuple[int, int] = (1400, 2800),
    patch_size: int = 100,
    model_size: str = 'base',
    **kwargs
) -> PatchTransformerClassifier:
    """
    Factory function for creating patch transformers.
    
    Args:
        image_size: (H, W) tuple
        patch_size: Patch size
        model_size: 'tiny', 'small', 'base', 'large'
        **kwargs: Override default parameters
        
    Returns:
        PatchTransformerClassifier instance
    """
    configs = {
        'tiny': {
            'cnn_backbone': 'resnet18',
            'feature_dim': 256,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.1
        },
        'small': {
            'cnn_backbone': 'resnet34',
            'feature_dim': 384,
            'num_heads': 6,
            'num_layers': 4,
            'dropout': 0.1
        },
        'base': {
            'cnn_backbone': 'resnet18',
            'feature_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1
        },
        'large': {
            'cnn_backbone': 'resnet34',
            'feature_dim': 768,
            'num_heads': 12,
            'num_layers': 8,
            'dropout': 0.1
        }
    }
    
    config = configs.get(model_size, configs['base'])
    
    # Filter out parameters not accepted by PatchTransformerClassifier
    # (pretrained is handled internally by PatchCNNEncoder)
    valid_params = {'cnn_backbone', 'feature_dim', 'num_heads', 'num_layers', 'dropout', 'aggregation'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    config.update(filtered_kwargs)
    
    return PatchTransformerClassifier(
        image_size=image_size,
        patch_size=patch_size,
        **config
    )


# Testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING PATCH TRANSFORMER CLASSIFIER")
    print("="*70)
    
    # Test with panoramic image size
    model = create_patch_transformer(
        image_size=(1400, 2800),
        patch_size=100,
        model_size='base',
        aggregation='max'
    )
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 1400, 2800)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        patch_preds, nh, nw = model.get_patch_predictions(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Patch predictions: {patch_preds.shape} ({nh}x{nw} grid)")
    
    # Test GPU if available
    if torch.cuda.is_available():
        print("\nTesting on GPU...")
        model = model.cuda()
        dummy_input_gpu = torch.randn(1, 3, 1400, 2800).cuda()
        
        with torch.no_grad():
            output_gpu = model(dummy_input_gpu)
        
        print(f"GPU output shape: {output_gpu.shape}")
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
