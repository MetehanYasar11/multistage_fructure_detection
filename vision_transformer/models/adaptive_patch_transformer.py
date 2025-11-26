"""
Adaptive Patch Transformer - Dynamic image sizes

Automatically adjusts patch grid based on input image size.
Perfect for RCT crops with varying dimensions.

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

from models.patch_transformer import PatchTransformerClassifier


class AdaptivePatchTransformer(nn.Module):
    """
    Patch Transformer with adaptive patch grid.
    
    Automatically computes optimal patch grid based on input size.
    Suitable for variable-sized RCT tooth crops.
    """
    
    def __init__(
        self,
        target_patch_size: int = 112,
        min_patches: int = 2,
        max_patches: int = 8,
        backbone: str = "resnet18",
        pretrained: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        aggregation: str = "max"
    ):
        """
        Initialize Adaptive Patch Transformer.
        
        Args:
            target_patch_size: Target patch size (will adapt based on input)
            min_patches: Minimum patches per dimension
            max_patches: Maximum patches per dimension
            backbone: CNN backbone for patch encoding
            pretrained: Use pretrained backbone
            d_model: Transformer feature dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            aggregation: Feature aggregation method
        """
        super().__init__()
        
        self.target_patch_size = target_patch_size
        self.min_patches = min_patches
        self.max_patches = max_patches
        
        # Create base model with default patch config
        # Will be dynamically adjusted during forward pass
        self.base_model = PatchTransformerClassifier(
            image_height=224,  # Dummy values
            image_width=224,
            patch_size=target_patch_size,
            num_patches_h=2,
            num_patches_w=2,
            backbone=backbone,
            pretrained=pretrained,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            aggregation=aggregation
        )
        
        # Cache for different input sizes
        self.size_cache = {}
    
    def _compute_patch_grid(
        self,
        height: int,
        width: int
    ) -> Tuple[int, int, int]:
        """
        Compute optimal patch grid for given image size.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            (num_patches_h, num_patches_w, patch_size)
        """
        # Try to use target patch size
        num_patches_h = height // self.target_patch_size
        num_patches_w = width // self.target_patch_size
        
        # Clamp to min/max
        num_patches_h = max(self.min_patches, min(self.max_patches, num_patches_h))
        num_patches_w = max(self.min_patches, min(self.max_patches, num_patches_w))
        
        # Adjust patch size to fit exactly
        patch_size_h = height // num_patches_h
        patch_size_w = width // num_patches_w
        
        # Use smaller patch size for consistency
        patch_size = min(patch_size_h, patch_size_w)
        
        # Recompute grid with consistent patch size
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        
        return num_patches_h, num_patches_w, patch_size
    
    def _get_model_for_size(
        self,
        height: int,
        width: int
    ) -> PatchTransformerClassifier:
        """
        Get or create model for specific image size.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Model configured for this size
        """
        size_key = (height, width)
        
        if size_key in self.size_cache:
            return self.size_cache[size_key]
        
        # Compute patch grid
        num_patches_h, num_patches_w, patch_size = self._compute_patch_grid(height, width)
        
        # Create new model with this configuration
        model = PatchTransformerClassifier(
            image_height=height,
            image_width=width,
            patch_size=patch_size,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
            backbone=self.base_model.patch_encoder.backbone_name,
            pretrained=False,  # Copy weights from base
            d_model=self.base_model.transformer.layers[0].self_attn.embed_dim,
            nhead=self.base_model.transformer.layers[0].self_attn.num_heads,
            num_layers=len(self.base_model.transformer.layers),
            dim_feedforward=self.base_model.transformer.layers[0].linear1.out_features,
            dropout=self.base_model.transformer.layers[0].dropout.p,
            aggregation=self.base_model.aggregation
        )
        
        # Copy weights from base model (shared components)
        model.patch_encoder.load_state_dict(self.base_model.patch_encoder.state_dict())
        model.transformer.load_state_dict(self.base_model.transformer.state_dict())
        model.classifier.load_state_dict(self.base_model.classifier.state_dict())
        
        # Move to same device as base model
        model = model.to(next(self.base_model.parameters()).device)
        
        # Cache for future use
        self.size_cache[size_key] = model
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive patch grid.
        
        Args:
            x: Input images (B, C, H, W) - can have varying H, W in batch
            
        Returns:
            Logits (B,) or (B, 1)
        """
        batch_size, channels, height, width = x.shape
        
        # If all images in batch have same size, use optimized path
        if batch_size > 1:
            # For batched training, require uniform size
            # (Can be relaxed for inference)
            model = self._get_model_for_size(height, width)
            return model(x)
        else:
            # Single image inference
            model = self._get_model_for_size(height, width)
            return model(x)
    
    def train(self, mode: bool = True):
        """Set training mode for all cached models."""
        super().train(mode)
        self.base_model.train(mode)
        for model in self.size_cache.values():
            model.train(mode)
        return self
    
    def eval(self):
        """Set eval mode for all cached models."""
        super().eval()
        self.base_model.eval()
        for model in self.size_cache.values():
            model.eval()
        return self


def create_adaptive_patch_transformer(
    target_patch_size: int = 112,
    min_patches: int = 2,
    max_patches: int = 8,
    **kwargs
) -> AdaptivePatchTransformer:
    """
    Factory function for Adaptive Patch Transformer.
    
    Args:
        target_patch_size: Target patch size
        min_patches: Minimum patches per dimension
        max_patches: Maximum patches per dimension
        **kwargs: Additional arguments for PatchTransformerClassifier
        
    Returns:
        AdaptivePatchTransformer model
    """
    return AdaptivePatchTransformer(
        target_patch_size=target_patch_size,
        min_patches=min_patches,
        max_patches=max_patches,
        **kwargs
    )


# Test
if __name__ == "__main__":
    print("Testing Adaptive Patch Transformer...")
    
    model = create_adaptive_patch_transformer(
        target_patch_size=112,
        min_patches=2,
        max_patches=8,
        backbone="resnet18",
        d_model=256,
        nhead=8,
        num_layers=4
    )
    
    # Test different sizes
    test_sizes = [
        (224, 224),
        (300, 200),
        (180, 240),
        (256, 256)
    ]
    
    model.eval()
    with torch.no_grad():
        for h, w in test_sizes:
            x = torch.randn(1, 3, h, w)
            output = model(x)
            print(f"Input: {h}x{w} → Output: {output.shape}")
    
    print("\n✅ Adaptive Patch Transformer test passed!")
