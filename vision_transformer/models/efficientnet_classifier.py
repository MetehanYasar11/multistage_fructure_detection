"""
EfficientNet Baseline Classifier for Dental X-Ray Fracture Detection

This module implements a baseline classification model using EfficientNet
backbones from the timm library, with a custom classification head optimized
for binary classification of dental X-rays.

Architecture:
    - Backbone: EfficientNet-B0 (pretrained on ImageNet)
    - Global Average Pooling
    - Custom MLP head with dropout regularization
    - Binary classification output (sigmoid activation)

Performance Target (Baseline):
    - Accuracy: >70%
    - Precision: >75%
    - Recall: >70%
    - F1 Score: >70%

Author: Master's Thesis Project
Date: October 28, 2025
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based binary classifier for dental X-ray fracture detection.
    
    This model uses a pretrained EfficientNet backbone from timm library
    with a custom classification head for binary classification.
    
    Attributes:
        backbone: EfficientNet backbone (from timm)
        classifier: Custom MLP classification head
        num_classes: Number of output classes (1 for binary classification)
        model_name: Name of the EfficientNet variant
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        hidden_dim: int = 512,
        freeze_backbone: bool = False
    ):
        """
        Initialize EfficientNet classifier.
        
        Args:
            model_name: EfficientNet variant name (efficientnet_b0, efficientnet_b1, etc.)
            pretrained: Load ImageNet pretrained weights
            num_classes: Number of output classes (1 for binary classification with BCE)
            dropout: Dropout rate for regularization
            hidden_dim: Hidden layer dimension in classification head
            freeze_backbone: Freeze backbone weights (only train classifier head)
        """
        super(EfficientNetClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Load pretrained EfficientNet backbone (without classifier)
        print(f"Loading {model_name} backbone (pretrained={pretrained})...")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            global_pool='avg'  # Global average pooling
        )
        
        # Get number of features from backbone
        in_features = self.backbone.num_features
        print(f"Backbone output features: {in_features}")
        
        # Freeze backbone if requested (for transfer learning)
        if freeze_backbone:
            print("Freezing backbone weights...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            # First dropout
            nn.Dropout(p=dropout),
            
            # Hidden layer
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Second dropout (reduced rate)
            nn.Dropout(p=dropout / 2),
            
            # Output layer (no activation - BCEWithLogitsLoss handles sigmoid)
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _initialize_weights(self):
        """Initialize classifier head weights using He initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output logits (B, num_classes) - use sigmoid for probabilities
        """
        # Extract features from backbone
        features = self.backbone(x)  # (B, in_features)
        
        # Classification head
        logits = self.classifier(features)  # (B, num_classes)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone (for visualization, embeddings, etc.).
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, in_features)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self, layers_to_unfreeze: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            layers_to_unfreeze: Number of last layers to unfreeze (None = unfreeze all)
        """
        if layers_to_unfreeze is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Unfroze all backbone layers")
        else:
            # Unfreeze last N layers (not implemented for all architectures)
            print(f"Unfreezing last {layers_to_unfreeze} layers not fully implemented")
            # Simple implementation: unfreeze all for now
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def get_classifier_params(self):
        """Get parameters of classifier head only (for different learning rates)."""
        return self.classifier.parameters()
    
    def get_backbone_params(self):
        """Get parameters of backbone only (for different learning rates)."""
        return self.backbone.parameters()


class EfficientNetClassifierMultiHead(EfficientNetClassifier):
    """
    EfficientNet classifier with multiple heads for multi-task learning.
    
    This can be used for:
    - Binary classification (fractured vs healthy)
    - Auxiliary tasks (location prediction, severity estimation, etc.)
    
    Currently implements basic structure for future extensions.
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        hidden_dim: int = 512,
        auxiliary_tasks: Optional[dict] = None
    ):
        """
        Initialize multi-head classifier.
        
        Args:
            auxiliary_tasks: Dictionary of auxiliary task configurations
                Example: {'severity': 3, 'location': 4}
        """
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=hidden_dim
        )
        
        # Add auxiliary task heads if specified
        self.auxiliary_tasks = auxiliary_tasks or {}
        self.auxiliary_heads = nn.ModuleDict()
        
        in_features = self.backbone.num_features
        
        for task_name, task_classes in self.auxiliary_tasks.items():
            self.auxiliary_heads[task_name] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2),
                nn.Linear(hidden_dim, task_classes)
            )
            print(f"Added auxiliary head: {task_name} ({task_classes} classes)")
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass with multiple outputs.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary with 'main' output and auxiliary task outputs
        """
        # Extract features
        features = self.backbone(x)
        
        # Main classification
        main_output = self.classifier(features)
        
        # Auxiliary outputs
        outputs = {'main': main_output}
        
        for task_name, head in self.auxiliary_heads.items():
            outputs[task_name] = head(features)
        
        return outputs


# Model factory function
def create_efficientnet_classifier(
    variant: str = 'b0',
    pretrained: bool = True,
    num_classes: int = 1,
    dropout: float = 0.3,
    **kwargs
) -> EfficientNetClassifier:
    """
    Factory function to create EfficientNet classifiers.
    
    Args:
        variant: EfficientNet variant ('b0', 'b1', 'b2', ..., 'b7')
        pretrained: Load ImageNet pretrained weights
        num_classes: Number of output classes
        dropout: Dropout rate
        **kwargs: Additional arguments for EfficientNetClassifier
        
    Returns:
        EfficientNetClassifier instance
    """
    model_name = f'efficientnet_{variant}'
    
    return EfficientNetClassifier(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("TESTING EFFICIENTNET CLASSIFIER")
    print("="*70)
    
    # Test model creation
    print("\n1. Creating EfficientNet-B0 classifier...")
    model = EfficientNetClassifier(
        model_name='efficientnet_b0',
        pretrained=True,
        num_classes=1,
        dropout=0.3
    )
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 640, 640)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test feature extraction
    print("\n3. Testing feature extraction...")
    with torch.no_grad():
        features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n4. Testing on GPU...")
        model = model.cuda()
        dummy_input_gpu = dummy_input.cuda()
        
        with torch.no_grad():
            output_gpu = model(dummy_input_gpu)
        
        print(f"GPU output shape: {output_gpu.shape}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Test different variants
    print("\n5. Testing different EfficientNet variants...")
    variants = ['b0', 'b1', 'b2']
    
    for variant in variants:
        model_variant = create_efficientnet_classifier(
            variant=variant,
            pretrained=False,  # Faster for testing
            num_classes=1
        )
        total_params = sum(p.numel() for p in model_variant.parameters())
        print(f"  EfficientNet-{variant.upper()}: {total_params:,} parameters")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
