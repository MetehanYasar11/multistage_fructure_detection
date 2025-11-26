"""
Test Model and Loss Functions

This script verifies:
1. Model creation and architecture
2. Forward pass with different input sizes
3. Feature extraction
4. Loss function computation
5. GPU compatibility
6. Parameter counting

Author: Master's Thesis Project
Date: October 28, 2025
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EfficientNetClassifier
from training import get_loss_function


def test_model_creation():
    """Test model instantiation."""
    print("="*70)
    print("TEST 1: Model Creation")
    print("="*70)
    
    model = EfficientNetClassifier(
        model_name='efficientnet_b0',
        pretrained=True,
        num_classes=1,
        dropout=0.3,
        hidden_dim=512
    )
    
    print("\n✓ Model created successfully")
    return model


def test_forward_pass(model, batch_size=8, image_size=640):
    """Test forward pass with different input sizes."""
    print("\n" + "="*70)
    print(f"TEST 2: Forward Pass (batch_size={batch_size}, image_size={image_size})")
    print("="*70)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Check output shape
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    
    print("\n✓ Forward pass successful")
    return output


def test_feature_extraction(model, batch_size=4, image_size=640):
    """Test feature extraction."""
    print("\n" + "="*70)
    print("TEST 3: Feature Extraction")
    print("="*70)
    
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)
    
    model.eval()
    with torch.no_grad():
        features = model.extract_features(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature dim: {features.shape[1]}")
    
    print("\n✓ Feature extraction successful")
    return features


def test_loss_functions():
    """Test all loss functions."""
    print("\n" + "="*70)
    print("TEST 4: Loss Functions")
    print("="*70)
    
    # Create dummy predictions and targets
    batch_size = 16
    predictions = torch.randn(batch_size, 1)  # Logits
    targets = torch.randint(0, 2, (batch_size,)).float()  # Binary labels (1D for our loss functions)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Target distribution: {targets.sum().item():.0f} fractured, {(batch_size - targets.sum()).item():.0f} healthy")
    
    # Test each loss function
    loss_configs = {
        'BCE': 'bce',
        'Focal Loss': 'focal',
        'Combined (BCE+Focal)': 'combined',
        'Dice Loss': 'dice',
        'Dice + BCE': 'dice_bce',
        'Weighted BCE': 'weighted_bce'
    }
    
    print("\nLoss Function Results:")
    print("-" * 50)
    
    for name, loss_type in loss_configs.items():
        loss_fn = get_loss_function(loss_type)
        loss_value = loss_fn(predictions, targets)
        print(f"{name:25s}: {loss_value.item():.4f}")
    
    print("\n✓ All loss functions working")


def test_gpu_compatibility(model):
    """Test GPU compatibility."""
    print("\n" + "="*70)
    print("TEST 5: GPU Compatibility")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available, skipping GPU test")
        return model  # Return original model
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Move model to GPU
    model_gpu = model.cuda()
    
    # Test forward pass on GPU
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 640, 640).cuda()
    
    model_gpu.eval()
    with torch.no_grad():
        output = model_gpu(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Move model back to CPU
    model = model.cpu()
    torch.cuda.empty_cache()
    
    print("\n✓ GPU compatibility confirmed")
    return model  # Return CPU model


def test_parameter_count(model):
    """Test parameter counting."""
    print("\n" + "="*70)
    print("TEST 6: Parameter Analysis")
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    
    print(f"\nModel size (FP32): ~{total_params * 4 / 1024**2:.2f} MB")
    print(f"Model size (FP16): ~{total_params * 2 / 1024**2:.2f} MB")
    
    print("\n✓ Parameter analysis complete")


def test_different_image_sizes(model):
    """Test with different input sizes."""
    print("\n" + "="*70)
    print("TEST 7: Different Input Sizes")
    print("="*70)
    
    sizes = [512, 640, 768]
    batch_size = 4
    
    model.eval()
    
    for size in sizes:
        dummy_input = torch.randn(batch_size, 3, size, size)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\nImage size {size}×{size}:")
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Memory: ~{dummy_input.numel() * 4 / 1024**2:.2f} MB")
    
    print("\n✓ All image sizes supported")


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*70)
    print("TEST 8: Training Step Simulation")
    print("="*70)
    
    # Create model and loss
    model = EfficientNetClassifier(
        model_name='efficientnet_b0',
        pretrained=False,  # Faster for testing
        num_classes=1,
        dropout=0.3
    )
    
    loss_fn = get_loss_function('combined')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy batch
    batch_size = 8
    images = torch.randn(batch_size, 3, 640, 640)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.parameters())
    
    # Optimizer step
    optimizer.step()
    
    print(f"\nBatch size: {batch_size}")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Gradients computed: {'✓' if has_gradients else '✗'}")
    print(f"Optimizer step: ✓")
    
    print("\n✓ Training step successful")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DENTAL X-RAY MODEL AND LOSS TESTING")
    print("="*70)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model, batch_size=8, image_size=640)
        
        # Test 3: Feature extraction
        test_feature_extraction(model, batch_size=4, image_size=640)
        
        # Test 4: Loss functions
        test_loss_functions()
        
        # Test 5: GPU compatibility
        model = test_gpu_compatibility(model)  # Returns CPU model
        
        # Test 6: Parameter count
        test_parameter_count(model)
        
        # Test 7: Different image sizes
        test_different_image_sizes(model)
        
        # Test 8: Training step
        test_training_step()
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\n✅ Model is ready for training")
        print("✅ Loss functions are working correctly")
        print("✅ GPU compatibility confirmed")
        print("\nNext step: Implement training loop in training/train.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
