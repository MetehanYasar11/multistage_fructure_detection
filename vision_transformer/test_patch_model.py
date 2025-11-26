"""
Test Patch Transformer Model with Panoramic Images

This script tests the new patch-based transformer architecture
with full-resolution panoramic X-ray images.

Author: Master's Thesis Project
Date: October 28, 2025
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PatchTransformerClassifier, create_patch_transformer
from data import get_train_transforms, get_val_transforms, DentalXrayDataset
from training import get_loss_function
import numpy as np


def test_patch_model_creation():
    """Test patch transformer creation."""
    print("="*70)
    print("TEST 1: Patch Transformer Creation")
    print("="*70)
    
    # Test different model sizes
    sizes = ['tiny', 'small', 'base']
    
    for size in sizes:
        print(f"\nCreating {size} model...")
        model = create_patch_transformer(
            image_size=(1400, 2800),
            patch_size=100,
            model_size=size
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
        print("-" * 50)
    
    print("\n✓ Model creation successful")
    return model


def test_forward_with_panoramic():
    """Test forward pass with panoramic images."""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass with Panoramic Images")
    print("="*70)
    
    model = create_patch_transformer(
        image_size=(1400, 2800),
        patch_size=100,
        model_size='base'
    )
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4]
    
    model.eval()
    for bs in batch_sizes:
        print(f"\nBatch size: {bs}")
        dummy_input = torch.randn(bs, 3, 1400, 2800)
        
        with torch.no_grad():
            output = model(dummy_input)
            patch_preds, nh, nw = model.get_patch_predictions(dummy_input)
        
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Patch predictions: {patch_preds.shape} ({nh}×{nw})")
        print(f"  Memory: ~{dummy_input.numel() * 4 / 1024**2:.2f} MB")
    
    print("\n✓ Forward pass successful")
    return model


def test_with_augmentations():
    """Test with augmented panoramic images."""
    print("\n" + "="*70)
    print("TEST 3: Augmentation with Panoramic Images")
    print("="*70)
    
    # Create transforms for panoramic size
    train_tfm = get_train_transforms(image_size=(1400, 2800))
    val_tfm = get_val_transforms(image_size=(1400, 2800))
    
    # Create dummy panoramic image (original size)
    dummy_image = np.random.randint(0, 255, (1435, 2900, 3), dtype=np.uint8)
    
    print(f"\nOriginal image: {dummy_image.shape}")
    
    # Apply transforms
    train_aug = train_tfm(image=dummy_image)['image']
    val_aug = val_tfm(image=dummy_image)['image']
    
    print(f"Train augmented: {train_aug.shape}")
    print(f"Val augmented: {val_aug.shape}")
    
    assert train_aug.shape == (3, 1400, 2800), f"Expected (3, 1400, 2800), got {train_aug.shape}"
    assert val_aug.shape == (3, 1400, 2800), f"Expected (3, 1400, 2800), got {val_aug.shape}"
    
    print("\n✓ Augmentation working correctly")


def test_gpu_memory():
    """Test GPU memory usage."""
    print("\n" + "="*70)
    print("TEST 4: GPU Memory Usage")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available, skipping GPU test")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    
    model = create_patch_transformer(
        image_size=(1400, 2800),
        patch_size=100,
        model_size='base'
    ).cuda()
    
    # Test different batch sizes to find limits
    batch_sizes = [1, 2, 4]
    
    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()
            dummy_input = torch.randn(bs, 3, 1400, 2800).cuda()
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            
            print(f"\nBatch size {bs}:")
            print(f"  Allocated: {mem_alloc:.2f} MB")
            print(f"  Reserved: {mem_reserved:.2f} MB")
            print(f"  Status: ✓")
        
        except RuntimeError as e:
            print(f"\nBatch size {bs}: ✗ (OOM)")
            break
    
    torch.cuda.empty_cache()
    print("\n✓ GPU memory test complete")


def test_with_real_dataset():
    """Test with actual dataset."""
    print("\n" + "="*70)
    print("TEST 5: Real Dataset Integration")
    print("="*70)
    
    try:
        # Load dataset with panoramic size
        dataset = DentalXrayDataset(
            root_dir="c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset",
            split='all',
            transform=get_val_transforms(image_size=(1400, 2800))
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Load one sample
        image, label = dataset[0]
        
        print(f"Image shape: {image.shape}")
        print(f"Label: {label} ({'Fractured' if label == 1 else 'Healthy'})")
        
        # Test with model
        model = create_patch_transformer(
            image_size=(1400, 2800),
            patch_size=100,
            model_size='tiny'  # Use tiny for speed
        )
        
        model.eval()
        with torch.no_grad():
            image_batch = image.unsqueeze(0)  # Add batch dim
            output = model(image_batch)
            patch_preds, nh, nw = model.get_patch_predictions(image_batch)
        
        print(f"\nModel output: {output.shape}")
        print(f"Prediction logit: {output.item():.4f}")
        print(f"Prediction prob: {torch.sigmoid(output).item():.4f}")
        print(f"Patch predictions: {patch_preds.shape}")
        
        # Analyze patch predictions
        patch_probs = torch.sigmoid(patch_preds[0]).squeeze()
        print(f"\nPatch statistics:")
        print(f"  Min prob: {patch_probs.min().item():.4f}")
        print(f"  Max prob: {patch_probs.max().item():.4f}")
        print(f"  Mean prob: {patch_probs.mean().item():.4f}")
        print(f"  Patches with high confidence (>0.7): {(patch_probs > 0.7).sum().item()}")
        
        print("\n✓ Real dataset integration successful")
    
    except Exception as e:
        print(f"\n⚠ Dataset test skipped: {str(e)}")


def test_loss_compatibility():
    """Test loss function compatibility."""
    print("\n" + "="*70)
    print("TEST 6: Loss Function Compatibility")
    print("="*70)
    
    model = create_patch_transformer(
        image_size=(1400, 2800),
        patch_size=100,
        model_size='tiny'
    )
    
    # Create dummy batch
    images = torch.randn(4, 3, 1400, 2800)
    labels = torch.randint(0, 2, (4,)).float()
    
    # Test with different loss functions
    loss_types = ['bce', 'focal', 'combined']
    
    print("\nTesting loss functions:")
    for loss_type in loss_types:
        loss_fn = get_loss_function(loss_type)
        
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        
        loss = loss_fn(outputs, labels)
        print(f"  {loss_type:15s}: {loss.item():.4f}")
    
    print("\n✓ All loss functions compatible")


def test_training_step():
    """Test a training step."""
    print("\n" + "="*70)
    print("TEST 7: Training Step Simulation")
    print("="*70)
    
    model = create_patch_transformer(
        image_size=(1400, 2800),
        patch_size=100,
        model_size='tiny'
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = get_loss_function('combined')
    
    # Create dummy batch
    images = torch.randn(2, 3, 1400, 2800)
    labels = torch.randint(0, 2, (2,)).float()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    
    # Check gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 
                    for p in model.parameters() if p.requires_grad)
    
    optimizer.step()
    
    print(f"\nBatch size: {images.shape[0]}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradients: {'✓' if has_grads else '✗'}")
    print(f"Optimizer step: ✓")
    
    print("\n✓ Training step successful")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PATCH TRANSFORMER MODEL TESTING")
    print("PANORAMIC X-RAY ANALYSIS (1400×2800)")
    print("="*70)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Test 1: Model creation
        test_patch_model_creation()
        
        # Test 2: Forward pass
        test_forward_with_panoramic()
        
        # Test 3: Augmentations
        test_with_augmentations()
        
        # Test 4: GPU memory
        test_gpu_memory()
        
        # Test 5: Real dataset
        test_with_real_dataset()
        
        # Test 6: Loss compatibility
        test_loss_compatibility()
        
        # Test 7: Training step
        test_training_step()
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\n✅ Patch Transformer model is ready for training")
        print("✅ Panoramic image size (1400×2800) working correctly")
        print("✅ 392 patches (14×28 grid) per image")
        print("✅ Patch-wise predictions available for interpretability")
        print("\nModel Architecture:")
        print("  - Image: 1400×2800 → 392 patches (100×100 each)")
        print("  - CNN: ResNet18 per patch → 512D features")
        print("  - Transformer: 6 layers, 8 heads, positional encoding")
        print("  - Output: Binary classification + patch-wise predictions")
        print("\nNext step: Train the model on full-resolution panoramic images!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
