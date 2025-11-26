"""
Compare model complexity and GPU usage
"""
import torch
import torch.nn as nn
from models.patch_transformer_localization import PatchTransformerWithLocalization

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
).cuda()

print("\n" + "="*80)
print("MODEL ANALYSIS")
print("="*80)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Baseline had ~11M parameters
baseline_params = 11_000_000
print(f"\nBaseline model: ~{baseline_params:,} parameters")
print(f"This model: {total_params:,} parameters")
print(f"Ratio: {total_params / baseline_params:.2f}x")

# Test forward pass and measure memory
print("\n" + "="*80)
print("MEMORY & COMPUTE TEST")
print("="*80)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Dummy batch
batch_size = 4
dummy_input = torch.randn(batch_size, 3, 1400, 2800).cuda()

print(f"\nInput shape: {dummy_input.shape}")
print(f"Input size: {dummy_input.numel() * 4 / 1024**2:.2f} MB")

# Forward pass
print("\nRunning forward pass...")
with torch.no_grad():
    output = model(dummy_input)

mem_allocated = torch.cuda.memory_allocated() / 1024**2
mem_reserved = torch.cuda.memory_reserved() / 1024**2
peak_mem = torch.cuda.max_memory_allocated() / 1024**2

print(f"\nMemory allocated: {mem_allocated:.2f} MB")
print(f"Memory reserved: {mem_reserved:.2f} MB")
print(f"Peak memory: {peak_mem:.2f} MB")

# Check output shapes
print("\n" + "="*80)
print("OUTPUT ANALYSIS")
print("="*80)
print(f"\nGlobal logits: {output['global_logits'].shape}")
print(f"Patch logits: {output['patch_logits'].shape}")
print(f"Patch probs: {output['patch_probs'].shape}")
print(f"Number of patches: {output['num_patches_h']} x {output['num_patches_w']} = {output['num_patches_h'] * output['num_patches_w']}")

# Estimate FLOPS
num_patches = output['num_patches_h'] * output['num_patches_w']
print("\n" + "="*80)
print("COMPUTE ESTIMATION")
print("="*80)
print(f"\nNumber of patches per image: {num_patches}")
print(f"Patch size: 100x100 = 10,000 pixels")
print(f"Per batch ({batch_size} images): {num_patches * batch_size} patches")

# ResNet18 per patch
resnet_flops_per_patch = 1.8e9  # ~1.8 GFLOPS for 224x224, roughly similar for 100x100
total_resnet_flops = resnet_flops_per_patch * num_patches * batch_size / 1e9
print(f"\nResNet18 encoding: ~{total_resnet_flops:.2f} GFLOPS")

# Transformer
transformer_flops = (6 * num_patches * 512 * 512 * batch_size) / 1e9  # Rough estimate
print(f"Transformer: ~{transformer_flops:.2f} GFLOPS")

total_flops = total_resnet_flops + transformer_flops
print(f"Total (estimated): ~{total_flops:.2f} GFLOPS")

# Compare with baseline
baseline_flops = 5  # Baseline EfficientNet-B0 is ~5 GFLOPS for 512x512
print(f"\nBaseline (EfficientNet-B0): ~{baseline_flops:.2f} GFLOPS")
print(f"This model: ~{total_flops:.2f} GFLOPS")
print(f"Ratio: {total_flops / baseline_flops:.2f}x")

print("\n" + "="*80)
print("POSSIBLE REASONS FOR LOW GPU USAGE:")
print("="*80)
print("\n1. DATA LOADING BOTTLENECK:")
print("   - num_workers=0 (single threaded)")
print("   - Large images (1400x2800) take time to load")
print("   - Solution: Increase num_workers (but Windows multiprocessing issues)")

print("\n2. BATCH SIZE TOO SMALL:")
print("   - Current: 4 images/batch")
print("   - GPU underutilized with small batches")
print("   - Solution: Increase batch_size to 8-16")

print("\n3. CPU PREPROCESSING:")
print("   - CLAHE, resizing, augmentation on CPU")
print("   - GPU waits for data")
print("   - Solution: GPU preprocessing or larger batches")

print("\n4. MODEL EFFICIENCY:")
print("   - ResNet18 is efficient (not compute-heavy)")
print("   - Transformer with 392 patches is moderate")
print("   - Not a problem, just efficient architecture")

print("\n5. AMP (Mixed Precision):")
print("   - FP16 uses less memory → lower GPU utilization")
print("   - Faster compute → GPU waits for data more")
print("   - This is actually GOOD (efficient training)")
