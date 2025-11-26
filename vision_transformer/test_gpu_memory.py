"""
GPU Memory Test - Check optimal batch size

Tests different batch sizes to find the optimal configuration
for training without OOM errors.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import create_patch_transformer
from training import get_loss_function
from torch.cuda.amp import autocast, GradScaler


def test_batch_size(batch_size, image_size=(1400, 2800)):
    """Test if batch size fits in GPU memory."""
    print(f"\nTesting batch size {batch_size}...")
    
    try:
        torch.cuda.empty_cache()
        
        # Create model
        model = create_patch_transformer(
            image_size=image_size,
            patch_size=100,
            model_size='base'
        ).cuda()
        
        # Create optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = get_loss_function('combined')
        scaler = GradScaler()
        
        # Create dummy batch
        images = torch.randn(batch_size, 3, *image_size).cuda()
        labels = torch.randint(0, 2, (batch_size,)).float().cuda()
        
        # Forward pass with AMP
        model.train()
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Memory stats
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        mem_free = (torch.cuda.get_device_properties(0).total_memory / 1024**2) - mem_reserved
        
        print(f"  ✅ Success!")
        print(f"  Allocated: {mem_allocated:.0f} MB")
        print(f"  Reserved: {mem_reserved:.0f} MB")
        print(f"  Free: {mem_free:.0f} MB")
        
        # Clean up
        del model, optimizer, images, labels, outputs, loss
        torch.cuda.empty_cache()
        
        return True, mem_allocated, mem_reserved
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ❌ OOM Error")
            torch.cuda.empty_cache()
            return False, 0, 0
        else:
            raise


def main():
    """Test different batch sizes."""
    print("="*70)
    print("GPU MEMORY TEST - FINDING OPTIMAL BATCH SIZE")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\nGPU: {gpu_name}")
    print(f"Total Memory: {gpu_memory:.2f} GB")
    print(f"Image size: 1400×2800")
    print(f"Model: PatchTransformer Base (30.2M params)")
    
    # Test batch sizes
    batch_sizes = [1, 2, 4, 6, 8]
    results = []
    
    for bs in batch_sizes:
        success, mem_alloc, mem_res = test_batch_size(bs)
        results.append((bs, success, mem_alloc, mem_res))
        
        if not success:
            print(f"\n⚠️  Batch size {bs} causes OOM, stopping tests...")
            break
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r[1]]
    
    if successful:
        max_bs = successful[-1][0]
        max_mem = successful[-1][2]
        
        print(f"\n✅ Maximum batch size: {max_bs}")
        print(f"   Memory usage: {max_mem:.0f} MB")
        
        # Recommendations
        recommended_bs = max(1, max_bs // 2)  # Use half of max for safety
        
        print(f"\n💡 Recommended batch size: {recommended_bs}")
        print(f"   Reason: Leaves headroom for gradient accumulation and safety")
        
        print("\n📊 All successful batch sizes:")
        print(f"{'Batch Size':<12} {'Memory (MB)':<15} {'Memory per sample (MB)':<20}")
        print("-" * 70)
        for bs, success, mem_alloc, mem_res in successful:
            mem_per_sample = mem_alloc / bs
            print(f"{bs:<12} {mem_alloc:<15.0f} {mem_per_sample:<20.0f}")
    else:
        print("\n❌ Even batch size 1 causes OOM!")
        print("   Consider:")
        print("   - Using 'tiny' or 'small' model size")
        print("   - Reducing image size")
        print("   - Using gradient checkpointing")
    
    print("="*70)


if __name__ == "__main__":
    main()
