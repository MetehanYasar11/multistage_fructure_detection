"""
Start Full Training - PatchTransformer Base Model

This script starts full training of the PatchTransformer model
on the complete dental X-ray dataset.

Configuration:
- Model: PatchTransformer Base (30.2M params)
- Image size: 1400×2800 (panoramic)
- Batch size: 4
- Epochs: 50
- Loss: Combined (BCE + Focal)
- Optimizer: AdamW with Cosine LR
- Mixed Precision: Enabled

Expected training time: ~2-3 hours on RTX 5070 Ti
"""

import sys
import os

# Ensure we're using the conda environment
if 'CONDA_DEFAULT_ENV' not in os.environ or os.environ.get('CONDA_DEFAULT_ENV') != 'dental-ai':
    print("⚠️  Warning: conda environment 'dental-ai' is not active!")
    print("Please run: conda activate dental-ai")
    sys.exit(1)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train import main

if __name__ == "__main__":
    print("="*70)
    print("STARTING FULL TRAINING - PATCH TRANSFORMER")
    print("="*70)
    print("\n📊 Training Configuration:")
    print("  - Model: PatchTransformer Base")
    print("  - Parameters: 30.2M")
    print("  - Image size: 1400×2800")
    print("  - Patch size: 100×100 (14×28 grid = 392 patches)")
    print("  - Batch size: 4")
    print("  - Epochs: 50")
    print("  - Dataset: 340 train, 73 val, 74 test")
    print("  - Loss: Combined (BCE + Focal)")
    print("  - Optimizer: AdamW (lr=1e-4, wd=0.01)")
    print("  - Scheduler: Cosine Annealing")
    print("  - Mixed Precision: Enabled")
    print("  - Early Stopping: Patience=15")
    print("\n⏱️  Expected time: ~2-3 hours")
    print("💾 Checkpoints will be saved to: checkpoints/patch_transformer_*/")
    print("\n" + "="*70)
    
    input("\n🚀 Press Enter to start training (or Ctrl+C to cancel)...")
    print("\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
