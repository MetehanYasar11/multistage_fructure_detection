"""
Quick training test - 3 epochs with Tiny model

Tests the training pipeline without WandB.
"""

import torch
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import main as train_main


if __name__ == "__main__":
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings for quick test
    config['training']['epochs'] = 3
    config['training']['batch_size'] = 2  # Small batch for speed
    config['training']['use_wandb'] = False
    config['training']['early_stopping_patience'] = 10
    config['model']['name'] = 'patch_transformer'
    config['model']['size'] = 'tiny'  # Fastest model
    config['image']['default_size'] = [1400, 2800]
    
    # Save temporary config
    with open('config_test.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("=" * 70)
    print("QUICK TRAINING TEST")
    print("=" * 70)
    print("Model: PatchTransformer (Tiny)")
    print("Image size: 1400×2800")
    print("Epochs: 3")
    print("Batch size: 2")
    print("=" * 70)
    print()
    
    # Temporarily replace config path
    original_argv = sys.argv.copy()
    
    try:
        # Run training
        train_main()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING TEST SUCCESSFUL!")
        print("=" * 70)
        print("\nTraining pipeline is working correctly.")
        print("Ready for full training runs!")
        
    except Exception as e:
        print(f"\n❌ Training test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        sys.argv = original_argv
        # Clean up test config
        if os.path.exists('config_test.yaml'):
            os.remove('config_test.yaml')
