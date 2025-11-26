"""
Convert Baseline Patch Transformer to Attention-Guided Version

This script:
1. Loads baseline checkpoint (best.pth)
2. Creates new attention-guided model
3. Transfers weights (patch encoder + transformer)
4. Initializes attention head randomly
5. Saves as attention-ready checkpoint

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import yaml
from pathlib import Path

from models.patch_transformer import PatchTransformerClassifier
from models.attention_patch_transformer import AttentionGuidedPatchTransformer


def convert_baseline_to_attention(
    baseline_path: str,
    output_path: str,
    config_path: str = "config.yaml"
):
    """
    Convert baseline model to attention-guided version
    
    Args:
        baseline_path: Path to baseline checkpoint
        output_path: Path to save converted checkpoint
        config_path: Config file path
    """
    print("="*80)
    print("CONVERTING BASELINE TO ATTENTION-GUIDED MODEL")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    print(f"\nLoading baseline checkpoint: {baseline_path}")
    checkpoint = torch.load(baseline_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        baseline_state = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        best_f1 = checkpoint.get('best_val_f1', 0.0)
        print(f"  Epoch: {epoch}")
        print(f"  Best Val F1: {best_f1:.4f}")
    else:
        baseline_state = checkpoint
        print("  Warning: No metadata found in checkpoint")
    
    # Create attention-guided model
    print("\nCreating attention-guided model...")
    
    # Default settings (baseline used these)
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    
    attention_model = AttentionGuidedPatchTransformer(
        image_size=tuple(config['image']['default_size']),
        patch_size=model_config['patch_size'],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=model_config['dropout']
    )
    
    attention_state = attention_model.state_dict()
    
    # Transfer weights
    print("\nTransferring weights...")
    transferred = 0
    skipped = 0
    
    for key, value in baseline_state.items():
        if key in attention_state:
            # Check shape compatibility
            if attention_state[key].shape == value.shape:
                attention_state[key] = value
                transferred += 1
            else:
                print(f"  [!] Shape mismatch for {key}: "
                      f"{value.shape} vs {attention_state[key].shape}")
                skipped += 1
        else:
            # Key not in attention model (expected for old classifier head)
            skipped += 1
    
    print(f"\n  Transferred: {transferred} parameters")
    print(f"  Skipped: {skipped} parameters")
    print(f"  New (random): {len(attention_state) - transferred} parameters")
    
    # Load transferred state
    attention_model.load_state_dict(attention_state)
    
    # Save checkpoint
    print(f"\nSaving converted checkpoint: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': attention_state,
        'epoch': checkpoint.get('epoch', 0) if 'epoch' in checkpoint else 0,
        'best_val_f1': checkpoint.get('best_val_f1', 0.0) if 'best_val_f1' in checkpoint else 0.0,
        'converted_from': baseline_path,
        'note': 'Converted from baseline to attention-guided model. Attention head is randomly initialized.'
    }, output_path)
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETED!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Fine-tune attention head:")
    print(f"   python train_attention_finetuning.py")
    print(f"2. Test inference:")
    print(f"   python inference_attention_rct.py --image <path>")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert baseline to attention model")
    parser.add_argument(
        '--baseline',
        type=str,
        default='checkpoints/patch_transformer_full/best.pth',
        help="Path to baseline checkpoint"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='checkpoints/attention_patch_transformer/baseline_converted.pth',
        help="Output path for converted checkpoint"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help="Config file path"
    )
    
    args = parser.parse_args()
    
    convert_baseline_to_attention(
        baseline_path=args.baseline,
        output_path=args.output,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
