"""
Debug RCT Training - Check model predictions

Author: Master's Thesis Project  
Date: November 23, 2025
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

from data.rct_dataset import RCTDataset
from data import get_val_transforms
from models import create_patch_transformer


def check_model_predictions():
    """Check what model is predicting"""
    
    # Load checkpoint
    checkpoint_path = Path("checkpoints/rct_classifier/checkpoint_epoch_010.pth")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = create_patch_transformer(
        num_patches_h=4,
        num_patches_w=4,
        patch_size=56,
        backbone="resnet18",
        pretrained=False,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load validation dataset
    val_dataset = RCTDataset(
        root_dir="RCT_classification_dataset",
        split='val',
        transform=get_val_transforms(224),
        image_size=224
    )
    
    print(f"\nValidation dataset: {len(val_dataset)} images")
    print(f"  Fractured: {sum(val_dataset.labels)}")
    print(f"  Healthy: {len(val_dataset.labels) - sum(val_dataset.labels)}")
    
    # Test predictions
    predictions = []
    targets = []
    logits_list = []
    
    with torch.no_grad():
        for i in range(min(50, len(val_dataset))):
            image, label = val_dataset[i]
            image = image.unsqueeze(0)  # Add batch dimension
            
            output = model(image)
            logit = output.item() if output.dim() == 1 else output[0].item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            pred = int(prob >= 0.5)
            
            predictions.append(pred)
            targets.append(label)
            logits_list.append(logit)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    logits_list = np.array(logits_list)
    
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)
    print(f"Samples checked: {len(predictions)}")
    print(f"\nPredictions:")
    print(f"  Always 0 (healthy): {np.sum(predictions == 0)}")
    print(f"  Always 1 (fractured): {np.sum(predictions == 1)}")
    print(f"\nTargets:")
    print(f"  Healthy (0): {np.sum(targets == 0)}")
    print(f"  Fractured (1): {np.sum(targets == 1)}")
    print(f"\nLogits statistics:")
    print(f"  Mean: {np.mean(logits_list):.4f}")
    print(f"  Std: {np.std(logits_list):.4f}")
    print(f"  Min: {np.min(logits_list):.4f}")
    print(f"  Max: {np.max(logits_list):.4f}")
    print(f"\nProbabilities (after sigmoid):")
    probs = 1 / (1 + np.exp(-logits_list))
    print(f"  Mean: {np.mean(probs):.4f}")
    print(f"  Std: {np.std(probs):.4f}")
    print(f"  Min: {np.min(probs):.4f}")
    print(f"  Max: {np.max(probs):.4f}")
    
    # Show some examples
    print(f"\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    for i in range(min(10, len(predictions))):
        print(f"Sample {i}: Target={targets[i]} ({'Fractured' if targets[i]==1 else 'Healthy'}), "
              f"Logit={logits_list[i]:.4f}, Prob={probs[i]:.4f}, "
              f"Pred={predictions[i]} ({'Fractured' if predictions[i]==1 else 'Healthy'})")
    
    # Diagnosis
    print(f"\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    if np.std(logits_list) < 0.01:
        print("❌ PROBLEM: Model outputs are nearly constant!")
        print("   → Model not learning, outputting same value for all inputs")
        print("   → Possible causes:")
        print("      1. Loss function issue (pos_weight too strong?)")
        print("      2. Learning rate too low")
        print("      3. Model initialization problem")
        print("      4. Gradient vanishing")
    elif np.std(probs) < 0.1:
        print("⚠️  WARNING: Low prediction variance")
        print("   → Model is learning but not confident")
    else:
        print("✅ Model is producing diverse predictions")
        
        if np.sum(predictions == 0) / len(predictions) > 0.9:
            print("⚠️  WARNING: Model biased towards Healthy class")
            print("   → Need stronger pos_weight or different sampling")
        elif np.sum(predictions == 1) / len(predictions) > 0.9:
            print("⚠️  WARNING: Model biased towards Fractured class")
            print("   → pos_weight too strong?")


if __name__ == "__main__":
    check_model_predictions()
