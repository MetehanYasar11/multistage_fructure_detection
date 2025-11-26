"""
Re-evaluate test set with PROPER patch predictions.
Uses model.get_patch_predictions() which gives INDIVIDUAL patch scores.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Model
from models.patch_transformer import create_patch_transformer

# Data
from data.dataset import DentalXrayDataset
from torch.utils.data import DataLoader

# Paths
CHECKPOINT_PATH = r"checkpoints\patch_transformer_full\best.pth"
DATA_ROOT = Path(r"c:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset")
OUTPUT_DIR = Path(r"outputs\test_evaluation")

def main():
    """Re-extract patch predictions properly."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n📦 Loading model...")
    model = create_patch_transformer(
        image_size=(1400, 2800),
        patch_size=100,
        model_size='base'
    )
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    
    # Load test data
    print("\n📂 Loading test data...")
    
    # Import transforms
    from data import get_val_transforms
    
    test_dataset = DentalXrayDataset(
        root_dir=DATA_ROOT,
        split='test',
        transform=get_val_transforms(image_size=(1400, 2800)),
        split_file="outputs/splits/train_val_test_split.json"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one at a time for clarity
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Test set: {len(test_dataset)} images")
    
    # Extract patch predictions
    print("\n🔍 Extracting patch predictions...")
    all_patch_logits = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Processing"):
            images = images.to(device)
            
            # Get INDIVIDUAL patch predictions
            patch_logits, nh, nw = model.get_patch_predictions(images)
            # patch_logits: (1, 392, 1)
            
            patch_logits_np = patch_logits.cpu().numpy()
            all_patch_logits.append(patch_logits_np[0])  # (392, 1)
    
    # Stack and save
    all_patch_logits = np.array(all_patch_logits)  # (74, 392, 1)
    
    print(f"\n📊 Patch predictions shape: {all_patch_logits.shape}")
    print(f"   Min logit: {all_patch_logits.min():.4f}")
    print(f"   Max logit: {all_patch_logits.max():.4f}")
    print(f"   Mean logit: {all_patch_logits.mean():.4f}")
    
    # Apply sigmoid and check variance
    patch_probs = 1 / (1 + np.exp(-all_patch_logits))
    print(f"\n   After sigmoid:")
    print(f"   Min prob: {patch_probs.min():.4f}")
    print(f"   Max prob: {patch_probs.max():.4f}")
    print(f"   Mean prob: {patch_probs.mean():.4f}")
    
    # Check variance per image
    print(f"\n📈 Variance check (first 5 images):")
    for i in range(min(5, len(patch_probs))):
        img_probs = patch_probs[i].flatten()
        print(f"   Image {i}: min={img_probs.min():.4f}, max={img_probs.max():.4f}, "
              f"range={img_probs.max()-img_probs.min():.4f}, std={img_probs.std():.4f}")
    
    # Save
    output_path = OUTPUT_DIR / "test_patch_predictions_fixed.npy"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_patch_logits)
    print(f"\n💾 Saved to: {output_path}")
    
    print("\n✅ Done! Now patch predictions should have proper variance.")

if __name__ == "__main__":
    main()
