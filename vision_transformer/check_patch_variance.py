import numpy as np

# Load patch predictions
p = np.load(r'outputs\test_evaluation\test_patch_predictions.npy')
p_sig = 1 / (1 + np.exp(-p.squeeze()))  # Sigmoid

print("=" * 60)
print("PATCH PREDICTION VARIANCE ANALYSIS")
print("=" * 60)

# Check a few images
test_indices = [0, 11, 17, 37, 44]  # Mix of correct/incorrect

for idx in test_indices:
    patches = p_sig[idx]
    print(f"\nImage {idx}:")
    print(f"  Min:  {patches.min():.4f}")
    print(f"  Max:  {patches.max():.4f}")
    print(f"  Mean: {patches.mean():.4f}")
    print(f"  Std:  {patches.std():.4f}")
    print(f"  Range: {patches.max() - patches.min():.4f}")
    
    # Count high/low patches
    high = (patches > 0.8).sum()
    medium = ((patches >= 0.6) & (patches <= 0.8)).sum()
    low = (patches < 0.6).sum()
    print(f"  High risk (>0.8): {high}/392 ({high/392*100:.1f}%)")
    print(f"  Medium (0.6-0.8): {medium}/392 ({medium/392*100:.1f}%)")
    print(f"  Low risk (<0.6): {low}/392 ({low/392*100:.1f}%)")

print("\n" + "=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)
print(f"Global min: {p_sig.min():.4f}")
print(f"Global max: {p_sig.max():.4f}")
print(f"Global mean: {p_sig.mean():.4f}")
print(f"Global std: {p_sig.std():.4f}")
