"""
Debug why model is not learning
"""
import torch
import sys
from pathlib import Path
import yaml

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.patch_transformer_localization import PatchTransformerWithLocalization
from training.loss_localization import MultiTaskLocalizationLoss
from data.dataset import DentalXrayDataset
from torch.utils.data import DataLoader

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create small dataset
dataset = DentalXrayDataset(
    root_dir=r"C:\Users\maspe\OneDrive\Masaüstü\masterthesis\Dataset_2021\Dataset_2021\Dataset",
    split='train',
    image_size=(1400, 2800),  # Tuple not list
    split_file='outputs/splits/train_val_test_split.json'
)

loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

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

criterion = MultiTaskLocalizationLoss(
    global_weight=1.0,
    patch_weight=1.0,
    diversity_weight=0.0,
    use_focal_loss=True
)

# Get one batch
images, targets = next(iter(loader))
images = images.cuda()
targets = targets.cuda()

print("\n" + "="*80)
print("DEBUG: Model Output Analysis")
print("="*80)

# Forward pass
with torch.no_grad():
    output = model(images)
    
print(f"\nBatch size: {images.shape[0]}")
print(f"Targets: {targets.cpu().numpy()}")

print(f"\nGlobal logits shape: {output['global_logits'].shape}")
print(f"Global logits: {output['global_logits'].squeeze().cpu().numpy()}")
print(f"Global probs: {torch.sigmoid(output['global_logits']).squeeze().cpu().numpy()}")

print(f"\nPatch logits shape: {output['patch_logits'].shape}")
print(f"Patch logits - min: {output['patch_logits'].min().item():.4f}")
print(f"Patch logits - max: {output['patch_logits'].max().item():.4f}")
print(f"Patch logits - mean: {output['patch_logits'].mean().item():.4f}")
print(f"Patch logits - std: {output['patch_logits'].std().item():.4f}")

print(f"\nPatch probs - min: {output['patch_probs'].min().item():.4f}")
print(f"Patch probs - max: {output['patch_probs'].max().item():.4f}")
print(f"Patch probs - mean: {output['patch_probs'].mean().item():.4f}")
print(f"Patch probs - std: {output['patch_probs'].std().item():.4f}")

# Loss computation
losses = criterion(output, targets)
print(f"\n" + "="*80)
print("Loss Components:")
print(f"Total loss: {losses['total_loss'].item():.6f}")
print(f"Global loss: {losses['global_loss'].item():.6f}")
print(f"Patch loss: {losses['patch_loss'].item():.6f}")
print(f"Diversity loss: {losses['diversity_loss'].item():.6f}")

# Check if predictions change with training
print(f"\n" + "="*80)
print("Simulating Training Step:")
print("="*80)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(5):
    optimizer.zero_grad()
    output = model(images)
    losses = criterion(output, targets)
    losses['total_loss'].backward()
    optimizer.step()
    
    global_probs = torch.sigmoid(output['global_logits']).squeeze()
    print(f"Step {step}: Loss={losses['total_loss'].item():.4f}, "
          f"Global probs={global_probs.detach().cpu().numpy()}")

print(f"\n" + "="*80)
print("Checking F1 Computation:")
print("="*80)

# Simulate predictions
with torch.no_grad():
    output = model(images)
    global_probs = torch.sigmoid(output['global_logits']).cpu().numpy()
    preds = (global_probs > 0.5).astype(int).flatten()
    targets_np = targets.cpu().numpy()
    
    print(f"Predictions: {preds}")
    print(f"Targets: {targets_np}")
    print(f"Predictions > 0.5: {preds.sum()}/{len(preds)}")
    
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(targets_np, preds, zero_division=0)
    acc = accuracy_score(targets_np, preds)
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if global_probs.mean() < 0.1 or global_probs.mean() > 0.9:
    print("❌ PROBLEM: Model predicting same class for all samples!")
    print(f"   Mean probability: {global_probs.mean():.4f}")
    if global_probs.mean() < 0.1:
        print("   → Model always predicting HEALTHY (class 0)")
    else:
        print("   → Model always predicting FRACTURED (class 1)")
    print("\nPossible causes:")
    print("  1. Class imbalance too strong (260 fractured vs 80 healthy)")
    print("  2. Loss function issue")
    print("  3. Learning rate too low/high")
    print("  4. Model architecture issue")
else:
    print("✅ Model making diverse predictions")
