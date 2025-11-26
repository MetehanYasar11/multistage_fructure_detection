"""
Check what predictions look like from checkpoint
"""
import torch
import numpy as np

# Load last checkpoint
checkpoint = torch.load('outputs/localization_model/checkpoint_epoch_050.pth', map_location='cpu', weights_only=False)

print("Checkpoint epoch:", checkpoint['epoch'])
#print("F1 score:", checkpoint.get('f1', 'N/A'))

# Load model
from models.patch_transformer_localization import PatchTransformerWithLocalization

model = PatchTransformerWithLocalization(
    image_size=(1400, 2800),
    patch_size=100,
    cnn_backbone='resnet18',
    feature_dim=512,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    use_global_head=True
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create dummy input
dummy_input = torch.randn(2, 3, 1400, 2800)

print("\nTesting model predictions:")
with torch.no_grad():
    output = model(dummy_input)
    global_logits = output['global_logits']
    global_probs = torch.sigmoid(global_logits)
    
    print(f"Global logits: {global_logits.squeeze().numpy()}")
    print(f"Global probs: {global_probs.squeeze().numpy()}")
    print(f"Predictions (>0.5): {(global_probs > 0.5).int().squeeze().numpy()}")

# Check global head final layer
print("\nGlobal head final layer bias:")
global_head_bias = checkpoint['model_state_dict']['global_head.4.bias']
print(f"  Value: {global_head_bias.item():.6f}")

print("\nGlobal head final layer weights:")
global_head_weight = checkpoint['model_state_dict']['global_head.4.weight']
print(f"  Shape: {global_head_weight.shape}")
print(f"  Mean: {global_head_weight.mean().item():.6f}")
print(f"  Std: {global_head_weight.std().item():.6f}")
print(f"  Min: {global_head_weight.min().item():.6f}")
print(f"  Max: {global_head_weight.max().item():.6f}")

print("\n" + "="*80)
print("DIAGNOSIS:")
if global_probs.mean() < 0.1:
    print("❌ Model ALWAYS predicts HEALTHY (class 0)")
    print(f"   Mean probability: {global_probs.mean().item():.4f}")
    print("   Cause: Model biased toward negative class")
elif global_probs.mean() > 0.9:
    print("❌ Model ALWAYS predicts FRACTURED (class 1)")
    print(f"   Mean probability: {global_probs.mean().item():.4f}")
    print("   Cause: Model biased toward positive class")
else:
    print("✅ Model making varied predictions")
    print(f"   Mean probability: {global_probs.mean().item():.4f}")
