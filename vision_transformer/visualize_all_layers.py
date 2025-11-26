"""
Layer-wise Feature Visualization

Extract and visualize features from each layer:
1. Patch CNN encoder layers
2. Transformer encoder layers
3. Patch classifier

Goal: Find which layer preserves spatial information about fractures

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from typing import Dict, List
import json

from models.patch_transformer import PatchTransformerClassifier


class LayerWiseVisualizer:
    """Extract and visualize features from each layer"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {self.device}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Loading model...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Storage for activations
        self.activations = {}
        self.hooks = []
        
        print("✓ Visualizer ready!")
    
    def _load_model(self, checkpoint_path: str):
        """Load model"""
        model_config = self.config['model']
        image_config = self.config['image']
        
        model = PatchTransformerClassifier(
            image_size=tuple(image_config['default_size']),
            patch_size=model_config['patch_size'],
            aggregation=model_config['aggregation'],
            dropout=model_config['dropout']
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Epoch {checkpoint['epoch']}, F1: {checkpoint.get('best_val_f1', 0):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def _register_hooks(self):
        """Register hooks on all interesting layers"""
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Clear previous hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        
        # 1. Patch CNN Encoder layers (ResNet18)
        if hasattr(self.model.patch_encoder, 'backbone'):
            backbone = self.model.patch_encoder.backbone
            
            # Early layers (conv1, layer1)
            if hasattr(backbone, 'conv1'):
                self.hooks.append(backbone.conv1.register_forward_hook(get_activation('cnn_conv1')))
            if hasattr(backbone, 'layer1'):
                self.hooks.append(backbone.layer1.register_forward_hook(get_activation('cnn_layer1')))
            if hasattr(backbone, 'layer2'):
                self.hooks.append(backbone.layer2.register_forward_hook(get_activation('cnn_layer2')))
            if hasattr(backbone, 'layer3'):
                self.hooks.append(backbone.layer3.register_forward_hook(get_activation('cnn_layer3')))
            if hasattr(backbone, 'layer4'):
                self.hooks.append(backbone.layer4.register_forward_hook(get_activation('cnn_layer4')))
            if hasattr(backbone, 'avgpool'):
                self.hooks.append(backbone.avgpool.register_forward_hook(get_activation('cnn_avgpool')))
        
        # 2. Patch encoder output
        self.hooks.append(self.model.patch_encoder.register_forward_hook(get_activation('patch_features')))
        
        # 3. After positional encoding
        self.hooks.append(self.model.pos_encoding.register_forward_hook(get_activation('pos_encoded')))
        
        # 4. Transformer layers
        if hasattr(self.model.transformer, 'layers'):
            for i, layer in enumerate(self.model.transformer.layers):
                self.hooks.append(layer.register_forward_hook(get_activation(f'transformer_layer_{i}')))
        
        # PyTorch TransformerEncoder uses 'layers' attribute internally
        # We'll capture the whole transformer output instead
        self.hooks.append(self.model.transformer.register_forward_hook(get_activation('transformer_output')))
        
        # 5. Patch classifier output
        self.hooks.append(self.model.patch_classifier.register_forward_hook(get_activation('patch_logits')))
        
        print(f"✓ Registered {len(self.hooks)} hooks")
    
    def load_image(self, image_path: str):
        """Load and preprocess image"""
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if self.config['image'].get('apply_clahe', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        target_size = tuple(self.config['image']['default_size'])
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def load_ground_truth(self, image_path: str):
        """Load GT points"""
        txt_path = Path(image_path).with_suffix('.txt')
        
        if not txt_path.exists():
            return []
        
        points = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    points.append((x, y))
        
        return points
    
    def extract_features(self, image_tensor: torch.Tensor):
        """Extract features from all layers"""
        self._register_hooks()
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        return self.activations, output
    
    def visualize_layer(
        self,
        layer_name: str,
        features: torch.Tensor,
        gt_points: List,
        save_dir: Path
    ):
        """Visualize features from one layer"""
        
        # Handle different feature shapes
        if len(features.shape) == 4:
            # CNN features: (B, C, H, W)
            self._visualize_cnn_features(layer_name, features, gt_points, save_dir)
        
        elif len(features.shape) == 3:
            # Transformer features: (B, N, D) where N=num_patches
            self._visualize_transformer_features(layer_name, features, gt_points, save_dir)
        
        elif len(features.shape) == 2:
            # Pooled features: (B, D)
            self._visualize_pooled_features(layer_name, features, save_dir)
    
    def _visualize_cnn_features(self, layer_name, features, gt_points, save_dir):
        """Visualize CNN convolutional features"""
        B, C, H, W = features.shape
        features_np = features[0].cpu().numpy()  # (C, H, W)
        
        # Average across channels
        avg_activation = features_np.mean(axis=0)  # (H, W)
        
        # Normalize
        avg_activation = (avg_activation - avg_activation.min()) / (avg_activation.max() - avg_activation.min() + 1e-8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Average activation
        im1 = axes[0].imshow(avg_activation, cmap='jet')
        axes[0].set_title(f"{layer_name}\nAverage Activation (C={C})", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # 2. Max activation across channels
        max_activation = features_np.max(axis=0)
        max_activation = (max_activation - max_activation.min()) / (max_activation.max() - max_activation.min() + 1e-8)
        im2 = axes[1].imshow(max_activation, cmap='hot')
        axes[1].set_title(f"Max Activation\nShape: ({C}, {H}, {W})", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # 3. Top-5 channels
        channel_energy = features_np.reshape(C, -1).std(axis=1)
        top_channels = np.argsort(channel_energy)[-5:]
        
        combined = features_np[top_channels].mean(axis=0)
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        im3 = axes[2].imshow(combined, cmap='viridis')
        axes[2].set_title(f"Top-5 Channels\nChannels: {top_channels.tolist()}", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{layer_name}_cnn.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {layer_name}: CNN features ({C} channels, {H}x{W})")
    
    def _visualize_transformer_features(self, layer_name, features, gt_points, save_dir):
        """Visualize transformer patch features"""
        B, N, D = features.shape
        
        # Reshape to 2D grid
        num_patches_h = self.config['image']['default_size'][0] // self.config['model']['patch_size']
        num_patches_w = self.config['image']['default_size'][1] // self.config['model']['patch_size']
        
        # Compute attention-like scores
        # Method 1: L2 norm of each patch embedding
        patch_norms = torch.norm(features[0], dim=1).cpu().numpy()  # (N,)
        patch_norms_2d = patch_norms.reshape(num_patches_h, num_patches_w)
        
        # Method 2: Standard deviation across dimensions
        patch_std = features[0].std(dim=1).cpu().numpy()
        patch_std_2d = patch_std.reshape(num_patches_h, num_patches_w)
        
        # Method 3: Mean activation
        patch_mean = features[0].mean(dim=1).cpu().numpy()
        patch_mean_2d = patch_mean.reshape(num_patches_h, num_patches_w)
        
        # Normalize
        patch_norms_2d = (patch_norms_2d - patch_norms_2d.min()) / (patch_norms_2d.max() - patch_norms_2d.min() + 1e-8)
        patch_std_2d = (patch_std_2d - patch_std_2d.min()) / (patch_std_2d.max() - patch_std_2d.min() + 1e-8)
        patch_mean_2d = (patch_mean_2d - patch_mean_2d.min()) / (patch_mean_2d.max() - patch_mean_2d.min() + 1e-8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. L2 Norm
        im1 = axes[0].imshow(patch_norms_2d, cmap='jet')
        axes[0].set_title(f"{layer_name}\nL2 Norm per Patch", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # 2. Standard Deviation
        im2 = axes[1].imshow(patch_std_2d, cmap='hot')
        axes[1].set_title(f"Std Dev per Patch\nShape: ({N} patches, {D}D)", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # 3. Mean Activation
        im3 = axes[2].imshow(patch_mean_2d, cmap='viridis')
        axes[2].set_title(f"Mean Activation per Patch\nGrid: {num_patches_h}x{num_patches_w}", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{layer_name}_transformer.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {layer_name}: Transformer features ({N} patches, {D}D)")
    
    def _visualize_pooled_features(self, layer_name, features, save_dir):
        """Visualize pooled/global features"""
        B, D = features.shape
        features_np = features[0].cpu().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.bar(range(D), features_np, color='steelblue', alpha=0.7)
        ax.set_title(f"{layer_name}\nGlobal Feature Vector (D={D})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Activation")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{layer_name}_pooled.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {layer_name}: Pooled features ({D}D)")
    
    def visualize_all_layers(
        self,
        image_path: str,
        output_dir: str = "outputs/layer_analysis"
    ):
        """Extract and visualize all layers"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("LAYER-WISE FEATURE EXTRACTION")
        print("="*80)
        
        # Load image and GT
        print(f"\nLoading: {image_path}")
        image_tensor = self.load_image(image_path)
        gt_points = self.load_ground_truth(image_path)
        print(f"  GT points: {len(gt_points)}")
        
        # Extract features
        print("\nExtracting features from all layers...")
        activations, output = self.extract_features(image_tensor)
        
        prob = torch.sigmoid(output).item()
        print(f"\nClassification: {'Fractured' if prob > 0.5 else 'Healthy'} ({prob:.1%})")
        print(f"Total layers captured: {len(activations)}")
        
        # Visualize each layer
        print("\nVisualizing layers:")
        for layer_name, features in activations.items():
            try:
                self.visualize_layer(layer_name, features, gt_points, output_dir)
            except Exception as e:
                print(f"  ✗ {layer_name}: Error - {e}")
        
        # Create summary
        self._create_summary(activations, gt_points, output_dir)
        
        print("\n" + "="*80)
        print(f"✓ COMPLETED! Results saved to: {output_dir}")
        print("="*80)
    
    def _create_summary(self, activations: Dict, gt_points: List, save_dir: Path):
        """Create summary JSON"""
        summary = {
            'total_layers': len(activations),
            'layers': {}
        }
        
        for layer_name, features in activations.items():
            summary['layers'][layer_name] = {
                'shape': list(features.shape),
                'dtype': str(features.dtype),
                'min': float(features.min()),
                'max': float(features.max()),
                'mean': float(features.mean()),
                'std': float(features.std())
            }
        
        with open(save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Summary saved: summary.json")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Layer-wise Feature Visualization")
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model', type=str, default='checkpoints/patch_transformer_full/best.pth')
    parser.add_argument('--output', type=str, default='outputs/layer_analysis')
    
    args = parser.parse_args()
    
    visualizer = LayerWiseVisualizer(args.model)
    visualizer.visualize_all_layers(args.image, args.output)


if __name__ == "__main__":
    main()
