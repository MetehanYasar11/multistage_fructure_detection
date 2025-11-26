"""
Simplified Localization: GradCAM on Baseline Patch Transformer

Instead of modifying architecture:
1. Use existing baseline (F1=0.9091) 
2. Extract attention via GradCAM on patch embeddings
3. Visualize high-attention patches as fracture regions

This preserves the proven model while adding localization.

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml

from models.patch_transformer import PatchTransformerClassifier


class GradCAMLocalizer:
    """
    GradCAM-based localization for Patch Transformer
    
    Extracts attention from patch embeddings to identify fracture locations
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        print("\nLoading Patch Transformer...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Hook for gradients
        self.gradients = None
        self.activations = None
        
        print("\n[*] Localizer ready!")
    
    def _load_model(self, checkpoint_path: str):
        """Load trained model"""
        from models.patch_transformer import PatchTransformerClassifier
        
        # Get model config
        model_config = self.config['model']
        image_config = self.config['image']
        
        # Create model
        model = PatchTransformerClassifier(
            image_size=tuple(image_config['default_size']),
            patch_size=model_config['patch_size'],
            cnn_backbone='resnet18',
            feature_dim=512,
            num_heads=8,
            num_layers=6,
            dropout=model_config['dropout'],
            aggregation=model_config['aggregation']
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch {checkpoint['epoch']}")
            print(f"  Best Val F1: {checkpoint.get('best_val_f1', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Hook on transformer encoder output
        # This gives us patch-level features
        target_layer = self.model.transformer
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load and preprocess image
        
        Returns:
            tensor: (1, 3, H, W) preprocessed tensor
            original: (H, W, 3) original image
        """
        # Load image (handle Unicode paths)
        import numpy as np
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply CLAHE
        if self.config['image'].get('apply_clahe', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        # Resize
        target_size = tuple(self.config['image']['default_size'])
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # To tensor
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor, original
    
    def generate_gradcam(
        self,
        image_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Generate GradCAM attention map
        
        Returns:
            attention_map: (H, W) normalized attention scores
        """
        # Register hooks
        self._register_hooks()
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(image_tensor)
        
        # Backward pass (for positive class)
        output.backward()
        
        # GradCAM calculation
        # gradients: (B, N, D) where N=num_patches
        # activations: (B, N, D)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check hooks.")
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=-1, keepdim=True)  # (B, N, 1)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=-1)  # (B, N)
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Reshape to 2D grid
        B = cam.size(0)
        num_patches_h = self.config['image']['default_size'][0] // self.config['model']['patch_size']
        num_patches_w = self.config['image']['default_size'][1] // self.config['model']['patch_size']
        
        cam_2d = cam.view(B, num_patches_h, num_patches_w)
        
        return cam_2d[0].cpu().numpy()
    
    def get_fracture_bboxes(
        self,
        attention_map: np.ndarray,
        threshold: float = 0.7,
        min_cluster_size: int = 2
    ) -> List[Tuple[int, int, int, int]]:
        """
        Convert attention map to bounding boxes
        
        Args:
            attention_map: (nh, nw) attention scores
            threshold: Attention threshold (percentile-based)
            min_cluster_size: Minimum connected patches
            
        Returns:
            bboxes: List of (x1, y1, x2, y2) in pixel coordinates
        """
        # Dynamic threshold
        threshold_value = np.percentile(attention_map, threshold * 100)
        binary_mask = (attention_map > threshold_value).astype(np.uint8)
        
        # Find connected components
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary_mask)
        
        patch_size = self.config['model']['patch_size']
        bboxes = []
        
        for i in range(1, num_features + 1):
            component = (labeled == i)
            if component.sum() < min_cluster_size:
                continue
            
            # Get bounding box
            rows, cols = np.where(component)
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            
            # Convert to pixel coordinates
            x1 = c_min * patch_size
            y1 = r_min * patch_size
            x2 = (c_max + 1) * patch_size
            y2 = (r_max + 1) * patch_size
            
            bboxes.append((x1, y1, x2, y2))
        
        return bboxes
    
    def predict(
        self,
        image_path: str,
        attention_threshold: float = 0.7
    ) -> Dict:
        """
        Complete prediction with localization
        
        Returns:
            Dictionary with classification and localization info
        """
        # Load image
        tensor, original = self.preprocess_image(image_path)
        
        # Classification
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()
            pred_class = int(prob > 0.5)
        
        # GradCAM localization
        attention_map = self.generate_gradcam(tensor)
        
        # Get bboxes
        fracture_bboxes = []
        if pred_class == 1:  # Only for fractured
            fracture_bboxes = self.get_fracture_bboxes(
                attention_map,
                threshold=attention_threshold
            )
        
        return {
            'classification': 'Fractured' if pred_class == 1 else 'Healthy',
            'confidence': prob if pred_class == 1 else (1 - prob),
            'attention_map': attention_map,
            'fracture_bboxes': fracture_bboxes,
            'original_image': original
        }
    
    def visualize_results(
        self,
        results: Dict,
        save_path: str
    ):
        """Visualize results"""
        img = results['original_image']
        H, W = img.shape[:2]
        
        # Upsample attention map
        attention_map = results['attention_map']
        attn_resized = cv2.resize(
            attention_map,
            (W, H),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # 1. Original
        axes[0].imshow(img)
        axes[0].set_title("Original X-ray", fontsize=14)
        axes[0].axis('off')
        
        # 2. Attention Map
        axes[1].imshow(img)
        axes[1].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[1].set_title("GradCAM Attention (Red=High)", fontsize=14)
        axes[1].axis('off')
        
        # 3. Fracture Localization
        img_bbox = img.copy()
        for bbox in results['fracture_bboxes']:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(
                img_bbox,
                "FRACTURE",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
        
        axes[2].imshow(img_bbox)
        title = f"{results['classification']} (Conf: {results['confidence']:.2%})"
        if results['fracture_bboxes']:
            title += f"\n{len(results['fracture_bboxes'])} fracture region(s)"
        axes[2].set_title(title, fontsize=14, 
                         color='red' if results['classification'] == 'Fractured' else 'green')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"\n[*] Visualization saved to {save_path}")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GradCAM Fracture Localization")
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model', type=str, default='checkpoints/patch_transformer_full/best.pth')
    parser.add_argument('--output', type=str, default='outputs/gradcam/result.png')
    parser.add_argument('--threshold', type=float, default=0.7)
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GRADCAM FRACTURE LOCALIZATION")
    print("="*80)
    
    # Initialize localizer
    localizer = GradCAMLocalizer(args.model)
    
    # Run prediction
    print(f"\nProcessing: {args.image}")
    results = localizer.predict(args.image, args.threshold)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Classification: {results['classification']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"Fracture Regions: {len(results['fracture_bboxes'])}")
    if results['fracture_bboxes']:
        for i, bbox in enumerate(results['fracture_bboxes'], 1):
            print(f"  Region {i}: x={bbox[0]}-{bbox[2]}, y={bbox[1]}-{bbox[3]}")
    
    # Visualize
    localizer.visualize_results(results, args.output)
    
    print("\n" + "="*80)
    print("LOCALIZATION COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
