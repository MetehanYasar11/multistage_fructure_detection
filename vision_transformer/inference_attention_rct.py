"""
End-to-End Inference Pipeline: Attention + RCT Integration

Pipeline:
1. Load panoramic X-ray
2. Patch Transformer → Classification + Attention Map
3. YOLOv11x RCT Detector → RCT tooth bboxes
4. Intersection Analysis → Which RCT tooth has fracture?
5. Visualization → Annotated image

Key Features:
- Attention-guided localization (patch-level)
- RCT detection (tooth-level)
- Smart intersection: IoU-based matching
- Color-coded visualization: Green=healthy RCT, Red=fractured RCT

Author: Master's Thesis Project
Date: November 23, 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
from ultralytics import YOLO
from tqdm import tqdm

from models.attention_patch_transformer import AttentionGuidedPatchTransformer


class FractureLocalizationPipeline:
    """
    Complete pipeline for fracture detection and localization
    """
    
    def __init__(
        self,
        patch_transformer_path: str,
        rct_detector_path: str,
        config_path: str = "config.yaml",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load Patch Transformer
        print("\nLoading Patch Transformer...")
        self.patch_transformer = self._load_patch_transformer(patch_transformer_path)
        
        # Load RCT Detector
        print("\nLoading RCT Detector (YOLOv11x)...")
        self.rct_detector = YOLO(rct_detector_path)
        
        print("\n[*] Pipeline ready!")
    
    def _load_patch_transformer(self, checkpoint_path: str):
        """Load trained Patch Transformer"""
        model_config = self.config['model']
        
        model = AttentionGuidedPatchTransformer(
            image_size=tuple(self.config['image']['default_size']),
            patch_size=model_config['patch_size'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout']
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch {checkpoint['epoch']}")
            print(f"  Best Val F1: {checkpoint.get('best_val_f1', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load and preprocess image
        
        Returns:
            tensor: (1, 3, H, W) preprocessed tensor
            original: (H, W, 3) original image for visualization
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply CLAHE if specified
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
    
    def detect_rct_teeth(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect RCT teeth using YOLO
        
        Returns:
            List of detections with {bbox, conf, class_id}
        """
        results = self.rct_detector(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': conf,
                    'class_id': cls
                })
        
        return detections
    
    def calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_fractures_to_rct(
        self,
        fracture_bboxes: List[Tuple[int, int, int, int]],
        rct_detections: List[Dict],
        iou_threshold: float = 0.3
    ) -> List[int]:
        """
        Match fracture regions to RCT teeth
        
        Returns:
            List of RCT indices that contain fractures
        """
        fractured_rct_indices = []
        
        for rct_idx, rct_det in enumerate(rct_detections):
            rct_bbox = rct_det['bbox']
            
            # Check if any fracture region overlaps with this RCT
            max_iou = 0.0
            for frac_bbox in fracture_bboxes:
                iou = self.calculate_iou(frac_bbox, rct_bbox)
                max_iou = max(max_iou, iou)
            
            if max_iou > iou_threshold:
                fractured_rct_indices.append(rct_idx)
        
        return fractured_rct_indices
    
    def predict(
        self,
        image_path: str,
        attention_threshold: float = 0.7,
        rct_conf_threshold: float = 0.5,
        iou_threshold: float = 0.3
    ) -> Dict:
        """
        Complete prediction pipeline
        
        Returns:
            Dictionary with:
            - classification: Fractured or not
            - confidence: Classification confidence
            - attention_map: (H, W) attention scores
            - fracture_bboxes: Attention-based fracture regions
            - rct_detections: RCT tooth detections
            - fractured_rct_indices: Which RCT teeth have fractures
        """
        # Load and preprocess image
        tensor, original = self.preprocess_image(image_path)
        
        # Patch Transformer prediction
        with torch.no_grad():
            logits, attention_map = self.patch_transformer(
                tensor,
                return_attention=True
            )
            
            # Classification
            prob = torch.sigmoid(logits).item()
            pred_class = int(prob > 0.5)
            
            # Get fracture bboxes from attention
            fracture_bboxes = self.patch_transformer.get_fracture_bboxes(
                attention_map,
                threshold=attention_threshold
            )[0]  # First batch
        
        # RCT Detection
        rct_detections = self.detect_rct_teeth(original, rct_conf_threshold)
        
        # Match fractures to RCT teeth
        fractured_rct_indices = []
        if pred_class == 1 and len(fracture_bboxes) > 0 and len(rct_detections) > 0:
            fractured_rct_indices = self.match_fractures_to_rct(
                fracture_bboxes,
                rct_detections,
                iou_threshold
            )
        
        return {
            'classification': 'Fractured' if pred_class == 1 else 'Healthy',
            'confidence': prob if pred_class == 1 else (1 - prob),
            'attention_map': attention_map[0].cpu().numpy(),
            'fracture_bboxes': fracture_bboxes,
            'rct_detections': rct_detections,
            'fractured_rct_indices': fractured_rct_indices,
            'original_image': original
        }
    
    def visualize_results(
        self,
        results: Dict,
        save_path: str
    ):
        """
        Visualize complete results
        
        Creates figure with:
        1. Original image with RCT bboxes
        2. Attention map overlay
        3. Final result: Fractured RCT highlighted
        """
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
        
        # 1. RCT Detections
        img_rct = img.copy()
        for i, det in enumerate(results['rct_detections']):
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0)  # Green for all RCT
            cv2.rectangle(img_rct, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img_rct,
                f"RCT {i+1}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        axes[0].imshow(img_rct)
        axes[0].set_title(f"RCT Detection ({len(results['rct_detections'])} teeth)", fontsize=14)
        axes[0].axis('off')
        
        # 2. Attention Map
        axes[1].imshow(img)
        axes[1].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[1].set_title("Attention Map (Red=High Attention)", fontsize=14)
        axes[1].axis('off')
        
        # 3. Final Result: Fractured RCT
        img_final = img.copy()
        
        # Draw all RCT in green
        for i, det in enumerate(results['rct_detections']):
            x1, y1, x2, y2 = det['bbox']
            
            # Check if fractured
            if i in results['fractured_rct_indices']:
                color = (255, 0, 0)  # RED for fractured
                label = f"FRACTURED RCT {i+1}"
                thickness = 4
            else:
                color = (0, 255, 0)  # GREEN for healthy
                label = f"Healthy RCT {i+1}"
                thickness = 2
            
            cv2.rectangle(img_final, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                img_final,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        axes[2].imshow(img_final)
        title = f"{results['classification']} (Conf: {results['confidence']:.2%})"
        if results['fractured_rct_indices']:
            title += f"\nFractured: RCT {results['fractured_rct_indices']}"
        axes[2].set_title(title, fontsize=14, color='red' if results['classification'] == 'Fractured' else 'green')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"\n[*] Visualization saved to {save_path}")
        plt.close()


def main():
    """Test pipeline on sample images"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fracture Localization Pipeline")
    parser.add_argument('--image', type=str, required=True, help="Path to panoramic X-ray")
    parser.add_argument('--patch_model', type=str, default='checkpoints/patch_transformer_full/best.pth',
                        help="Path to Patch Transformer checkpoint")
    parser.add_argument('--rct_model', type=str, default='checkpoints/rct_detector/best.pt',
                        help="Path to RCT detector checkpoint")
    parser.add_argument('--output', type=str, default='outputs/localization/result.png',
                        help="Output visualization path")
    parser.add_argument('--attention_threshold', type=float, default=0.7,
                        help="Attention threshold for fracture detection")
    parser.add_argument('--rct_conf', type=float, default=0.5,
                        help="RCT detection confidence threshold")
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help="IoU threshold for matching fractures to RCT")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FRACTURE LOCALIZATION PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = FractureLocalizationPipeline(
        patch_transformer_path=args.patch_model,
        rct_detector_path=args.rct_model
    )
    
    # Run prediction
    print(f"\nProcessing: {args.image}")
    results = pipeline.predict(
        args.image,
        attention_threshold=args.attention_threshold,
        rct_conf_threshold=args.rct_conf,
        iou_threshold=args.iou_threshold
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Classification: {results['classification']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"RCT Teeth Detected: {len(results['rct_detections'])}")
    print(f"Fracture Regions: {len(results['fracture_bboxes'])}")
    if results['fractured_rct_indices']:
        print(f"Fractured RCT Indices: {results['fractured_rct_indices']}")
        print(f"  -> RCT {', RCT '.join(map(str, [i+1 for i in results['fractured_rct_indices']]))} contains fracture!")
    else:
        if results['classification'] == 'Fractured':
            print("  -> Fracture detected but no RCT teeth found in image")
        else:
            print("  -> No fractures detected")
    
    # Visualize
    pipeline.visualize_results(results, args.output)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
