"""
PyTorch Dataset for Dental X-Ray Fracture Detection

Binary Classification:
- Fractured (label=1): Images with broken endodontic instruments
- Healthy (label=0): Hard negative examples (no fractures)

Key Features:
- PIL-based image loading (handles Turkish characters in Windows paths)
- CLAHE preprocessing option
- Stratified split support
- Configurable image size

Author: Master's Thesis Project
Date: October 28, 2025
"""

import os
import json
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class DentalXrayDataset(Dataset):
    """
    PyTorch Dataset for Dental X-ray Fracture Detection.
    
    Directory Structure:
        root_dir/
            Fractured/
                image1.jpg
                image2.jpg
                ...
            Healthy/
                image1.jpg
                image2.jpg
                ...
    
    Attributes:
        root_dir: Path to dataset root directory
        split: Dataset split ('train', 'val', 'test', or 'all')
        transform: Albumentations transform pipeline
        image_size: Target image size (if not using transform)
        use_clahe: Enable CLAHE preprocessing (if not using transform)
        split_file: Path to JSON file containing split indices
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'all',
        transform: Optional[Callable] = None,
        image_size: int = 640,
        use_clahe: bool = True,
        split_file: Optional[str] = None,
        fractured_dir: str = 'Fractured',
        healthy_dir: str = 'Healthy'
    ):
        """
        Initialize Dental X-ray Dataset.
        
        Args:
            root_dir: Path to dataset root directory
            split: Dataset split ('train', 'val', 'test', or 'all')
            transform: Albumentations transform pipeline (if None, uses default)
            image_size: Target image size for resizing
            use_clahe: Enable CLAHE preprocessing
            split_file: Path to JSON file with split indices (required if split != 'all')
            fractured_dir: Name of fractured images directory
            healthy_dir: Name of healthy images directory
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.fractured_dir = fractured_dir
        self.healthy_dir = healthy_dir
        
        # Load all images and labels
        self.image_paths, self.labels = self._load_dataset()
        
        # Apply split filter if needed
        if split != 'all':
            if split_file is None:
                raise ValueError(f"split_file must be provided when split='{split}'")
            
            self._apply_split(split_file)
        
        print(f"Loaded {len(self.image_paths)} images for split '{split}'")
        print(f"  Fractured: {sum(self.labels)} | Healthy: {len(self.labels) - sum(self.labels)}")
    
    def _load_dataset(self) -> Tuple[List[str], List[int]]:
        """
        Load all image paths and labels from dataset directories.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        # Load Fractured images (label=1)
        fractured_dir = self.root_dir / self.fractured_dir
        if fractured_dir.exists():
            for img_file in fractured_dir.glob('*.jpg'):
                image_paths.append(str(img_file))
                labels.append(1)  # Fractured = positive class
        else:
            print(f"Warning: Fractured directory not found: {fractured_dir}")
        
        # Load Healthy images (label=0)
        healthy_dir = self.root_dir / self.healthy_dir
        if healthy_dir.exists():
            for img_file in healthy_dir.glob('*.jpg'):
                image_paths.append(str(img_file))
                labels.append(0)  # Healthy = negative class (hard negatives)
        else:
            print(f"Warning: Healthy directory not found: {healthy_dir}")
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}")
        
        return image_paths, labels
    
    def _apply_split(self, split_file: str):
        """
        Filter dataset to only include images from specified split.
        
        Args:
            split_file: Path to JSON file containing split indices
        """
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        if self.split not in split_data:
            raise ValueError(f"Split '{self.split}' not found in {split_file}")
        
        split_indices = split_data[self.split]
        
        # Filter image_paths and labels
        self.image_paths = [self.image_paths[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Load image using PIL (handles Turkish characters in Windows paths)
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")
        
        # Convert to numpy array for Albumentations
        image = np.array(image)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: resize only (no augmentation)
            image = Image.fromarray(image).resize((self.image_size, self.image_size))
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        label = self.labels[idx]
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset.
        
        Formula: weight = total_samples / (num_classes * class_samples)
        
        Returns:
            Tensor of class weights [weight_healthy, weight_fractured]
        """
        num_fractured = sum(self.labels)
        num_healthy = len(self.labels) - num_fractured
        total = len(self.labels)
        
        # Weight inversely proportional to class frequency
        weight_healthy = total / (2 * num_healthy)
        weight_fractured = total / (2 * num_fractured)
        
        weights = torch.tensor([weight_healthy, weight_fractured], dtype=torch.float32)
        
        print(f"Class weights: Healthy={weight_healthy:.3f}, Fractured={weight_fractured:.3f}")
        
        return weights
    
    def get_sample_weights(self) -> List[float]:
        """
        Get per-sample weights for WeightedRandomSampler.
        
        Returns:
            List of weights (one per sample)
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label].item() for label in self.labels]
        return sample_weights


class DentalXrayDatasetWithAnnotations(DentalXrayDataset):
    """
    Extended dataset that also loads annotation files.
    
    This can be used for future enhancements:
    - ROI extraction based on annotation boxes
    - Attention-guided training
    - Visualization of fracture locations
    
    Annotation Format (text file):
        x1 y1  ← Line 1 start
        x2 y2  ← Line 1 end
        x3 y3  ← Line 2 start
        x4 y4  ← Line 2 end
        ...
    """
    
    def __init__(self, *args, annotation_ext: str = '.txt', **kwargs):
        """
        Initialize dataset with annotation loading.
        
        Args:
            annotation_ext: Annotation file extension
            *args, **kwargs: Arguments for parent DentalXrayDataset
        """
        super().__init__(*args, **kwargs)
        self.annotation_ext = annotation_ext
    
    def _load_annotation(self, img_path: str) -> np.ndarray:
        """
        Load annotation file for given image.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Numpy array of annotation points (N, 2) for N points
        """
        # Get annotation file path (same name, different extension)
        ann_path = Path(img_path).with_suffix(self.annotation_ext)
        
        if not ann_path.exists():
            return np.array([])
        
        # Read annotation file
        try:
            with open(ann_path, 'r') as f:
                lines = f.readlines()
            
            # Parse points (x, y)
            points = []
            for line in lines:
                line = line.strip()
                if line:
                    x, y = map(float, line.split())
                    points.append([x, y])
            
            return np.array(points)
        
        except Exception as e:
            print(f"Warning: Failed to load annotation {ann_path}: {str(e)}")
            return np.array([])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, np.ndarray]:
        """
        Get dataset item with annotations.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image_tensor, label, annotation_points)
        """
        image, label = super().__getitem__(idx)
        
        # Load annotations
        img_path = self.image_paths[idx]
        annotations = self._load_annotation(img_path)
        
        return image, label, annotations


# Example usage
if __name__ == "__main__":
    from augmentation import get_train_transforms, get_val_transforms
    
    # Dataset path
    root_dir = "c:/Users/maspe/OneDrive/Masaüstü/masterthesis/Dataset_2021/Dataset_2021/Dataset"
    
    # Test basic dataset loading
    print("=== Testing Basic Dataset ===")
    dataset = DentalXrayDataset(
        root_dir=root_dir,
        split='all',
        transform=get_train_transforms(image_size=640)
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test data loading
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label} ({'Fractured' if label == 1 else 'Healthy'})")
    
    # Test class weights
    class_weights = dataset.get_class_weights()
    print(f"\nClass weights: {class_weights}")
    
    # Test sample weights
    sample_weights = dataset.get_sample_weights()
    print(f"Sample weights (first 5): {sample_weights[:5]}")
    
    print("\n=== Dataset loading test PASSED ===")
