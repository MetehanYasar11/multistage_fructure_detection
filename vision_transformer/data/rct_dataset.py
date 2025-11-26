"""
PyTorch Dataset for RCT Classification Dataset

Single-tooth classification:
- Fractured (label=1): Teeth with broken endodontic instruments
- Healthy (label=0): Teeth without fractures

Author: Master's Thesis Project
Date: November 23, 2025
"""

import os
import json
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class RCTDataset(Dataset):
    """
    PyTorch Dataset for RCT Classification.
    
    Directory Structure:
        root_dir/
            fractured/
                0001_tooth00.jpg
                0002_tooth01.jpg
                ...
            healthy/
                0003_tooth00.jpg
                0004_tooth01.jpg
                ...
            train_val_test_split.json
    
    Attributes:
        root_dir: Path to RCT dataset root directory
        split: Dataset split ('train', 'val', 'test', or 'all')
        transform: Albumentations transform pipeline
        image_size: Target image size (if not using transform)
    """
    
    def __init__(
        self,
        root_dir: str = "RCT_classification_dataset",
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 640,
        split_file: Optional[str] = None
    ):
        """
        Initialize RCT Dataset.
        
        Args:
            root_dir: Path to RCT dataset root directory
            split: Dataset split ('train', 'val', 'test', or 'all')
            transform: Albumentations transform pipeline
            image_size: Target image size for resizing
            split_file: Path to split JSON (default: root_dir/train_val_test_split.json)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Default split file location
        if split_file is None:
            split_file = self.root_dir / "train_val_test_split.json"
        
        # Load all images and labels
        self.image_paths, self.labels = self._load_dataset(split_file)
        
        print(f"RCT Dataset - Split '{split}':")
        print(f"  Total: {len(self.image_paths)} images")
        print(f"  Fractured: {sum(self.labels)} | Healthy: {len(self.labels) - sum(self.labels)}")
    
    def _load_dataset(self, split_file: Path) -> Tuple[List[str], List[int]]:
        """
        Load image paths and labels from split file.
        
        Args:
            split_file: Path to train_val_test_split.json
            
        Returns:
            Tuple of (image_paths, labels)
        """
        # Load split
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        if self.split == 'all':
            # Load all images
            image_names = split_data['train'] + split_data['val'] + split_data['test']
        else:
            if self.split not in split_data:
                raise ValueError(f"Split '{self.split}' not found in {split_file}")
            image_names = split_data[self.split]
        
        # Build full paths and labels
        image_paths = []
        labels = []
        
        fractured_dir = self.root_dir / "fractured"
        healthy_dir = self.root_dir / "healthy"
        
        for img_name in image_names:
            # Check if in fractured or healthy folder
            fractured_path = fractured_dir / img_name
            healthy_path = healthy_dir / img_name
            
            if fractured_path.exists():
                image_paths.append(str(fractured_path))
                labels.append(1)  # Fractured
            elif healthy_path.exists():
                image_paths.append(str(healthy_path))
                labels.append(0)  # Healthy
            else:
                print(f"Warning: Image not found: {img_name}")
        
        return image_paths, labels
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get single item from dataset.
        
        Args:
            idx: Index of item
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Load image using PIL (handles Windows paths with Turkish characters)
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Convert to numpy
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: normalize (and optionally resize)
            if self.image_size is not None:
                image = Image.fromarray(image).resize((self.image_size, self.image_size))
                image = np.array(image)
            
            image = image / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        label = self.labels[idx]
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for WeightedRandomSampler.
        
        Returns:
            Tensor of class weights [weight_healthy, weight_fractured]
        """
        n_fractured = sum(self.labels)
        n_healthy = len(self.labels) - n_fractured
        
        # Inverse frequency weighting
        weight_fractured = len(self.labels) / (2 * n_fractured) if n_fractured > 0 else 0
        weight_healthy = len(self.labels) / (2 * n_healthy) if n_healthy > 0 else 0
        
        return torch.tensor([weight_healthy, weight_fractured], dtype=torch.float32)
    
    def get_sample_weights(self) -> List[float]:
        """
        Get sample weights for WeightedRandomSampler.
        
        Returns:
            List of weights for each sample
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label].item() for label in self.labels]
        return sample_weights


def test_rct_dataset():
    """Test RCT dataset loading"""
    print("Testing RCT Dataset...")
    
    dataset = RCTDataset(
        root_dir="RCT_classification_dataset",
        split='train',
        image_size=224
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading first item
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label} ({'Fractured' if label == 1 else 'Healthy'})")
    
    # Test class weights
    class_weights = dataset.get_class_weights()
    print(f"\nClass weights: {class_weights}")
    
    print("\n✅ RCT Dataset test passed!")


if __name__ == "__main__":
    test_rct_dataset()
