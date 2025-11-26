"""
Data Augmentation for Dental X-Ray Fracture Detection

This module provides Albumentations-based augmentation pipelines
optimized for panoramic dental X-ray images.

Key Features:
- CLAHE preprocessing for brightness normalization (critical for 72% variance)
- Medical imaging-appropriate geometric transforms
- Intensity augmentations for robustness
- Separate train/val/test pipelines

Author: Master's Thesis Project
Date: October 28, 2025
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(image_size: int = 640, clahe_enabled: bool = True) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        image_size: Target image size. Can be:
            - int: Square resize (e.g., 640 → 640×640)
            - tuple: (height, width) for panoramic (e.g., (1400, 2800))
        clahe_enabled: Enable CLAHE preprocessing (recommended: True)
        
    Returns:
        Albumentations Compose object for training transforms
        
    Pipeline:
        1. CLAHE - Adaptive histogram equalization for brightness normalization
        2. Geometric - Rotation, flip, elastic transforms
        3. Intensity - Brightness/contrast, noise
        4. Resize and normalize (ImageNet stats for pretrained models)
    """
    transforms = []
    
    # 1. CLAHE - CRITICAL for dental X-rays (72% brightness variance in Fractured class)
    if clahe_enabled:
        transforms.append(
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=1.0  # Always apply
            )
        )
    
    # 2. Geometric Augmentations
    # Panoramic X-rays tolerate small rotations
    transforms.extend([
        A.Rotate(
            limit=15,  # ±15 degrees (dental structures are mostly horizontal)
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        
        # Horizontal flip - Dental structures are symmetric
        A.HorizontalFlip(p=0.3),
        
        # Elastic transforms - Simulate dental anatomy variations
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            alpha_affine=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.3
        ),
        
        # Grid distortion - Subtle geometric variations
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.3
        ),
    ])
    
    # 3. Intensity Augmentations
    # Handle brightness variance between classes
    transforms.extend([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Gaussian noise - Simulate X-ray sensor noise
        A.GaussNoise(
            var_limit=(10.0, 30.0),
            p=0.3
        ),
    ])
    
    # 4. Resize and Normalize
    # Handle both square and panoramic sizes
    if isinstance(image_size, (list, tuple)):
        height, width = image_size
    else:
        height, width = image_size, image_size
    
    transforms.extend([
        A.Resize(height=height, width=width),
        
        # ImageNet normalization (for pretrained models)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # Convert to PyTorch tensor
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)


def get_val_transforms(image_size: int = 640, clahe_enabled: bool = True) -> A.Compose:
    """
    Get validation/test augmentation pipeline.
    
    Args:
        image_size: Target image size. Can be:
            - int: Square resize (e.g., 640 → 640×640)
            - tuple: (height, width) for panoramic (e.g., (1400, 2800))
        clahe_enabled: Enable CLAHE preprocessing (recommended: True)
        
    Returns:
        Albumentations Compose object for validation/test transforms
        
    Pipeline:
        No augmentations - Only preprocessing and normalization
        1. CLAHE (if enabled)
        2. Resize
        3. Normalize
    """
    transforms = []
    
    # 1. CLAHE - Apply same preprocessing as training
    if clahe_enabled:
        transforms.append(
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=1.0
            )
        )
    
    # 2. Resize and Normalize
    # Handle both square and panoramic sizes
    if isinstance(image_size, (list, tuple)):
        height, width = image_size
    else:
        height, width = image_size, image_size
    
    transforms.extend([
        A.Resize(height=height, width=width),
        
        # ImageNet normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # Convert to PyTorch tensor
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)


def get_test_transforms(image_size: int = 640, clahe_enabled: bool = True) -> A.Compose:
    """
    Get test augmentation pipeline (same as validation).
    
    Args:
        image_size: Target image size (height and width)
        clahe_enabled: Enable CLAHE preprocessing (recommended: True)
        
    Returns:
        Albumentations Compose object for test transforms
    """
    return get_val_transforms(image_size=image_size, clahe_enabled=clahe_enabled)


def get_tta_transforms(image_size: int = 640, clahe_enabled: bool = True) -> list:
    """
    Get Test-Time Augmentation (TTA) pipelines.
    
    TTA Strategy:
        - Original image
        - Horizontal flip
        - ±5° rotation
        
    Average predictions from all variants for robust inference.
    
    Args:
        image_size: Target image size (height and width)
        clahe_enabled: Enable CLAHE preprocessing (recommended: True)
        
    Returns:
        List of Albumentations Compose objects for TTA
    """
    tta_list = []
    
    # Base transforms (CLAHE + resize + normalize)
    base_transforms = []
    if clahe_enabled:
        base_transforms.append(
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
        )
    
    # 1. Original
    tta_list.append(
        A.Compose(
            base_transforms + [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    )
    
    # 2. Horizontal flip
    tta_list.append(
        A.Compose(
            base_transforms + [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    )
    
    # 3. Rotate +5°
    tta_list.append(
        A.Compose(
            base_transforms + [
                A.Rotate(limit=(5, 5), border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    )
    
    # 4. Rotate -5°
    tta_list.append(
        A.Compose(
            base_transforms + [
                A.Rotate(limit=(-5, -5), border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    )
    
    return tta_list


# Example usage
if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    
    # Create a dummy X-ray image
    dummy_image = np.random.randint(0, 255, (1435, 2900, 3), dtype=np.uint8)
    
    # Test train transforms
    train_tfm = get_train_transforms(image_size=640)
    augmented = train_tfm(image=dummy_image)
    print(f"Train augmented image shape: {augmented['image'].shape}")
    
    # Test val transforms
    val_tfm = get_val_transforms(image_size=640)
    augmented = val_tfm(image=dummy_image)
    print(f"Val augmented image shape: {augmented['image'].shape}")
    
    # Test TTA
    tta_tfms = get_tta_transforms(image_size=640)
    print(f"TTA variants: {len(tta_tfms)}")
    for i, tfm in enumerate(tta_tfms):
        augmented = tfm(image=dummy_image)
        print(f"  TTA {i+1} shape: {augmented['image'].shape}")
