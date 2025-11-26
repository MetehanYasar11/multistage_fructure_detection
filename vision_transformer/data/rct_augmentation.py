"""
Heavy Augmentation for RCT Classification

Strong augmentation to prevent overfitting on small dataset (1041 samples)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_rct_train_transforms_heavy():
    """
    Heavy augmentation for preventing overfitting
    
    Applied to training set only
    """
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=30, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.6
        ),
        A.RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),
        
        # Color transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),
        
        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Cutout / Erasing
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.5
        ),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(),
    ])


def get_rct_val_transforms():
    """Validation transforms - minimal, just normalize"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    
    # Test augmentation
    print("Testing heavy augmentation...")
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    transform = get_rct_train_transforms_heavy()
    
    print("\nApplying augmentation 5 times:")
    for i in range(5):
        augmented = transform(image=dummy_img)
        img_tensor = augmented['image']
        print(f"  Aug {i+1}: shape={img_tensor.shape}, min={img_tensor.min():.3f}, max={img_tensor.max():.3f}")
    
    print("\nAugmentation test completed!")
