"""
Utility functions for Dental X-Ray Fracture Detection.

This module provides common utility functions for reproducibility,
device management, and helper functions.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get available device (CUDA if available, else CPU).
    
    Returns:
        torch.device: Device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    
    return device


__all__ = [
    'set_seed',
    'get_device',
]
