# src/data/__init__.py
"""
Data package for HTR contrastive learning.
Contains dataset classes and augmentations.
"""

from .dataset import FinalHTRDataset, create_optimized_dataloaders
from .augmentations import OptimizedHTRAugmentation, SmartResize

__all__ = [
    'RealHTRDataset',
    'create_optimized_dataloaders',
    'OptimizedHTRAugmentation',
    'SmartResize'
]