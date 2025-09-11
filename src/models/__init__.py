# src/models/__init__.py
"""
Models package for HTR contrastive learning.
Contains encoder architectures and loss functions.
"""

from .encoder import OptimizedHTREncoder
from .losses import OptimizedContrastiveLoss, InfoNCE

__all__ = [
    'OptimizedHTREncoder',
    'OptimizedContrastiveLoss', 
    'InfoNCE'
]