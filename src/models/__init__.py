# src/models/__init__.py
"""
Models package for HTR contrastive learning.
Contains encoder architectures and loss functions.
"""

from .crnn_backbone import CRNN_CNN_Backbone
from .encoder import ConfigurableEncoder
from .losses_composite import CompositeLoss
from .losses_light import LightweightContrastiveLoss
from .losses import OptimizedContrastiveLoss, InfoNCE

__all__ = [
    'OptimizedHTREncoder',
    'OptimizedContrastiveLoss', 
    'InfoNCE'
]