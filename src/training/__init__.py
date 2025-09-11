# src/training/__init__.py
"""
Training package for HTR contrastive learning.
Contains trainer and monitoring classes.
"""


from .trainer import OptimizedContrastiveTrainer
from .monitor import ContrastiveMonitor, ContrastiveMetrics

__all__ = [
    'OptimizedContrastiveTrainer',
    'ContrastiveTrainer',  # Alias for compatibility
    'ContrastiveMonitor',
    'ContrastiveMetrics'
]
