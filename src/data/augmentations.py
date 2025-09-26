# Fichier : src/data/augmentations.py

import torch
from torchvision import transforms
from typing import Dict, Tuple

def get_htr_transform(config: Dict, view_config_name: str) -> transforms.Compose:
    """
    Construit un pipeline de transformation robuste basé sur torchvision.
    """
    aug_cfg = config['augmentation']['configs'][view_config_name]
    data_cfg = config['data']
    
    # Utiliser RandomChoice pour appliquer une seule des augmentations HTR "complexes"
    # Cela évite de sur-déformer l'image.
    # Note: Pour l'instant, nous nous concentrons sur les augmentations de base.
    # On pourra ajouter les augmentations HTR ici plus tard si nécessaire.

    pipeline = transforms.Compose([
        # --- Transformations Structurelles ---
        transforms.RandomResizedCrop(
            size=(data_cfg['target_height'], data_cfg['target_height']), 
            scale=tuple(aug_cfg.get('crop_scale', (0.5, 1.0))), 
            antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        
        # --- Transformations Géométriques (HTR-spécifiques) ---
        transforms.RandomAffine(
            degrees=aug_cfg.get('rotation', 0),
            shear=aug_cfg.get('shear', 0)
        ),
        
        # --- Transformations Photométriques ---
        transforms.ColorJitter(
            brightness=aug_cfg.get('brightness', 0.4),
            contrast=aug_cfg.get('contrast', 0.4)
        ),
        
        # --- Conversion Finale ---
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalise à [-1, 1]
    ])
    return pipeline