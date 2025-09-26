# Fichier : src/data/augmentations.py (VERSION FINALE ET COMPLÈTE)

import torch
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image
from torchvision import transforms

# ==============================================================================
# BLOCS DE BASE DES AUGMENTATIONS
# ==============================================================================

class ElasticDistortion:
    def __init__(self, alpha: float, sigma: float):
        self.alpha = alpha; self.sigma = sigma
    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.sigma) * self.alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return map_coordinates(image, indices, order=1, mode='reflect').reshape((h, w))

class InkVariation:
    def __init__(self, intensity_range: Tuple[float, float], thickness_range: int):
        self.intensity_range = intensity_range; self.thickness_range = thickness_range
    def __call__(self, image: np.ndarray) -> np.ndarray:
        res = image.copy()
        res[res < 0.8] = np.clip(res[res < 0.8] * random.uniform(*self.intensity_range), 0, 1)
        if random.random() < 0.5 and self.thickness_range > 1:
            k = random.randint(2, self.thickness_range)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            op = cv2.dilate if random.random() < 0.5 else cv2.erode
            res = op(res, kernel, iterations=1)
        return np.clip(res, 0, 1)

class PaperNoise:
    def __init__(self, noise_intensity: float):
        self.noise_intensity = noise_intensity
    def __call__(self, image: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_intensity, image.shape)
        return np.clip(image + noise, 0, 1)

class HorizontalCrop:
    def __init__(self, min_width_ratio: float):
        self.min_width_ratio = min_width_ratio
    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        crop_w = random.randint(int(w * self.min_width_ratio), w)
        start_x = random.randint(0, w - crop_w)
        cropped = image[:, start_x : start_x + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

# ==============================================================================
# LE PIPELINE D'AUGMENTATION FINAL QUE NOUS UTILISERONS
# ==============================================================================

class HTRContrastiveTransform:
    """
    Pipeline de transformation contrastive spécialisé pour le HTR.
    Prend une image PIL, retourne un tenseur PyTorch.
    """
    def __init__(self, height: int, crop_scale: Tuple[float, float], rotation: float, 
                 shear: float, elastic_alpha: int, brightness: float, 
                 ink_variation: bool, paper_noise: bool, horizontal_crop: bool):
        
        self.height = height
        
        # Transformations Torchvision (rapides et appliquées sur des tenseurs)
        self.torch_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=rotation, shear=shear),
            transforms.ColorJitter(brightness=brightness, contrast=brightness),
        ])

        # Transformations Numpy/CV2 (plus complexes)
        self.elastic = ElasticDistortion(alpha=elastic_alpha, sigma=max(1, elastic_alpha / 5)) if elastic_alpha > 0 else None
        self.ink_variation = InkVariation(intensity_range=(0.7, 1.3), thickness_range=2) if ink_variation else None
        self.paper_noise = PaperNoise(noise_intensity=0.02) if paper_noise else None
        self.horizontal_crop = HorizontalCrop(min_width_ratio=0.7) if horizontal_crop else None
        
        # Transformation finale
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # 1. Redimensionnement initial
        w, h = image.size
        new_width = int(w * self.height / h)
        image_resized = image.resize((new_width, self.height), Image.LANCZOS)
        
        # 2. Conversion en Numpy et inversion
        image_np = (255 - np.array(image_resized)).astype(np.float32) / 255.0
        
        # 3. Application des augmentations HTR-spécifiques (Numpy)
        if self.horizontal_crop and random.random() < 0.3: image_np = self.horizontal_crop(image_np)
        if self.ink_variation: image_np = self.ink_variation(image_np)
        if self.paper_noise: image_np = self.paper_noise(image_np)
        if self.elastic and random.random() < 0.5: image_np = self.elastic(image_np)

        # 4. Reconversion en PIL pour les transformations Torchvision
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        
        # 5. Application des transformations Torchvision
        image_transformed_pil = self.torch_transforms(image_pil)
        
        # 6. Conversion finale en tenseur et normalisation
        final_tensor = self.to_tensor(image_transformed_pil)
        final_tensor = self.normalize(final_tensor)
        
        return final_tensor