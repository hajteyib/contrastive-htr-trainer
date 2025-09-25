# Fichier : src/data/dataset_contrastive.py (VERSION FINALE CORRIGÉE)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import random
from torchvision import transforms

# --- IMPORTATION DES BRIQUES NÉCESSAIRES ---
from .augmentations import ElasticDistortion, InkVariation, PaperNoise, HorizontalCrop

class HTRViewGenerator:
    """
    Crée une vue augmentée d'une image avec des paramètres spécifiques.
    """
    def __init__(self, height: int, crop_scale: Tuple[float, float], rotation: float, 
                 shear: float, elastic_alpha: int, brightness: float, 
                 ink_variation: bool, paper_noise: bool, horizontal_crop: bool):
        
        self.height = height
        
        transforms_list = [
            transforms.RandomResizedCrop(size=(height, height), scale=crop_scale, antialias=True),
            transforms.RandomAffine(degrees=rotation, shear=shear),
            transforms.ColorJitter(brightness=brightness, contrast=brightness),
        ]
        self.torch_transforms = transforms.Compose(transforms_list)

        self.elastic = ElasticDistortion(alpha=elastic_alpha, sigma=max(1, elastic_alpha / 5)) if elastic_alpha > 0 else None
        self.ink_variation = InkVariation(intensity_range=(0.7, 1.3), thickness_range=2) if ink_variation else None
        self.paper_noise = PaperNoise(noise_intensity=0.02) if paper_noise else None
        self.horizontal_crop = HorizontalCrop(min_width_ratio=0.7) if horizontal_crop else None

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image_np = np.array(image).astype(np.float32) / 255.0

        if self.elastic: image_np = self.elastic(image_np)
        if self.ink_variation: image_np = self.ink_variation(image_np)
        if self.paper_noise: image_np = self.paper_noise(image_np)

        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        
        # Inversion des couleurs avant ToTensor si le modèle attend du texte blanc
        # Pour l'instant, on suppose que le modèle gère ça.
        
        view = self.torch_transforms(image_pil)
        view = (view - 0.5) * 2.0
        
        # Retourner un tenseur avec 1 canal
        return view.mean(dim=0, keepdim=True)


class MultiViewDataset(Dataset):
    def __init__(self, data_list_file: str, split: str, config: Dict, smoke_test: bool = False):
        self.config = config
        
        list_file = Path(data_list_file)
        if not list_file.exists(): raise FileNotFoundError(f"{list_file} not found.")
            
        with open(list_file, "r") as f: all_paths = [Path(line.strip()) for line in f if line.strip()]
        if smoke_test: all_paths = all_paths[:200]

        self.image_paths = self._create_splits(all_paths, split)
        print(f"-> Split '{split}' initialized with {len(self.image_paths)} images.")

        self.view_generators = []
        aug_configs = self.config['augmentation']['configs']
        for view_type in self.config['augmentation']['view_types']:
            cfg = aug_configs.get(view_type, {})
            self.view_generators.append(
                HTRViewGenerator(
                    height=self.config['data']['target_height'],
                    crop_scale=tuple(cfg.get('crop_scale', [0.8, 1.0])),
                    rotation=cfg.get('rotation', 0),
                    shear=cfg.get('shear', 0),
                    elastic_alpha=cfg.get('elastic_alpha', 0),
                    brightness=cfg.get('brightness', 0),
                    ink_variation=cfg.get('ink_variation', False),
                    paper_noise=cfg.get('paper_noise', False),
                    horizontal_crop=cfg.get('horizontal_crop', False)
                )
            )
        print(f"-> Créé {len(self.view_generators)} générateurs de vues.")

    def _create_splits(self, paths: List[Path], split: str):
        random.seed(42); random.shuffle(paths)
        train_size, val_size = int(0.8 * len(paths)), int(0.1 * len(paths))
        if split == 'train': return paths[:train_size]
        elif split == 'val': return paths[train_size : train_size + val_size]
        else: return paths[train_size + val_size:]

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("L")
            
            views = [gen(image) for gen in self.view_generators]
            
            width_ratio = torch.tensor([image.width / self.config['data']['target_height']], dtype=torch.float32)
            density = torch.tensor([1 - (np.array(image).astype(np.float32) / 255.0).mean()], dtype=torch.float32)

            return tuple(views) + (width_ratio, density)
        except Exception as e:
            print(f"ERREUR au chargement de {image_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

def create_contrastive_dataloaders(config: Dict, smoke_test: bool):
    """Crée les DataLoaders pour l'entraînement contrastif."""
    data_cfg = config['data']
    train_dataset = MultiViewDataset(data_cfg['data_list_file'], 'train', config, smoke_test)
    val_dataset = MultiViewDataset(data_cfg['data_list_file'], 'val', config, smoke_test)
    
    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True, 
                              num_workers=data_cfg['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg['batch_size'], 
                            num_workers=data_cfg['num_workers'] // 2, pin_memory=True)
                             
    return train_loader, val_loader