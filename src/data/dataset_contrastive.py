# Fichier : src/data/dataset_contrastive.py

import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import random
import numpy as np

# On importe la NOUVELLE classe qui crée notre pipeline
from .augmentations import HTRContrastiveTransform

class MultiViewDataset(Dataset):
    def __init__(self, data_list_file: str, split: str, config: Dict, smoke_test: bool = False):
        self.config = config
        
        with open(data_list_file, "r") as f: all_paths = [Path(line.strip()) for line in f if line.strip()]
        if smoke_test: all_paths = all_paths[:200]

        self.image_paths = self._create_splits(all_paths, split)
        print(f"-> Split '{split}' initialisé avec {len(self.image_paths)} images.")

        # Création des pipelines de transformation pour chaque vue
        self.view_generators = []
        aug_configs = self.config['augmentation']['configs']
        for view_type in self.config['augmentation']['view_types']:
            cfg = aug_configs.get(view_type, {})
            self.view_generators.append(
                HTRContrastiveTransform(
                    height=self.config['data']['target_height'],
                    crop_scale=tuple(cfg.get('crop_scale', [0.8, 1.0])), # N'est plus utilisé ici, mais gardé pour la forme
                    rotation=cfg.get('rotation', 0),
                    shear=cfg.get('shear', 0),
                    elastic_alpha=cfg.get('elastic_alpha', 0),
                    brightness=cfg.get('brightness', 0),
                    ink_variation=cfg.get('ink_variation', False),
                    paper_noise=cfg.get('paper_noise', False),
                    horizontal_crop=cfg.get('horizontal_crop', False)
                )
            )
        print(f"-> Créé {len(self.view_generators)} générateurs de vues HTR-spécifiques.")

    def _create_splits(self, paths: List[Path], split: str):
        random.seed(42); random.shuffle(paths)
        train_size, val_size = int(0.8 * len(paths)), int(0.1 * len(paths))
        if split == 'train': return paths[:train_size]
        else: return paths[train_size : train_size + val_size]

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[tuple]:
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("L")
            
            views = [gen(image) for gen in self.view_generators]
            
            width_ratio = torch.tensor([image.width / self.config['data']['target_height']], dtype=torch.float32)
            density = torch.tensor([1 - (np.array(image).astype(np.float32) / 255.0).mean()], dtype=torch.float32)

            return tuple(views) + (width_ratio, density)
        except Exception as e:
            print(f"ERREUR Critique au chargement de {image_path}: {e}")
            return None

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return default_collate(batch)

def create_contrastive_dataloaders(config: Dict, smoke_test: bool):
    data_cfg = config['data']
    train_dataset = MultiViewDataset(data_cfg['data_list_file'], 'train', config, smoke_test)
    val_dataset = MultiViewDataset(data_cfg['data_list_file'], 'val', config, smoke_test)
    
    train_loader = DataLoader(train_dataset, batch_size=data_cfg['batch_size'], shuffle=True, 
                              num_workers=data_cfg['num_workers'], pin_memory=True, drop_last=True, collate_fn=safe_collate)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg['batch_size'], 
                            num_workers=data_cfg['num_workers'] // 2, pin_memory=True, collate_fn=safe_collate)
                             
    return train_loader, val_loader