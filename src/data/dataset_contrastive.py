# Fichier : src/data/dataset_contrastive.py

import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from PIL import Image
from pathlib import Path
from typing import List, Optional, Dict
import random

# On importe la NOUVELLE fonction qui crée notre pipeline
from .augmentations import get_htr_transform

class MultiViewDataset(Dataset):
    def __init__(self, data_list_file: str, split: str, config: Dict, smoke_test: bool = False):
        self.config = config
        
        with open(data_list_file, "r") as f:
            all_paths = [Path(line.strip()) for line in f if line.strip()]
        if smoke_test: all_paths = all_paths[:200]

        self.image_paths = self._create_splits(all_paths, split)
        print(f"-> Split '{split}' initialisé avec {len(self.image_paths)} images.")

        # Création des pipelines de transformation pour chaque vue
        self.view_transforms = [get_htr_transform(config, vt) for vt in config['augmentation']['view_types']]
        print(f"-> Créé {len(self.view_transforms)} pipelines de transformation.")

    def _create_splits(self, paths: List[Path], split: str):
        random.seed(42); random.shuffle(paths)
        train_size, val_size = int(0.8 * len(paths)), int(0.1 * len(paths))
        if split == 'train': return paths[:train_size]
        else: return paths[train_size:train_size + val_size] # On combine val et test pour la validation

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[tuple]:
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("L")
            views = [transform(image) for transform in self.view_transforms]
            
            # Cibles pour les loss auxiliaires
            width_ratio = torch.tensor([image.width / self.config['data']['target_height']], dtype=torch.float32)
            density = torch.tensor([1 - (np.array(image).astype(np.float32) / 255.0).mean()], dtype=torch.float32)

            return tuple(views) + (width_ratio, density)
        except Exception:
            return None # Le collate s'en chargera

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