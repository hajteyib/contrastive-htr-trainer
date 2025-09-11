# src/data/dataset.py (VERSION FINALE, DÉFINITIVE ET PROPRE)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import random
import os
from PIL import Image
from torchvision import transforms

from .augmentations import OptimizedHTRAugmentation

class SimCLRTransform:
    def __init__(self, size: int, htr_augmenter: Optional[Any]):
        self.htr_augmenter = htr_augmenter
        self.size = size

        # Les transformations de torchvision sont plus robustes et standardisées
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), # Convertit en Tenseur [C, H, W] et normalise à [0, 1]
        ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Appliquer les augmentations complexes d'abord, si elles existent
        if self.htr_augmenter:
            image_np = np.array(image)
            image_np_aug = self.htr_augmenter(image=image_np)['image']
            image = Image.fromarray((image_np_aug * 255).astype(np.uint8))
        
        return self.base_transform(image)

class FinalHTRDataset(Dataset):
    def __init__(self, data_list_file: str, split: str = 'train', transform: Optional[Any] = None):
        self.split = split
        self.transform = transform
        list_file = Path(data_list_file)
        if not list_file.exists(): raise FileNotFoundError(f"{list_file} not found.")
            
        with open(list_file, "r") as f: all_paths = [Path(line.strip()) for line in f if line.strip()]
        self.image_paths = self._create_splits(all_paths)
        print(f"-> Split '{split}' initialized with {len(self.image_paths)} images.")

    def _create_splits(self, paths: List[Path]):
        random.seed(42); random.shuffle(paths)
        train_size, val_size = int(0.8 * len(paths)), int(0.15 * len(paths))
        if self.split == 'train': return paths[:train_size]
        elif self.split == 'val': return paths[train_size:train_size + val_size]
        else: return paths[train_size + val_size:]

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        image_path = self.image_paths[idx]
        try:
            # On charge avec PIL car c'est ce que torchvision attend
            image = Image.open(image_path).convert("L")
            
            if self.transform:
                view1 = self.transform(image)
                view2 = self.transform(image)
            else:
                view1 = transforms.ToTensor()(image)
                view2 = transforms.ToTensor()(image)

            return {"anchor": view1, "positive": view2}
        except Exception as e:
            print(f"ERROR: Could not load/process {image_path}, skipping. Reason: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

# La collate_fn n'est plus nécessaire car SimCLRTransform produit des images de taille fixe
# mais nous la gardons au cas où le test set aurait des tailles variables
def pad_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.default_collate(batch)


def create_optimized_dataloaders(data_list_file, batch_size, num_workers, augmentation_strength, target_height, pin_memory, **kwargs):
    htr_aug = OptimizedHTRAugmentation(
        geometric_prob=augmentation_strength * 0.6,
        photometric_prob=augmentation_strength * 0.8,
        structural_prob=augmentation_strength * 0.4
    )
    
    train_transform = SimCLRTransform(size=target_height, htr_augmenter=htr_aug)
    val_transform = SimCLRTransform(size=target_height, htr_augmenter=htr_aug)
    test_transform = transforms.Compose([transforms.Resize((target_height, target_height)), transforms.ToTensor()])

    train_dataset = FinalHTRDataset(data_list_file, 'train', transform=train_transform)
    val_dataset = FinalHTRDataset(data_list_file, 'val', transform=val_transform)
    test_dataset = FinalHTRDataset(data_list_file, 'test', transform=test_transform)
    
    persistent_workers = num_workers > 0
    # On retire collate_fn car les tenseurs ont une taille fixe
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers//2, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers//2, pin_memory=pin_memory)
                             
    return train_loader, val_loader, test_loader