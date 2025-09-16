# Fichier : src/data/dataset_contrastive.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional , Dict 
import random
from PIL import Image
from torchvision import transforms
import os 

class SimCLR_Transform:
    """
    Pipeline de transformation SimCLR qui génère deux vues augmentées d'une image.
    C'est le cœur de la stratégie pour forcer l'apprentissage sémantique.
    """
    def __init__(self, size: int):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(size, size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            # Augmentations de couleur agressives pour détruire les raccourcis
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            # Un flou gaussien est souvent utilisé dans SimCLR
            transforms.GaussianBlur(kernel_size=23),
            # Normalisation standard
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Appliquer la transformation deux fois pour obtenir deux vues indépendantes
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2


class ContrastiveDataset(Dataset):
    """Dataset pour le pré-entraînement contrastif."""
    def __init__(self, data_list_file: str, split: str, transform: SimCLR_Transform, smoke_test: bool = False):
        self.transform = transform
        
        list_file = Path(data_list_file)
        if not list_file.exists():
            raise FileNotFoundError(f"Fichier de liste non trouvé : {list_file}")

        with open(list_file, "r") as f:
            all_paths = [Path(line.strip()) for line in f if line.strip()]

        if smoke_test:
            print("--- MODE SMOKE TEST : Utilisation de 100 images seulement. ---")
            all_paths = all_paths[:100]

        self.image_paths = self._create_splits(all_paths, split)
        print(f"-> Split '{split}' initialisé avec {len(self.image_paths)} images.")

    def _create_splits(self, paths: List[Path], split: str):
        random.seed(42)
        random.shuffle(paths)
        train_size = int(0.8 * len(paths))
        # Pour le pré-entraînement, on peut se permettre un plus petit set de validation
        val_size = int(0.1 * len(paths))
        
        if split == 'train': return paths[:train_size]
        elif split == 'val': return paths[train_size : train_size + val_size]
        else: return paths[train_size + val_size:]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        image_path = self.image_paths[idx]
        try:
            # Charger avec PIL, convertir en niveaux de gris
            image = Image.open(image_path).convert("L")
            
            # Appliquer les transformations pour obtenir les deux vues
            view1, view2 = self.transform(image)

            return {"view1": view1, "view2": view2}
        except Exception as e:
            print(f"ERREUR au chargement de {image_path}: {e}. On passe à l'image suivante.")
            return self.__getitem__(random.randint(0, len(self) - 1))


def create_contrastive_dataloaders(data_list_file: str, batch_size: int, num_workers: int, target_height: int, smoke_test: bool) -> Tuple[DataLoader, DataLoader]:
    """Crée les DataLoaders pour l'entraînement contrastif."""
    
    # Créer le pipeline de transformation
    transform = SimCLR_Transform(size=target_height)

    # Créer les datasets
    train_dataset = ContrastiveDataset(data_list_file, 'train', transform, smoke_test)
    val_dataset = ContrastiveDataset(data_list_file, 'val', transform, smoke_test)
    
    # Options pour les DataLoaders sur le cluster
    persistent_workers = True if num_workers > 0 else False
    if smoke_test:
        batch_size = min(batch_size, 16) # Utiliser un petit batch size pour le test
        print(f"--- SMOKE TEST: Batch size réduit à {batch_size} ---") 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=True,
        persistent_workers=persistent_workers
    )
                             
    return train_loader, val_loader