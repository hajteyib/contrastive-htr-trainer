# src/data/dataset.py

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

# On importe notre NOUVEAU pipeline d'augmentation HTR-spécifique
from .augmentations import HTRContrastiveTransform

class FinalHTRDataset(Dataset):
    """
    Dataset qui charge les images pré-traitées et applique les transformations
    contrastives HTR-spécifiques pour générer deux vues.
    """
    def __init__(self, data_list_file: str, split: str = 'train', transform: Optional[Any] = None):
        print(f"Initializing dataset for split: {split}")
        self.split = split
        self.transform = transform # Le transformateur est maintenant l'objet HTRContrastiveTransform
        
        list_file = Path(data_list_file)
        if not list_file.exists():
            raise FileNotFoundError(f"Fichier de liste non trouvé : {list_file}. Assurez-vous d'avoir lancé prepare_dataset.py.")
            
        print(f"Loading image paths from {list_file}...")
        with open(list_file, "r") as f:
            all_paths = [Path(line.strip()) for line in f if line.strip() and Path(line.strip()).exists()]
        
        self.image_paths = self._create_splits(all_paths)
        print(f"-> Split '{split}' initialized with {len(self.image_paths)} images.")

    def _create_splits(self, paths: List[Path]):
        random.seed(42)
        random.shuffle(paths)
        train_size = int(0.8 * len(paths))
        val_size = int(0.15 * len(paths))
        if self.split == 'train': return paths[:train_size]
        elif self.split == 'val': return paths[train_size:train_size + val_size]
        else: return paths[train_size + val_size:]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        image_path = self.image_paths[idx]
        try:
            # Charger avec PIL, car notre nouveau pipeline d'augmentation commence avec PIL
            image = Image.open(image_path).convert("L")
            
            if self.transform:
                # Chaque appel à self.transform crée une vue aléatoire différente
                # car les opérations internes (random.random, etc.) sont stochastiques.
                view1 = self.transform(image)
                view2 = self.transform(image)
            else: # Cas sans augmentation (pour le test set final)
                # Simple redimensionnement et conversion en tenseur
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                # On redimensionne à la hauteur cible, mais on garde la largeur variable
                w, h = image.size
                new_width = int(w * self.transform.height / h)
                image = image.resize((new_width, self.transform.height), Image.LANCZOS)
                view1 = transform_test(image)
                view2 = view1.clone() # Pour le test, les deux vues sont identiques

            return {"anchor": view1, "positive": view2}
        except Exception as e:
            print(f"ERROR: Could not load/process {image_path}, skipping. Reason: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

def pad_collate_fn(batch):
    """
    Assemble un batch d'images de largeurs variables en les paddant.
    Cette fonction est de nouveau CRUCIALE car nos augmentations HTR préservent la largeur.
    """
    batch = [item for item in batch if item is not None]
    if not batch: return None
    
    anchors = [item['anchor'] for item in batch]
    positives = [item['positive'] for item in batch]
    
    # Trouver la largeur maximale dans tout le batch (ancres et positifs)
    max_w = 0
    for img_list in [anchors, positives]:
        for img in img_list:
            if img.shape[2] > max_w:
                max_w = img.shape[2]

    # Appliquer le padding
    padded_anchors = [torch.nn.functional.pad(img, (0, max_w - img.shape[2]), value=0.0) for img in anchors]
    padded_positives = [torch.nn.functional.pad(img, (0, max_w - img.shape[2]), value=0.0) for img in positives]

    return {'anchor': torch.stack(padded_anchors), 'positive': torch.stack(padded_positives)}

def create_optimized_dataloaders(data_list_file: str, batch_size: int, num_workers: int, target_height: int, pin_memory: bool, **kwargs):
    
    # On instancie notre NOUVEAU pipeline de transformation HTR-spécifique
    # Il n'a pas besoin de 'augmentation_strength', il est déjà configuré
    htr_transform = HTRContrastiveTransform(height=target_height)
    
    train_dataset = FinalHTRDataset(data_list_file, 'train', transform=htr_transform)
    val_dataset = FinalHTRDataset(data_list_file, 'val', transform=htr_transform)
    test_dataset = FinalHTRDataset(data_list_file, 'test', transform=htr_transform) # On passe la hauteur cible
    
    persistent_workers = True if num_workers > 0 else False
    
    # Le collate_fn est de nouveau indispensable
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=pin_memory, drop_last=True, collate_fn=pad_collate_fn, 
                              persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers//2, 
                            pin_memory=pin_memory, collate_fn=pad_collate_fn, 
                            persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers//2, 
                             pin_memory=pin_memory, collate_fn=pad_collate_fn, 
                             persistent_workers=persistent_workers)
                             
    return train_loader, val_loader, test_loader