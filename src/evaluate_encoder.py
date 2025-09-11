# src/evaluate_encoder.py

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse

# Importer les classes nécessaires de votre projet
from models.encoder import OptimizedHTREncoder
from data.dataset import FinalHTRDataset, pad_collate_fn
from data.augmentations import OptimizedHTRAugmentation

def evaluate_contrastive_accuracy(model, dataloader, device):
    """
    Évalue la capacité du modèle à faire correspondre correctement deux vues
    augmentées de la même image parmi un batch.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Contrastive Accuracy"):
            if batch is None or batch['anchor'].nelement() == 0:
                continue
            
            # Anchor et Positive sont maintenant deux vues augmentées différentes
            anchors = batch['anchor'].to(device)
            positives = batch['positive'].to(device)
            
            batch_size = anchors.size(0)
            if batch_size < 2:
                continue

            # On ne réactive pas AMP ici pour une évaluation simple et stable
            anchor_embeds = model(anchors)['global']
            positive_embeds = model(positives)['global']
            
            anchor_embeds = torch.nn.functional.normalize(anchor_embeds, dim=1)
            positive_embeds = torch.nn.functional.normalize(positive_embeds, dim=1)
            
            # Calcul de la matrice de similarité [batch_size, batch_size]
            all_sims = torch.matmul(anchor_embeds, positive_embeds.T)
            
            # La prédiction est l'index de la plus haute similarité pour chaque ancre
            predictions = torch.argmax(all_sims, dim=1)
            
            # Les labels corrects sont sur la diagonale (0, 1, 2, ...)
            correct_labels = torch.arange(batch_size, device=device)
            
            total_correct += (predictions == correct_labels).sum().item()
            total_samples += batch_size

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained HTR encoder.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--data_list", type=str, default="/home/jovyan/buckets/ehsebou-manuscrits/final_valid_paths.txt", help="Path to the master list of valid image paths.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Charger le modèle
    print(f"--- Loading Model ---")
    print(f"Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    
    model_config = checkpoint['config']['model']
    model = OptimizedHTREncoder(
        global_dim=model_config['global_dim'],
        patch_dim=model_config['patch_dim']
    )
    
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    print("Model loaded successfully.")

    # 2. Créer le DataLoader de validation avec la même logique à 2 transformations
    print("\n--- Preparing Validation Data ---")
    augmentation_strength = 0.7 
    
    # On crée deux transformations distinctes, comme pour l'entraînement
    aug = OptimizedHTRAugmentation(
        geometric_prob=augmentation_strength * 0.6,
        photometric_prob=augmentation_strength * 0.8,
        structural_prob=augmentation_strength * 0.4
    )
    
    # On passe le tuple au dataset de validation
    val_dataset = FinalHTRDataset(args.data_list, 'val', augmentations=aug)
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=pad_collate_fn
    )

    # 3. Lancer l'évaluation
    print("\n--- Starting Evaluation ---")
    accuracy = evaluate_contrastive_accuracy(model, val_dataloader, DEVICE)

    print("\n--- Evaluation Complete ---")
    print(f"Model: {args.checkpoint}")
    print(f"Contrastive Accuracy (Top-1): {accuracy:.2f}%")
    print("---------------------------")
    print(f"Interpretation: Le modèle a correctement identifié la bonne paire (vue 1, vue 2) parmi {args.batch_size} possibilités dans {accuracy:.2f}% des cas.")