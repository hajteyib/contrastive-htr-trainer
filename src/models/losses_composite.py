# Fichier : src/models/losses_composite.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# On réutilise les briques de base de notre premier fichier de loss
from .losses import InfoNCE, EnhancedMomentumQueue

class CompositeLoss(nn.Module):
    """
    Calcule une perte composite combinant la loss contrastive InfoNCE
    avec des loss auxiliaires pour guider l'apprentissage.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config['loss']
        
        # --- Loss Principale : Contrastive ---
        self.temp = self.config.get('temperature', 0.07)
        self.contrastive_loss = InfoNCE(temperature=self.temp)
        
        # --- Losses Auxiliaires ---
        self.width_loss = nn.MSELoss()
        self.density_loss = nn.MSELoss()
        
        # --- Pondérations ---
        self.lambda_contrast = self.config.get('lambda_contrast', 1.0)
        self.lambda_width = self.config.get('lambda_width', 0.0)
        self.lambda_density = self.config.get('lambda_density', 0.0)

    def forward(self,
                anchor_projections: List[torch.Tensor],
                positive_projections: List[torch.Tensor],
                predicted_widths: List[torch.Tensor],
                predicted_densities: List[torch.Tensor],
                target_widths: torch.Tensor,
                target_densities: torch.Tensor,
                queue: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        total_contrast_loss = 0
        
        # --- Calcul de la Loss Contrastive ---
        # On calcule la loss entre chaque ancre et son positif correspondant
        for anchor_proj, positive_proj in zip(anchor_projections, positive_projections):
            keys = torch.cat([positive_proj, queue], dim=0)
            total_contrast_loss += self.contrastive_loss(anchor_proj, keys)
        
        # Moyenne sur le nombre de vues
        contrast_loss = total_contrast_loss / len(anchor_projections)

        # --- Calcul des Losses Auxiliaires ---
        # On ne prend que la prédiction de la première vue (ancre) pour les tâches auxiliaires
        loss_w = self.width_loss(predicted_widths[0], target_widths) if self.lambda_width > 0 else 0
        loss_d = self.density_loss(predicted_densities[0], target_densities) if self.lambda_density > 0 else 0
        
        # --- Calcul de la Loss Totale Pondérée ---
        total_loss = (self.lambda_contrast * contrast_loss +
                      self.lambda_width * loss_w +
                      self.lambda_density * loss_d)
                      
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrast_loss.detach(),
            'width_loss': loss_w.detach() if isinstance(loss_w, torch.Tensor) else torch.tensor(0),
            'density_loss': loss_d.detach() if isinstance(loss_d, torch.Tensor) else torch.tensor(0)
        }