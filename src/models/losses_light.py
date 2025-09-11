# Fichier à créer : src/models/losses_light.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# On peut réutiliser les briques de base de notre fichier de loss original
# Assurez-vous que ce fichier existe et est importable
from .losses import InfoNCE, EnhancedMomentumQueue 

class LightweightContrastiveLoss(nn.Module):
    """
    Une loss contrastive simplifiée qui ne fonctionne que sur les features globales,
    adaptée à notre CRNN_CNN_Backbone.
    """
    def __init__(self, global_dim: int, loss_config: Dict[str, Any]):
        super().__init__()
        
        self.global_loss = InfoNCE(loss_config.get('global_temp', 0.2))
        
        self.global_queue = EnhancedMomentumQueue(
            dim=global_dim, 
            queue_size=loss_config.get('queue_size', 8192),
            diversity_threshold=loss_config.get('diversity_threshold', 0.95)
        )

    def forward(self, anchor_features: Dict[str, torch.Tensor], positive_features: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        anchor_global = anchor_features['global']
        positive_global = positive_features['global']

        # Mettre à jour la queue avec les nouvelles features
        self.global_queue.update(anchor_global)
        
        # Obtenir les négatifs
        queue_negatives = self.global_queue.get_negatives(anchor_global.device)
        
        # Construire les clés pour la loss InfoNCE
        keys = torch.cat([positive_global, queue_negatives], dim=0)
        
        # Calculer la loss
        loss = self.global_loss(anchor_global, keys)
        
        return {'total_loss': loss, 'global_loss': loss.detach()}