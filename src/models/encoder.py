# Fichier : src/models/encoder.py (VERSION FINALE FLEXIBLE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from typing import Dict

# --- Modules d'Attention ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x

# --- Encodeur Principal ---
class ConfigurableEncoder(nn.Module):
    """
    Architecture ResNet34 modifiée, configurable pour les expériences.
    Prend la configuration complète pour accéder à toutes les sections nécessaires.
    """
    def __init__(self, config: Dict):
        super().__init__()
        # On stocke la configuration complète pour y accéder facilement
        self.config = config 
        
        # --- Récupération des sous-configurations ---
        model_config = self.config['model']
        loss_config = self.config['loss']

        # --- Backbone ResNet34 ---
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # --- Pooling OCR-Spécifique ---
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # --- Module d'Attention (Configurable) ---
        attention_type = model_config.get('attention_type', 'none').lower()
        if attention_type == 'cbam':
            self.attention = CBAM(512)
        elif attention_type == 'se':
            self.attention = SEBlock(512)
        else:
            self.attention = nn.Identity()
            
        # --- Tête de Projection Profonde (Configurable) ---
        head_config = model_config['projection_head']
        layers = []
        input_dim = 512 # Sortie de Layer4 et Attention
        
        for i, layer_conf in enumerate(head_config['layers']):
            output_dim = layer_conf['dim']
            layers.append(nn.Linear(input_dim, output_dim))
            # Pas de BN/Activation sur la toute dernière couche
            if i < len(head_config['layers']) - 1:
                layers.append(nn.BatchNorm1d(output_dim))
                layers.append(nn.ReLU(inplace=True))
                if layer_conf.get('dropout', 0) > 0:
                    layers.append(nn.Dropout(p=layer_conf['dropout']))
            input_dim = output_dim
            
        self.projection_head = nn.Sequential(*layers)
        
        # --- Têtes pour les Loss Auxiliaires (liées à la config de la loss) ---
        self.width_predictor = nn.Linear(512, 1) if loss_config.get('lambda_width', 0) > 0 else nn.Identity()
        self.density_predictor = nn.Linear(512, 1) if loss_config.get('lambda_density', 0) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x est une vue [B, 1, H, W]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.attention(x)
        
        # Pooling global pour les têtes
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        # Calcul des sorties
        projection = self.projection_head(pooled)
        predicted_width = self.width_predictor(pooled)
        predicted_density = self.density_predictor(pooled)
        
        return {
            'projection': projection,       # Pour la loss contrastive
            'features': pooled,             # Pour les loss auxiliaires
            'predicted_width': predicted_width,
            'predicted_density': predicted_density
        }