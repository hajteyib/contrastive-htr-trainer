# src/models/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from typing import Tuple, Dict

class SpatialAttention(nn.Module):
    """Module d'attention spatiale pour se concentrer sur les régions d'écriture."""
    def __init__(self, in_channels: int):
        super().__init__()
        hidden_channels = max(16, in_channels // 16)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        attention = F.relu(self.conv1(x))
        attention = self.sigmoid(self.conv2(attention))
        return identity * attention

class DilatedFPN(nn.Module):
    """Feature Pyramid Network amélioré avec des convolutions dilatées pour le HTR."""
    def __init__(self, in_channels_list: list, out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2))
    
    def forward(self, features: list) -> list:
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i-1] = laterals[i-1] + upsampled
        
        return [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]

class SimplePatchExtractor(nn.Module):
    """
    Extraction de patchs simple sur une grille fixe, adaptée aux images de taille fixe.
    """
    def __init__(self, patch_dim: int = 128, fpn_channels: int = 256, grid_size: int = 7):
        super().__init__()
        self.patch_dim = patch_dim
        self.grid_size = grid_size
        self.patch_processor = nn.Sequential(
            nn.Conv2d(fpn_channels, patch_dim, 1),
            nn.ReLU(inplace=True),
        )
        
        ### CORRECTION FINALE ###
        # Au lieu d'une projection linéaire fixe, nous utilisons un pooling adaptatif
        # pour forcer chaque patch à une taille fixe (1x1), puis nous l'aplatissons.
        # Cela garantit que la sortie aura toujours la bonne dimension (patch_dim).
        self.projection = nn.AdaptiveAvgPool2d((1, 1)) # Adapte la taille
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = features.shape
        
        # 1. Première projection convolutive
        processed_features = self.patch_processor(features) # Taille: [B, patch_dim, H', W']
        
        # 2. Découpage en une grille de patchs
        # unfold crée des "vues" sans copier la mémoire, c'est très efficace.
        patch_size_h, patch_size_w = H // self.grid_size, W // self.grid_size
        patches = processed_features.unfold(2, patch_size_h, patch_size_h).unfold(3, patch_size_w, patch_size_w)
        # patches a maintenant la taille [B, patch_dim, grid_size, grid_size, patch_size_h, patch_size_w]

        # 3. On fusionne les dimensions du batch et de la grille
        patches = patches.contiguous().view(B * self.grid_size**2, self.patch_dim, patch_size_h, patch_size_w)
        
        # 4. On applique le pooling adaptatif sur chaque patch individuellement
        # Chaque patch [patch_dim, patch_size_h, patch_size_w] devient [patch_dim, 1, 1]
        pooled_patches = self.projection(patches)
        
        # 5. On retire les dimensions inutiles et on reforme le batch
        final_patches = pooled_patches.view(B, self.grid_size**2, self.patch_dim)
        
        return {'features': final_patches, 'num_patches': self.grid_size**2}


class OptimizedHTREncoder(nn.Module):
    """Encodeur HTR optimisé pour la stratégie SimCLR avec des images de taille fixe."""
    def __init__(self, global_dim: int = 512, patch_dim: int = 128, fpn_channels: int = 256):
        super().__init__()
        self.global_dim = global_dim
        self.patch_dim = patch_dim
        
        self.backbone = self._build_enhanced_backbone()
        
        self.attention_modules = nn.ModuleList([
            SpatialAttention(64),
            SpatialAttention(64),
            SpatialAttention(128),
            SpatialAttention(256),
        ])
        
        self.fpn = DilatedFPN([64, 64, 128, 256], out_channels=fpn_channels)
        
        # Tête de projection robuste pour la représentation globale
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fpn_channels, fpn_channels),
            nn.BatchNorm1d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fpn_channels, global_dim)
        )
        
        # Remplacement de l'extracteur adaptatif par un extracteur simple à grille fixe
        self.patch_extractor = SimplePatchExtractor(patch_dim, fpn_channels, grid_size=7)
        
        self._init_weights()
    
    def _build_enhanced_backbone(self):
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # Stride réduit pour mieux gérer les images de taille fixe (ex: 400x400)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        for block in backbone.layer3:
            if hasattr(block, 'conv2'):
                block.conv2.dilation = (2, 2); block.conv2.padding = (2, 2)
        backbone.fc = nn.Identity(); backbone.avgpool = nn.Identity()
        return backbone
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _extract_backbone_features(self, x: torch.Tensor) -> list:
        """Extrait les features multi-échelles, en appliquant l'attention au bon endroit."""
        features = []
        
        # Étape 0
        l0_out = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        features.append(l0_out)
        
        # Étape 1
        p0 = self.backbone.maxpool(self.attention_modules[0](l0_out))
        l1_out = self.backbone.layer1(p0)
        features.append(l1_out)
        
        # Étape 2
        l2_out = self.backbone.layer2(self.attention_modules[1](l1_out))
        features.append(l2_out)
        
        # Étape 3
        l3_out = self.backbone.layer3(self.attention_modules[2](l2_out))
        features.append(l3_out)
        
        return features
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_features = self._extract_backbone_features(x)
        fpn_features = self.fpn(backbone_features)
        
        # La représentation globale est extraite des features les plus sémantiques
        global_features = self.global_head(fpn_features[-1])
        
        # La représentation locale (patchs) est extraite de features de niveau intermédiaire
        patch_info = self.patch_extractor(fpn_features[1])
        
        return {
            'global': global_features,
            'patches': patch_info['features'],
            'patch_info': {'num_patches': patch_info['num_patches']}
                    }