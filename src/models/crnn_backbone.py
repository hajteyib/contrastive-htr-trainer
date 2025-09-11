# Fichier : src/models/crnn_backbone.py

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Un bloc résiduel simple mais efficace, la brique de base de notre CNN."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CRNN_CNN_Backbone(nn.Module):
    """
    Notre architecture de référence : un CNN puissant, rapide, et adapté à l'OCR.
    Il sera utilisé pour le pré-entraînement ET le fine-tuning.
    """
    def __init__(self, global_dim=512, supervised_output_dim=512):
        super().__init__()
        self.global_dim = global_dim
        
        # Corps principal du CNN
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # H: 128 -> 64
        )
        self.layer2 = ResidualBlock(64, 128, stride=2) # H: 64 -> 32
        self.layer3 = nn.Sequential(
             ResidualBlock(128, 256, stride=1),
             nn.MaxPool2d(kernel_size=2, stride=2) # H: 32 -> 16
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # H: 16 -> 8, W inchangé
        )
        self.layer5 = nn.Sequential(
            ResidualBlock(512, 512, stride=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) # H: 8 -> 4, W inchangé
        )
        
        # --- Têtes spécifiques ---
        
        # Tête de projection pour la loss contrastive
        self.contrastive_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.global_dim)
        )
        
        # Tête de projection pour la tâche OCR supervisée (fine-tuning)
        # Elle aplatit la hauteur restante (4) et les canaux (512)
        self.supervised_head = nn.Linear(512 * 4, supervised_output_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant commune pour extraire la carte de features finale."""
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features = self.layer5(out) # Sortie ex: [B, 512, 4, W_out]
        return features

    def forward_contrastive(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pour le pré-entraînement. Ne retourne qu'une feature globale."""
        features = self.forward_features(x)
        global_feature = self.contrastive_head(features)
        # On ne produit plus de patchs pour simplifier et accélérer
        return {'global': global_feature, 'patches': torch.empty(0)}
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Forward pour la tâche OCR supervisée."""
        # Pour le fine-tuning, on prend une image RGB
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
            
        features = self.forward_features(x)
        B, C, H, W = features.shape
        
        # Permuter pour avoir la largeur comme dimension de séquence : [B, W, H, C]
        features = features.permute(0, 3, 1, 2).contiguous()
        # Aplatir les dimensions de hauteur et de canaux : [B, W, H*C]
        features = features.view(B, W, H * C)
        
        # Projection finale
        features = self.supervised_head(features) # [B, W, supervised_output_dim]
        
        return features, 1, W # On retourne H=1 car elle est maintenant encodée dans la dimension des features