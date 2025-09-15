# Fichier : src/training/trainer_contrastive.py

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
import time
from typing import Dict, Any, Optional
import logging
from collections import defaultdict
import copy
from tqdm import tqdm
import numpy as np

# Imports depuis notre projet
from models.crnn_backbone import CRNN_CNN_Backbone
from models.losses_light import LightweightContrastiveLoss
# Note: On a besoin du monitor, assurez-vous que `src/training/monitor.py` existe
# from training.monitor import ContrastiveMonitor 

@torch.no_grad()
def calculate_contrastive_accuracy(model, dataloader, device, use_amp):
    """Calcule la précision contrastive Top-1 sur un dataloader."""
    model.eval()
    total_correct, total_samples = 0, 0
    for batch in dataloader:
        if batch is None: continue
        view1, view2 = batch['view1'].to(device), batch['view2'].to(device)
        batch_size = view1.size(0)
        if batch_size < 2: continue

        with autocast(enabled=use_amp):
            # Utiliser la bonne méthode du modèle pour le contrastif
            embed1 = model.forward_contrastive(view1)['global']
            embed2 = model.forward_contrastive(view2)['global']
        
        embed1 = torch.nn.functional.normalize(embed1, dim=1)
        embed2 = torch.nn.functional.normalize(embed2, dim=1)
        
        sims = torch.matmul(embed1, embed2.T)
        predictions = torch.argmax(sims, dim=1)
        labels = torch.arange(batch_size, device=device)
        
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        
    return (total_correct / total_samples) * 100 if total_samples > 0 else 0


class ContrastiveTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, config, device, output_dir, log_dir):
        self.device = torch.device(device)
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        # Initialisation de la Loss
        self.criterion = LightweightContrastiveLoss(
            global_dim=model.global_dim,
            loss_config=config['loss']
        ).to(self.device)
        
        # Initialisation de l'Optimiseur et du Scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Gestion de la précision mixte (AMP)
        self.use_amp = config['advanced'].get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Momentum Encoder
        self.momentum_encoder = copy.deepcopy(self.model)
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        self.momentum = config['training']['momentum']
        
        # État de l'entraînement
        self.current_epoch = 0
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early Stopping
        es_config = config['training'].get('early_stopping', {})
        self.es_enabled = es_config.get('enabled', False)
        self.es_patience = es_config.get('patience', 10)
        self.es_monitor_metric = es_config.get('monitor', 'val_contrastive_acc')
        self.es_mode = es_config.get('mode', 'max')
        self.es_counter = 0
        self.es_best_metric = -1.0 if self.es_mode == 'max' else float('inf')
        
        self.logger = logging.getLogger(__name__)
        print(f"🚀 Trainer initialisé. Early stopping: {self.es_enabled}, Métrique: {self.es_monitor_metric}")

    def _create_optimizer(self):
        cfg = self.config['optimizer']
        lr = float(self.config['training']['learning_rate'])
        wd = float(self.config['training']['weight_decay'])
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd, betas=cfg.get('betas', (0.9, 0.999)))

    def _create_scheduler(self):
        cfg = self.config['scheduler']
        num_epochs = self.config['training']['num_epochs']
        if cfg['type'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=float(cfg.get('min_lr', 1e-6)))
        else:
            return None

    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_q, param_k in zip(self.model.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch} Training", leave=False)
        for batch in pbar:
            if batch is None: continue
            view1, view2 = batch['view1'].to(self.device), batch['view2'].to(self.device)

            with autocast(enabled=self.use_amp):
                # Utiliser la bonne méthode .forward_contrastive()
                anchor_features = self.model.forward_contrastive(view1)
                with torch.no_grad():
                    self._update_momentum_encoder()
                    positive_features = self.momentum_encoder.forward_contrastive(view2)
                
                loss_dict = self.criterion(anchor_features, positive_features)
                loss = loss_dict['total_loss']
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        return {'train_loss': total_loss / len(self.train_dataloader)}

    def validate_epoch(self) -> Dict[str, float]:
        """Évalue le modèle sur le set de validation pour la perte ET la précision."""
        self.model.eval()
        val_losses = defaultdict(list)
        
        # --- CALCUL DE LA VAL LOSS ---
        with torch.no_grad():
            pbar_loss = tqdm(self.val_dataloader, desc="Calculating Val Loss", leave=False)
            for batch in pbar_loss:
                if batch is None: continue
                
                view1 = batch['view1'].to(self.device, non_blocking=True)
                view2 = batch['view2'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.use_amp):
                    anchor_features = self.model.forward_contrastive(view1)
                    # Utiliser le momentum_encoder pour une mesure cohérente avec l'entraînement
                    positive_features = self.momentum_encoder.forward_contrastive(view2)
                    loss_dict = self.criterion(anchor_features, positive_features)

                for key, value in loss_dict.items():
                    val_losses[key].append(value.item())
        
        val_metrics = {f"val_{key}": np.mean([v for v in values if not np.isnan(v)]) for key, values in val_losses.items()}
        
        # --- CALCUL DE LA VAL ACCURACY ---
        # On appelle la fonction séparée qui est déjà optimisée pour cela
        self.logger.info("Calculating Contrastive Accuracy on Validation Set...")
        contrastive_acc = calculate_contrastive_accuracy(self.model, self.val_dataloader, self.device, self.use_amp)
        val_metrics['val_contrastive_acc'] = contrastive_acc
        
        self.logger.info(f"📊 Val Loss: {val_metrics.get('val_total_loss', 'N/A'):.4f} | Val Acc: {val_metrics.get('val_contrastive_acc', 'N/A'):.2f}%")
        return val_metrics
        
    def save_checkpoint(self, is_best=False):
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{self.current_epoch}.pth"
        filepath = self.checkpoint_dir / filename
        torch.save({'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict()}, filepath)
        self.logger.info(f"💾 Checkpoint sauvegardé : {filepath}")

    def train(self):
        start_time = time.time()
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            current_metric = val_metrics.get(self.es_monitor_metric)
            if current_metric is not None:
                improved = (self.es_mode == 'max' and current_metric > self.es_best_metric) or \
                           (self.es_mode == 'min' and current_metric < self.es_best_metric)
                if improved:
                    self.es_best_metric = current_metric
                    self.es_counter = 0
                    self.logger.info(f"✅ Nouveau meilleur score ({self.es_monitor_metric}): {current_metric:.2f}%. Sauvegarde du modèle.")
                    self.save_checkpoint(is_best=True)
                else:
                    self.es_counter += 1
                
                if self.es_enabled and self.es_counter >= self.es_patience:
                    self.logger.info(f"🛑 Early stopping déclenché.")
                    break
            
            if self.scheduler:
                self.scheduler.step()

            # Logging
            log_str = f"Epoch {epoch} | Train Loss: {train_metrics['train_loss']:.4f}"
            log_str += f" | Val Acc: {val_metrics['val_contrastive_acc']:.2f}%"
            self.logger.info(log_str)
        
        total_time = time.time() - start_time
        self.logger.info(f"🎉 Entraînement terminé en {total_time/3600:.2f} heures.")
        self.logger.info(f"🏆 Meilleur score ({self.es_monitor_metric}): {self.es_best_metric:.2f}%")