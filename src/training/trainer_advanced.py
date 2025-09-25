# Fichier : src/training/trainer_advanced.py

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
from models.encoder import ConfigurableEncoder
from models.losses_composite import CompositeLoss
from models.losses import EnhancedMomentumQueue

@torch.no_grad()
def calculate_contrastive_accuracy(model, dataloader, device, use_amp):
    """Calcule la précision contrastive en comparant la première vue (ancre) à la deuxième (positif)."""
    model.eval()
    total_correct, total_samples = 0, 0
    # Utiliser un sous-ensemble du dataloader de validation pour une évaluation plus rapide
    num_batches_to_eval = min(50, len(dataloader))
    
    for i, batch_views in enumerate(tqdm(dataloader, desc="Calculating Accuracy", leave=False, total=num_batches_to_eval)):
        if i >= num_batches_to_eval:
            break
        if batch_views is None: continue
        
        # Pour le test de précision, on ne prend que les deux premières vues
        view1, view2 = batch_views[0].to(device), batch_views[1].to(device)
        batch_size = view1.size(0)
        if batch_size < 2: continue

        with autocast(enabled=use_amp):
            embed1 = model(view1)['projection']
            embed2 = model(view2)['projection']
        
        embed1 = torch.nn.functional.normalize(embed1, dim=1)
        embed2 = torch.nn.functional.normalize(embed2, dim=1)
        
        sims = torch.matmul(embed1, embed2.T)
        predictions = torch.argmax(sims, dim=1)
        labels = torch.arange(batch_size, device=device)
        
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        
    return (total_correct / total_samples) * 100 if total_samples > 0 else 0


class AdvancedTrainer:
    def __init__(self, model: ConfigurableEncoder, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                 config: Dict, output_dir: Path, log_dir: Path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.output_dir = output_dir
        
        self.criterion = CompositeLoss(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters()) # Sera reconfiguré
        self.scheduler = None
        self.use_amp = config['advanced'].get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.momentum_encoder = copy.deepcopy(self.model)
        for param in self.momentum_encoder.parameters(): param.requires_grad = False
        self.momentum = 0.996
        
        projection_dim = config['model']['projection_head']['layers'][-1]['dim']
        self.queue = EnhancedMomentumQueue(
            dim=projection_dim,
            queue_size=config['loss'].get('queue_size', 16384),
            diversity_threshold=0.95
        )
        
        self.current_epoch = 0
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        es_config = self.config['training'].get('early_stopping', {})
        self.es_enabled = es_config.get('enabled', True)
        self.es_patience = es_config.get('patience', 15)
        self.es_monitor_metric = es_config.get('monitor', 'val_contrastive_acc')
        self.es_mode = es_config.get('mode', 'max')
        self.es_counter = 0
        self.es_best_metric = -1.0 if self.es_mode == 'max' else float('inf')
        self.logger.info(f"🚀 Trainer initialisé. Early stopping: {self.es_enabled}, Métrique: {self.es_monitor_metric}")

    def _setup_phase(self, phase_config: Dict):
        self.logger.info(f"--- Démarrage de la Phase d'Entraînement : {phase_config['name']} ---")
        lr = float(phase_config['learning_rate'])
        self.momentum = float(phase_config.get('momentum', self.momentum))
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=float(self.config['optimizer']['weight_decay']))
        
        num_epochs_in_phase = phase_config['epochs']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs_in_phase * len(self.train_dataloader), eta_min=1e-6)
        
        if 'loss_weights' in phase_config:
            self.criterion.lambda_contrast = float(phase_config['loss_weights'].get('contrast', 1.0))
            self.criterion.lambda_width = float(phase_config['loss_weights'].get('width', 0.0))
            self.criterion.lambda_density = float(phase_config['loss_weights'].get('density', 0.0))

    def train_epoch(self):
        self.model.train()
        epoch_losses = defaultdict(list)
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch} Training", leave=False)
        
        for batch_views in pbar:
            if batch_views is None: continue
            
            views = [v.to(self.device) for v in batch_views[:-2]]
            target_widths = batch_views[-2].to(self.device)
            target_densities = batch_views[-1].to(self.device)

            with autocast(enabled=self.use_amp):
                num_views = len(views)
                anchor_views = views[:num_views//2]
                positive_views = views[num_views//2:]
                
                outputs = [self.model(v) for v in anchor_views]
                with torch.no_grad():
                    self._update_momentum_encoder()
                    positive_outputs = [self.momentum_encoder(v) for v in positive_views]
                
                anchor_projs = [out['projection'] for out in outputs]
                pos_projs = [out['projection'] for out in positive_outputs]
                pred_widths = [out['predicted_width'] for out in outputs]
                pred_densities = [out['predicted_density'] for out in outputs]

                loss_dict = self.criterion(anchor_projs, pos_projs, pred_widths, pred_densities,
                                           target_widths, target_densities, self.queue.get_negatives(self.device))
                loss = loss_dict['total_loss']
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.queue.update(outputs[0]['features'])
            
            for key, value in loss_dict.items(): epoch_losses[key].append(value.item())
            pbar.set_postfix({k: f"{v.item():.3f}" for k, v in loss_dict.items() if isinstance(v, torch.Tensor)})

        if self.scheduler: self.scheduler.step()
        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def validate_epoch(self):
        self.model.eval()
        # Pour une validation rapide, nous ne calculons que la précision, qui est notre métrique principale.
        # Le calcul de la Val Loss est redondant si nous ne l'utilisons pas pour l'early stopping.
        self.logger.info("Calcul de la précision contrastive sur le set de validation...")
        acc = calculate_contrastive_accuracy(self.model, self.val_dataloader, self.device, self.use_amp)
        return {'val_contrastive_acc': acc}

    def save_checkpoint(self, is_best=False):
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{self.current_epoch}.pth"
        filepath = self.checkpoint_dir / filename
        
        # On ne sauvegarde que le state_dict du modèle pour garder les checkpoints légers.
        torch.save(self.model.state_dict(), filepath)
        self.logger.info(f"💾 Checkpoint sauvegardé : {filepath}")

    def train(self):
        """
        Boucle d'entraînement principale qui gère les phases et l'early stopping.
        """
        training_phases = self.config['training']['phases']
        self.logger.info(f"🚀 Démarrage de l'entraînement pour {len(training_phases)} phase(s).")
        
        for phase_idx, phase_config in enumerate(training_phases):
            self._setup_phase(phase_config)
            
            for _ in range(phase_config['epochs']):
                # --- Entraînement et Validation ---
                train_metrics = self.train_epoch()
                val_metrics = self.validate_epoch()
                
                # --- Logique d'Early Stopping et Sauvegarde ---
                current_metric = val_metrics.get(self.es_monitor_metric)
                
                if current_metric is not None and not np.isnan(current_metric):
                    improved = (self.es_mode == 'max' and current_metric > self.es_best_metric) or \
                               (self.es_mode == 'min' and current_metric < self.es_best_metric)
                               
                    if improved:
                        self.es_best_metric = current_metric
                        self.es_counter = 0
                        self.logger.info(f"✅ Nouveau meilleur score ({self.es_monitor_metric}): {current_metric:.2f}%. Sauvegarde du modèle.")
                        self.save_checkpoint(is_best=True)
                    else:
                        self.es_counter += 1
                        self.logger.info(f"⚠️ Validation metric did not improve for {self.es_counter} epoch(s). Patience: {self.es_patience}")

                # --- Logging de Fin d'Époque ---
                train_loss = train_metrics.get('total_loss', float('nan'))
                val_acc = val_metrics.get('val_contrastive_acc', float('nan'))
                log_str = f"Epoch {self.current_epoch} (Phase {phase_idx+1}) | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%"
                self.logger.info(log_str)

                # --- Vérification de l'Arrêt ---
                if self.es_enabled and self.es_counter >= self.es_patience:
                    self.logger.info(f"🛑 Early stopping déclenché après {self.es_patience} époques sans amélioration.")
                    self.logger.info(f"🏆 Meilleur score obtenu ({self.es_monitor_metric}): {self.es_best_metric:.2f}%")
                    return # Terminer proprement l'entraînement

                self.current_epoch += 1

        self.logger.info("🎉 Entraînement terminé (nombre maximum d'époques atteint).")
        self.logger.info(f"🏆 Meilleur score obtenu ({self.es_monitor_metric}): {self.es_best_metric:.2f}%")