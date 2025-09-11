# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple
import logging
import yaml
from collections import defaultdict
import copy
from tqdm import tqdm

import sys
sys.path.append('/home/jovyan/Contrastive-project/src')
from models.encoder import OptimizedHTREncoder
from models.losses import OptimizedContrastiveLoss
from training.monitor import ContrastiveMonitor

@torch.no_grad()
def calculate_contrastive_accuracy(model, dataloader, device):
    """Calcule la précision contrastive sur un dataloader donné."""
    model.eval()
    total_correct, total_samples = 0, 0
    for batch in tqdm(dataloader, desc="Calculating Contrastive Accuracy", leave=False):
        if batch is None or batch['anchor'].nelement() == 0: continue
        
        anchors = batch['anchor'].to(device)
        positives = batch['positive'].to(device)
        
        batch_size = anchors.size(0)
        if batch_size < 2: continue

        with autocast(enabled=True):
            anchor_embeds = model(anchors)['global']
            positive_embeds = model(positives)['global']
        
        anchor_embeds = torch.nn.functional.normalize(anchor_embeds, dim=1)
        positive_embeds = torch.nn.functional.normalize(positive_embeds, dim=1)
        
        sims = torch.matmul(anchor_embeds, positive_embeds.T)
        predictions = torch.argmax(sims, dim=1)
        labels = torch.arange(batch_size, device=device)
        
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        
    return (total_correct / total_samples) * 100 if total_samples > 0 else 0


class OptimizedContrastiveTrainer:

    def __init__(self,
             model: OptimizedHTREncoder,
             train_dataloader: DataLoader,
             val_dataloader: Optional[DataLoader] = None,
             config: Optional[Dict[str, Any]] = None,
             device: str = 'cuda',
             experiment_name: str = 'htr_optimized_contrastive'):
    
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.experiment_name = experiment_name
    
        self.criterion = OptimizedContrastiveLoss(
            global_dim=self.model.global_dim,
            patch_dim=self.model.patch_dim,
            loss_config=self.config['loss']
        ).to(self.device)
    
        self.optimizer = self._create_optimizer()
        
        # --- CORRECTION : Définir accumulation_steps AVANT d'appeler _create_scheduler ---
        self.accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        self.scheduler = self._create_scheduler()
        
        self.use_amp = self.config['training'].get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
    
        self.monitor = ContrastiveMonitor(log_dir=f"logs/{experiment_name}", config=self.config)
        
        self.momentum_encoder = copy.deepcopy(self.model)
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        self.momentum = self.config['training']['momentum']
        
        self.current_epoch = 0
        self.global_step = 0
    
        es_config = self.config['training'].get('early_stopping', {})
        self.es_enabled = es_config.get('enabled', False)
        self.es_patience = es_config.get('patience', 10)
        self.es_monitor_metric = es_config.get('monitor', 'val_contrastive_acc')
        self.es_mode = es_config.get('mode', 'max')
        self.es_counter = 0
        self.es_best_metric = -1.0 if self.es_mode == 'max' else float('inf')
    
        self.checkpoint_dir = Path(f"outputs/{experiment_name}/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
    
        if self.config.get('advanced', {}).get('channels_last_memory_format', False):
            self.model = self.model.to(memory_format=torch.channels_last)

        print(f"🚀 Trainer initialized for {experiment_name}. Early stopping is {'ENABLED' if self.es_enabled else 'DISABLED'} on metric '{self.es_monitor_metric}'.")
    def _create_optimizer(self) -> optim.Optimizer:
        """Crée un optimiseur simple et robuste pour l'ensemble du modèle."""
        opt_config = self.config['optimizer']
        
        try:
            lr = float(self.config['training']['learning_rate'])
            weight_decay = float(self.config['training']['weight_decay'])
        except (ValueError, TypeError) as e:
            self.logger.error(f"Impossible de convertir learning_rate ou weight_decay en nombre. Vérifiez votre fichier YAML.")
            raise e

        params = self.model.parameters()
        
        if opt_config['type'].lower() == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=opt_config.get('betas', [0.9, 0.999]))
        else:
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        sched_config = self.config['scheduler']
        num_epochs = self.config['training']['num_epochs']
        warmup_epochs = self.config['training']['warmup_epochs']
        
        # Le nombre de pas pour le scheduler est calculé après la phase de warmup
        main_schedule_steps = (num_epochs - warmup_epochs) * (len(self.train_dataloader) // self.accumulation_steps)

        if sched_config['type'].lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=main_schedule_steps, eta_min=float(sched_config.get('min_lr', 1e-6)))
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_q, param_k in zip(self.model.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def _warmup_learning_rate(self):
        warmup_epochs = self.config['training']['warmup_epochs']
        if self.current_epoch >= warmup_epochs: return

        warmup_steps = warmup_epochs * (len(self.train_dataloader) // self.accumulation_steps)
        if self.global_step < warmup_steps:
            lr_scale = self.global_step / warmup_steps
            base_lr = float(self.config['training']['learning_rate'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * lr_scale

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        # La configuration du logging est gérée dans main.py

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_losses = defaultdict(list)
        
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch} Training", leave=False)):
            if batch is None: continue
            
            self._warmup_learning_rate()

            anchor_images = batch['anchor'].to(self.device, non_blocking=True)
            positive_images = batch['positive'].to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                anchor_features = self.model(anchor_images)
                with torch.no_grad():
                    self._update_momentum_encoder()
                    positive_features = self.momentum_encoder(positive_images)
                
                loss_dict = self.criterion(anchor_features, positive_features)
                total_loss = loss_dict['total_loss'] / self.accumulation_steps
            
            self.scaler.scale(total_loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            for key, value in loss_dict.items():
                epoch_losses[key].append(value.item())
        
        if self.current_epoch >= self.config['training']['warmup_epochs']:
            self.scheduler.step()
            
        return {key: np.mean([v for v in values if not np.isnan(v)]) for key, values in epoch_losses.items()}

    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        val_losses = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc=f"Epoch {self.current_epoch} Validation", leave=False):
                if batch is None: continue
                anchor_images = batch['anchor'].to(self.device, non_blocking=True)
                positive_images = batch['positive'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.use_amp):
                    anchor_features = self.model(anchor_images)
                    positive_features = self.momentum_encoder(positive_images)
                    loss_dict = self.criterion(anchor_features, positive_features)

                for key, value in loss_dict.items():
                    val_losses[key].append(value.item())
        
        val_metrics = {f"val_{key}": np.mean([v for v in values if not np.isnan(v)]) for key, values in val_losses.items()}
        
        # --- AJOUT DE LA NOUVELLE MÉTRIQUE ---
        self.logger.info("Calculating Contrastive Accuracy on Validation Set...")
        contrastive_acc = calculate_contrastive_accuracy(self.model, self.val_dataloader, self.device)
        val_metrics['val_contrastive_acc'] = contrastive_acc
        self.logger.info(f"📊 Contrastive Accuracy: {contrastive_acc:.2f}%")

        return val_metrics

    def save_checkpoint(self, is_best: bool = False, suffix: str = ""):
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{self.current_epoch}{suffix}.pt"
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(), 'config': self.config}
        torch.save(checkpoint, filepath)
        self.logger.info(f"💾 Saved {'best' if is_best else ''} checkpoint to {filepath}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"📂 Resumed from checkpoint: {checkpoint_path} at epoch {self.current_epoch}")

    def train(self, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info(f"🚀 Starting training for {self.config['training']['num_epochs']} epochs")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = {}
            if (epoch + 1) % self.config['training']['validate_every_n_epochs'] == 0:
                val_metrics = self.validate_epoch()
            
            epoch_metrics = {**train_metrics, **val_metrics}
            self.monitor.log_epoch(epoch, epoch_metrics, model=self.model)
            
            if self.es_enabled and val_metrics:
                current_metric = val_metrics.get(self.es_monitor_metric, -1.0 if self.es_mode == 'max' else float('inf'))
                
                improved = (self.es_mode == 'max' and current_metric > self.es_best_metric) or \
                           (self.es_mode == 'min' and current_metric < self.es_best_metric)
                           
                if improved:
                    self.es_best_metric = current_metric
                    self.es_counter = 0
                    self.logger.info(f"✅ New best validation metric ({self.es_monitor_metric}): {current_metric:.4f}.")
                    self.save_checkpoint(is_best=True)
                else:
                    self.es_counter += 1
                    self.logger.info(f"⚠️ Validation metric did not improve for {self.es_counter} epoch(s). Patience: {self.es_patience}")
                
                if self.es_counter >= self.es_patience:
                    self.logger.info(f"🛑 Early stopping triggered after {self.es_patience} epochs.")
                    break
            
            if (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint()

            train_loss = train_metrics.get('total_loss', 'N/A')
            val_info = ""
            if val_metrics:
                val_loss = val_metrics.get('val_total_loss', 'N/A')
                val_acc = val_metrics.get('val_contrastive_acc', 'N/A')
                val_info = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"

            self.logger.info(f"⏱️  Epoch {epoch} completed - Train Loss: {train_loss:.4f}{val_info}")
            
        self.monitor.finalize()
        self.logger.info("🎯 Training finished.")
