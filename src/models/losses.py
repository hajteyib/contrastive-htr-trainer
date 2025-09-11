# losses.py (VERSION FINALE, COMPLÈTE ET STABILISÉE)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import numpy as np
from collections import deque
import math



class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, positive_mask: Optional[torch.Tensor] = None):
        query = F.normalize(query, dim=-1, eps=1e-8)
        keys = F.normalize(keys, dim=-1, eps=1e-8)
        logits = torch.matmul(query, keys.T) / self.temperature
        
        if positive_mask is None:
            labels = torch.arange(len(query), device=query.device)
            return F.cross_entropy(logits, labels)
        else: # Garder la flexibilité pour le futur
            exp_logits = torch.exp(logits)
            pos_mask_float = positive_mask.float()
            neg_mask_float = (~positive_mask).float()
            diag_mask = torch.eye(logits.shape[0], logits.shape[1], device=logits.device, dtype=torch.bool)
            neg_mask_float[diag_mask] = 0.0
            pos_exp = exp_logits * pos_mask_float
            neg_exp = exp_logits * neg_mask_float
            pos_sum = torch.sum(pos_exp, dim=1)
            neg_sum = torch.sum(neg_exp, dim=1)
            loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
            return loss.mean()


class EnhancedMomentumQueue:
    def __init__(self, dim: int, queue_size: int, diversity_threshold: float):
        self.dim=dim; self.queue_size=queue_size; self.diversity_threshold=diversity_threshold
        self.queue=deque(maxlen=queue_size); self.initialized=False
    
    def update(self, features: torch.Tensor):
        features = F.normalize(features, dim=-1, eps=1e-8)
        for feat in features:
            if self.initialized and len(self.queue) > 100:
                sample_indices = np.random.choice(len(self.queue), 100, replace=False)
                sample_queue_cpu = torch.stack([self.queue[i] for i in sample_indices])
                sample_queue_gpu = sample_queue_cpu.to(feat.device)
                if F.cosine_similarity(feat.unsqueeze(0), sample_queue_gpu).max() > self.diversity_threshold: continue
            self.queue.append(feat.detach().cpu())
        self.initialized = len(self.queue) >= self.queue_size // 4
    
    def get_negatives(self, device: torch.device) -> torch.Tensor:
        if not self.initialized or len(self.queue) == 0: return torch.empty(0, self.dim, device=device)
        return F.normalize(torch.stack(list(self.queue), dim=0).to(device), dim=-1, eps=1e-8)


class HardNegativeMiner:
    """Enhanced hard negative mining with adaptive weighting."""
    
    def __init__(self, percentile: float = 90.0, min_negatives: int = 64, adaptive_weighting: bool = True):
        self.percentile=percentile; self.min_negatives=min_negatives; self.adaptive_weighting=adaptive_weighting
    
    def mine_hard_negatives(self, query: torch.Tensor, negatives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if negatives.shape[0] == 0: return negatives, torch.tensor([], device=query.device)
        
        query_norm = F.normalize(query, dim=-1, eps=1e-8)
        neg_norm = F.normalize(negatives, dim=-1, eps=1e-8)
        similarities = torch.matmul(query_norm, neg_norm.T)
        threshold = torch.quantile(similarities, self.percentile / 100.0, dim=1, keepdim=True)
        hard_mask = similarities >= threshold
        if hard_mask.sum(dim=1).min() < self.min_negatives:
            k = min(self.min_negatives, similarities.shape[1])
            _, topk_indices = torch.topk(similarities, k, dim=1)
            hard_mask = torch.zeros_like(similarities, dtype=torch.bool).scatter_(1, topk_indices, True)
        hard_indices = torch.where(hard_mask)
        hard_negatives = negatives[hard_indices[1]]
        if self.adaptive_weighting:
            weights = torch.sigmoid(similarities[hard_indices] * 10)
        else:
            weights = torch.ones(len(hard_negatives), device=query.device)
        return hard_negatives, weights


class SemanticGuidanceLoss(nn.Module):
    """Semantic guidance loss for HTR contrastive learning."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, patch_features: torch.Tensor, semantic_similarities: torch.Tensor) -> torch.Tensor:
        ### CORRECTION DE STABILITÉ ###
        sample_features = F.normalize(patch_features.mean(dim=1), dim=-1, eps=1e-8)
        visual_sims = torch.matmul(sample_features, sample_features.T)
        visual_probs = F.log_softmax(visual_sims / self.temperature, dim=1)
        semantic_probs = F.softmax(semantic_similarities / self.temperature, dim=1)
        return F.kl_div(visual_probs, semantic_probs, reduction='batchmean')


class OptimizedContrastiveLoss(nn.Module):
    def __init__(self, global_dim: int, patch_dim: int, loss_config: Dict[str, Any]):
        super().__init__()
        
        self.global_loss = InfoNCE(loss_config.get('global_temp', 0.15))
        self.patch_loss = InfoNCE(loss_config.get('patch_temp', 0.15))
        self.semantic_loss = SemanticGuidanceLoss(loss_config.get('semantic_temp', 0.2))
        
        # --- MODIFICATION 1 : Retirer lambda_style ---
        self.lambda_global = loss_config.get('lambda_global', 1.0)
        self.lambda_patch = loss_config.get('lambda_patch', 0.8)
        self.lambda_semantic = loss_config.get('lambda_semantic', 0.0)
        
        queue_size = loss_config.get('queue_size', 8192)
        diversity_threshold = loss_config.get('diversity_threshold', 0.95)
        self.global_queue = EnhancedMomentumQueue(global_dim, queue_size, diversity_threshold)
        self.patch_queue = EnhancedMomentumQueue(patch_dim, queue_size, diversity_threshold)

    def _compute_global_loss(self, anchor_global: torch.Tensor, positive_global: torch.Tensor) -> torch.Tensor:
        queue_negatives = self.global_queue.get_negatives(anchor_global.device)
        keys = torch.cat([positive_global, queue_negatives], dim=0)
        return self.global_loss(anchor_global, keys)
    
    def _compute_patch_loss(self, anchor_patches: torch.Tensor, positive_patches: torch.Tensor) -> torch.Tensor:
        if anchor_patches.shape[1] == 0: return torch.tensor(0.0, device=anchor_patches.device)
        anchor_flat = anchor_patches.flatten(0, 1); positive_flat = positive_patches.flatten(0, 1)
        if anchor_flat.shape[0] > 1024:
            indices = torch.randperm(anchor_flat.shape[0], device=anchor_flat.device)[:1024]
            anchor_flat, positive_flat = anchor_flat[indices], positive_flat[indices]
        queue_negatives = self.patch_queue.get_negatives(anchor_patches.device)
        keys = torch.cat([positive_flat, queue_negatives], dim=0)
        return self.patch_loss(anchor_flat, keys)
    
    
    
    # --- MODIFICATION 2 : Retirer la méthode _compute_style_consistency_loss ---
    # def _compute_style_consistency_loss(...)
    
    def forward(self, anchor_features: Dict[str, torch.Tensor], positive_features: Dict[str, torch.Tensor], semantic_similarities: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        anchor_global = anchor_features['global']
        positive_global = positive_features['global']
        anchor_patches = anchor_features['patches']
        positive_patches = positive_features['patches']

        self.global_queue.update(anchor_global)
        if anchor_patches.shape[1] > 0:
            self.patch_queue.update(anchor_patches.flatten(0, 1))
        
        global_loss = self._compute_global_loss(anchor_global, positive_global)
        patch_loss = self._compute_patch_loss(anchor_patches, positive_patches)
        
        semantic_loss = torch.tensor(0.0, device=anchor_global.device)
        if self.lambda_semantic > 0 and semantic_similarities is not None and anchor_patches.shape[1] > 0:
            semantic_loss = self.semantic_loss(anchor_patches, semantic_similarities)
        
        # --- MODIFICATION 3 : Retirer style_loss du calcul total ---
        total_loss = (
            self.lambda_global * global_loss +
            self.lambda_patch * patch_loss +
            self.lambda_semantic * semantic_loss
        )
        
        return {
            'total_loss': total_loss,
            'global_loss': global_loss.detach(),
            'patch_loss': patch_loss.detach(),
            'semantic_loss': semantic_loss.detach()
            # On ne retourne plus de 'style_loss'
        }