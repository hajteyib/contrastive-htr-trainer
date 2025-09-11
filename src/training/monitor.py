import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from collections import defaultdict, deque
import time
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import logging
import json
import warnings
warnings.filterwarnings('ignore')


class ContrastiveMetrics:
    """Enhanced contrastive learning metrics for the optimized system."""
    
    @staticmethod
    def compute_alignment(features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Compute alignment between positive pairs."""
        features1_norm = F.normalize(features1, dim=-1)
        features2_norm = F.normalize(features2, dim=-1)
        similarities = torch.sum(features1_norm * features2_norm, dim=-1)
        return similarities.mean().item()
    
    @staticmethod
    def compute_uniformity(features: torch.Tensor) -> float:
        """Compute uniformity of feature distribution."""
        features_norm = F.normalize(features, dim=-1)
        similarities = torch.matmul(features_norm, features_norm.T)
        
        # Remove diagonal
        mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
        similarities = similarities[mask]
        
        uniformity = torch.log(torch.exp(similarities * 2).mean())
        return uniformity.item()
    
    @staticmethod
    def compute_patch_diversity(patch_features: torch.Tensor) -> Dict[str, float]:
        """
        Compute patch diversity metrics.
        
        Args:
            patch_features: Patch features [B, num_patches, D]
        """
        if patch_features.shape[1] == 0:  # No patches
            return {'diversity': 0.0, 'coherence': 0.0}
        
        batch_size, num_patches, feature_dim = patch_features.shape
        
        # Reshape for analysis
        patches_flat = patch_features.reshape(-1, feature_dim)  # [B*P, D]
        patches_norm = F.normalize(patches_flat, dim=-1)
        
        # Compute pairwise similarities
        similarities = torch.matmul(patches_norm, patches_norm.T)
        
        # Diversity: lower similarity = higher diversity
        # Remove self-similarities
        mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
        diversity = 1 - similarities[mask].mean()
        
        # Coherence: similarity within same image
        coherence_scores = []
        for b in range(batch_size):
            start_idx = b * num_patches
            end_idx = (b + 1) * num_patches
            
            if end_idx <= len(patches_norm):
                batch_patches = patches_norm[start_idx:end_idx]
                if len(batch_patches) > 1:
                    batch_similarities = torch.matmul(batch_patches, batch_patches.T)
                    # Remove diagonal
                    batch_mask = ~torch.eye(len(batch_patches), dtype=torch.bool, device=similarities.device)
                    coherence_scores.append(batch_similarities[batch_mask].mean().item())
        
        coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return {
            'diversity': diversity.item(),
            'coherence': coherence
        }
    
    ### CORRECTION START ###
    # Cette métrique est incorrecte car elle compare des tenseurs de dimensions différentes (512 vs 128).
    # La vraie mesure de cette cohérence est déjà capturée par la `cross_loss`, qui utilise une projection.
    # Nous allons donc la commenter pour ne plus l'utiliser.
    # @staticmethod
    # def compute_cross_scale_consistency(global_features: torch.Tensor,
    #                                   patch_features: torch.Tensor) -> float:
    #     """Compute consistency between global and patch representations."""
    #     if patch_features.shape[1] == 0:
    #         return 0.0
    #     patch_global = patch_features.mean(dim=1)
    #     global_norm = F.normalize(global_features, dim=-1)
    #     patch_global_norm = F.normalize(patch_global, dim=-1)
    #     # L'ERREUR SE PRODUIT ICI :
    #     consistency = F.cosine_similarity(global_norm, patch_global_norm).mean()
    #     return consistency.item()
    ### CORRECTION END ###


class ContrastiveMonitor:
    """
    Enhanced monitoring system for the optimized contrastive learning.
    Adapted for Global + Patches architecture with real data insights.
    """
    
    def __init__(self,
                 log_dir: str,
                 config: Dict[str, Any],
                 tsne_interval: int = 8,
                 max_tsne_samples: int = 1500):
        """
        Initialize the enhanced contrastive monitor.
        
        Args:
            log_dir: Directory for saving logs and visualizations
            config: Training configuration
            tsne_interval: Frequency of t-SNE visualization (epochs)
            max_tsne_samples: Maximum samples for t-SNE
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.log_dir / 'plots').mkdir(exist_ok=True)
        (self.log_dir / 'analysis').mkdir(exist_ok=True)
        
        
        self.config = config
        self.tsne_interval = tsne_interval
        self.max_tsne_samples = max_tsne_samples
        
        # Enhanced metrics storage
        self.epoch_metrics = defaultdict(list)
        self.batch_metrics = defaultdict(list)
        self.similarity_history = []
        self.patch_diversity_history = []
        
        # Cache performance tracking
        self.cache_performance = defaultdict(list)
        
        # Feature storage for analysis
        self.feature_buffer = deque(maxlen=max_tsne_samples)
        self.label_buffer = deque(maxlen=max_tsne_samples)
        
        # Timing and performance
        self.start_time = time.time()
        self.epoch_times = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Save configuration
        with open(self.log_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📊 Enhanced monitoring initialized: {self.log_dir}")
    
    def log_batch(self,
                  epoch: int,
                  batch_idx: int,
                  losses: Dict[str, torch.Tensor],
                  learning_rate: float,
                  features: Dict[str, Dict[str, torch.Tensor]],
                  cache_stats: Optional[Dict[str, Any]] = None,
                  log_interval: int = 25):
        """
        Log batch-level metrics with enhanced analysis.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            losses: Dictionary of loss components
            learning_rate: Current learning rate
            features: Dictionary with anchor and positive features
            cache_stats: Optional cache performance statistics
            log_interval: Frequency of detailed logging
        """
        # Store basic metrics
        for key, value in losses.items():
            self.batch_metrics[f"batch_{key}"].append(value.item())
        
        self.batch_metrics["batch_lr"].append(learning_rate)
        
        # Cache performance tracking
        if cache_stats:
            for key, value in cache_stats.items():
                self.cache_performance[key].append(value)
        
        # Detailed analysis every log_interval batches
        if batch_idx % log_interval == 0:
            anchor_features = features['anchor']
            positive_features = features['positive']
            
            # Global feature analysis
            global_anchor = anchor_features['global']
            global_positive = positive_features['global']
            
            # Alignment metric
            alignment = ContrastiveMetrics.compute_alignment(global_anchor, global_positive)
            self.batch_metrics["alignment"].append(alignment)
            
            # Uniformity metric
            uniformity = ContrastiveMetrics.compute_uniformity(global_anchor)
            self.batch_metrics["uniformity"].append(uniformity)
            
            # Patch diversity analysis
            patch_anchor = anchor_features['patches']
            patch_diversity = ContrastiveMetrics.compute_patch_diversity(patch_anchor)
            
            for key, value in patch_diversity.items():
                self.batch_metrics[f"patch_{key}"].append(value)
            
            ### CORRECTION START ###
            # On supprime l'appel à la fonction défectueuse.
            # La `cross_loss` est déjà loggée et est la bonne métrique à suivre.
            # cross_consistency = ContrastiveMetrics.compute_cross_scale_consistency(
            #     global_anchor, patch_anchor
            # )
            # self.batch_metrics["cross_consistency"].append(cross_consistency)
            ### CORRECTION END ###
            
            # Store features for t-SNE (sample to avoid memory issues)
            if len(global_anchor) > 0:
                sample_size = min(16, len(global_anchor))  # Reduced sample size
                sample_indices = torch.randperm(len(global_anchor))[:sample_size]
                sampled_features = global_anchor[sample_indices].detach().cpu()
                
                for i, feat in enumerate(sampled_features):
                    self.feature_buffer.append(feat.numpy())
                    self.label_buffer.append(f"epoch_{epoch}_batch_{batch_idx}_sample_{i}")
    
    def log_epoch(self,
                  epoch: int,
                  metrics: Dict[str, float],
                  model: torch.nn.Module):
        """
        Log epoch-level metrics with enhanced visualizations.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of epoch metrics
            model: The model being trained
        """
        # Store epoch metrics
        for key, value in metrics.items():
            self.epoch_metrics[key].append(value)
        
        # Track epoch timing
        if hasattr(self, '_epoch_start_time'):
            epoch_time = time.time() - self._epoch_start_time
            self.epoch_times.append(epoch_time)
        self._epoch_start_time = time.time()
        
        # Store patch diversity trends
        if 'patch_diversity' in self.batch_metrics:
            recent_diversity = np.mean(self.batch_metrics['patch_diversity'][-20:])  # Last 20 batches
            recent_coherence = np.mean(self.batch_metrics['patch_coherence'][-20:])
            
            self.patch_diversity_history.append({
                'epoch': epoch,
                'diversity': recent_diversity,
                'coherence': recent_coherence
            })
        
        # Create visualizations
        plot_interval = self.config.get('monitoring', {}).get('plot_interval', 3)
        if epoch % plot_interval == 0:
            self._create_enhanced_loss_plots(epoch)
            self._create_performance_plots(epoch)
        
        # t-SNE visualization
        if epoch % self.tsne_interval == 0 and len(self.feature_buffer) > 50:
            self._create_tsne_visualization(epoch)
        
        # Cache analysis
        if epoch % 5 == 0 and self.cache_performance:
            self._create_cache_analysis(epoch)
        
        # Log summary
        self.logger.info(f"📊 Epoch {epoch} metrics: {metrics}")
    
    def _create_enhanced_loss_plots(self, epoch: int):
        """Create enhanced loss visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Enhanced Training Analysis - Epoch {epoch}', fontsize=16)
        
        # Total loss
        if 'total_loss' in self.epoch_metrics:
            axes[0, 0].plot(self.epoch_metrics['total_loss'], label='Train', color='blue', linewidth=2)
            if 'val_total_loss' in self.epoch_metrics:
                val_losses = [v for v in self.epoch_metrics['val_total_loss'] if v is not None]
                if val_losses:
                    val_epochs = [i * self.config['training']['validate_every_n_epochs'] 
                                for i in range(len(val_losses))]
                    axes[0, 0].plot(val_epochs, val_losses, label='Val', color='red', linewidth=2, marker='o')
            
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Component losses
        loss_components = ['global_loss', 'patch_loss', 'cross_loss', 'style_loss', 'semantic_loss']
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for i, (component, color) in enumerate(zip(loss_components[:4], colors[:4])):
            if component in self.epoch_metrics:
                row, col = ((i + 1) // 3, (i + 1) % 3) if i < 2 else (1, i - 1)
                axes[row, col].plot(self.epoch_metrics[component], color=color, linewidth=2)
                axes[row, col].set_title(f'{component.replace("_", " ").title()}')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Loss')
                axes[row, col].grid(True, alpha=0.3)
        
        # Semantic loss (if available)
        if 'semantic_loss' in self.epoch_metrics:
            axes[1, 2].plot(self.epoch_metrics['semantic_loss'], color='purple', linewidth=2)
            axes[1, 2].set_title('Semantic Guidance Loss')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'plots' / f'enhanced_loss_curves_epoch_{epoch}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_performance_plots(self, epoch: int):
        """Create performance and quality metrics plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Performance Metrics - Epoch {epoch}', fontsize=16)
        
        # Alignment and Uniformity
        if 'alignment' in self.batch_metrics:
            alignment_smooth = self._smooth_metric(self.batch_metrics['alignment'])
            uniformity_smooth = self._smooth_metric(self.batch_metrics['uniformity'])
            
            axes[0, 0].plot(alignment_smooth, color='blue', linewidth=2)
            axes[0, 0].set_title('Feature Alignment (Higher is Better)')
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Alignment')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(uniformity_smooth, color='orange', linewidth=2)
            axes[0, 1].set_title('Feature Uniformity (Lower is Better)')
            axes[0, 1].set_xlabel('Batch')
            axes[0, 1].set_ylabel('Uniformity')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Patch diversity metrics
        if self.patch_diversity_history:
            epochs = [h['epoch'] for h in self.patch_diversity_history]
            diversity = [h['diversity'] for h in self.patch_diversity_history]
            coherence = [h['coherence'] for h in self.patch_diversity_history]
            
            axes[0, 2].plot(epochs, diversity, 'g-', label='Diversity', marker='o', linewidth=2)
            axes[0, 2].plot(epochs, coherence, 'b-', label='Coherence', marker='s', linewidth=2)
            axes[0, 2].set_title('Patch Analysis')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        ### CORRECTION START ###
        # On commente le graphique qui utilisait la métrique défectueuse.
        # Vous pouvez le réactiver plus tard si vous trouvez une meilleure façon
        # de visualiser la `cross_loss`.
        # if 'cross_consistency' in self.batch_metrics:
        #     consistency_smooth = self._smooth_metric(self.batch_metrics['cross_consistency'])
        #     axes[1, 0].plot(consistency_smooth, color='purple', linewidth=2)
        #     axes[1, 0].set_title('Cross-Scale Consistency')
        #     axes[1, 0].set_xlabel('Batch')
        #     axes[1, 0].set_ylabel('Consistency')
        #     axes[1, 0].grid(True, alpha=0.3)
        # À la place, on peut laisser ce graphique vide ou afficher la `cross_loss`
        if 'batch_cross_loss' in self.batch_metrics:
            cross_loss_smooth = self._smooth_metric(self.batch_metrics['batch_cross_loss'])
            axes[1, 0].plot(cross_loss_smooth, color='purple', linewidth=2)
            axes[1, 0].set_title('Cross-Scale Loss')
            axes[1, 0].set_xlabel('Batch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)

        ### CORRECTION END ###
        
        # Learning rate schedule
        if 'batch_lr' in self.batch_metrics:
            axes[1, 1].plot(self.batch_metrics['batch_lr'], color='green', linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Training speed
        if self.epoch_times:
            axes[1, 2].plot(self.epoch_times, color='red', linewidth=2, marker='o')
            axes[1, 2].set_title('Training Speed')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Time (seconds)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'plots' / f'performance_metrics_epoch_{epoch}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_cache_analysis(self, epoch: int):
        """Create cache performance analysis."""
        if not self.cache_performance.get('hit_rate'):
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Cache Performance Analysis - Epoch {epoch}', fontsize=16)
        
        # Hit rate over time
        hit_rates = self.cache_performance['hit_rate']
        axes[0].plot(hit_rates, color='green', linewidth=2)
        axes[0].set_title('Cache Hit Rate')
        axes[0].set_xlabel('Batch')
        axes[0].set_ylabel('Hit Rate')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        
        # Cache size evolution
        if 'size' in self.cache_performance:
            cache_sizes = self.cache_performance['size']
            axes[1].plot(cache_sizes, color='blue', linewidth=2)
            axes[1].set_title('Cache Size')
            axes[1].set_xlabel('Batch')
            axes[1].set_ylabel('Number of Cached Items')
            axes[1].grid(True, alpha=0.3)
        
        # Hit/Miss distribution
        if len(hit_rates) > 0:
            recent_hit_rate = np.mean(hit_rates[-100:]) if len(hit_rates) > 100 else np.mean(hit_rates)
            miss_rate = 1 - recent_hit_rate
            
            axes[2].pie([recent_hit_rate, miss_rate], 
                       labels=['Hits', 'Misses'], 
                       colors=['green', 'red'],
                       autopct='%1.1f%%')
            axes[2].set_title('Recent Hit/Miss Ratio')
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'plots' / f'cache_analysis_epoch_{epoch}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_tsne_visualization(self, epoch: int):
        """Create enhanced t-SNE visualization."""
        if len(self.feature_buffer) < 50:
            return
        
        try:
            # Prepare data
            features = np.array(list(self.feature_buffer))
            labels = list(self.label_buffer)
            
            # Sample if too many points
            if len(features) > self.max_tsne_samples:
                indices = np.random.choice(len(features), self.max_tsne_samples, replace=False)
                features = features[indices]
                labels = [labels[i] for i in indices]
            
            # Compute t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            features_2d = tsne.fit_transform(features)
            
            # Create enhanced visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f't-SNE Analysis - Epoch {epoch}', fontsize=16)
            
            # Color by epoch
            epoch_numbers = []
            for label in labels:
                try:
                    epoch_num = int(label.split('_')[1])
                    epoch_numbers.append(epoch_num)
                except:
                    epoch_numbers.append(0)
            
            # Plot 1: Colored by epoch
            scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                     c=epoch_numbers, cmap='viridis', alpha=0.6, s=20)
            axes[0].set_title('Features Colored by Epoch')
            axes[0].set_xlabel('t-SNE 1')
            axes[0].set_ylabel('t-SNE 2')
            fig.colorbar(scatter1, ax=axes[0], label='Epoch')
            
            # Plot 2: Density visualization
            axes[1].hexbin(features_2d[:, 0], features_2d[:, 1], gridsize=30, cmap='Blues', alpha=0.7)
            axes[1].set_title('Feature Density')
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            
            # Compute and display silhouette score
            if len(set(epoch_numbers)) > 1:
                sil_score = silhouette_score(features_2d, epoch_numbers)
                axes[0].text(0.02, 0.98, f'Silhouette Score: {sil_score:.3f}', 
                           transform=axes[0].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'plots' / f'tsne_analysis_epoch_{epoch}.png', 
                        dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create t-SNE visualization: {str(e)}")
    
    def _smooth_metric(self, values: List[float], window_size: int = 50) -> List[float]:
        """Apply moving average smoothing to metrics."""
        if len(values) < window_size:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            smoothed.append(np.mean(values[start_idx:end_idx]))
        
        return smoothed
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all collected metrics for analysis."""
        return {
            'epoch_metrics': dict(self.epoch_metrics),
            'batch_metrics': dict(self.batch_metrics),
            'similarity_history': self.similarity_history,
            'patch_diversity_history': self.patch_diversity_history,
            'cache_performance': dict(self.cache_performance),
            'epoch_times': self.epoch_times,
            'total_training_time': time.time() - self.start_time
        }
    
    def save_comprehensive_summary(self):
        """Save comprehensive training summary with insights."""
        metrics = self.export_metrics()
        
        # Save raw metrics
        with open(self.log_dir / 'comprehensive_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Create detailed summary with insights
        summary = {
            'training_overview': {
                'total_epochs': len(self.epoch_metrics.get('total_loss', [])),
                'total_time_hours': (time.time() - self.start_time) / 3600,
                'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            },
            'performance_metrics': {},
            'cache_performance': {},
            'model_insights': {}
        }
        
        # Performance analysis
        if 'total_loss' in self.epoch_metrics:
            losses = self.epoch_metrics['total_loss']
            summary['performance_metrics'].update({
                'initial_loss': losses[0],
                'final_loss': losses[-1],
                'best_loss': min(losses),
                'loss_reduction_percent': ((losses[0] - losses[-1]) / losses[0]) * 100,
                'convergence_stability': np.std(losses[-10:]) if len(losses) > 10 else 0
            })
        
        if 'val_total_loss' in self.epoch_metrics:
            val_losses = [v for v in self.epoch_metrics['val_total_loss'] if v is not None]
            if val_losses:
                summary['performance_metrics'].update({
                    'best_val_loss': min(val_losses),
                    'final_val_loss': val_losses[-1],
                    'overfitting_indicator': val_losses[-1] - min(val_losses)
                })
        
        # Cache analysis
        if self.cache_performance.get('hit_rate'):
            hit_rates = self.cache_performance['hit_rate']
            summary['cache_performance'] = {
                'avg_hit_rate': np.mean(hit_rates),
                'final_hit_rate': hit_rates[-1] if hit_rates else 0,
                'cache_efficiency': 'Excellent' if np.mean(hit_rates) > 0.85 else 'Good' if np.mean(hit_rates) > 0.7 else 'Needs Improvement'
            }
        
        # Model insights
        if 'alignment' in self.batch_metrics:
            recent_alignment = np.mean(self.batch_metrics['alignment'][-100:])
            recent_uniformity = np.mean(self.batch_metrics['uniformity'][-100:])
            
            summary['model_insights'] = {
                'feature_quality': {
                    'alignment': recent_alignment,
                    'uniformity': recent_uniformity,
                    'balance_score': recent_alignment - recent_uniformity
                }
            }
        
        if self.patch_diversity_history:
            final_diversity = self.patch_diversity_history[-1]
            summary['model_insights']['patch_analysis'] = {
                'diversity': final_diversity['diversity'],
                'coherence': final_diversity['coherence'],
                'patch_effectiveness': 'Good' if final_diversity['diversity'] > 0.3 and final_diversity['coherence'] > 0.5 else 'Needs Tuning'
            }
        
        # Save detailed summary
        with open(self.log_dir / 'training_insights.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        self._create_markdown_report(summary)
        
        self.logger.info(f"📊 Comprehensive summary saved to {self.log_dir}")
    
    def _create_markdown_report(self, summary: Dict[str, Any]):
        """Create a markdown training report."""
        report_path = self.log_dir / 'TRAINING_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# HTR Contrastive Training Report\n\n")
            
            # Overview
            overview = summary['training_overview']
            f.write("## Training Overview\n")
            f.write(f"- **Total Epochs**: {overview['total_epochs']}\n")
            f.write(f"- **Training Time**: {overview['total_time_hours']:.1f} hours\n")
            f.write(f"- **Average Epoch Time**: {overview['avg_epoch_time']:.1f} seconds\n\n")
            
            # Performance
            if 'performance_metrics' in summary:
                perf = summary['performance_metrics']
                f.write("## Performance Metrics\n")
                f.write(f"- **Loss Reduction**: {perf.get('loss_reduction_percent', 0):.1f}%\n")
                f.write(f"- **Final Training Loss**: {perf.get('final_loss', 0):.4f}\n")
                if 'best_val_loss' in perf:
                    f.write(f"- **Best Validation Loss**: {perf['best_val_loss']:.4f}\n")
                f.write("\n")
            
            # Cache Performance
            if 'cache_performance' in summary:
                cache = summary['cache_performance']
                f.write("## Cache Performance\n")
                f.write(f"- **Average Hit Rate**: {cache.get('avg_hit_rate', 0):.1%}\n")
                f.write(f"- **Efficiency**: {cache.get('cache_efficiency', 'Unknown')}\n\n")
            
            # Model Insights
            if 'model_insights' in summary:
                insights = summary['model_insights']
                f.write("## Model Analysis\n")
                
                if 'feature_quality' in insights:
                    fq = insights['feature_quality']
                    f.write(f"- **Feature Alignment**: {fq.get('alignment', 0):.3f}\n")
                    f.write(f"- **Feature Uniformity**: {fq.get('uniformity', 0):.3f}\n")
                
                if 'patch_analysis' in insights:
                    pa = insights['patch_analysis']
                    f.write(f"- **Patch Effectiveness**: {pa.get('patch_effectiveness', 'Unknown')}\n")
                
                f.write("\n")
            
            f.write("## Files Generated\n")
            f.write("- Loss curves and performance plots in `plots/`\n")
            f.write("- Detailed metrics in `comprehensive_metrics.json`\n")
            f.write("- Training insights in `training_insights.json`\n")
        
        print(f"📋 Training report saved: {report_path}")
    
    def finalize(self):
        """Finalize monitoring and create comprehensive analysis."""
        self.save_comprehensive_summary()
        
        # Create final comprehensive visualization
        self._create_final_dashboard()
        
        print(f"🎯 Enhanced monitoring completed. All analysis saved to: {self.log_dir}")
    
    def _create_final_dashboard(self):
        """Create a comprehensive final training dashboard."""
        if not any(self.epoch_metrics.values()):
            return
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)
        
        # Loss evolution
        ax1 = fig.add_subplot(gs[0, :2])
        if 'total_loss' in self.epoch_metrics:
            ax1.plot(self.epoch_metrics['total_loss'], label='Train', linewidth=2, color='blue')
            if 'val_total_loss' in self.epoch_metrics:
                val_losses = [v for v in self.epoch_metrics['val_total_loss'] if v is not None]
                if val_losses:
                    val_epochs = [i * self.config['training']['validate_every_n_epochs'] 
                                for i in range(len(val_losses))]
                    ax1.plot(val_epochs, val_losses, label='Val', linewidth=2, color='red', marker='o')
            ax1.set_title('Loss Evolution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Component losses
        ax2 = fig.add_subplot(gs[0, 2:])
        loss_components = ['global_loss', 'patch_loss', 'cross_loss', 'style_loss']
        colors = ['blue', 'orange', 'green', 'red']
        
        for component, color in zip(loss_components, colors):
            if component in self.epoch_metrics:
                ax2.plot(self.epoch_metrics[component], label=component.replace('_', ' ').title(), 
                        color=color, linewidth=2)
        
        ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance metrics
        ax3 = fig.add_subplot(gs[1, :2])
        if 'alignment' in self.batch_metrics:
            alignment_smooth = self._smooth_metric(self.batch_metrics['alignment'], 100)
            uniformity_smooth = self._smooth_metric(self.batch_metrics['uniformity'], 100)
            
            ax3.plot(alignment_smooth, label='Alignment', color='green', linewidth=2)
            ax3_twin = ax3.twinx()
            ax3_twin.plot(uniformity_smooth, label='Uniformity', color='orange', linewidth=2)
            
            ax3.set_title('Feature Quality Evolution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Batch')
            ax3.set_ylabel('Alignment', color='green')
            ax3_twin.set_ylabel('Uniformity', color='orange')
            ax3.grid(True, alpha=0.3)
        
        # Cache performance
        ax4 = fig.add_subplot(gs[1, 2:])
        if self.cache_performance.get('hit_rate'):
            hit_rate_smooth = self._smooth_metric(self.cache_performance['hit_rate'], 50)
            ax4.plot(hit_rate_smooth, color='purple', linewidth=2)
            ax4.set_title('Cache Hit Rate', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Batch')
            ax4.set_ylabel('Hit Rate')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
        
        # Training speed
        ax5 = fig.add_subplot(gs[2, :2])
        if self.epoch_times:
            ax5.plot(self.epoch_times, color='red', linewidth=2, marker='o', markersize=4)
            ax5.set_title('Training Speed', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Time (seconds)')
            ax5.grid(True, alpha=0.3)
        
        # Patch analysis
        ax6 = fig.add_subplot(gs[2, 2:])
        if self.patch_diversity_history:
            epochs = [h['epoch'] for h in self.patch_diversity_history]
            diversity = [h['diversity'] for h in self.patch_diversity_history]
            coherence = [h['coherence'] for h in self.patch_diversity_history]
            
            ax6.plot(epochs, diversity, 'g-', label='Diversity', marker='o', linewidth=2)
            ax6.plot(epochs, coherence, 'b-', label='Coherence', marker='s', linewidth=2)
            ax6.set_title('Patch Analysis Evolution', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Score')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Summary statistics
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create summary text
        summary_text = []
        
        if 'total_loss' in self.epoch_metrics:
            losses = self.epoch_metrics['total_loss']
            improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
            summary_text.append(f"Loss Improvement: {improvement:.1f}%")
            summary_text.append(f"Final Loss: {losses[-1]:.4f}")
        
        if 'val_total_loss' in self.epoch_metrics:
            val_losses = [v for v in self.epoch_metrics['val_total_loss'] if v is not None]
            if val_losses:
                summary_text.append(f"Best Val Loss: {min(val_losses):.4f}")
        
        if self.cache_performance.get('hit_rate'):
            avg_hit_rate = np.mean(self.cache_performance['hit_rate'])
            summary_text.append(f"Avg Cache Hit Rate: {avg_hit_rate:.1%}")
        
        if self.epoch_times:
            total_time = sum(self.epoch_times) / 3600
            summary_text.append(f"Total Training: {total_time:.1f}h")
        
        # Display summary
        summary_str = " | ".join(summary_text)
        ax7.text(0.5, 0.5, summary_str, ha='center', va='center', 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('HTR Contrastive Learning - Training Dashboard', fontsize=20, fontweight='bold')
        plt.savefig(self.log_dir / 'final_training_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()


def test_enhanced_monitor():
    """Test the enhanced monitoring system."""
    print("Testing enhanced contrastive monitor...")
    
    # Create test monitor
    monitor = ContrastiveMonitor(
        log_dir="test_logs_enhanced",
        config={'test': True, 'training': {'validate_every_n_epochs': 2}},
        tsne_interval=3
    )
    
    # Simulate enhanced training data
    for epoch in range(6):
        # Simulate epoch metrics
        epoch_metrics = {
            'total_loss': 2.5 - epoch * 0.3 + np.random.normal(0, 0.1),
            'global_loss': 1.2 - epoch * 0.15 + np.random.normal(0, 0.05),
            'patch_loss': 0.9 - epoch * 0.12 + np.random.normal(0, 0.05),
            'cross_loss': 0.4 - epoch * 0.05 + np.random.normal(0, 0.03),
            'style_loss': 0.3 - epoch * 0.03 + np.random.normal(0, 0.02),
            'semantic_loss': 0.2 - epoch * 0.02 + np.random.normal(0, 0.01)
        }
        
        # Add validation metrics every 2 epochs
        if epoch % 2 == 0:
            epoch_metrics.update({
                'val_total_loss': epoch_metrics['total_loss'] + 0.1,
                'val_global_loss': epoch_metrics['global_loss'] + 0.05
            })
        
        # Simulate batch data with enhanced features
        for batch in range(15):
            # Create fake enhanced features
            anchor_features = {
                'global': torch.randn(4, 128),
                'patches': torch.randn(4, 32, 64),  # Variable patch count
                'patch_info': {'num_patches': 32, 'grid_shape': (4, 8)}
            }
            positive_features = {
                'global': torch.randn(4, 128),
                'patches': torch.randn(4, 32, 64),
                'patch_info': {'num_patches': 32, 'grid_shape': (4, 8)}
            }
            
            batch_losses = {key: torch.tensor(value + np.random.normal(0, 0.05)) 
                          for key, value in epoch_metrics.items() if not key.startswith('val_')}
            
            # Simulate cache stats
            cache_stats = {
                'hit_rate': min(0.95, 0.3 + epoch * 0.1 + np.random.normal(0, 0.05)),
                'size': min(2000, 100 + epoch * 200 + batch * 10),
                'hit_count': 100 + batch * 50,
                'miss_count': 20 + batch * 5
            }
            
            monitor.log_batch(
                epoch=epoch,
                batch_idx=batch,
                losses=batch_losses,
                learning_rate=1e-3 * (0.95 ** epoch),
                features={'anchor': anchor_features, 'positive': positive_features},
                cache_stats=cache_stats
            )
        
        # Create mock model for epoch logging
        class MockModel:
            pass
        
        mock_model = MockModel()
        monitor.log_epoch(epoch, epoch_metrics, mock_model)
    
    # Finalize with comprehensive analysis
    monitor.finalize()
    
    print("✅ Enhanced monitor test completed!")
    print("Check the 'test_logs_enhanced' directory for comprehensive analysis.")
    
    # List generated files
    log_dir = Path("test_logs_enhanced")
    if log_dir.exists():
        print("\n📊 Generated files:")
        for file in sorted(log_dir.rglob("*")):
            if file.is_file():
                print(f"  📄 {file.relative_to(log_dir)}")


if __name__ == "__main__":
    test_enhanced_monitor() 
