#!/usr/bin/env python3
"""
Optimized training script for HTR Multi-Scale Contrastive Learning.

This script implements the complete training pipeline for real HTR data:
- Optimized model with global + adaptive patches
- Real data loading with intelligent caching  
- Enhanced monitoring and visualization
- A40 GPU optimizations

Usage:
    python src/main.py --preset full_training
    python src/main.py --preset quick_test
    python src/main.py --config src/configs/default.yaml --experiment-name my_model
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import logging
import os
import sys
from typing import Dict, Any, Optional
import time
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from models.encoder import OptimizedHTREncoder
from models.losses import OptimizedContrastiveLoss
from data.dataset import create_optimized_dataloaders
from training.trainer import OptimizedContrastiveTrainer
from training.monitor import ContrastiveMonitor


def load_config(config_path: str, preset: Optional[str] = None) -> Dict[str, Any]:
    """Load and apply configuration with preset."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply preset if specified
    if preset and preset in config.get('presets', {}):
        preset_config = config['presets'][preset]
        
        def update_nested_dict(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_nested_dict(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_nested_dict(config, preset_config)
        print(f"✅ Applied preset: {preset}")
    
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup comprehensive logging system."""
    experiment_name = config['experiment']['name']
    log_dir = Path(f"logs/{experiment_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting HTR Contrastive Learning: {experiment_name}")
    logger.info(f"📝 Logs saved to: {log_dir}")
    return logger


def setup_reproducibility(config: Dict[str, Any]):
    """Setup reproducibility with enhanced settings."""
    repro_config = config.get('reproducibility', {})
    seed = repro_config.get('seed', 42)
    deterministic = repro_config.get('deterministic', True)
    benchmark = repro_config.get('benchmark', True)
    
    # Set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    # Configure PyTorch
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark
    
    # Memory format optimization for A40
    if config.get('advanced', {}).get('channels_last_memory_format', False):
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for A40
    
    print(f"🔧 Reproducibility: seed={seed}, deterministic={deterministic}, benchmark={benchmark}")


def create_optimized_model(config: Dict[str, Any], device: torch.device) -> OptimizedHTREncoder:
    """Create optimized HTR encoder model."""
    model_config = config['model']
    
    model = OptimizedHTREncoder(
        global_dim=model_config['global_dim'],
        patch_dim=model_config['patch_dim']
    )
    
    model = model.to(device)
    
    # Enable memory format optimization
    if config.get('advanced', {}).get('channels_last_memory_format', False):
        model = model.to(memory_format=torch.channels_last)
        print("📊 Enabled channels_last memory format")
    
    # Model compilation for PyTorch 2.0+
    if config.get('advanced', {}).get('compile_model', False):
        compile_mode = config['advanced'].get('compile_mode', 'default')
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"⚡ Model compiled with mode: {compile_mode}")
        except Exception as e:
            print(f"⚠️  Model compilation failed: {e}")
    
    return model


def create_dataloaders(config: Dict[str, Any]) -> tuple:
    """Create optimized dataloaders for real HTR data."""
    data_config = config['data']
    
    # Le chemin pointe maintenant vers le fichier texte
    data_list_file = data_config['data_list_file']
    
    if not os.path.exists(data_list_file):
        raise FileNotFoundError(f"Data list file not found: {data_list_file}. Please run the 'src/data/prepare_dataset.py' script first.")
    
    print(f"📂 Loading data paths from: {data_list_file}")
    
    # On passe maintenant tous les arguments nécessaires à la fonction
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        data_list_file=data_list_file,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        augmentation_strength=config['augmentation']['strength'],
        pin_memory=data_config['pin_memory'],
        target_height=data_config['target_height'] # Cet argument est nécessaire pour SimCLRTransform
    )
    
    return train_loader, val_loader, test_loader


def setup_monitoring(config: Dict[str, Any]) -> Optional[SummaryWriter]:
    """Setup monitoring and logging services."""
    monitoring_config = config.get('monitoring', {})
    experiment_name = config['experiment']['name']
    
    # TensorBoard setup
    tensorboard_writer = None
    if monitoring_config.get('use_tensorboard', True):
        log_dir = Path(f"runs/{experiment_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir)
        print(f"📈 TensorBoard logging: {log_dir}")
    
    # Weights & Biases (disabled by default for simplicity)
    if monitoring_config.get('use_wandb', False):
        try:
            import wandb
            wandb.init(
                project=monitoring_config.get('wandb_project', 'htr-contrastive'),
                name=experiment_name,
                config=config
            )
            print("📊 W&B logging initialized")
        except ImportError:
            print("⚠️  wandb not installed, skipping W&B logging")
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
    
    return tensorboard_writer


def validate_gpu_setup(device: torch.device):
    """Validate and optimize GPU setup."""
    if device.type == 'cuda':
        print(f"🎯 GPU Device: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU functionality
        try:
            test_tensor = torch.randn(100, 100, device=device)
            _ = torch.matmul(test_tensor, test_tensor)
            torch.cuda.synchronize()
            print("✅ GPU functionality verified")
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            raise
        
        # Clear cache for clean start
        torch.cuda.empty_cache()
    else:
        print("⚠️  Running on CPU - consider using GPU for better performance")


def export_final_model(model: OptimizedHTREncoder, 
                      config: Dict[str, Any],
                      training_history: Dict[str, Any],
                      output_dir: Path):
    """Export trained model for transfer learning."""
    export_dict = {
        'encoder_state_dict': model.state_dict(),
        'model_config': config['model'],
        'training_config': config['training'],
        'data_config': config['data'],
        'training_history': training_history,
        'architecture': 'OptimizedHTREncoder',
        'version': config['experiment']['version'],
        'description': config['experiment']['description'],
        'training_completed': True,
        'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Calculate final performance metrics
    if training_history.get('train_losses'):
        final_train_loss = training_history['train_losses'][-1].get('total_loss', 0)
        export_dict['final_train_loss'] = final_train_loss
    
    if training_history.get('val_losses'):
        val_losses = [v for v in training_history['val_losses'] if v]
        if val_losses:
            final_val_loss = val_losses[-1].get('val_total_loss', 0)
            export_dict['final_val_loss'] = final_val_loss
    
    # Export paths
    encoder_path = output_dir / 'htr_encoder_for_transfer_learning.pt'
    full_model_path = output_dir / 'complete_trained_model.pt'
    
    # Save encoder for transfer learning
    torch.save(export_dict, encoder_path)
    
    # Save complete model with training state
    complete_export = export_dict.copy()
    complete_export['full_model_state'] = model.state_dict()
    torch.save(complete_export, full_model_path)
    
    print(f"💾 Encoder exported: {encoder_path}")
    print(f"💾 Complete model exported: {full_model_path}")
    
    return encoder_path


def save_training_summary(config: Dict[str, Any],
                         training_history: Dict[str, Any],
                         output_dir: Path):
    """Save comprehensive training summary."""
    summary = {
        'experiment': config['experiment'],
        'model_architecture': config['model'],
        'training_config': config['training'],
        'data_statistics': {},
        'performance_metrics': {},
        'training_duration': {},
        'hardware_info': {
            'device': str(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__
        }
    }
    
    # Add performance metrics
    if training_history.get('train_losses'):
        losses = [loss['total_loss'] for loss in training_history['train_losses']]
        summary['performance_metrics']['initial_train_loss'] = losses[0]
        summary['performance_metrics']['final_train_loss'] = losses[-1]
        summary['performance_metrics']['best_train_loss'] = min(losses)
        summary['performance_metrics']['loss_reduction'] = (losses[0] - losses[-1]) / losses[0]
    
    if training_history.get('val_losses'):
        val_losses = [v['val_total_loss'] for v in training_history['val_losses'] if v and 'val_total_loss' in v]
        if val_losses:
            summary['performance_metrics']['best_val_loss'] = min(val_losses)
            summary['performance_metrics']['final_val_loss'] = val_losses[-1]
    
    # Save summary
    summary_path = output_dir / 'training_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    print(f"📋 Training summary saved: {summary_path}")


def main():
    """Main training function with enhanced error handling."""
    parser = argparse.ArgumentParser(description='HTR Optimized Contrastive Learning')
    parser.add_argument('--config', type=str, default='src/configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--preset', type=str, default='full_training',
                       help='Configuration preset to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Override experiment name')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Override data path')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Test setup without training')
    
    args = parser.parse_args()
    
    print("🚀 HTR Multi-Scale Contrastive Learning")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config(args.config, args.preset)
        
        # Override with command line arguments
        if args.experiment_name:
            config['experiment']['name'] = args.experiment_name
        if args.data_path:
            config['data']['real_data_path'] = args.data_path
        
        # Setup device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        print(f"🎯 Device: {device}")
        
        # Validate GPU setup
        validate_gpu_setup(device)
        
        # Setup reproducibility
        setup_reproducibility(config)
        
        # Setup logging
        logger = setup_logging(config)
        
        # Create output directory
        output_dir = Path(args.output_dir) / config['experiment']['name']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup monitoring
        tensorboard_writer = setup_monitoring(config)
        
        print("\n📊 Creating model and data loaders...")
        
        # Create optimized model
        model = create_optimized_model(config, device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"🧠 Model created: {total_params:,} total params, {trainable_params:,} trainable")
        try:
            mp.set_start_method('spawn', force=True)
            print("✅ Multiprocessing start method set to 'spawn'.")
        except RuntimeError:
            print("ℹ️ Multiprocessing context already set.")
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        # Early exit for dry run
        if args.dry_run:
            print("✅ Dry run completed successfully!")
            return
        
        print("\n🔥 Starting training...")
        
        # Create trainer with optimized settings
        trainer = OptimizedContrastiveTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=config,
            device=str(device),
            experiment_name=config['experiment']['name']
        )
        
        # Start training
        start_time = time.time()
        training_history = trainer.train(resume_from=args.resume)
        training_duration = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_duration/3600:.1f} hours!")
        
        # Export models and results
        print("\n💾 Exporting results...")
        
        encoder_path = export_final_model(model, config, training_history, output_dir)
        save_training_summary(config, training_history, output_dir)
        
        # Save final configuration
        with open(output_dir / 'final_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Print summary
        print("\n🎉 Training Summary:")
        print("=" * 30)
        
        if training_history.get('train_losses'):
            final_loss = training_history['train_losses'][-1]['total_loss']
            initial_loss = training_history['train_losses'][0]['total_loss']
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            print(f"📉 Loss improvement: {improvement:.1f}%")
            print(f"📊 Final training loss: {final_loss:.4f}")
        
        if training_history.get('val_losses'):
            val_losses = [v for v in training_history['val_losses'] if v and 'val_total_loss' in v]
            if val_losses:
                best_val = min(v['val_total_loss'] for v in val_losses)
                print(f"🏆 Best validation loss: {best_val:.4f}")
        
        print(f"⏱️  Training duration: {training_duration/3600:.1f} hours")
        print(f"💾 Model ready for transfer learning: {encoder_path}")
        print(f"📁 All outputs saved to: {output_dir}")
        
        # Cache statistics
        if hasattr(train_loader.dataset, 'get_cache_stats'):
            cache_stats = train_loader.dataset.get_cache_stats()
            print(f"📊 Cache hit rate: {cache_stats['hit_rate']:.1%}")
        
        logger.info("🎯 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup
        if tensorboard_writer:
            tensorboard_writer.close()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()