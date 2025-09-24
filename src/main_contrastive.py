# Fichier : src/main_contrastive.py (VERSION FINALE AVEC HYDRA)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import logging
import os
import sys

# Ajouter le chemin src pour les imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- NOUVEAUX IMPORTS ---
from models.encoder import ConfigurableEncoder
from models.losses_composite import CompositeLoss
from data.dataset_contrastive import create_contrastive_dataloaders
from training.trainer_advanced import AdvancedTrainer

def setup_logging(log_dir: Path, experiment_name: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_dir / 'training.log'), logging.StreamHandler()]
    )
    logging.info(f"🚀 Démarrage de l'expérience : {experiment_name}")

@hydra.main(config_path="configs", config_name="exp_A_baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    print("--- Configuration de l'Expérience ---")
    print(OmegaConf.to_yaml(cfg))
    
    exp_config = cfg.experiment
    output_dir = Path(exp_config.output_dir) / exp_config.name
    log_dir = Path(exp_config.log_dir) / exp_config.name
    setup_logging(log_dir, exp_config.name)

    torch.manual_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConfigurableEncoder(cfg.model).to(device)
    logging.info(f"🧠 Modèle créé ({sum(p.numel() for p in model.parameters()):,} paramètres).")

    if cfg.advanced.get('compile_model', False):
        model = torch.compile(model)

    train_loader, val_loader = create_contrastive_dataloaders(cfg, 'smoke_test' in cfg and cfg.smoke_test)

    trainer = AdvancedTrainer(model, train_loader, val_loader, cfg, output_dir, log_dir)
    trainer.train()

if __name__ == "__main__":
    main()