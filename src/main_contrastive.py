# Fichier : src/main_contrastive.py (VERSION FINALE SANS HYDRA)

import argparse
import yaml
import torch
from pathlib import Path
import logging
import os
import sys

# Ajouter le chemin src pour les imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.crnn_backbone import CRNN_CNN_Backbone
from models.losses_light import LightweightContrastiveLoss
from data.dataset_contrastive import create_contrastive_dataloaders
from training.trainer_contrastive import ContrastiveTrainer

def load_config(config_path: str) -> dict:
    """Charge une configuration depuis un fichier YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ Configuration chargée depuis : {config_path}")
    return config

def setup_logging(log_dir: Path, experiment_name: str):
    """Configure le système de logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"🚀 Démarrage de l'expérience : {experiment_name}")
    logging.info(f"📝 Logs sauvegardés dans : {log_dir}")

def main():
    parser = argparse.ArgumentParser(description="Contrastive Pre-training for HTR.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment's YAML configuration file.")
    parser.add_argument("--smoke-test", action='store_true', help="Run a quick smoke test with a few batches.")
    args = parser.parse_args()

    # Charger la configuration de l'expérience
    config = load_config(args.config)
    exp_config = config['experiment']
    
    # Définir les chemins de sortie
    output_dir = Path(exp_config['output_dir']) / exp_config['name']
    log_dir = Path(exp_config['log_dir']) / exp_config['name']
    setup_logging(log_dir, exp_config['name'])

    # Configurer la reproductibilité
    torch.manual_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"🎯 Utilisation du device : {device}")

    # Créer le modèle
    model = CRNN_CNN_Backbone(global_dim=config['model']['global_dim'])
    model = model.to(device)
    logging.info(f"🧠 Modèle CRNN_CNN_Backbone créé ({sum(p.numel() for p in model.parameters()):,} paramètres).")

    if config['advanced'].get('compile_model', False):
        logging.info("⚡ Activation de torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning(f"⚠️ La compilation du modèle a échoué : {e}")

    # Créer les DataLoaders
    train_loader, val_loader = create_contrastive_dataloaders(
        data_list_file=config['data']['data_list_file'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        target_height=config['data']['target_height'],
        smoke_test=args.smoke_test
    )

    # Créer le Trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=str(device),
        output_dir=output_dir,
        log_dir=log_dir
    )

    logging.info("\n🔥 Démarrage de l'entraînement...")
    trainer.train()
    logging.info("🎉 Entraînement terminé !")

if __name__ == "__main__":
    main()