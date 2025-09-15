#!/bin/bash

# --- HEADER SLURM ---
# Nom du job et fichiers de log
#SBATCH --job-name=exp_C_slower_lr
#SBATCH --output=slurm_logs/exp_C_slower_lr_%j.out
#SBATCH --error=slurm_logs/exp_C_slower_lr_%j.err


# Partition : On doit spécifier la partition GPU. "gpu" est un nom standard.
# Si cela ne marche pas, il faudra peut-être utiliser "gpu_h100" ou autre.
#SBATCH --partition=gpu

# Ressources de calcul
#SBATCH --time=2-00:00:00        # Temps max : 2 jours (pour être large)
#SBATCH --gres=gpu:h100:2      # 2 GPU H100
#SBATCH --cpus-per-task=10     # 10 coeurs CPU (pour 2 GPUs, respecte la limite de 5/GPU)
#SBATCH --mem=64G              # 64 Go de RAM CPU

# Notifications par e-mail
#SBATCH --mail-user=el-haj.ebou@insa-lyon.fr
#SBATCH --mail-type=ALL

# Déclaration de l'utilisation du stockage /sps
#SBATCH --licenses=sps

# --- SCRIPT D'EXÉCUTION ---
# Gestion des erreurs robuste
set -euxo pipefail

# Créer le dossier de logs Slurm
mkdir -p slurm_logs

echo "--- Job Information ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "-----------------------"

# Préparation de l'environnement
echo "Chargement des modules..."
module purge
module load python 

echo "Activation de l'environnement virtuel..."
source /sps/liris/eebou/htr_env/bin/activate

# Lancement du script Python principal
echo "Lancement de l'Expérience A..."
python src/main_contrastive.py --config src/configs/exp_C_slower_lr.yaml

echo "Fin du job."