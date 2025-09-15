#!/bin/bash

# --- HEADER SLURM ---
#SBATCH --job-name=exp_A_baseline
#SBATCH --output=slurm_logs/exp_A_baseline_%j.out
#SBATCH --error=slurm_logs/exp_A_baseline_%j.err
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --mail-user=el-haj.ebou@insa-lyon.fr
#SBATCH --mail-type=ALL
#SBATCH --licenses=sps

# --- SCRIPT D'EXÉCUTION DIRECT ---

# Créer le dossier de logs Slurm
mkdir -p slurm_logs

echo "--- Démarrage du Job ---"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"

# 1. Préparation de l'environnement
echo "Activation de l'environnement..."
module purge
module load python
source /sps/liris/eebou/htr_env/bin/activate

# 2. Lancement du script Python
echo "Lancement de l'expérience A..."
python src/main_contrastive.py --config src/configs/exp_A_baseline.yaml

echo "--- Fin du Job ---"