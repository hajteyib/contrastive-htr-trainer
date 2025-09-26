#!/bin/bash

# --- HEADER SLURM ---
# (Le header est parfait, aucun changement)
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

# --- SCRIPT D'EXÉCUTION ROBUSTE ---
set -euxo pipefail # C'est une bonne pratique de garder ceci

mkdir -p slurm_logs

echo "--- Job Information ---"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "-----------------------"

# --- NOUVEAUTÉ : Se déplacer dans le répertoire du projet ---
# La variable $SLURM_SUBMIT_DIR contient le chemin d'où vous avez lancé sbatch
cd $SLURM_SUBMIT_DIR
echo "Répertoire de travail : $(pwd)"

# Préparation de l'environnement
echo "Chargement du module Python..."
module purge
module load python

echo "Activation de l'environnement virtuel..."
source /sps/liris/eebou/htr_env/bin/activate

# Lancement de l'Expérience
echo "Lancement de l'Expérience A..."
python src/main_contrastive.py --config-name=exp_A_baseline

echo "Fin du job."