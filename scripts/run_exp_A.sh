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
echo "--- Job Information ---"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "-----------------------"

echo "Chargement du module Python..."
module purge
module load python

echo "Activation de l'environnement virtuel..."
source /sps/liris/eebou/htr_env/bin/activate

echo "Lancement de l'Expérience A..."
echo "--- DEBUT DE LA SORTIE PYTHON ---"

# Utilisation du chemin absolu pour une robustesse maximale
/sps/liris/eebou/htr_env/bin/python src/main_contrastive.py --config src/configs/exp_A_baseline.yaml

echo "--- FIN DE LA SORTIE PYTHON ---"
echo "Fin du job."