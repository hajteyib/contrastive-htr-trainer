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

# --- SCRIPT D'EXÉCUTION ROBUSTE ---
set -euxo pipefail

mkdir -p slurm_logs

echo "--- Job Information ---"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "-----------------------"

# --- PRÉPARATION DE L'ENVIRONNEMENT ---
echo "1. Chargement du module Python..."
module purge
module load python
echo "Python
version: $(python --version)"

# --- VÉRIFICATION ET ACTIVATION DE L'ENVIRONNEMENT VIRTUEL ---
VENV_PATH="/sps/liris/eebou/htr_env"
echo "2. Vérification de l'environnement virtuel à : $VENV_PATH"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "ERREUR: L'environnement virtuel n'a pas été trouvé. Veuillez le créer."
    exit 1
fi
source "$VENV_PATH/bin/activate"
echo "Environnement activé. Python executable: $(which python)"

# --- VÉRIFICATION DES DÉPENDANCES ---
echo "3. Vérification de la présence de PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "ERREUR: PyTorch n'est pas installé dans l'environnement virtuel."
    exit 1
fi

# --- LANCEMENT DU SCRIPT PYTHON ---
echo "4. Lancement de l'Expérience A..."
python src/main_contrastive.py --config src/configs/exp_A_baseline.yaml

echo "5. Fin du job."