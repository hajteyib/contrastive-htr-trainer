#!/bin/bash
#SBATCH --job-name=exp_A_baseline
#SBATCH --output=slurm_logs/exp_A_baseline_%j.out
#SBATCH --error=slurm_logs/exp_A_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# --- Configuration pour les GPU H100 ---
# Demander 2 GPUs de type H100
# NOTE : La syntaxe peut être --partition=gpu_h100 ou --constraint=h100
# On utilise la syntaxe --gres qui est la plus standard.
#SBATCH --gres=gpu:h100:2

# Demander 10 coeurs CPU (2 GPUs * 5 CPU/GPU max = 10)
#SBATCH --cpus-per-task=10

# Demander de la RAM CPU
#SBATCH --mem=64G

# --- Notifications par E-mail ---
#SBATCH --mail-user=el-haj.ebou@insa-lyon.fr
#SBATCH --mail-type=ALL # Notifie au début (BEGIN), à la fin (END), et en cas d'erreur (FAIL)


# --- Exécution du Job ---
# Créer le dossier de logs Slurm s'il n'existe pas
mkdir -p slurm_logs

echo "----------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on node: $SLURMD_NODENAME"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------------------"

echo "Chargement des modules..."
module purge
module load python cuda

echo "Activation de l'environnement virtuel..."
source /sps/liris/eebou/htr_env/bin/activate

echo "Lancement du script Python pour l'Expérience A..."
# Utiliser srun est une bonne pratique pour lancer le code Python dans l'allocation Slurm
srun python src/main_contrastive.py --config src/configs/exp_A_baseline.yaml

echo "Fin du job."