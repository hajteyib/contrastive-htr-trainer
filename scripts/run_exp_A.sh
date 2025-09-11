#!/bin/bash
#SBATCH --job-name=exp_A_baseline
#SBATCH --output=slurm_logs/exp_A_baseline_%j.out
#SBATCH --error=slurm_logs/exp_A_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=el-haj.ebou@insa-lyon.fr
#SBATCH --mail-type=ALL

mkdir -p slurm_logs

echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

echo "Chargement des modules..."
module purge # Nettoyer l'environnement
module load python/3.9.0 cuda/12.1 # Adaptez si nécessaire

echo "Activation de l'environnement virtuel..."
source /sps/liris/eebou/htr_env/bin/activate

echo "Lancement de l'Expérience A..."
python src/main_contrastive.py --config src/configs/exp_A_baseline.yaml