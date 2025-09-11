#!/bin/bash
#SBATCH --job-name=exp_C_slower_lr
#SBATCH --output=slurm_logs/exp_C_slower_lr_%j.out
#SBATCH --error=slurm_logs/exp_C_slower_lr_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=el-haj.ebou@insa-lyon.fr
#SBATCH --mail-type=ALL

mkdir -p slurm_logs
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"
module purge; module load python/3.9.0 cuda/12.1
source /sps/liris/eebou/htr_env/bin/activate

echo "Lancement de l'Expérience C..."
python src/main_contrastive.py --config src/configs/exp_C_slower_lr.yaml