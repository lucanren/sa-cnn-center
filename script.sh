#!/usr/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH -t 3-00:00:00
#SBATCH -p debug 
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=mind-1-23
#SBATCH --output=/user_data/isaacl/sa-cnn-center/model6b.out

cd /user_data/isaacl/sa-cnn-center
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tangvit
python train_model.py
