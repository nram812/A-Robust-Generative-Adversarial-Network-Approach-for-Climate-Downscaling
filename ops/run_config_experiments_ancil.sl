#!/bin/bash -l
#SBATCH --job-name=GPU_job
#SBATCH --partition=niwa_work
#SBATCH --time=23:59:00
#SBATCH --cluster=maui_ancil
#SBATCH --mem=500G
#SBATCH --gpus-per-node=A100:1
#SBATCH --account=niwap03712
#SBATCH --mail-user=neelesh.rampal@niwa.co.nz
#SBATCH --mail-type=ALL
#SBATCH --output log/%j-%x.out
#SBATCH --error log/%j-%x.out




module purge # optional
module load NeSI
module load gcc/9.3.0
#module load CDO/1.9.5-GCC-7.1.0
#module load Miniconda3/4.12.0
module load cuDNN/8.1.1.33-CUDA-11.2.0
#conda activate ml_env
nvidia-smi
# set the experiment name that we are implementing


# change to your working directory
cd "/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling"

/nesi/project/niwa00004/rampaln/bin/python ops/new_training.py $1



