#!/bin/bash -l
#SBATCH --job-name=GPU_job
#SBATCH --partition=niwa_work
#SBATCH --time=23:59:00
#SBATCH --cluster=maui_ancil
#SBATCH --mem=300G
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
cd /nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/
/nesi/project/niwa00004/rampaln/bin/python ./outputs/Val_metrics/val_metrics.py /nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/Baseline/CCAM_NIWA-REMS_ACCESS-CM2_hist_ssp370_pr.nc /nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/Val_metrics