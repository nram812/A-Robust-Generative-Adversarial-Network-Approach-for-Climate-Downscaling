#!/bin/bash -l

module purge # optional
module load NeSI
module load cuDNN/8.1.1.33-CUDA-11.2.0
#conda activate ml_env
nvidia-smi

cd "/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling"
# change directory to the working directory of interest

directory="experiment_configs/GAN_intensity_penalty"

if [ ! -d "$directory" ]; then
    echo "Directory does not exist: $directory"
    exit 1
fi


for file in "$directory"/*.json; do
    if [ -f "$file" ]; then
        # Process each .json file as needed
        echo "Processing file: $file"
        sbatch ops/run_config_experiments.sl $file

        # Add your custom logic here to work with each file
        # For example, you could read the content of the file:
        # content=$(cat "$file")
        # echo "File content: $content"
    fi
done


directory="experiment_configs/GAN"

if [ ! -d "$directory" ]; then
    echo "Directory does not exist: $directory"
    exit 1
fi


for file in "$directory"/*.json; do
    if [ -f "$file" ]; then
        # Process each .json file as needed
        echo "Processing file: $file"
        sbatch ops/run_config_experiments.sl $file

        # Add your custom logic here to work with each file
        # For example, you could read the content of the file:
        # content=$(cat "$file")
        # echo "File content: $content"
    fi
done



