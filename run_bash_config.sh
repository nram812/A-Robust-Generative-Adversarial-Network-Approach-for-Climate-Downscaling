#!/bin/bash -l

module purge # optional
module load NeSI
module load gcc/9.3.0
#module load CDO/1.9.5-GCC-7.1.0
#module load Miniconda3/4.12.0
module load cuDNN/8.1.1.33-CUDA-11.2.0
#conda activate ml_env
nvidia-smi


directory="/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/paper_experiments/intensity_penalty"

if [ ! -d "$directory" ]; then
    echo "Directory does not exist: $directory"
    exit 1
fi


for file in "$directory"/*.json; do
    if [ -f "$file" ]; then
        # Process each .json file as needed
        echo "Processing file: $file"
        sbatch //nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/run_config_experiments.sl $file

        # Add your custom logic here to work with each file
        # For example, you could read the content of the file:
        # content=$(cat "$file")
        # echo "File content: $content"
    fi
done


