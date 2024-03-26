# A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling
This repository contains the code and data (link to Zenodo) for "A Robust Generative Adversarial Network Approach for Climate Downscaling"

## Understanding the Repository:

## Add some information about the Zenodo outputs



## Setting Up A Python Environment
These experiments have been performed on an Nvidia A100 GPU, and thus the code has been developed in a
slurm (job scheduling) and unix enivironment.

These experiments can be run without slurm, but will require special configurations.


## Experiment Set Up

The experiments are configured with .json files, where each .json file contains information about each experiment.
There are two experiments used in this study, one without intensity penalty and one with.

All these configuration files (.json files) are stored in the following directory.
There should be in total 6 configuration files for each experiment, where each configuration file contains a specific value of $\lambda_{adv}$.

The values of $\lambda_{adv}$ explored in this study are: 0.0, 0.0001, 0.00125, 0.0025, 0.005, 0.01 and 0.1.

* experiment_configs/
    * GAN
        *config_experiments_all_extreme.json
        * ....

    * GAN_intensity penalty
        *config_experiments_all_extreme.json
        * ....

To re-run the experiments on your local/HPC environment, you will need to modify each (all of them)
to your specific requirements.

An example Json file is shown below.
Please modify "mean" (data to normalize the predictors), "std" (data to normalize the predictors),
 "train_x" (predictor variables),"train_y" (target variables), "src_path" (the path where you have cloned the repo),
 "output_folder" (where you would like to store the model files), "static_predictors" (the location where the topography files are stored).

 The other arguments are most likely not required to be changed if you would like to run the experiments as is.

```json
{"mean": "/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/inputs/ERA5/mean_1974_2011.nc",
  "std": "/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/inputs/ERA5/std_1974_2011.nc",
  "train_x": "/scale_wlg_persistent/filesets/project/niwa00018/ML_downscaling_CCAM/Training_CNN/inputs/CCAM_emulator_precip_fields/Combined_ACCESS-CM2/ACCESS-CM2_hist.nc",
  "train_y": "/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/inputs/CCAM_emulator_precip_fields/Combined_ACCESS-CM2/ACCESS_CM2_pr_hist.nc",
  "src_path": "/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/ops/training_GAN",
  "var_names": ["q_500", "q_850", "u_500", "u_850", "v_500", "v_850", "t_500", "t_850"],
  "output_folder": "/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/paper_experiments/models",
  "model_name": "cascaded_perfect_framework_extreme_intensity_constraint",
  "notes": "Just trained on the ACCESS-CM2 gcm only, both historical training",
  "ad_loss_factor": 0.1, "n_filters": [32, 64, 128, 256],
  "kernel_size": 5, "n_input_channels": 8, "n_output_channels": 1,
  "output_shape": [172, 179], "input_shape": [23, 26], "learning_rate": 0.0002,
  "beta_1": 0.5, "beta_2": 0.9, "precip_conv_factor": 1, "log_beta": 3.5,
  "fine_tuning": "False", "gp_weight": 10,
  "discrim_steps": 3,
  "static_predictors": "/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/ancil_fields/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc",
  "learning_rate_unet": 0.0007, "decay_steps":1000,
  "decay_rate_gan":0.9945, "decay_rate":0.989,
  "batch_size":32,"epochs":240,"discrim_steps":3, "norm":"log", "delta":0.01}
```

Once the configuration files have been modified for each of the experiments (please note all need to be manually modified), we can run these experiments by slurm.

## Running the Experiments
The experiments have been run using Python 3.8.5, but the experiments are generally compatible with >Python 3.6 and Tensorflow >2.5.

There are three important scripts when running the experiments
* run_bash_config.sh (submits slurm jobs for every config file in parallel)
* run_config_experiments.sl (the slurm configuration for each job)
* train_model.py (the script that train the model)

### Run_Bash_Config.sh

While only 128GB of memory is typically required for all these experiments (slurm jobs), we run all experiments, with a default of 256GB of memory on slurm.

To run all the experiments (with or without the intensity constraint), please execute the "run_bash_config.sh" file.
You will need to modify the following, for your specific operation.

Please see the comments below, on how to modify the file (run_bash_config.sh).

```bash
#!/bin/bash -l

module purge # optional
module load NeSI
module load cuDNN/8.1.1.33-CUDA-11.2.0 # make sure you check whether CUDA and cuDNN have been successfully loaded.
nvidia-smi

# Please change this to where your experiments are located. This will need to be repeated for both with and without the intensity constraint
# please modify the experiments individually by changing the directory.
directory="/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/paper_experiments/intensity_penalty"

if [ ! -d "$directory" ]; then
    echo "Directory does not exist: $directory"
    exit 1
fi


for file in "$directory"/*.json; do
    if [ -f "$file" ]; then
        # Process each .json file as needed
        echo "Processing file: $file"
        # please modify this line to ensure that you are running the correct script
        sbatch //nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/run_config_experiments.sl $file

        # Add your custom logic here to work with each file
        # For example, you could read the content of the file:
        # content=$(cat "$file")
        # echo "File content: $content"
    fi
done

### Run_Config_Experiments.sl
To modify the slurm file see below the configuration


#!/bin/bash -l
#SBATCH --job-name=GPU_job
#SBATCH --partition=hgx
#SBATCH --time=48:59:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=A100:1
#SBATCH --account=11111#(your account)
#SBATCH --mail-user=11111#(your email)
#SBATCH --mail-type=ALL
#SBATCH --output log/%j-%x.out #(please create a log folder or this will fail)
#SBATCH --error log/%j-%x.out #(please create a log folder or this will fail)

module purge # optional
module load NeSI
module load cuDNN/8.1.1.33-CUDA-11.2.0

nvidia-smi
# set the experiment name that we are implementing
# change directory into your folder
cd /nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/
# select your python interpreter and run the code
# please note that $1 allows us to pass a configuration file (.json) into the script.
/nesi/project/niwa00004/rampaln/bin/python /nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/unet_pretraining_nobn.py $1

### Train_unet.py
Before the GAN is trained a U-NET regression model is trained for a certain number of epochs (240). This same U-Net is then used
for all experiemnts.

\textbf{Please note}: The U-Net model configuration file has not been defined, but it is possible to simply use a pre-defined configuration file
from "GAN_intensity_penalty" or "GAN" folder directly.



### Train_model.py

This function simply loads the configuration file, and then runs all the experiments. (see the src folder for detailed information about the functions).

By default the configuration folders need to have "itensity" in their name, as the intensity weight is an integer, see line 29

```python
# configuration file is simply a string
if 'itensity' in config_file:
    itensity_weight = 1
else:
    itensity_weight =0
```




```

* src/
    * models.py
    * utils.py
* notebooks/
    * exploratory_analysis.ipynb
* README.md
* environment.yml
