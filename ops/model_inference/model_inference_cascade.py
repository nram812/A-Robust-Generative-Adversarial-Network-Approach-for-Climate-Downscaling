import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import tqdm
from tqdm import tqdm
import tqdm
import numpy as np
from functools import partial
import json
# changed activation function to hyperbolic tangent
import tensorflow as tf
from tensorflow.keras import layers
AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pandas as pd
import sys
sys.path.append(os.getcwd())
from src.layers import *
from src.models import *
from src.gan import *
from ops.model_inference.src_eval_inference import *

with open(r'./ops/model_inference/metadata.json') as f:
    config = json.load(f)



model_dir = '/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/paper_experiments/models/'
filepath = '/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/ancil_fields/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc'
output_dir = '/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/paper_experiments/outputs'
means_filepath = "/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/inputs/ERA5/mean_1974_2011.nc"
stds_filepath = "/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/inputs/ERA5/std_1974_2011.nc"
variables = ['q_500', 'q_850', 'u_500', 'u_850', 'v_500', 'v_850', 't_500', 't_850']
models = ['EC-Earth3', 'ACCESS-CM2', 'NorESM2-MM']

# note first experiment is for the u-net architecture
# experiments = ['cascaded_perfect_framework_extreme_intensity_constraint','cascaded_perfect_framework_extreme_intensity_constraint',
#                 'cascaded_perfect_framework_high_adv_intensity_constraint', 
#                 'cascaded_perfect_framework_very_high_adv_intensity_constraint',
#                 'cascaded_perfect_framework_midrange_itensity_constraint',
#                 'cascaded_perfect_framework_low_adv_intensity_constraint',
#                 'cascaded_perfect_framework_lowrange_itensity_constraint']


# note first experiment is for the u-net architecture
# experiments = ['cascaded_perfect_framework_extreme_intensity_constraint_no_bn_fixed_unet_red_intensity',
#                'cascaded_perfect_framework_extreme_intensity_constraint_no_bn_fixed_unet_red_intensity',
#                'cascaded_perfect_framework_high_adv_intensity_constraint_no_bn_fixed_unet_red_intensity',
#                'cascaded_perfect_framework_low_adv_intensity_constraint_no_bn_fixed_unet_red_intensity',
#                'cascaded_perfect_framework_lowrange_itensity_constraint_no_bn_fixed_unet_red_intensity',
#                'cascaded_perfect_framework_midrange_itensity_constraint_no_bn_fixed_unet_red_intensity',
#                'cascaded_perfect_framework_very_high_adv_intensity_constraint_no_bn_fixed_unet_red_intensity',
#                ]

experiments = ['cascaded_perfect_framework_extreme_intensity_constraint_no_bn_fixed_unet_red_intensity',
               'cascaded_perfect_framework_extreme_intensity_constraint_no_bn_fixed_unet_red_intensity',
               'cascaded_perfect_framework_high_adv_intensity_constraint_no_bn_fixed_unet_red_intensity',
               'cascaded_perfect_framework_low_adv_intensity_constraint_no_bn_fixed_unet_red_intensity',
               'cascaded_perfect_framework_lowrange_itensity_constraint_no_bn_fixed_unet_red_intensity',
               'cascaded_perfect_framework_midrange_itensity_constraint_no_bn_fixed_unet_red_intensity',
               'cascaded_perfect_framework_very_high_adv_intensity_constraint_no_bn_fixed_unet_red_intensity',
               ]

experiments = ['cascaded_perfect_framework_extreme_linear_act',
               'cascaded_perfect_framework_extreme_linear_act',
               'cascaded_perfect_framework_high_adv_linear_act',
               'cascaded_perfect_framework_low_adv_linear_act',
               'cascaded_perfect_framework_lowrange_linear_act',
               'cascaded_perfect_framework_midrange_linear_act',
               'cascaded_perfect_framework_very_high_adv_linear_act',
               ]
output_model_name = 'LeakyRelu_Activation'

# experiments = ['cascaded_perfect_framework_extreme','base_model_extreme',
#                 'base_model_high_adv', 
#                 'base_model_very_high_adv',
#                 'base_model_low_adv',
#                 'base_model_lowrange', 'base_model_verylow_adv']
epochs = 130

n_members = 10
timeslice = slice("1986", "2005")
unet_path = f"/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/paper_experiments/models/cascaded_perfect_framework_very_high_adv_intensity_constraint/unet_epoch_380.h5"
fixed_unet = False

# need to write an os.listdir for all these experiments


for model in models:
    # ensuring that the data is cleared
    # load the imperfect_conditions
    vegt, orog, he = load_and_normalize_topography_data(config["static_predictors"])
    predictor, ground_truth = load_data_historical(model, config)
    stacked_predictors = normalize_and_stack(predictor, config["mean"], config["std"],
                                              config["var_names"])
    common_time_imperfect = stacked_predictor.sel(time=timeslice).time.to_index().intersection(ground_truth.time.to_index())

    output_prediction = run_experiments(experiments, epochs, model_dir,
                                        stacked_predictors, common_time_imperfect, ground_truth, orog, he, vegt, n_members,
                                        batch_size=64)
    output_prediction.to_netcdf(
        f'{output_dir}/{output_model_name}/{model}_{output_model_name}_hist_1986_2005_cascaded_imperfect_applied.nc')
    # computing validation metrics
    # validation_metrics = ValidationMetric(output_prediction)
    # validation_metrics = validation_metrics(
    #              thresh =1)
    # validation_metrics.to_netcdf(f'{output_dir}/{model}_hist_1986_2005_cascaded_imperfect_applied_val_metrics.nc')

    # load the perfect conditions