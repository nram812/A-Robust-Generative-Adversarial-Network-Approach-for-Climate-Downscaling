import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import tqdm
from tqdm import tqdm
import tqdm
import tensorflow as tf
import numpy as np
# DisableqdGPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from comet_ml import Experiment
import numpy as np
import tensorflow as tf
import albumentations as A
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar

import pandas as pd

# Create an experiment with your api key
sys.path.append(r'/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN')
from src_unet_init_step import *
from paper_experiments.analyse_experiments_src import *
from tensorflow.keras import layers
import pandas as pd
import json
# changed activation function to hyperbolic tangent
import tensorflow as tf

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
unet_activation_function = lambda x: 8.25 * tf.keras.activations.tanh(x / 2.5)


def run_experiments(experiments, epoch_list, model_dir,
                    input_predictors, common_times, output_shape, orog, he, vegt, n_members, batch_size=64,
                    fixed_unet=True, fixed_unet_model_path=unet_path,
                    fixed_unet_activation=unet_activation_function):
    """
    Runs inference on some predictor fields in/ out-of-sample


    experiments: list of experiment names in the model_dir folder
    input_predictors: stacked netcdf with dims (time, lat, lon, channel) and normalized data
    common_times: common_times between output_shape data and input_predictors
    output_shape: a netcdf (y_true) that is the same shape as the output prediction, it contains the time metadata
    orog, he, vegt: auxiliary files from ccam
    n_member: the number of ensemble members

    """

    # if the epoch list is only a float convert to a list
    if isinstance(epoch_list, int):
        epoch_list = [epoch_list] * len(experiments)

    # creating empty lists to save outputs            
    dsets = []
    lambda_var = []
    for i, experiment in enumerate(experiments):
        if 'cascade' in experiment:
            if fixed_unet:
                gan, lambdas = load_model_cascade(experiment,
                                                    epoch_list[i], model_dir, load_unet = False)

                unet = tf.keras.models.load_model(fixed_unet_model_path,
                                                  custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D,
                                                                  "<lambda>": fixed_unet_activation}, compile=False)
            else:

                gan, unet, lambdas = load_model_cascade(experiment,
                                                    epoch_list[i], model_dir)

            if i == 0:
                # first instance is always a unet model
                lambdas = 0.0
                preds = xr.concat([predict_parallel_resid(gan, unet,
                                                          input_predictors.sel(time=common_times).values,
                                                          output_shape.sel(time=common_times),
                                                          batch_size, orog, he, vegt, model_type='unet')
                                   for i in range(n_members)],
                                  dim="member")
            else:
                # do not change lambdas value otherwise
                lambdas = lambdas
                preds = xr.concat([predict_parallel_resid(gan, unet,
                                                          input_predictors.sel(time=common_times).values,
                                                          output_shape.sel(time=common_times),
                                                          batch_size, orog, he, vegt, model_type='GAN')
                                   for i in range(n_members)],
                                  dim="member")


        else:
            gan, lambdas = load_model_reg(experiment,
                                          epoch_list[i], model_dir)
            preds = xr.concat([predict_parallel_v1(gan,
                                                   input_predictors.sel(time=common_times).values,
                                                   output_shape.sel(time=common_times),
                                                   batch_size, orog, he, vegt, model_type='GAN')
                               for i in range(n_members)],
                              dim="member")
        lambda_var.append(lambdas)
        dsets.append(preds)
        # finish the experiment and concatenate the data
    dsets = xr.concat(dsets, dim="experiment")
    dsets['experiment'] = (('experiment'), lambda_var)
    dsets = dsets.reindex(experiment=sorted(dsets.experiment.values))
    return dsets


for model in models:
    # ensuring that the data is cleared
    # load the imperfect_conditions
    vegt, orog, he = load_and_normalize_topography_data(filepath)
    df_raw, df_imperfect, y_true = load_and_concatenate_hist(model)
    stacked_X_imperfect = normalize_and_stack(df_imperfect, means_filepath, stds_filepath, variables)
    common_time_imperfect = stacked_X_imperfect.sel(time=timeslice).time.to_index().intersection(y_true.time.to_index())

    output_prediction = run_experiments(experiments, epochs, model_dir,
                                        stacked_X_imperfect, common_time_imperfect, y_true, orog, he, vegt, n_members,
                                        batch_size=64)
    output_prediction.to_netcdf(
        f'{output_dir}/{output_model_name}/{model}_{output_model_name}_hist_1986_2005_cascaded_imperfect_applied.nc')
    # computing validation metrics
    # validation_metrics = ValidationMetric(output_prediction)
    # validation_metrics = validation_metrics(
    #              thresh =1)
    # validation_metrics.to_netcdf(f'{output_dir}/{model}_hist_1986_2005_cascaded_imperfect_applied_val_metrics.nc')

    # load the perfect conditions
    df_raw_perfect, df_perfect, y_true = load_trainig_data_historical(model)
    stacked_X_perfect = normalize_and_stack(df_perfect, means_filepath, stds_filepath, variables)
    common_time_perfect = stacked_X_perfect.sel(time=timeslice).time.to_index().intersection(y_true.time.to_index())

    # running the experiments
    output_prediction = run_experiments(experiments, epochs, model_dir,
                                        stacked_X_perfect, common_time_perfect, y_true, orog, he, vegt, n_members,
                                        batch_size=64)
    output_prediction.to_netcdf(
        f'{output_dir}/{output_model_name}/{model}_{output_model_name}_hist_1986_2005_cascaded_perfect_applied.nc')

    # lets compute a series of validation metrics for these

# # LOAD THE metadata for inference
# # looping for imperfect conditions
# for model in models:

#     dsets = []
#     lambda_var =[]
#     # ensuring that the data is cleared
#     # load the imperfect_conditions
#     vegt, orog, he = load_and_normalize_topography_data(filepath)
#     df_raw, df_imperfect, y_true = load_and_concatenate_hist(model)
#     stacked_X_imperfect = normalize_and_stack(df_imperfect, means_filepath, stds_filepath, variables)

#     common_time_imperfect = stacked_X_imperfect.sel(time = timeslice).time.to_index().intersection(y_true.time.to_index())


#     for i, experiment in enumerate(experiments):
#         gan, unet, lambdas = load_model_cascade(experiment,
#                                                 220, model_dir)
#         # getting the model at 110 epochs
#         if i ==0:
#             lambdas = 0.0
#             preds = xr.concat([predict_parallel_resid(gan,unet,
#                                     stacked_X_imperfect.sel(time = common_time_imperfect).values,
#                                     y_true.sel(time = common_time_imperfect),
#                                     64,orog, he, vegt,model_type ='unet') 
#                                     for i in range(n_members)], dim ="member")

#         else:
#             preds = xr.concat([predict_parallel_resid(gan,unet,
#                                     stacked_X_imperfect.sel(time = common_time_imperfect).values,
#                                     y_true.sel(time = common_time_imperfect),
#                                     64,orog, he, vegt, model_type ='GAN') 
#                                     for i in range(n_members)], dim ="member")
#         lambda_var.append(lambdas)
#         dsets.append(preds)
#     # finish the experiment and concatenate the data
#     dsets = xr.concat(dsets, dim ="experiment")
#     dsets['experiment'] = (('experiment'), lambda_var)
#     dsets = dsets.reindex(experiment = sorted(dsets.experiment.values)) 
#     dsets.to_netcdf(f'{output_dir}/{model}_hist_1986_2005_cascaded_imperfect_applied.nc')
#     # computing validation metrics
#     validation_metrics = ValidationMetric(dsets)
#     validation_metrics = validation_metrics(output_grid = dsets.isel(member =0, time =0, experiment =0),
#                  thresh =1)
#     validation_metrics.to_netcdf(f'{output_dir}/{model}_hist_1986_2005_cascaded_imperfect_applied_val_metrics.nc')

#     # load the perfect conditions
#     df_raw_perfect, df_perfect, y_true = load_trainig_data_historical(model)
#     stacked_X_perfect = normalize_and_stack(df_perfect, means_filepath, stds_filepath, variables)
#     common_time_perfect = stacked_X_perfect.sel(time = timeslice).time.to_index().intersection(y_true.time.to_index())


#     for i, experiment in enumerate(experiments):
#         gan, unet, lambdas = load_model_cascade(experiment,
#                                                 220, model_dir)
#         # getting the model at 110 epochs
#         if i ==0:
#             lambdas = 0.0
#             preds = xr.concat([predict_parallel_resid(gan,unet,
#                                     stacked_X_perfect.sel(time = common_time_perfect).values,
#                                     y_true.sel(time = common_time_perfect),
#                                     64,orog, he, vegt,model_type ='unet') 
#                                     for i in range(n_members)], dim ="member")

#         else:
#             preds = xr.concat([predict_parallel_resid(gan,unet,
#                                     stacked_X_imperfect.sel(time = common_time_perfect).values,
#                                     y_true.sel(time = common_time_perfect),
#                                     64,orog, he, vegt, model_type ='GAN') 
#                                     for i in range(n_members)], dim ="member")
#         lambda_var.append(lambdas)
#         dsets.append(preds)
#     # finish the experiment and concatenate the data
#     dsets = xr.concat(dsets, dim ="experiment")
#     dsets['experiment'] = (('experiment'), lambda_var)
#     dsets = dsets.reindex(experiment = sorted(dsets.experiment.values)) 
#     dsets.to_netcdf(f'{output_dir}/{model}_hist_1986_2005_cascaded_perfect_applied.nc')

#     # lets compute a series of validation metrics for these
#     validation_metrics = ValidationMetric(dsets)
#     validation_metrics = validation_metrics(output_grid = dsets.isel(member =0, time =0, experiment =0),
#                  thresh =1)
#     validation_metrics.to_netcdf(f'{output_dir}/{model}_hist_1986_2005_cascaded_perfect_applied_val_metrics.nc')
