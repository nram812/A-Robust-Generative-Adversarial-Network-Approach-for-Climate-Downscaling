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

with open(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/experiment_configs/GAN_intensity_penalty/modified_intensity_constraint_value_very_high_adv.json') as f:
    config = json.load(f)
config["train_x"] ="/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/predictor_fields/predictor_fields_hist_ssp370_merged_updated.nc"
stacked_X, y, vegt, orog, he = preprocess_input_data(config, match_index =False)
gan, unet, adv_factor = load_model_cascade('Rain_Model_Mod_Intensity_Constraint', None, './models', load_unet=True)
try:
    y = y.isel(GCM =0)[['pr']]
except:
    y =y[['pr']]
for gcm in stacked_X.GCM.values:
    print(f"prepraring data fpr a GCM {gcm}")
    output_shape = create_output(stacked_X, y)
    output_shape.pr.values = output_shape.pr.values * 0.0
    #with tf.device('/GPU:0'):
    outputs = predict_parallel_resid(gan, unet,
                                   stacked_X.sel( GCM =gcm).transpose('time','lat','lon','channel').values,
                                   output_shape, 128, orog.values, he.values, vegt.values, model_type='GAN')
    outputs.attrs['title'] = outputs.attrs['title'] + f'   /n ML Emulated NIWA-REMS GAN v1 GCM: {gcm}'
    #outputs['ML ver'] = 'NIWA-REMS Rain V2 hist ssp370, ACCESS-CM2 only trained'
    outputs.to_netcdf(f'./outputs/Reduced_Constraint/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr.nc')

    # with tf.device('/CPU:0'):
    #     outputs = predict_parallel_resid(gan, unet,
    #                                stacked_X.sel( GCM =gcm).isel(time = slice(0,800)).transpose('time','lat','lon','channel').values,
    #                                output_shape.isel(time = slice(0,800)), 32, orog.values, he.values, vegt.values, model_type='GAN')
    #     outputs.attrs['title'] = outputs.attrs['title'] + f'   /n ML Emulated NIWA-REMS GAN v1 GCM: {gcm}'
    # outputs['ML ver'] = 'NIWA-REMS Rain V2 hist ssp370, ACCESS-CM2 only trained'
    #     # computing validation metrics
        # validation_metrics = ValidationMetric(output_prediction)
        # validation_metrics = validation_metrics(
        #              thresh =1)
        # validation_metrics.to_netcdf(f'{output_dir}/{model}_hist_1986_2005_cascaded_imperfect_applied_val_metrics.nc')

        # load the perfect conditions