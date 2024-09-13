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
config_file = sys.argv[-1]


os.chdir(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling')

with open(config_file) as f:
    config = json.load(f)
quantiles = [ 0.5 , 0.7, 0.9, 0.925,
             0.95, 0.975, 0.98, 0.99,
             0.995, 0.998, 0.999]
historical_period = slice("1985","2014")
future_period = slice("2070","2099")
#config["means_output"] = "/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/normalization/target_spatial_norm_all_gcm_mean.nc"
#config["stds_output"] = "/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/normalization/target_spatial_norm_all_gcm_std.nc"
output_means = xr.open_dataset(config["means_output"])
output_stds = xr.open_dataset(config["stds_output"])
#config['tmin_min_value'] = -6.535853862762451
# the above is true for access-cm2
def compute_quantiles(df,quantiles, period):
    df = df.sel(time = period)
    seasonal_rainfall = df.groupby('time.season').mean()
    quantiled_rain = df.quantile(q = quantiles, dim =["time"], skipna =True)
    return quantiled_rain, seasonal_rainfall


def compute_signal(df, quantiles, historical_period, future_period):

    historical_quantiles, seasonal_rainfall = compute_quantiles(df, quantiles, historical_period)
    future_quantiles, future_rainfall = compute_quantiles(df, quantiles, future_period)

    cc_signal = future_rainfall - seasonal_rainfall
    signal = (future_quantiles - historical_quantiles)
    historical_quantiles = historical_quantiles.rename({"pr":"hist_quantiles"})
    future_quantiles = future_quantiles.rename({"pr": "future_quantiles"})
    seasonal_rainfall = seasonal_rainfall.rename({"pr":"hist_clim_rainfall"})
    future_rainfall = future_rainfall.rename({"pr":"future_clim_rainfall"})
    signal = signal.rename({"pr":"cc_signal"})
    cc_signal = cc_signal.rename({"pr":"seas_cc_signal"})
    dset = xr.merge([historical_quantiles, future_quantiles,
                     signal, cc_signal, seasonal_rainfall, future_rainfall])
    return dset

config["train_x"] ="/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/predictor_fields/predictor_fields_hist_ssp370_merged_updated.nc"
config["train_y"] = "/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/target_fields/target_fields_hist_ssp370_concat.nc"
stacked_X, y, vegt, orog, he = preprocess_input_data(config, match_index =False)
print("Normalized by the Mean (of the Spatial fields)")

gan, unet, adv_factor = load_model_cascade(config["model_name"], None, './models', load_unet=True)
try:
    y = y.isel(GCM =0)[['pr']]
except:
    y =y[['pr']]
for gcm in stacked_X.GCM.values:
    print(f"prepraring data fpr a GCM {gcm}")
    output_shape = create_output(stacked_X, y)
    output_shape.pr.values = output_shape.pr.values * 0.0
    output_hist = xr.concat([predict_parallel_resid_t(gan, unet,
                                   stacked_X.sel( GCM =gcm).transpose('time','lat','lon','channel').sel(time = historical_period).values,
                                       output_shape.sel(time = historical_period), 16, orog.values, he.values, vegt.values, model_type='GAN',
                                                      output_means = output_means, output_stds = output_stds, config = config) for i in range(1)],
                            dim ="member")

    output_future = xr.concat([predict_parallel_resid_t(gan, unet,
                                         stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
                                             time=future_period).values,
                                         output_shape.sel(time=future_period), 16, orog.values, he.values,
                                         vegt.values, model_type='GAN',
                                         output_means = output_means, output_stds = output_stds, config = config) for i in range(1)], dim ="member")
    outputs = xr.concat([output_hist, output_future], dim ="time")
    if not os.path.exists(f'./outputs/{config["model_name"]}'):
        os.makedirs(f'./outputs/{config["model_name"]}')
    outputs.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_tmin_raw_output_GAN.nc')
    outputs = compute_signal(outputs[['pr']], quantiles, historical_period, future_period)
    outputs.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_tmin_unet_for_GAN.nc')
    with open(f'./outputs/{config["model_name"]}/config_info.json', 'w') as f:
        json.dump(config, f)

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