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
os.chdir(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling')
sys.path.append(os.getcwd())
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *
from ops.model_inference.src_eval_inference import *
config_file = r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/models/historical_run_precip_residual_new_unet_shuffle_off_new_iten_weight_new_padding_new_unet_V2_resid_now_01_iten_ACCESS-CM2/config_info.json'

# y['tasmin'] = (y['tasmin'] - output_means['tasmin'])/output_stds['tasmin']
# min_value = y.tasmin.min()
# y['tasmin'] = y['tasmin'] - min_value

os.chdir(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling')

with open(config_file) as f:
    config = json.load(f)
quantiles = [ 0.5 , 0.7, 0.9, 0.925,
             0.95, 0.975, 0.98, 0.99,
             0.995, 0.998, 0.999]
historical_period = slice("1985","2014")
future_period = slice("2070","2099")
config["means_output"] = "/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/normalization/target_spatial_norm_all_gcm_mean.nc"
config["stds_output"] = "/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/normalization/target_spatial_norm_all_gcm_std.nc"
output_means = xr.open_dataset(config["means_output"])
output_stds = xr.open_dataset(config["stds_output"])

def compute_quantiles(df,quantiles, period):
    df = df.sel(time = period)
    seasonal_rainfall = df.groupby('time.season').mean()
    df = df.where(df>0, np.nan)
    quantiled_rain = df.quantile(q = quantiles, dim =["time"], skipna =True)
    return quantiled_rain, seasonal_rainfall


def compute_signal(df, quantiles, historical_period, future_period):

    historical_quantiles, seasonal_rainfall = compute_quantiles(df, quantiles, historical_period)
    future_quantiles, future_rainfall = compute_quantiles(df, quantiles, future_period)

    cc_signal = 100 * (future_rainfall - seasonal_rainfall)/seasonal_rainfall
    signal = 100 * (future_quantiles - historical_quantiles)/historical_quantiles
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
with tf.device('/CPU:0'):
    gan, unet, adv_factor = load_model_cascade(config["model_name"], None, './models', load_unet=True)
# try:
#     y = y.isel(GCM =0)[['pr']]
# except:
#     y =y[['pr']]
for gcm in stacked_X.GCM.values:
    print(f"prepraring data fpr a GCM {gcm}")
    with tf.device('/CPU:0'):
        output_shape = create_output(stacked_X.isel(time = slice(0, 300)), y.isel(time = slice(0, 300)))
        output_shape.pr.values = output_shape.pr.values * 0.0
        output_hist = xr.concat([predict_parallel_resid(gan, unet,
                                       stacked_X.sel( GCM =gcm).transpose('time','lat','lon','channel').isel(time = slice(0, 300)).values,
                                       output_shape[['pr']].isel(time = slice(0, 300)).sel(GCM = gcm), 8, orog.values, he.values, vegt.values, model_type='GAN') for i in range(1)],
                                dim ="member")

    # output_future = xr.concat([predict_parallel_resid(gan, unet,
    #                                      stacked_X.sel(GCM=gcm).transpose('time', 'lat', 'lon', 'channel').sel(
    #                                          time=future_period).values,
    #                                      output_shape.sel(time=future_period), 64, orog.values, he.values,
    #                                      vegt.values, model_type='GAN') for i in range(1)], dim ="member")
    # outputs = xr.concat([output_hist, output_future], dim ="time")
    # outputs = compute_signal(outputs[['pr']], quantiles, historical_period, future_period)
    # #outputs.attrs['title'] = outputs.attrs['title'] + f'   /n ML Emulated NIWA-REMS GAN v1 GCM: {gcm}'
    # if not os.path.exists(f'./outputs/{config["model_name"]}'):
    #     os.makedirs(f'./outputs/{config["model_name"]}')
    # outputs.to_netcdf(f'./outputs/{config["model_name"]}/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr_ens.nc')
    # with open(f'./outputs/{config["model_name"]}/config_info.json', 'w') as f:
    #     json.dump(config, f)

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
times = output_hist.time.to_index().intersection(y.time.to_index())
output_hist = output_hist.sel(time = times)
y1 = y.sel(time = times)
time =31
fig, ax = plt.subplots(1,2, figsize = (12, 8))
output_hist.pr.isel(time =time).plot(ax=ax[0], cmap ='jet', vmin =0, vmax =90)

((y1).pr.isel(time =time).sel( GCM =gcm)*86400).plot(ax=ax[1], cmap ='jet', vmin =0, vmax =90)
fig.show()

corrs = xr.corr(output_hist.pr, y1.pr.sel(GCM =gcm), dim ="time")
fig, ax = plt.subplots()
corrs.plot(ax = ax)
fig.show()

fig, ax = plt.subplots(1,2, figsize = (12, 8))
output_hist.groupby('time.season').mean().pr.isel(season =1).plot(cmap ='BrBG', ax = ax[0], vmax =6)
(y1.pr *86400).sel(time = output_hist.time, GCM =gcm).groupby('time.season').mean().isel(season =1).plot(cmap ='BrBG', ax = ax[1], vmax =6)
fig.show()



import seaborn as sns
bins = np.arange(0,600,10)
fig,ax = plt.subplots()
sns.histplot(output_hist.pr.values.ravel(), bins = bins, ax = ax, label ='Emulator', color ='r', alpha =0.5)
sns.histplot(y1.pr.sel(time = output_hist.time, GCM =gcm).values.ravel() * 86400, bins = bins, ax = ax, label ='Emulator', color ='b', alpha =0.5)
ax.set_yscale('log')
ax.legend()
fig.show()