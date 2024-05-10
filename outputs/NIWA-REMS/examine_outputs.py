import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import json
sys.path.append(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling')
from src.process_input_training_data import *
gcm = 'ACCESS-CM2'
df = xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/NIWA_REMS_high_adv_with_constraint/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr.nc')[['pr']]
y_true = xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/target_fields/{gcm}_hist_ssp370_pr_psl_tasmin_tasmax_sfcwind_sfcwindmax.nc')
config_file = r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/cascaded_perfect_framework_extreme_intensity_constraint/config_info.json'
with open(config_file, 'r') as f:
    config = json.load(f)

__, orog, __ = prepare_static_fields(config)
import seaborn as sns
import matplotlib.pyplot as plt
bins = np.arange(0,500,10)
fig, ax= plt.subplots()
sns.histplot(df.isel(time = slice(0,600)).pr.values.ravel(), bins = bins,
             ax=ax, alpha =0.6)
sns.histplot(y_true.isel(time = slice(0,600)).pr.values.ravel()*86400, bins = bins, ax=ax, color ='r',
             alpha =0.5)

ax.set_yscale('log')
fig.show()
quantile_signal_hist = df.sel(time = slice("1985","2014"))
z =quantile_signal_hist.groupby('time.season').mean()
fig, ax= plt.subplots()
z.pr.isel(season =1).plot(ax = ax, vmax =7, cmap ='BrBG')
fig.show()

# #
# fig, ax= plt.subplots()
# z2.pr.isel(season =1).plot(ax = ax, vmax =7, cmap ='BrBG')
# fig.show()

quantile_signal_hist = quantile_signal_hist.pr.where(quantile_signal_hist.pr>0.0, np.nan).quantile(q = [0.5, 0.7, 0.9,0.95,0.97, 0.98, 0.99, 0.992, 0.995, 0.998,0.999],
                                                                                             dim ="time", skipna=True)

quantile_signal_future = df.sel(time = slice("2070","2099"))
quantile_signal_future = quantile_signal_future.pr.where(quantile_signal_future.pr>0.0, np.nan).quantile(q = [0.5, 0.7, 0.9,0.95,0.97, 0.98, 0.99, 0.992, 0.995, 0.998,0.999],
                                                                                             dim ="time", skipna=True)

y_true = y_true[['pr']]
y_true['time'] = pd.to_datetime(y_true.time.dt.strftime("%Y-%m-%d"))

quantile_signal_hist_gt = y_true.sel(time=slice("1985", "2014")) *86400
quantile_signal_hist_gt = quantile_signal_hist_gt.pr.where(quantile_signal_hist_gt.pr > 0.0, np.nan).quantile(
    q=[0.5,0.7, 0.9,0.95, 0.97, 0.98, 0.99, 0.992, 0.995, 0.998, 0.999],
    dim="time", skipna=True)

quantile_signal_future_gt = y_true.sel(time=slice("2070", "2099"))*86400
quantile_signal_future_gt = quantile_signal_future_gt.pr.where(quantile_signal_future_gt.pr > 0.0, np.nan).quantile(
    q=[0.5, 0.7, 0.9,0.95, 0.97, 0.98, 0.99, 0.992, 0.995, 0.998, 0.999],dim ="time",skipna=True)

signal = 100* (quantile_signal_future- quantile_signal_hist)/quantile_signal_hist
signal_gt = 100 * (quantile_signal_future_gt-quantile_signal_hist_gt)/quantile_signal_hist_gt

fig, ax = plt.subplots()
signal.mean(["lat","lon"]).plot(ax = ax, color ='r')
#(signal.mean(["lat","lon"]) +signal.std(["lat","lon"])).plot(ax = ax, color ='r')
#(signal.mean(["lat","lon"]) -signal.std(["lat","lon"])).plot(ax = ax, color ='r')
signal_gt.mean(["lat","lon"]).plot(ax =ax, color ='g')
ax.set_ylabel('NZ-averaged Climate Change Signal (%) \n (2080-2099) - (1985-2005)', fontsize =12, weight ='bold')
#fig.savefig(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/figures/cc_signal_ec_earth3_orig_model_new_model.png', dpi =400)
fig.show()

fig, ax = plt.subplots()
import xarray as xr
import matplotlib.pyplot as plt
gcms = ['ACCESS-CM2', 'AWI-CM-1-1-MR', 'CNRM-CM6-1', 'EC-Earth3','NorESM2-MM']
dfs = []
for gcm in gcms:
    df = xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/cascaded_perfect_framework_very_high_adv_intensity_constraint_mod_dis/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr.nc')
    dfs.append(df)
signals = xr.concat(dfs, dim ="GCM")
signals['GCM'] = (('GCM'), gcms)

fig, ax = plt.subplots()
for gcm in gcms:
    signals.cc_signal.mean(["lat","lon"]).sel(GCM =gcm).plot(ax = ax, label =gcm)
ax.legend()
fig.show()


model ='cascaded_perfect_framework_extreme_intensity_constraint'
#fig, ax = plt.subplots()
signal_gan = xr.open_dataset(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/cascaded_perfect_framework_extreme_intensity_constraint/CCAM_NIWA-REMS_ACCESS-CM2_hist_ssp370_pr_ens.nc')
signal_gan = xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/{model}/CCAM_NIWA-REMS_ACCESS-CM2_hist_ssp370_pr.nc')
signal_gt = xr.open_dataset(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/Val_metrics/cc_signal/gt_ACCESS-CM2.nc')
signal_unet = xr.open_dataset(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/Val_metrics/cc_signal/baseline_ACCESS-CM2.nc')

fig, ax = plt.subplots(figsize =(6, 6))
signal_gan.cc_signal.mean(["lat","lon"]).plot(ax = ax, color ='g', label ='GAN', marker ='o')
signal_unet.mean(["lat","lon"]).pr.plot(ax = ax, color ='b', label ='UNET', marker ='o')
signal_gt.mean(["lat","lon"]).pr.plot(ax = ax, color ='k', label ='CCAM', marker ='o')
ax.set_ylabel('NZ-averaged \n Climate Change Signal (%)', fontsize =12, weight ='bold')

ax.grid('on')
ax.axhline(0.0, ls ='--')
ax.axhline(3.25 * 7, ls ='--', label = 'Clausius Clapeyron', color ='r')
ax.legend()
ax.set_xlim(0.5, 1.01)
# signal_gan.seas_cc_signal.plot(col ="season", col_wrap =2, cmap ='BrBG', vmax =50, vmin =-50)
# plt.show()
#ax.set_xscale('log')
fig.show()


fig, ax = plt.subplots(figsize =(6, 6))
(signal_gan.where(orog>0, np.nan)).mean(["lat","lon"]).cc_signal.plot(ax = ax, color ='g', label ='GAN', marker ='o')
(signal_unet.where(orog>0, np.nan)).mean(["lat","lon"]).pr.plot(ax = ax, color ='b', label ='UNET', marker ='o')
(signal_gt.where(orog>0, np.nan)).mean(["lat","lon"]).pr.plot(ax = ax, color ='k', label ='CCAM', marker ='o')
ax.set_ylabel('Land \n Climate Change Signal (%)', fontsize =12, weight ='bold')

ax.grid('on')
ax.axhline(0.0, ls ='--')
ax.axhline(3.25 * 7, ls ='--', label = 'Clausius Clapeyron', color ='r')
ax.legend()
#ax.set_xscale('log')
fig.show()

fig, ax = plt.subplots(figsize =(6, 6))
(signal_gan.where(orog<=0, np.nan)).mean(["lat","lon"]).cc_signal.plot(ax = ax, color ='g', label ='GAN', marker ='o')
(signal_unet.where(orog<=0, np.nan)).mean(["lat","lon"]).pr.plot(ax = ax, color ='b', label ='UNET', marker ='o')
(signal_gt.where(orog<=0, np.nan)).mean(["lat","lon"]).pr.plot(ax = ax, color ='k', label ='CCAM', marker ='o')
ax.set_ylabel('Ocean \n Climate Change Signal (%)', fontsize =12, weight ='bold')

ax.grid('on')
ax.axhline(0.0, ls ='--')
ax.axhline(3.25 * 7, ls ='--', label = 'Clausius Clapeyron', color ='r')
ax.legend()
#ax.set_xscale('log')
fig.show()


fig, ax = plt.subplots(1,3, figsize = (18, 6))
(signal_gan.cc_signal).sel(quantile =0.995).plot(ax = ax[1], cmap ='BrBG', vmin =-30, vmax =30)
(signal_gt.pr).sel(quantile =0.995).plot(ax = ax[2], cmap ='BrBG', vmin =-30, vmax =30)
(signal_unet.pr).sel(quantile =0.995).plot(ax = ax[0], cmap ='BrBG', vmin =-30, vmax =30)
ax[0].set_title('UNet')
ax[1].set_title('GAN')
ax[2].set_title('CCAM')
#fig.savefig(r'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/figures/cc_signal_ec_earth3_spatial_0.7_new_model.png', dpi =400)

fig.show()

signal_gan.seas_cc_signal.plot(col ="season", cmap ='BrBG', vmax =50)
plt.show()

    rx1day = df.groupby('time.year').max()
y_true['time'] =pd.to_datetime(y_true.time.dt.strftime("%Y-%m-%d"))
rx1day_true = y_true.groupby('time.year').max()

cc_signal_hist = rx1day.sel(year= slice("1986","2005")).mean("year")
cc_signal_future = rx1day.sel(year = slice("2080","2099")).mean("year")
signal = 100 * (cc_signal_future - cc_signal_hist)/cc_signal_hist

cc_signal_hist_true = rx1day_true.sel(year= slice("1986","2005")).mean("year")
cc_signal_future_true = rx1day_true.sel(year = slice("2080","2099")).mean("year")
signal_true = 100 * (cc_signal_future_true - cc_signal_hist_true)/cc_signal_hist_true

fig, ax = plt.subplots(1, 2, figsize = (12, 6))
signal.pr.plot(cmap ='BrBG', vmin =-30, vmax =30, ax = ax[0])
signal_true.pr.plot(cmap ='BrBG', vmin =-30, vmax =30, ax = ax[1])
ax[0].set_title('Emulated')
ax[1].set_title('GT')
fig.show()

cc_signal_hist = rx1day.sel(year= slice("1986","2005")).mean("year")
cc_signal_future = rx1day#.sel(year = slice("2080","2099")).mean("year")
signal = 100 * (cc_signal_future - cc_signal_hist)/cc_signal_hist

fig, ax = plt.subplots()
signal.mean(["lat","lon"]).pr.plot()
fig.show()