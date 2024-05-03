import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
gcm = 'ACCESS-CM2'
df = xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/Reduced_Constraint/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr.nc')[['pr']]
y_true = xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/target_fields/{gcm}_hist_ssp370_pr_psl_tasmin_tasmax_sfcwind_sfcwindmax.nc')

quantile_signal_hist = df.sel(time = slice("1986","2005"))
quantile_signal_hist = quantile_signal_hist.pr.where(quantile_signal_hist.pr>0.0, np.nan).quantile(q = [0.5, 0.7, 0.9,0.95,0.97, 0.98, 0.99, 0.992, 0.995, 0.998,0.999],
                                                                                             dim ="time", skipna=True)

quantile_signal_future = df.sel(time = slice("2080","2099"))
quantile_signal_future = quantile_signal_future.pr.where(quantile_signal_future.pr>0.0, np.nan).quantile(q = [0.5, 0.7, 0.9,0.95,0.97, 0.98, 0.99, 0.992, 0.995, 0.998,0.999],
                                                                                             dim ="time", skipna=True)

y_true = y_true[['pr']]
y_true['time'] = pd.to_datetime(y_true.time.dt.strftime("%Y-%m-%d"))

quantile_signal_hist_gt = y_true.sel(time=slice("1986", "2005")) *86400
quantile_signal_hist_gt = quantile_signal_hist_gt.pr.where(quantile_signal_hist_gt.pr > 0.0, np.nan).quantile(
    q=[0.5,0.7, 0.9,0.95, 0.97, 0.98, 0.99, 0.992, 0.995, 0.998, 0.999],
    dim="time", skipna=True)

quantile_signal_future_gt = y_true.sel(time=slice("2080", "2099"))*86400
quantile_signal_future_gt = quantile_signal_future_gt.pr.where(quantile_signal_future_gt.pr > 0.0, np.nan).quantile(
    q=[0.5, 0.7, 0.9,0.95, 0.97, 0.98, 0.99, 0.992, 0.995, 0.998, 0.999],dim ="time",skipna=True)

signal = 100* (quantile_signal_future- quantile_signal_hist)/quantile_signal_hist
signal_gt = 100 * (quantile_signal_future_gt-quantile_signal_hist_gt)/quantile_signal_hist_gt

fig, ax = plt.subplots()
signal.mean(["lat","lon"]).plot(ax = ax, color ='r')
signal_gt.mean(["lat","lon"]).plot(ax =ax, color ='g')
fig.show()

fig, ax = plt.subplots(1,2, figsize = (12, 6))
signal.sel(quantile =0.998).plot(ax = ax[0], cmap ='BrBG', vmin =-50, vmax =50)
signal_gt.sel(quantile =0.998).plot(ax = ax[1], cmap ='BrBG', vmin =-50, vmax =50)
fig.show()

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