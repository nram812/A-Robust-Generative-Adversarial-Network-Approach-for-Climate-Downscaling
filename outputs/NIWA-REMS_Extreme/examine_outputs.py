import xarray as xr
import matplotlib.pyplot as plt
gcm = 'NorESM2-MM'
df = xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/A-Robust-Generative-Adversarial-Network-Approach-for-Climate-Downscaling/outputs/NIWA-REMS/CCAM_NIWA-REMS_{gcm}_hist_ssp370_pr.nc')
y_true = xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/target_fields/{gcm}_hist_ssp370_pr_psl_tasmin_tasmax_sfcwind_sfcwindmax.nc')
y_true = y_true[['pr']]
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