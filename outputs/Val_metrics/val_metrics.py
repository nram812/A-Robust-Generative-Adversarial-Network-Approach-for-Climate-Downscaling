import xarray as xr
import sys
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

fname = sys.argv[-2]
outfname = sys.argv[-1]
quantiles = [0.5, 0.75, 0.85, 0.9, 0.925, 0.95, 0.975, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999]
historical_period = slice("1985","2014")
future_period = slice("2070","2099")
# load the dataset


def calc_quantiles(df, quantiles, period):

    subset = df.sel(time = period)
    subset = subset.where(subset>0.0, np.nan)
    return subset.quantile(q = quantiles,  dim="time", skipna =True)


def compute_signal(fname, quantiles, historical_period, future_period, out_fname):
    try:
        df = xr.open_dataset(fname, chunks={"GCM": 1})
    except:
        df = xr.open_dataset(fname)
    hist_q = calc_quantiles(df, quantiles, historical_period)
    future_q = calc_quantiles(df, quantiles, future_period)
    signal = 100 * ((future_q - hist_q)/hist_q)
    fname_split = fname.split('/')[-1]
    signal.to_netcdf(f'{out_fname}/{fname_split}_climate_change_signal.nc')
    hist_q.to_netcdf(f'{out_fname}/{fname_split}_hist_q.nc')
    future_q.to_netcdf(f'{out_fname}/{fname_split}_future_q.nc')


with ProgressBar():
    compute_signal(fname, quantiles, historical_period, future_period, outfname)

