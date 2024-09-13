import xarray as xr
import numpy as np
import pandas as pd
import os
from dask.diagnostics import ProgressBar


def prepare_training_data(config, X, y, means, stds, match_index = True):
    """
    Normalizes the X training data, and stacks the features into a single dimension
    config: json file that contains a dictionary of the experimental files used in training
    X: training data, which is pre-loaded. Note this file is already in the config file, but has been loaded in another script
    mean:: normalize relative to a mean
    std: normalize relative to an std
    """

    list_of_vars = config["var_names"]
    # normalize data
    X_norm = (X[list_of_vars] - means[list_of_vars].mean()) / stds[list_of_vars].mean()

    stacked_X = xr.concat([X_norm[varname] for varname in list_of_vars], dim="channel")
    # stack features
    stacked_X['channel'] = (('channel'), list_of_vars)
    if match_index:
        times = stacked_X.time.to_index().intersection(y.time.to_index())
        # this part should be fine
        stacked_X = stacked_X.sel(time=times)
        y = y.sel(time=times)
    else:
        y = y
        stacked_X = stacked_X
    return stacked_X, y


def prepare_static_fields(config):
    topography_data = xr.open_dataset(config["static_predictors"])
    vegt = topography_data.vegt
    orog = topography_data.orog
    he = topography_data.he
    print(orog.max(), he.max(), vegt.max())

    # normazation to the range [0,1]
    vegt = (vegt - vegt.min()) / (vegt.max() - vegt.min())
    orog = (orog - orog.min()) / (orog.max() - orog.min())
    he = (he - he.min()) / (he.max() - he.min())
    return vegt, orog, he


# LOADING the mean values
def preprocess_input_data(config, match_index = True):
    vegt, orog, he = prepare_static_fields(config)
    means = xr.open_dataset(config["mean"])
    stds = xr.open_dataset(config["std"])

    X = xr.open_dataset(config["train_x"])  # .sel(time = slice("2016", None))
    X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d"))

    y = xr.open_dataset(config["train_y"])#, chunks={"time": 5000})
    y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d"))# .sel(time = slice("2016", None))
    try:
        y = y.drop("lat_bnds")
        y = y.drop("lon_bnds")

    except:
        pass

    # preare the training data
    stacked_X, y = prepare_training_data(config, X, y, means, stds, match_index = match_index)

    return stacked_X, y, vegt, orog, he