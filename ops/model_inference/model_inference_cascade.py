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

models = ['EC-Earth3', 'ACCESS-CM2', 'NorESM2-MM']


gans = ['GAN_Leaky_Relu','Intensity_Penalty']
n_members = 10
timeslice = slice("1986", "2005")


# need to write an os.listdir for all these experiments
for model in models:
    vegt, orog, he = load_and_normalize_topography_data(config["static_predictors"])
    predictor, ground_truth = load_data_historical(model, config)
    stacked_predictors = normalize_and_stack(predictor, config["mean"], config["std"],
                                             config["var_names"])
    for output_model_name in gans:

        common_times = stacked_predictors .sel(time=timeslice).time.to_index().intersection(ground_truth.time.to_index())

        experiments = [file for file in os.listdir(f"{config['model_dir']}/{output_model_name}")
                       if not (file =='base_unet')]
        # skipping files

        output_prediction = run_experiments(experiments, epochs,
                                            f"{config['model_dir']}/{output_model_name}",
                                            stacked_predictors, common_times[0:100],
                                            ground_truth, orog, he, vegt, n_members,
                                            batch_size=64)
        if not os.path.exists(f'{config["output_dir"]}/{output_model_name}'):
            os.makedirs(f'{config["output_dir"]}/{output_model_name}')

        output_prediction.to_netcdf(
            f'{config["output_dir"]}/{output_model_name}/{model}_{output_model_name}_hist_1986_2005_cascaded_imperfect_applied.nc')
        # computing validation metrics
        # validation_metrics = ValidationMetric(output_prediction)
        # validation_metrics = validation_metrics(
        #              thresh =1)
        # validation_metrics.to_netcdf(f'{output_dir}/{model}_hist_1986_2005_cascaded_imperfect_applied_val_metrics.nc')

        # load the perfect conditions