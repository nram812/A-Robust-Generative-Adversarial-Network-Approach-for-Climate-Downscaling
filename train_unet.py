import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from tensorflow.keras.callbacks import Callback
from comet_ml import Experiment
import numpy as np
import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import pandas as pd
import tensorflow.keras.layers as layers
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras import layers
import pandas as pd

config_file = sys.argv[-1]  # configuratoin file for training algorithm

with open(config_file, 'r') as f:
    config = json.load(f)


input_shape = config["input_shape"]  # the input shape of the reanalyses
output_shape = config["output_shape"]
# modified the output channels, filters, and output shape
n_filters = config["n_filters"]
kernel_size = config["kernel_size"]
n_channels = config["n_input_channels"]
n_output_channels = config["n_output_channels"]
BATCH_SIZE = config["batch_size"]

# creating a path to store the model outputs
if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
    os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')
# custom modules
sys.path.append(config["src_path"])
from src.layers import *
from src.models import *
from src.gan import *

stacked_X, y, vegt, orog, he = preprocess_input_data(config)

strategy = MirroredStrategy()

# Your model creation and training code goes here

# Define the generator and discriminator within the strategy scope
with strategy.scope():

    unet_model = unet_linear(input_shape, output_shape, n_filters,
                             kernel_size, n_channels, n_output_channels,
                             resize=True)
    # it does not matter if this is called discriminator checkpoint
    unet_checkpoint = DiscriminatorCheckpoint(
        discriminator=unet_model,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/unet',
        period=10  # Save every 5 epochs
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate_unet"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate"])

    unet_optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=config["beta_1"], beta_2=config["beta_2"])



    # Compile the UNET
    unet_model.compile(optimizer=unet_optimizer, loss=tf.keras.losses.mean_squared_error)

    # Start training the model.
    # we normalize by a fixed normalization value
    total_size = stacked_X.time.size
    eval_times = BATCH_SIZE * (total_size // BATCH_SIZE)

    data = tuple([tf.convert_to_tensor(np.log(y.pr[:eval_times].values + config["delta"]), 'float32'),
                  tf.convert_to_tensor(stacked_X[:eval_times].values, 'float32')])

    with open(f'{config["output_folder"]}/{config["model_name"]}/config_info.json', 'w') as f:
        json.dump(config, f)

    unet_model.fit(data, batch_size=BATCH_SIZE, epochs=config["epochs"], verbose=2, shuffle=True,
             callbacks=[unet_checkpoint])


