import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from tensorflow.keras.callbacks import Callback
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

config_file = sys.argv[-1]  # configuratoin file for training algorithm
#tmp_dir = sys.argv[-2]
pretrained_unet = False  # set this as true if you want to use the same U-Net or a specific unet everytime.
with open(config_file, 'r') as f:
    config = json.load(f)

input_shape = config["input_shape"]  # the input shape of the reanalyses
output_shape = config["output_shape"]
# modified the output channels, filters, and output shape
n_filters = config["n_filters"]
kernel_size = config["kernel_size"]
n_channels = config["n_input_channels"]
n_output_channels = config["n_output_channels"]
BATCH_SIZE = config["batch_size"]//2
unet_pretrained_path = config['unet_pretrained_path']
config["signal_weight"] = 2
config["itensity_weight"] = config["itensity_weight"]/4
# creating a path to store the model outputs
if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
    os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')
# custom modules
sys.path.append(os.getcwd())
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *
# config["train_x"] = f"{tmp_dir}/{config['train_x'].split('/')[-1]}"
# config["train_y"] = f"{tmp_dir}/{config['train_y'].split('/')[-1]}"
stacked_X, y, vegt, orog, he = preprocess_input_data(config)
stacked_X = stacked_X.sel(GCM =config["train_gcm"])
y = y.sel(GCM =config["train_gcm"])
y = y[['tasmin']]
config["means_output"] = "/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/normalization/target_spatial_norm_all_gcm_mean.nc"
config["stds_output"] = "/nesi/project/niwa00018/ML_downscaling_CCAM/multi-variate-gan/inputs/normalization/target_spatial_norm_all_gcm_std.nc"
output_means = xr.open_dataset(config["means_output"])
output_stds = xr.open_dataset(config["stds_output"])
y['tasmin'] = (y['tasmin'] - output_means['tasmin'].mean())/output_stds['tasmin'].mean()
#min_value = y.tasmin.min()
#config['tmin_min_value'] = float(min_value.values)
#y['tasmin'] = y['tasmin'] - min_value
# to stop the issues of negative values

with ProgressBar():
    y = y.load()
    stacked_X = stacked_X.transpose("time", "lat", "lon", "channel")
    stacked_X = stacked_X.load()
strategy = MirroredStrategy()

# Your model creation and training code goes here

# Define the generator and discriminator within the strategy scope
with strategy.scope():
    generator = res_linear_activation(input_shape, output_shape, n_filters,
                                      kernel_size, n_channels, n_output_channels,
                                      resize=True)


    unet_model = unet_linear(input_shape, output_shape, n_filters,
                                 kernel_size, n_channels, n_output_channels,
                                 resize=True)

    noise_dim = [tuple(generator.inputs[i].shape[1:]) for i in range(len(generator.inputs) - 1)]
    d_model = get_discriminator_model(tuple(output_shape) + (n_output_channels,),
                                      tuple(input_shape) + (n_channels,))


    generator_checkpoint = GeneratorCheckpoint(
        generator=generator,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/generator',
        period=3  # Save every 5 epochs
    )

    discriminator_checkpoint = DiscriminatorCheckpoint(
        discriminator=d_model,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/discriminator',
        period=3  # Save every 5 epochs
    )

    unet_checkpoint = DiscriminatorCheckpoint(
        discriminator=unet_model,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/unet',
        period=3  # Save every 5 epochs
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate_unet"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate"])

    lr_schedule_gan = tf.keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate"], decay_steps=config["decay_steps"], decay_rate=config["decay_rate_gan"])

    generator_optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule_gan, beta_1=config["beta_1"], beta_2=config["beta_2"])

    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=config["learning_rate"], beta_1=config["beta_1"], beta_2=config["beta_2"])
    unet_optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=config["beta_1"], beta_2=config["beta_2"])
    generator.summary()
    unet_model.summary()
    wgan = WGAN_Cascaded_Residual_IP_CC_pres(discriminator=d_model,
                                     generator=generator,
                                     latent_dim=noise_dim,
                                     discriminator_extra_steps=config["discrim_steps"],
                                     ad_loss_factor=config["ad_loss_factor"],
                                     orog=tf.convert_to_tensor(orog.values, 'float32'),
                                     vegt=tf.convert_to_tensor(vegt.values, 'float32'),
                                     he=tf.convert_to_tensor(he.values, 'float32'), gp_weight=config["gp_weight"],
                                     unet=unet_model,
                                     train_unet=True,
                                     intensity_weight=config["itensity_weight"], signal_weight = config["signal_weight"])

    # Compile the WGAN model.
    wgan.compile(d_optimizer=discriminator_optimizer,
                 g_optimizer=generator_optimizer,
                 g_loss_fn=generator_loss,
                 d_loss_fn=discriminator_loss,
                 u_optimizer=unet_optimizer,
                 u_loss_fn=tf.keras.losses.mean_squared_error)

    # Start training the model.
    # we normalize by a fixed normalization value
    total_size = stacked_X.time.size
    eval_times = (BATCH_SIZE * ((total_size//2) // BATCH_SIZE))


    def create_dataset(y, X, eval_times):
        output_vars = {
            'tasmin': tf.convert_to_tensor(y.tasmin.values[:eval_times][::-1], dtype=tf.float32),

            'tasmin_future': tf.convert_to_tensor(y.tasmin.values[eval_times:2 * eval_times], dtype=tf.float32)

        }
        X_tensor = {"X": tf.convert_to_tensor(X.values[:eval_times][::-1], dtype=tf.float32),
                    "X_future": tf.convert_to_tensor(X.values[eval_times:2 * eval_times], dtype=tf.float32)}

        return tf.data.Dataset.from_tensor_slices((output_vars, X_tensor))


    data = create_dataset(y, stacked_X, eval_times)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    data = data.with_options(options)
    data = data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    data = data.shuffle(16)

    with open(f'{config["output_folder"]}/{config["model_name"]}/config_info.json', 'w') as f:
        json.dump(config, f)

    wgan.fit(data, batch_size=BATCH_SIZE, epochs=config["epochs"], verbose=1,
             callbacks=[generator_checkpoint, discriminator_checkpoint, unet_checkpoint])


