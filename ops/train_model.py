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

config_file = sys.argv[-1] # configuratoin file for training algorithm
pretrained_unet = False # set this as true if you want to use the same U-Net or a specific unet everytime. 
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
unet_pretrained_path = config['unet_pretrained_path']

# creating a path to store the model outputs
if not os.path.exists(f'{config["output_folder"]}/{config["model_name"]}'):
    os.makedirs(f'{config["output_folder"]}/{config["model_name"]}')
# custom modules
sys.path.append(os.getcwd())
from src.layers import *
from src.models import *
from src.gan import *
from src.process_input_training_data import *


stacked_X, y, vegt, orog, he = preprocess_input_data(config)


strategy = MirroredStrategy()

# Your model creation and training code goes here

# Define the generator and discriminator within the strategy scope
with strategy.scope():

    generator = res_linear_activation(input_shape, output_shape, n_filters,
                                                      kernel_size, n_channels, n_output_channels,
                                                      resize=True)
    if pretrained_unet:
        # use an existing unet model
        unet_model = tf.keras.models.load_model(unet_training_path,
                                                custom_objects={"BicubicUpSampling2D":BicubicUpSampling2D},
                                                compile =False)
    else:
        # train the model from scratch
        unet_model = unet_linear(input_shape, output_shape, n_filters,
                                                          kernel_size, n_channels, n_output_channels,
                                                          resize=True)
    noise_dim = [tuple(generator.inputs[i].shape[1:]) for i in range(len(generator.inputs) - 1)]
    d_model = get_discriminator_model(tuple(output_shape) + (n_output_channels,),
                                      tuple(input_shape) + (n_channels,))

    generator_checkpoint = GeneratorCheckpoint(
        generator=generator,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/generator',
        period=10  # Save every 5 epochs
    )

    discriminator_checkpoint = DiscriminatorCheckpoint(
        discriminator=d_model,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/discriminator',
        period=10  # Save every 5 epochs
    )

    unet_checkpoint = DiscriminatorCheckpoint(
        discriminator=unet_model,
        filepath=f'{config["output_folder"]}/{config["model_name"]}/unet',
        period=10  # Save every 5 epochs
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

    wgan = WGAN_Cascaded_Residual_IP(discriminator=d_model,
                                    generator=generator,
                                    latent_dim=noise_dim,
                                    discriminator_extra_steps=config["discrim_steps"],
                                    ad_loss_factor=config["ad_loss_factor"],
                                    orog=tf.convert_to_tensor(orog.values, 'float32'),
                                    vegt=tf.convert_to_tensor(vegt.values, 'float32'),
                                    he=tf.convert_to_tensor(he.values, 'float32'), gp_weight=config["gp_weight"],
                                    unet = unet_model,
                                    train_unet=True,
                                    intensity_weight = config["itensity_weight"])

    # Compile the WGAN model.
    wgan.compile(d_optimizer=discriminator_optimizer,
                 g_optimizer=generator_optimizer,
                 g_loss_fn=generator_loss,
                 d_loss_fn=discriminator_loss,
                 u_optimizer=unet_optimizer,
                 u_loss_fn = tf.keras.losses.mean_squared_error)

    # Start training the model.
    # we normalize by a fixed normalization value
    total_size = stacked_X.time.size
    eval_times = BATCH_SIZE * (total_size // BATCH_SIZE)

    data = tuple([tf.convert_to_tensor(np.log(y.pr[:eval_times].values + config["delta"]), 'float32'),
              tf.convert_to_tensor(stacked_X[:eval_times].values, 'float32')])

    with open(f'{config["output_folder"]}/{config["model_name"]}/config_info.json', 'w') as f:
        json.dump(config, f)

    wgan.fit(data, batch_size=BATCH_SIZE, epochs=config["epochs"], verbose=2, shuffle=True,
             callbacks=[generator_checkpoint, discriminator_checkpoint, unet_checkpoint])


