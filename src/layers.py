
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd


def res_block_initial_gen(x, num_filters, kernel_size, strides, name,bn = False):
    """Residual Unet block layer for first layer
    In the residual unet the first residual block does not contain an
    initial batch normalization and activation so we create this separate
    block for it.
    Args:
        x: tensor, image or image activation
        num_filters: list, contains the number of filters for each subblock
        kernel_size: int, size of the convolutional kernel
        strides: list, contains the stride for each subblock convolution
        name: name of the layer
    Returns:
        x1: tensor, output from residual connection of x and x1
    """

    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]

    x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                kernel_size=kernel_size,
                                strides=strides[0],
                                padding='same',
                                name=name + '_1')(x)

    #x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU(0.05)(x1)#tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                kernel_size=kernel_size,
                                strides=strides[1],
                                padding='same',
                                name=name + '_2')(x1)

    x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               name=name + '_shortcut')(x)
    # if bn:
    #
    #     x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Add()([x, x1])
    x1 = tf.keras.layers.LeakyReLU(0.05)(x1)
    return x1


def res_block_initial(x, num_filters, kernel_size, strides, name,bn = False):
    """Residual Unet block layer for first layer
    In the residual unet the first residual block does not contain an
    initial batch normalization and activation so we create this separate
    block for it.
    Args:
        x: tensor, image or image activation
        num_filters: list, contains the number of filters for each subblock
        kernel_size: int, size of the convolutional kernel
        strides: list, contains the stride for each subblock convolution
        name: name of the layer
    Returns:
        x1: tensor, output from residual connection of x and x1
    """

    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]

    x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                kernel_size=kernel_size,
                                strides=strides[0],
                                padding='same',
                                name=name + '_1', kernel_regularizer = tf.keras.regularizers.l1(1e-3))(x)

    #x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU(0.05)(x1)#tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                kernel_size=kernel_size,
                                strides=strides[1],
                                padding='same',
                                name=name + '_2', kernel_regularizer = tf.keras.regularizers.l1(1e-3))(x1)

    x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               name=name + '_shortcut', kernel_regularizer = tf.keras.regularizers.l1(1e-3))(x)
    # if bn:
    #
    #     x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Add()([x, x1])
    x1 = tf.keras.layers.LeakyReLU(0.05)(x1)
    return x1

def res_block_initial_generator(x, num_filters, kernel_size, strides, name,bn = False):
    """Residual Unet block layer for first layer
    In the residual unet the first residual block does not contain an
    initial batch normalization and activation so we create this separate
    block for it.
    Args:
        x: tensor, image or image activation
        num_filters: list, contains the number of filters for each subblock
        kernel_size: int, size of the convolutional kernel
        strides: list, contains the stride for each subblock convolution
        name: name of the layer
    Returns:
        x1: tensor, output from residual connection of x and x1
    """

    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]

    x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                kernel_size=kernel_size,
                                strides=strides[0],
                                padding='same',
                                name=name + '_1')(x)

    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU(0.05)(x1)#tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                kernel_size=kernel_size,
                                strides=strides[1],
                                padding='same',
                                name=name + '_2')(x1)

    x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               name=name + '_shortcut')(x)
    # if bn:
    #
    #     x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Add()([x, x1])
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU(0.05)(x1)
    return x1


class BicubicUpSampling2D(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(BicubicUpSampling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        return tf.image.resize(inputs, [int(inputs.shape[1] * self.size[0]), int(inputs.shape[2] * self.size[1])],
                               method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size
        })
        return config


def upsample(x, target_size):
    """"Upsampling function, upsamples the feature map
    Deep Residual Unet paper does not describe the upsampling function
    in detail. Original Unet uses a transpose convolution that downsamples
    the number of feature maps. In order to restrict the number of
    parameters here we use a bilinear resampling layer. This results in
    the concatentation layer concatenting feature maps with n and n/2
    features as opposed to n/2  and n/2 in the original unet.
    Args:
        x: tensor, feature map
        target_size: size to resize feature map to
    Returns:
        x_resized: tensor, upsampled feature map
    """

    x_resized = BicubicUpSampling2D((target_size, target_size))(x)  # tf.keras.layers.Lambda(lambda x: tf.image.resize(x, target_size))(x)
    return x_resized


def conv_block(x, filters, activation, kernel_size=(7, 7), strides=(2, 2), padding="same",
               use_bias=True, use_bn=True, use_dropout=True, drop_value=0.5):
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding=padding, use_bias=use_bias)(x)
    #x = layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.05)(x)

    return x


def conv_block_generator(x, filters, activation, kernel_size=(7, 7), strides=(2, 2), padding="same",
               use_bias=True, use_bn=True, use_dropout=True, drop_value=0.5):
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding=padding, use_bias=use_bias)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.05)(x)

    return x


def decoder_noise(x, num_filters, kernel_size, noise = False,bn =True):
    """Unet decoder
    Args:
        x: tensor, output from previous layer
        encoder_output: list, output from all previous encoder layers
        num_filters: list, number of filters for each decoder layer
        kernel_size: int, size of the convolutional kernel
    Returns:
        x: tensor, output from last layer of decoder
    """
    noise_inputs = []# at some intermediate layers
    for i in range(1, len(num_filters) + 1):
        layer2 = 'decoder_layer_v2' + str(i)
        x = upsample(x, 2)
        x = res_block_initial(x, [num_filters[-i]], kernel_size, strides=[1, 1], name='decoder_layer_v2' + str(i))

    return x, noise_inputs


def decoder_noise_generator(x, num_filters, kernel_size, noise = False,bn =True):
    """Unet decoder
    Args:
        x: tensor, output from previous layer
        encoder_output: list, output from all previous encoder layers
        num_filters: list, number of filters for each decoder layer
        kernel_size: int, size of the convolutional kernel
    Returns:
        x: tensor, output from last layer of decoder
    """
    noise_inputs = []# at some intermediate layers
    for i in range(1, len(num_filters) + 1):
        layer2 = 'decoder_layer_v2' + str(i)
        x = upsample(x, 2)
        x = res_block_initial_generator(x, [num_filters[-i]], kernel_size, strides=[1, 1], name='decoder_layer_v2' + str(i))

    return x, noise_inputs
