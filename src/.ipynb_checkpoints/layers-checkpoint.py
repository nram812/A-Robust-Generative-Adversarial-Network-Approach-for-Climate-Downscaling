
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd


def res_block_initial(x, num_filters, kernel_size, strides, name, sym_padding =True):
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
    if sym_padding:
        x1 = SymmetricPadding2D(padding=[int((kernel_size-1)//2), int((kernel_size-1)//2)])(x)
        x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                    kernel_size=kernel_size,
                                    strides=strides[0],
                                    padding='valid',
                                    name=name + '_1')(x1)
    else:
        x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                    kernel_size=kernel_size,
                                    strides=strides[0],
                                    padding='same',
                                    name=name + '_1')(x)

    x1 = tf.keras.layers.LeakyReLU(0.01)(x1)
    if sym_padding:
        x1 = SymmetricPadding2D(padding=[int((kernel_size - 1) // 2), int((kernel_size - 1) // 2)])(x1)
        x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                    kernel_size=kernel_size,
                                    strides=strides[1],
                                    padding='valid',
                                    name=name + '_2')(x1)

        x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   name=name + '_shortcut')(x)
    else:
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
    x1 = tf.keras.layers.LeakyReLU(0.01)(x1)
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


class SymmetricPadding2D(tf.keras.layers.Layer):

    def __init__(self, padding=[1,1], **kwargs):

        super(SymmetricPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def build(self, input_shape):
        super(SymmetricPadding2D, self).build(input_shape)

    def call(self, inputs):
        if self.padding[0] >1:
            pad = [[0, 0]] + [[1, 1], [1, 1]] + [[0, 0]]
            paddings = tf.constant(pad)
            out = tf.pad(inputs, paddings, "SYMMETRIC")
            for i in range(self.padding[0]-1):
                pad = [[0, 0]] + [[1, 1], [1, 1]] + [[0, 0]]
                paddings = tf.constant(pad)
                out = tf.pad(out, paddings, "SYMMETRIC")
            return out
        else:

            pad = [[0, 0]] + [[1, 1], [1, 1]] + [[0, 0]]
            paddings = tf.constant(pad)
            out = tf.pad(inputs, paddings, "SYMMETRIC")
            return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.padding[0],
                input_shape[2] + self.padding[1], input_shape[-1])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding
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
    x = SymmetricPadding2D(padding=[int((kernel_size[0] - 1) // 2), int((kernel_size[0] - 1) // 2)])(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding='same', use_bias=use_bias)(x)
    x = tf.keras.layers.LeakyReLU(0.01)(x)

    return x


def decoder_noise(x, num_filters, kernel_size):
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
        x = res_block_initial(x, [num_filters[-i]], kernel_size, strides=[1, 1], name='decoder_layer_v2' + str(i),
                              sym_padding =False)
    return x, noise_inputs


def down_block(x, filters, kernel_size, i =1, use_pool=True, method ='unet', sym_padding =True):

    x = res_block_initial(x, [filters], kernel_size, strides=[1, 1],
                          name='decoder_layer_v2' + str(i),
                              sym_padding = sym_padding)
    if use_pool == True:
        return tf.keras.layers.AveragePooling2D(strides=(2, 2))(x), x
    else:
        return x


def up_block(x, y, filters, kernel_size, i =1, method ='unet', concat = True, sym_padding =True):
    x = upsample(x, 2)
    if concat:
        x = tf.keras.layers.Concatenate(axis=-1)([x, y])
    x = res_block_initial(x, [filters], kernel_size, strides=[1, 1],
                          name='encoder_layer_v2' + str(i),sym_padding = sym_padding)
    return x


