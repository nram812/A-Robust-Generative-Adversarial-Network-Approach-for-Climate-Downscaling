import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.layers import res_block_initial, \
    BicubicUpSampling2D,upsample, conv_block,decoder_noise,down_block,up_block,SymmetricPadding2D


def get_discriminator_model(high_resolution_fields_size,
                            low_resolution_fields_size, use_bn=False,
                            use_dropout=False, use_bias=True, low_resolution_feature_channels=(32, 64, 128),
                            low_resolution_dense_neurons =6,
                            high_resolution_feature_channels=(32, 64, 12)):
    """
    Discriminator no longer uses the unet model to demonstrate realism
    **Purpose:**
      * To create a discriminator model that takes two streams of inputs, one from the low resolution predictor fields(X)
      and auxilary inputs (topography), it also takes in the high-resolution "regression prediction",
      which is used for residuals

    **Parameters:**
      * **high_resolution_fields_size (tuple):**  The size of the 2D high-resolution RCM fields, over the NZ region this (172, 179)
      * **low_resolution_fields_size (tuple):**  The size of the 2D low-resolution predictor fields (23, 26) over the New Zealand domain
      * **use_bn (bool, optional):** whether to use batchnormalization or not (default no bn)
      * **use_dropout (bool, optional):** whether to use dropout or not(default no dropout)
      * **use_bias (bool, optional):** whether to use bias or not (default bias =True)

    **Returns:**
        * a tf.keras.models.Model class

    **Example Usage:**
    ```python
    discriminator_model = get_discriminator_model((172, 179), (23, 26))
    ```
    """
    IMG_SHAPE = high_resolution_fields_size
    IMG_SHAPE2 = low_resolution_fields_size

    img_input = layers.Input(shape=IMG_SHAPE) # real or fake predictions
    img_input2 = layers.Input(shape=IMG_SHAPE2) # boundary conditions or predictor fields

    # these are static inputs to the model
    img_input3 = layers.Input(shape=IMG_SHAPE) # Topography predictor variable
    img_input4 = layers.Input(shape=IMG_SHAPE) # other CCAM auxilary variables if used
    img_input5 = layers.Input(shape=IMG_SHAPE)
    img_input6 = layers.Input(shape=IMG_SHAPE) # UNET regressoin predictor.
    # now we concatenate these input a single vector
    img_inputs = tf.keras.layers.Concatenate(-1)([img_input3,img_input6])

    # Low resolution data stream
    x_init = conv_block(img_input2, low_resolution_feature_channels[0], kernel_size=(3, 3), strides=(2, 2), use_bn=use_bn, use_bias=use_bias,
                        use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    x_init = conv_block(x_init, low_resolution_feature_channels[0], kernel_size=(3, 3), strides=(2, 2), use_bn=use_bn, use_bias=use_bias,
                        use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # coarsen the data slightly
    x_init = conv_block(x_init, low_resolution_feature_channels[1], kernel_size=(3, 3), strides=(2, 2), use_bn=use_bn, use_bias=use_bias,
                        use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())

    x_init = conv_block(x_init, low_resolution_feature_channels[2], kernel_size=(3, 3), strides=(2, 2), use_bn=use_bn, use_bias=use_bias,
                        use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())

    flatten = tf.keras.layers.Flatten()(x_init)
    flatten = tf.keras.layers.Dense(low_resolution_dense_neurons)(flatten)

    # high-resolution data stream
    # first we put "real or fake data" with 32 channels, to allow it to be more important
    x = conv_block(img_input, high_resolution_feature_channels[0], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # topography and residuals only have one filter
    x2 = conv_block(img_inputs, 1, kernel_size=(5, 5), strides=(2, 2),
                    use_bn=use_bn, use_bias=use_bias,
                    use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # have two separate streams of conditional inputs into the model
    x = tf.keras.layers.Concatenate(-1)([x, x2])
    x = conv_block(x, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # reducing the dimensionality to speed up the computational cost
    # x = tf.keras.layers.AveragePooling2D((3,3))(x)

    x_init_raw = conv_block(x, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # x_init_raw = tf.keras.layers.AveragePooling2D((3,3))(x_init_raw)
    x_init_raw = conv_block(x_init_raw, high_resolution_feature_channels[2], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    flattened_output = layers.Flatten()(x_init_raw)
    concat = tf.keras.layers.Concatenate(-1)([flatten, flattened_output])
    dense2 = tf.keras.layers.Dense(64)(concat)

    x = layers.Dense(1)(dense2)

    d_model = keras.models.Model([img_input, img_input2, img_input3, img_input4, img_input5, img_input6], x,
                                 name="discriminator")
    return d_model


def get_discriminator_model_v1(high_resolution_fields_size,
                            low_resolution_fields_size, use_bn=False,
                            use_dropout=False, use_bias=True, low_resolution_feature_channels=(32, 64, 128),
                            low_resolution_dense_neurons =6,
                            high_resolution_feature_channels=(16, 32, 64, 128)):
    """
    Discriminator no longer uses the unet model to demonstrate realism
    **Purpose:**
      * To create a discriminator model that takes two streams of inputs, one from the low resolution predictor fields(X)
      and auxilary inputs (topography), it also takes in the high-resolution "regression prediction",
      which is used for residuals

    **Parameters:**
      * **high_resolution_fields_size (tuple):**  The size of the 2D high-resolution RCM fields, over the NZ region this (172, 179)
      * **low_resolution_fields_size (tuple):**  The size of the 2D low-resolution predictor fields (23, 26) over the New Zealand domain
      * **use_bn (bool, optional):** whether to use batchnormalization or not (default no bn)
      * **use_dropout (bool, optional):** whether to use dropout or not(default no dropout)
      * **use_bias (bool, optional):** whether to use bias or not (default bias =True)

    **Returns:**
        * a tf.keras.models.Model class

    **Example Usage:**
    ```python
    discriminator_model = get_discriminator_model((172, 179), (23, 26))
    ```
    """
    IMG_SHAPE = high_resolution_fields_size
    IMG_SHAPE2 = low_resolution_fields_size

    img_input = layers.Input(shape=IMG_SHAPE) # real or fake predictions
    img_input2 = layers.Input(shape=IMG_SHAPE2) # boundary conditions or predictor fields

    # these are static inputs to the model
    img_input3 = layers.Input(shape=IMG_SHAPE) # Topography predictor variable
    img_input4 = layers.Input(shape=IMG_SHAPE) # other CCAM auxilary variables if used
    img_input5 = layers.Input(shape=IMG_SHAPE)
    img_input6 = layers.Input(shape=IMG_SHAPE) # UNET regressoin predictor.
    # now we concatenate these input a single vector
    # high-resolution data stream
    # first we put "real or fake data" with 32 channels, to allow it to be more important
    inputs_high_res = res_block_initial(img_input, [high_resolution_feature_channels[0]], 3, [1, 1], "output_convbbb")
    x = conv_block(inputs_high_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())

    x = conv_block(x, high_resolution_feature_channels[2], kernel_size=(3, 3), strides=(2, 2),
                   use_bn=use_bn, use_bias=use_bias,
                   use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # reducing the dimensionality to speed up the computational cost
    # x = tf.keras.layers.AveragePooling2D((3,3))(x)

    x_init_raw = conv_block(x, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = conv_block(img_input2, high_resolution_feature_channels[0], kernel_size=(3, 3), strides=(1, 1),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = conv_block(images_low_res, high_resolution_feature_channels[1], kernel_size=(3, 3), strides=(1, 1),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    images_low_res = tf.image.resize(images_low_res, [x_init_raw.shape[1], x_init_raw.shape[2]],
                    method=tf.image.ResizeMethod.BILINEAR)
    concat_outputs = tf.keras.layers.Concatenate(-1)([x_init_raw, images_low_res])
    concat_outputs = res_block_initial(concat_outputs, [high_resolution_feature_channels[3]], 3, [1, 1], "output_convbbb1")
    x_init_raw = conv_block(concat_outputs, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    x_init_raw = conv_block(x_init_raw, high_resolution_feature_channels[3], kernel_size=(3, 3), strides=(2, 2),
                            use_bn=use_bn, use_bias=use_bias,
                            use_dropout=use_dropout, drop_value=0.0,
                            activation=tf.keras.layers.LeakyReLU())
    flattened_output = tf.keras.layers.Flatten()(x_init_raw)
    dense2 = tf.keras.layers.Dense(32)(flattened_output)

    x = layers.Dense(1)(dense2)

    d_model = keras.models.Model([img_input, img_input2, img_input3, img_input4, img_input5, img_input6], x,
                                 name="discriminator")
    return d_model


def res_linear_activation_v2(input_size, resize_output, num_filters, kernel_size, num_channels, num_classes, resize=True,
                          final_activation = tf.keras.layers.LeakyReLU(1)):

    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields4 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = tf.keras.layers.Concatenate(-1)([high_res_fields,  high_res_fields4])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=3, i =0)
    x, temp2 = down_block(x, num_filters[1], kernel_size=3, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=3, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    # decode
    x = up_block(x, temp3, kernel_size=3, filters = num_filters[2], i =0)
    x = up_block(x, temp2, kernel_size=3, filters = num_filters[1], i =2)
    x = up_block(x, temp1, kernel_size=3, filters = num_filters[0], i =3)
    output = tf.image.resize(x, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    output = res_block_initial(output, [64], 3, [1, 1], "output_convbbb123456", sym_padding=True)
    output = res_block_initial(output, [32], 3, [1, 1], "output_convbbb1234", sym_padding=True)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(32, 3, activation=final_activation, padding ='valid')(output)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(16, 3, activation=final_activation, padding ='valid')(output)
    output = tf.keras.layers.Conv2D(num_classes, 1, activation=final_activation, padding ='valid')(output)
    output = tf.image.resize(output, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    input_layers = [noise] + [low_res, high_res_fields, high_res_fields2,high_res_fields3, high_res_fields4]
    model = tf.keras.models.Model(input_layers, output, name='unet')
    model.summary()
    return model


def res_linear_activation(input_size, resize_output, num_filters, kernel_size, num_channels, num_classes, resize=True,
                          bn=True):
    """
    **Purpose:**
      * To create a generator model that takes two streams of inputs, one from the low resolution predictor fields(X)
      and auxilary inputs (topography), it also takes in the high-resolution "regression prediction",
      which is used for residuals

    **Parameters:**
      * **input_size (tuple):**  The size of the 2D predictor fields over NZ region (default 23, 26)
      * **resize_output (tuple):**  The size of the auxiliary fields (or the output fields), default is (172, 179)
      * **num_filters (tuple):** The number of filters or residual blokcs in the network
      * **num_classes (int): ** the number of output variables (i.e. for rainfall this is simply 1).
      * **bn (bool): ** whether to use batch normalization or not
      **num_channels (int): the number of predictor variables to be used to training the model.

    **Returns:**
        * a tf.keras.models.Model class

    **Example Usage:**
    ```python
    generator = res_linear_activation((172, 179), (23, 26), [32, 64, 128, 256], 3, 8, resize = True, bn =True)

    # note that resize and kernel size (3) are not currently used
    ```
    """
    x = tf.keras.Input(shape=[input_size[0], input_size[1], num_channels]) # predictor variables X
    img_input3 = layers.Input(shape=[resize_output[0], resize_output[1], 1]) # topography (RCM resolution)
    img_input4 = layers.Input(shape=[resize_output[0], resize_output[1], 1]) # other auxiilary variables if needed
    img_input5 = layers.Input(shape=[resize_output[0], resize_output[1], 1])
    img_input6 = layers.Input(shape=[resize_output[0], resize_output[1], 1])# U-Net prediction (regression-baseline)
    # input vectors
    img_inputs = tf.keras.layers.Concatenate(-1)([img_input3, img_input6])  # img_input4, img_input5

    # high-resolution information stream
    x_init_ref_fields_high_res = conv_block(img_inputs, 16, kernel_size=(5, 5), strides=(1, 1), use_bn=bn, use_bias=True,
                                            use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU(0.2))
    x_init_ref_fields_high_res = conv_block(x_init_ref_fields_high_res, 64, kernel_size=(5, 5), strides=(1, 1), use_bn=bn, use_bias=True,
                                            use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU(0.2))
    x_init_ref_fields = tf.keras.layers.AveragePooling2D((2, 2))(x_init_ref_fields_high_res)

    # lowering the importance of topography
    x_init_ref_fields = conv_block(x_init_ref_fields, 64, kernel_size=(5, 5), strides=(1, 1), use_bn=bn, use_bias=True,
                                   use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU(0.2))
    x_init_ref_fields = tf.keras.layers.AveragePooling2D((2, 2))(x_init_ref_fields)
    x_init_ref_fields = tf.image.resize(x_init_ref_fields, (input_size[0], input_size[1]),
                                        method=tf.image.ResizeMethod.BILINEAR)
    # this is now the same resolution as the input fields

    # concat noise with inputs in this layer
    noise = layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    concat_noise = tf.keras.layers.Concatenate(-1)([x, noise])  # this appears to be more stable
    # add some noise within the GAN framework
    x_output = res_block_initial(concat_noise, [num_filters[-1]], 3, [1, 1], "input_layer")
    # add the reference static fields as an input
    x_output = tf.keras.layers.Concatenate(-1)([x_init_ref_fields, x_output])
    decoder_output, noise_layers = decoder_noise(x_output, num_filters[:-1], 5, kernel_size_gauss =0, sigma=1 )

    # resizing the decoder output
    # x_init_ref_fields_high_res_resized = tf.image.resize(x_init_ref_fields_high_res,
    #                                                      (decoder_output.shape[-3], decoder_output.shape[-2]),
    #                                                      method=tf.image.ResizeMethod.BILINEAR)
    #decoder_output = tf.keras.layers.Concatenate(-1)([x_init_ref_fields_high_res_resized, decoder_outpu
    decoder_output = res_block_initial(decoder_output, [64], 3, [1, 1], "output_convbbb")
    # we are predicting log of precipitation

    # tf.keras.layers.Concatenate(-1)([alpha, beta, p_rainfall])
    output = tf.image.resize(decoder_output, (resize_output[0], resize_output[1]),
                             method=tf.image.ResizeMethod.BILINEAR)
    output = tf.keras.layers.Concatenate(-1)([output, x_init_ref_fields_high_res])
    output = res_block_initial(output, [64], 3, [1, 1], "output_conv2341")
    output = tf.keras.layers.Conv2D(32,
                                    3,
                                    strides=1,
                                    padding='same',
                                    name='custom_precip_layer', activation=tf.keras.layers.LeakyReLU(0.4))(output)
    output = tf.keras.layers.Conv2D(16,
                                    3,
                                    strides=1,
                                    padding='same',
                                    name='custom_precip_layerb',
                                    activation=tf.keras.layers.LeakyReLU(0.4))(output)

    output = tf.keras.layers.Conv2D(num_classes,
                                    3,
                                    strides=1,
                                    padding='same',
                                    name='custom_precip_layer2', activation='linear')(output)
    input_layers = [noise] + [x, img_input3, img_input4, img_input5, img_input6]
    # added multiple inputs into the tensorflow
    # output = output + x_input
    model = tf.keras.Model(input_layers, output)

    return model


def unet_linear_v2(input_size, resize_output, num_filters, kernel_size, num_channels, num_classes, resize=True,
                          final_activation = tf.keras.layers.LeakyReLU(1)):

    """have modified the architecture with a new concatenation layer"""

    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = high_res_fields#tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2,high_res_fields3])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = low_res#tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=5, i =0)
    x, temp2 = down_block(x, num_filters[1], kernel_size=5, i =1)
    x, temp3 = down_block(x, num_filters[2], kernel_size=5, i =2)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    # decode
    x = up_block(x, temp3, kernel_size=5, filters = num_filters[2], i =0, concat = False)
    x = up_block(x, temp2, kernel_size=5, filters = num_filters[1], i =2, concat = False)
    x = up_block(x, temp1, kernel_size=5, filters = num_filters[0], i =3, concat = False)
    output = tf.image.resize(x, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    output = res_block_initial(output, [64], 3, [1, 1], "output_convbbb1234567", sym_padding=True)
    output = res_block_initial(output, [32], 3, [1, 1], "output_convbbb12347", sym_padding=True)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(32, 3, activation=final_activation, padding ='valid')(output)
    output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(16, 3, activation=final_activation, padding ='valid')(output)
    output = tf.keras.layers.Conv2D(num_classes, 1, activation=final_activation, padding ='valid')(output)
    output = tf.image.resize(output, (resize_output[0], resize_output[1]),
                    method=tf.image.ResizeMethod.BILINEAR)
    input_layers = [noise] + [low_res, high_res_fields, high_res_fields2,high_res_fields3]
    model = tf.keras.models.Model(input_layers, output, name='unet')
    model.summary()
    return model


def unet_linear_temp(input_size, resize_output, num_filters, kernel_size, num_channels, num_classes, resize=True,
                          final_activation = tf.keras.layers.LeakyReLU(1)):

    high_res_fields = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields2 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])
    high_res_fields3 = tf.keras.layers.Input(shape=[resize_output[0], resize_output[1], 1])

    concat_image = high_res_fields#tf.keras.layers.Concatenate(-1)([high_res_fields, high_res_fields2,high_res_fields3])
    concatted_highres = tf.image.resize(concat_image, [int(np.ceil(resize_output[0]/8) * 8), int(np.ceil(resize_output[1]/8)) * 8],
                    method=tf.image.ResizeMethod.BILINEAR)

    low_res = tf.keras.layers.Input(shape =[input_size[0], input_size[1], num_channels])
    noise = tf.keras.layers.Input(shape=[input_size[0], input_size[1], num_channels])
    inputs_abstract = low_res#tf.keras.layers.Concatenate(-1)([low_res, noise])
    x, temp1 = down_block(concatted_highres, num_filters[0], kernel_size=5, i =0, sym_padding=False)
    x, temp2 = down_block(x, num_filters[1], kernel_size=5, i =1, sym_padding=False)
    x, temp3 = down_block(x, num_filters[2], kernel_size=5, i =2, sym_padding=False)
    x1 = down_block(inputs_abstract, num_filters[2], kernel_size=3, i=4, use_pool=False, sym_padding=False)
    x1 = tf.keras.layers.AveragePooling2D((2,2))(x1)
    x1 = tf.image.resize(x1, [int(x.shape[1]), int(x.shape[2])],
                    method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.Concatenate(-1)([x1, x])
    # decode
    x = up_block(x, [], kernel_size=5, filters = num_filters[2], i =0, concat = False, sym_padding=False)
    x = up_block(x, [], kernel_size=5, filters = num_filters[1], i =2, concat = False, sym_padding=False)
    x = up_block(x, [], kernel_size=5, filters = num_filters[0], i =3, concat = False, sym_padding=False)
    output = tf.image.resize(x, [172, 179],
                    method=tf.image.ResizeMethod.BILINEAR)
    output = res_block_initial(output, [64], 3, [1, 1], "output_convbbb1234567", sym_padding=False)
    output = res_block_initial(output, [32], 3, [1, 1], "output_convbbb12347", sym_padding=False)
    #output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(32, 3, activation=final_activation, padding ='same')(output)
    #output = SymmetricPadding2D(padding=[1, 1])(output)
    output = tf.keras.layers.Conv2D(16, 3, activation=final_activation, padding ='same')(output)
    output = tf.keras.layers.Conv2D(num_classes, 1, activation=final_activation, padding ='same')(output)
    # output = tf.image.resize(output, [172, 179],
    #                 method=tf.image.ResizeMethod.BILINEAR)
    input_layers = [noise] + [low_res, high_res_fields, high_res_fields2,high_res_fields3]
    model = tf.keras.models.Model(input_layers, output, name='unet')
    model.summary()
    return model