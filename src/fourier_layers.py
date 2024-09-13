"""Author Neelesh Rampal
SEE CLIMOGAN UPDATE
Modified 18.01.23 - added some code to align the borders and use cubic interpolation?
Modified 19.01.23 - remove bicubic interpolation and have added a further step in later parts of the code
Modified 19.01.23 - added a selu activation for each of the components
Modified 23.08.23 - have changed the implementation to Bilinear implementation for computational efficiency
Modified 29.08.23 - changed the amount of noise from 0.1 to 0.45
Modified 7.09.23 - changed the padding to valid
Modified 9.09.23 - changed the complexity of the generator and discriminator as a test here.
Modified 10.09.23 - changed the loss to a log loss (predicting log of the variable)
Modified 14.09.23 - Change noise back to additive noise
Modified 18.09.23 - modified the noise profile so that noise is only perturbed once, I realize that this could cause instability

Modified 9. 12.23 Have modified the unet as a predictor to be normalized into the GAN, as I realise that this could affect the weights in the network. Have also normalized the image input into the discrimunation for stability.
# BUG IN discriminator model I haven't removed vegetation as a predictor, so don't know what effect this has
# Add a penality to getting the write average precipitation itensity.


# NOTE i AM CURRENTLY UPDATING THE MULTITASK TRAINING PROCESS


Need to figure out how to generalize this code across multiple variables???


"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd

batch_size = 32
rx3day_file = None
import tensorflow.keras.backend as K

IMG_SHAPE = (172, 179, 1)


def res_block_initial(x, num_filters, kernel_size, strides, name):
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
    x1 = tf.keras.layers.Activation('relu')(x1)
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
    x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Add()([x, x1])

    return x1


def res_block(x, num_filters, kernel_size, strides, name):
    """Residual Unet block layer
    Consists of batch norm and relu, folowed by conv, batch norm and relu and
    final convolution. The input is then put through
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

    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.Activation('relu')(x)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                kernel_size=kernel_size,
                                strides=strides[0],
                                padding='same',
                                name=name + '_1')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                kernel_size=kernel_size,
                                strides=strides[1],
                                padding='same',
                                name=name + '_2')(x1)

    x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                               kernel_size=1,
                               strides=strides[0],
                               padding='same',
                               name=name + '_shortcut')(x)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Add()([x, x1])

    return x1


#
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

    x_resized = BicubicUpSampling2D((target_size, target_size))(
        x)  # tf.keras.layers.Lambda(lambda x: tf.image.resize(x, target_size))(x)
    return x_resized


def get_discriminator_model(IMG_SHAPE, IMG_SHAPE2):
    """
    8.09.23 modified the discriminator so that it becomes more conditional
    9.09.23 slightly modified some of the conditional elements.
    """
    img_input = layers.Input(shape=IMG_SHAPE)
    img_input2 = layers.Input(shape=IMG_SHAPE2)

    # these are static inputs to the model
    img_input3 = layers.Input(shape=IMG_SHAPE)
    img_input6 = layers.Input(shape=IMG_SHAPE)
    # now we concatenate these input a single vector
    img_inputs = tf.keras.layers.Concatenate(-1)([img_input3, img_input6])
    # these are now concatenated into a single vector

    # THE SECOND input corresponds to the reanalysis data
    # these are the convolutional blocks corresponding to the conditional input
    x_init = conv_block(img_input2, 16, kernel_size=(3, 3), strides=(2, 2), use_bn=False, use_bias=True,
                        use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    x_init = conv_block(x_init, 32, kernel_size=(3, 3), strides=(2, 2), use_bn=False, use_bias=True,
                        use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # coarsen the data slightly
    x_init = conv_block(x_init, 64, kernel_size=(3, 3), strides=(2, 2), use_bn=False, use_bias=True,
                        use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # x_init  = tf.keras.layers.AveragePooling2D((2,2))(x_init)

    # x_init = res_block(x_init, [64], 3, strides=[1, 1], name="initial_res2")
    x_init = res_block_initial(x_init, [64], 3, [1, 1], "resid_layer2")
    # x_init = conv_block(x_init, 128, kernel_size=(3, 3), strides=(2, 2), use_bn=False, use_bias=True,
    #                use_dropout=False, drop_value=0.3, activation =tf.keras.layers.LeakyReLU())
    # flattening the conditional outputs
    flatten = tf.keras.layers.Flatten()(x_init)
    flatten = tf.keras.layers.Dense(12)(flatten)
    # reduce the dimensionality

    # the second inputs which are the topography and vegetation inputs are also going into the
    # discriminator model
    x = conv_block(img_input, 32, kernel_size=(3, 3), strides=(2, 2), use_bn=False, use_bias=True,
                   use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())

    # we want to raise less importance to these fields
    x2 = conv_block(img_inputs, 1, kernel_size=(5, 5), strides=(2, 2), use_bn=False, use_bias=True,
                    use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    # have two separate streams of conditional inputs into the model
    x = tf.keras.layers.Concatenate(-1)([x, x2])
    x = res_block_initial(x, [64], 3, [1, 1], "resid_layer3")

    x_init_raw = conv_block(x, 32, kernel_size=(3, 3), strides=(2, 2), use_bn=False, use_bias=True,
                            use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    x_init_raw = conv_block(x_init_raw, 64, kernel_size=(3, 3), strides=(2, 2), use_bn=False, use_bias=True,
                            use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    flattened_output = layers.Flatten()(x_init_raw)
    concat = tf.keras.layers.Concatenate(-1)([flatten, flattened_output])
    dense2 = tf.keras.layers.Dense(64)(concat)
    x = layers.Dense(1)(dense2)

    d_model = keras.models.Model([img_input, img_input2, img_input3, img_input6], x, name="discriminator")
    return d_model


def conv_block(x, filters, activation, kernel_size=(7, 7), strides=(2, 2), padding="same",
               use_bias=True, use_bn=True, use_dropout=True, drop_value=0.5):
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding=padding, use_bias=use_bias)(x)

    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)

    return x


def encoder(x, num_filters, kernel_size):
    """Unet encoder
    Args:
        x: tensor, output from previous layer
        num_filters: list, number of filters for each decoder layer
        kernel_size: int, size of the convolutional kernel
    Returns:
        encoder_output: list, output from all encoder layers
    """

    x = res_block_initial(x, [num_filters[0]], kernel_size, strides=[1, 1], name='layer1')

    encoder_output = [x]
    for i in range(1, len(num_filters)):
        layer = 'encoder_layer' + str(i)
        x = res_block(x, [num_filters[i]], kernel_size, strides=[2, 1], name=layer)
        encoder_output.append(x)

    return encoder_output


def decoder_noise(x, num_filters, kernel_size, noise=False, names_end="temp_layers", fourier=True,
                  weight_thres=[0.25, 0.35]):
    """Unet decoder
    Args:
        x: tensor, output from previous layer
        encoder_output: list, output from all previous encoder layers
        num_filters: list, number of filters for each decoder layer
        kernel_size: int, size of the convolutional kernel
    Returns:
        x: tensor, output from last layer of decoder
    """
    noise_inputs = []  # at some intermediate layers
    for i in range(1, len(num_filters) + 1):
        layer = 'decoder_layer' + str(i)
        layer2 = 'decoder_layer_v2' + str(i)
        x = upsample(x, 2)
        if not fourier:
            x = res_block(x, [num_filters[-i]], kernel_size, strides=[1, 1], name=layer2 + names_end)
        else:
            x = FourierLayer(n_fourier_filters=[num_filters[-i]], weight_thres=weight_thres, name=layer2 + names_end)(
                x)  # (resized_output)fourier_layer(x, [num_filters[-i]], name=layer2 + names_end, weight_thres=weight_thres)

    return x, noise_inputs


def conv_aux_input(img_inputs, input_size):
    x_init_ref_fields_high_res = conv_block(img_inputs, 1, kernel_size=(5, 5), strides=(1, 1), use_bn=True,
                                            use_bias=True,
                                            use_dropout=False, drop_value=0.0,
                                            activation=tf.keras.layers.LeakyReLU())
    # twice convolutional
    x_init_ref_fields_high_res = conv_block(x_init_ref_fields_high_res, 1, kernel_size=(3, 3), strides=(1, 1),
                                            use_bn=True,
                                            use_bias=True,
                                            use_dropout=False, drop_value=0.0,
                                            activation=tf.keras.layers.LeakyReLU())
    x_init_ref_fields = tf.keras.layers.AveragePooling2D((2, 2))(x_init_ref_fields_high_res)

    # lowering the importance of topography
    x_init_ref_fields = conv_block(x_init_ref_fields, 1, kernel_size=(5, 5), strides=(1, 1), use_bn=True,
                                   use_bias=True,
                                   use_dropout=False, drop_value=0.0, activation=tf.keras.layers.LeakyReLU())
    x_init_ref_fields = tf.keras.layers.AveragePooling2D((2, 2))(x_init_ref_fields)
    x_init_ref_fields = tf.image.resize(x_init_ref_fields, (input_size[0], input_size[1]),
                                        method=tf.image.ResizeMethod.BILINEAR)
    return x_init_ref_fields, x_init_ref_fields_high_res


def interaction_of_aux_and_input_fields(x, x_init_ref_fields, x_init_ref_fields_high_res, num_filters,
                                        n_filters_output=3, names="temp_field", allow_interaction=True,
                                        n_fouier_filters=[6, 12],
                                        weight_thres=[0.25, 0.25]):
    x_output = FourierLayer(n_fourier_filters=n_fouier_filters, weight_thres=weight_thres, name=names + "v1")(
        x)  # fourier_layer(x, n_fouier_filters, names+"v1", weight_thres)

    if allow_interaction:
        x_output = tf.keras.layers.Concatenate(-1)([x_init_ref_fields, x_output])

    decoder_output, noise_layers = decoder_noise(x_output, num_filters[:-1], 3, names_end=names, fourier=True,
                                                 weight_thres=weight_thres)
    # resizing the decoder output
    if allow_interaction:
        x_init_ref_fields_high_res_resized = tf.image.resize(x_init_ref_fields_high_res,
                                                             (decoder_output.shape[-3], decoder_output.shape[-2]),
                                                             method=tf.image.ResizeMethod.BILINEAR)
        decoder_output = tf.keras.layers.Concatenate(-1)([x_init_ref_fields_high_res_resized, decoder_output])
    x_output = FourierLayer(n_fourier_filters=[4, 4], weight_thres=[0.25, 0.28], name=names + "updated_v2")(
        decoder_output)
    x_output = FourierLayer(n_fourier_filters=[8, 8], weight_thres=[0.31, 0.35], name=names + "updated_v3")(
        x_output)  # fourier_layer(x_output, [8, 16], names+"updated_v3", [0.7, 0.65])
    return x_output


def create_output_prediction_layer(decoder_output, resize_output, img_inputs, final_activation_function, layer_name,
                                   num_classes=1, concat=True, n_layers=8, initializer_zeros=True,
                                   n_fouier_filters=[9, 6]):
    resized_output = tf.image.resize(decoder_output, (resize_output[0], resize_output[1]),
                                     method=tf.image.ResizeMethod.BILINEAR)
    if concat:
        resized_output = tf.keras.layers.Concatenate(-1)([resized_output, img_inputs])
    else:
        resized_output = resized_output

    output = FourierLayer(n_fourier_filters=n_fouier_filters, weight_thres=[0.3, 0.3], name=layer_name + "updated_v5",
                          custom_activation=final_activation_function)(resized_output)
    output = FourierLayer(n_fourier_filters=n_fouier_filters, weight_thres=[0.3, 0.3],
                          name=layer_name + "updated_v6", custom_activation=final_activation_function)(output)
    output = FourierLayer(n_fourier_filters=[n_layers, 1], weight_thres=[0.4, 0.4], name=layer_name + "output",
                          custom_activation=final_activation_function)(
        output)  # fourier_layer(output, [n_layers,1], layer_name+"updated_v6", [0.7, 0.5])
    return output


def unet_multitask(input_size, resize_output, num_filters, kernel_size, num_channels, num_classes, resize=True):
    """Residual Unet
    removed the concatenation with the outputs
    Function that generates a residual unet
    Args:
        input_size: int, dimension of the input image
        num_layers: int, number of layers in the encoder half, excludes bridge
        num_filters: list, number of filters for each encoder layer
        kernel_size: size of the kernel, applied to all convolutions
        num_channels: int, number of channels for the input image
        num_classes: int, number of output classes for the output
    Returns:
        model: tensorflow keras model for residual unet architecture

        8.09.23: Modified the residual blocks to handle additive noise in the architecture
        12.09.23: Modified to handle conditional topography inputs, giving them less importance
        13.03.24 I've modified the input and output streams of information,

        I have a feeling that the predictor fields that predict temperature are different to wind, so I enable
        some low-level interaction to improve the interaction of the variables
    """

    x = tf.keras.Input(shape=[input_size[0], input_size[1], num_channels], name='input1')
    topography = layers.Input(shape=[resize_output[0], resize_output[1], 1], name='input2')
    noise = layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]), name='input3')
    activation_weights = {"rainfall": 1e-5, "sfcwind": 1e-5, "sfcwindmax": 1e-5, "tasmin": 1, "tasmax": 1e-6}
    n_filters_dict = {"rainfall": 64, "sfcwind": 64, "sfcwindmax": 32, "tasmin": 64, "tasmax": 32}

    decoder_output_wind_pr = interaction_of_aux_and_input_fields(x, [],
                                                                 [], num_filters, names="pr_sfcwind_fields",
                                                                 allow_interaction=False, n_fouier_filters=[3, 6],
                                                                 weight_thres=[0.4, 0.4])
    rainfall_output = create_output_prediction_layer(decoder_output_wind_pr, resize_output, [],
                                                     tf.keras.layers.LeakyReLU(0.3), layer_name='rainfall', n_layers=3,
                                                     concat=False, n_fouier_filters=[3, 6])

    input_layers = [noise] + [x, topography]
    output_layers = [rainfall_output]
    model = tf.keras.Model(input_layers, output_layers)
    return model


class ComplexLinear(layers.Layer):
    def __init__(self, filters=16, weight_thres=0.25, n_modes=20, **kwargs):
        super(ComplexLinear, self).__init__(**kwargs)
        self.n_modes = n_modes
        self.filters = filters
        self.weight_thres = weight_thres

    @staticmethod
    def create_weighting_matrix(width, height):
        # Create meshgrid coordinates
        x_indices, y_indices = tf.meshgrid(tf.range(width), tf.range(height), indexing='ij')

        # Calculate the center coordinates
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0

        # Compute the distances from the center for each point
        distances = tf.sqrt(tf.square(tf.cast(x_indices, tf.float32) - center_x) +
                            tf.square(tf.cast(y_indices, tf.float32) - center_y))

        # Normalize the distance matrix by dividing by the maximum distance
        max_distance = tf.reduce_max(distances)
        normalized_distances = distances / max_distance

        return normalized_distances

    def build(self, input_shape):
        # Initialize real and imaginary parts separately
        self.units = (input_shape[1], input_shape[2], self.filters)
        self.weight_matrix = self.create_weighting_matrix(self.units[0], self.units[1])
        self.condition = tf.less(self.weight_matrix, self.weight_thres)
        self.n_modes = int(tf.reduce_sum(tf.cast(self.condition, 'float32')).numpy())

        self.real_kernel = self.add_weight(
            shape=(1, input_shape[-1] * self.n_modes * self.filters, 1, 1),
            initializer='random_normal',
            trainable=True,
            name='real_kernel'
        )
        self.imag_kernel = self.add_weight(
            shape=(1, input_shape[-1] * self.n_modes * self.filters, 1, 1),
            initializer='random_normal',
            trainable=True,
            name='imag_kernel'
        )
        self.weighting_matrix = tf.cast(tf.repeat(
            tf.repeat(tf.expand_dims(tf.expand_dims(self.weight_matrix, axis=-1), axis=-2), input_shape[-1],
                      axis=-2), self.filters, axis=-1), 'float32')
        self.indices = tf.where(tf.less(self.weighting_matrix, self.weight_thres))
        self.complex_output_final = tf.cast(tf.zeros(self.weighting_matrix.shape), 'float32')

    # @tf.custom_gradient
    def custom_operation(self, real_kernel, imag_kernel, inputs):
        def custom_scatter_nd(indices, updates, shape):
            result = tf.scatter_nd(tf.cast(indices, 'int64'), tf.squeeze(updates), tf.cast(shape, 'int64'))
            result = tf.signal.fftshift(result, axes=[0, 1])
            return result

        fft_inputs = tf.signal.fft3d(tf.cast(inputs, 'complex64'))
        real_kernel_only = custom_scatter_nd(self.indices, real_kernel, self.complex_output_final.shape)
        complex_kernel_only = custom_scatter_nd(self.indices, imag_kernel, self.complex_output_final.shape)
        complex_kernel = tf.complex(real_kernel_only, complex_kernel_only),  # real_kernel, complex_kernel)
        complex_output = tf.einsum('abcd,bcde-> abce', fft_inputs, complex_kernel[0])
        real_output = tf.cast(tf.math.real(tf.signal.ifft3d(complex_output)), 'float32')

        # def grad(dy):
        #     grad_K = tf.cast(tf.math.real(tf.signal.ifft3d(complex_kernel)), 'float32')
        #     grad_K = tf.reduce_mean(grad_K, axis=-2)
        #     grad_inputs = tf.matmul(dy, grad_K, transpose_b=True)
        #
        #     real_fouier = tf.cast(real_kernel_only,'complex64') * fft_inputs
        #     inverse_real = tf.cast(tf.math.real(tf.signal.ifft3d(real_fouier)), 'float32')
        #     complex_fouier = tf.cast(complex_kernel_only,'complex64') * fft_inputs
        #     inverse_imag = tf.cast(tf.math.real(tf.signal.ifft3d(complex_fouier)), 'float32')
        #     grad_real = tf.reduce_mean(inverse_real * tf.expand_dims(dy, axis=3), axis =0)
        #     grad_real = tf.reshape(tf.gather_nd(grad_real, self.indices), real_kernel.shape)
        #     grad_imag= tf.reduce_mean(inverse_imag * tf.expand_dims(dy, axis=3), axis =0)
        #     grad_imag = tf.reshape(tf.gather_nd(grad_imag, self.indices), real_kernel.shape)
        #
        #     return grad_real, grad_imag, grad_inputs

        return real_output  # , grad
        # super(ComplexLinear, self).build(input_shape)

    def call(self, inputs):
        return self.custom_operation(self.real_kernel, self.imag_kernel, inputs)


class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, n_fourier_filters, weight_thres, name=None, custom_activation=None, bn=True, **kwargs):
        super(FourierLayer, self).__init__(name=name, **kwargs)
        self.n_fourier_filters = n_fourier_filters
        self.weight_thres = weight_thres
        self.bn = bn

        # Initialize layers here
        self.conv2d = tf.keras.layers.Conv2D(filters=n_fourier_filters[-1],
                                             kernel_size=1,
                                             padding='same',
                                             name=name + '_1')
        self.complex_linear1 = ComplexLinear(filters=n_fourier_filters[0],
                                             weight_thres=weight_thres[0], name=name + '_2')
        self.complex_linear2 = ComplexLinear(filters=n_fourier_filters[-1],
                                             weight_thres=weight_thres[1], name=name + '_3')
        self.add_layer = tf.keras.layers.Add()
        if custom_activation is not None:
            self.activation = custom_activation
        else:
            self.activation = tf.keras.layers.LeakyReLU(0.1)

    def call(self, input_vector):
        x1 = self.conv2d(input_vector)
        output = self.complex_linear1(input_vector)
        output = self.complex_linear2(output)
        residual_layer = self.add_layer([x1, output])
        activation = self.activation(residual_layer)
        # if self.bn:
        #    activation = tf.keras.layers.BatchNormalization()(activation)
        return activation

    def get_config(self):
        config = super(FourierLayer, self).get_config()
        config.update({
            'n_fourier_filters': self.n_fourier_filters,
            'weight_thres': self.weight_thres
        })
        return config


def res_unet_multitask(input_size, resize_output, num_filters, kernel_size, num_channels,
                       num_classes, resize=True, single_task=False,
                       single_layer_name='rainfall'):
    """Residual Unet
    Function that generates a residual unet
    Args:
        input_size: int, dimension of the input image
        num_layers: int, number of layers in the encoder half, excludes bridge
        num_filters: list, number of filters for each encoder layer
        kernel_size: size of the kernel, applied to all convolutions
        num_channels: int, number of channels for the input image
        num_classes: int, number of output classes for the output
    Returns:
        model: tensorflow keras model for residual unet architecture

        8.09.23: Modified the residual blocks to handle additive noise in the architecture
        12.09.23: Modified to handle conditional topography inputs, giving them less importance
    """

    x = tf.keras.Input(shape=[input_size[0], input_size[1], num_channels])
    # adding separate secondary inputs
    topography = layers.Input(shape=[resize_output[0], resize_output[1], 1])
    unet_rainfall = layers.Input(shape=[resize_output[0], resize_output[1], 1])


    img_inputs = tf.keras.layers.Concatenate(-1)([topography, unet_rainfall])

    x_init_ref_fields, x_init_ref_fields_high_res = conv_aux_input(img_inputs, input_size)
    noise = layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    concat_noise = tf.keras.layers.Concatenate(-1)([noise, x])
    decoder_output_wind_pr = interaction_of_aux_and_input_fields(x, x_init_ref_fields,
                                                                 x_init_ref_fields_high_res, num_filters,
                                                                 names="pr_sfcwind_fields")
    decoder_output_tmax = interaction_of_aux_and_input_fields(x, x_init_ref_fields,
                                                              x_init_ref_fields_high_res, num_filters,
                                                              n_filters_output=5, names="temp_fields")

    activation_weights = {"rainfall": 1e-5, "sfcwind": 1, "sfcwindmax": 1e-5, "tasmin": 0.7, "tasmax": 1e-5}
    n_filters_dict = {"rainfall": 64, "sfcwind": 64, "sfcwindmax": 32, "tasmin": 64, "tasmax": 32}

    rainfall_output = create_output_prediction_layer(decoder_output_wind_pr, resize_output, img_inputs,
                                                     tf.keras.layers.LeakyReLU(0.5), layer_name='rainfall', n_layers=64,
                                                     initializer_zeros=True)
    sfcwind_output = create_output_prediction_layer(decoder_output_wind_pr, resize_output, img_inputs,
                                                    tf.keras.layers.LeakyReLU(1), layer_name='sfcwind', n_layers=64,
                                                    initializer_zeros=True)
    sfcwindmax_output = create_output_prediction_layer(decoder_output_wind_pr, resize_output, img_inputs,
                                                       tf.keras.layers.LeakyReLU(1e-5), layer_name='sfcwindmax',
                                                       initializer_zeros=True)
    # hard_constraint
    sfcwindmax_output = sfcwindmax_output + sfcwind_output
    # weak constraint
    tasmax_output = create_output_prediction_layer(decoder_output_tmax, resize_output, img_inputs,
                                                   tf.keras.layers.LeakyReLU(1e-5), layer_name='tasmax',
                                                   initializer_zeros=True)
    tasmin_output = create_output_prediction_layer(decoder_output_tmax, resize_output, img_inputs,
                                                   tf.keras.layers.LeakyReLU(1), layer_name='tasmin', n_layers=64,
                                                   initializer_zeros=True)
    tasmax_output = tasmax_output + tasmin_output

    input_layers = [noise] + [x, topography, unet_rainfall, unet_sfcwind, unet_sfcwindmax,
                              unet_tasmax, unet_tasmin]
    output_layers = [rainfall_output, sfcwind_output, sfcwindmax_output, tasmax_output, tasmin_output]
    # added multiple inputs into the tensorflow
    # output = output + x_input
    model = tf.keras.Model(input_layers, output_layers)

    return model


class WGAN_Cascaded_Multi(keras.Model):
    """
    adapted from https://arxiv.org/pdf/2207.01561.pdf
    https://arxiv.org/pdf/1903.05628.pdf

    Also from https://arxiv.org/pdf/1903.05628.pdf

    It is also likely that our GAN suffers from MODE collapse and has an inability to generate diversity
s
    Added static vegetation inputs as a predictor
    """

    def __init__(self, discriminator, generator, latent_dim,
                 discriminator_extra_steps=3, gp_weight=10.0, ad_loss_factor=1e-3, latent_loss=5e-2, orog=None, he=None,
                 vegt=None, unet=None, train_gan=True, train_unet=True):
        super(WGAN_Cascaded_Multi, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.ad_loss_factor = ad_loss_factor
        self.latent_loss = latent_loss
        self.orog = orog
        self.he = he
        self.vegt = vegt
        self.unet = unet
        self.train_gan = train_gan
        self.train_unet = train_unet

    def compile(self, d_optimizer, g_optimizer,
                d_loss_fn, g_loss_fn, u_loss_fn, u_optimizer):
        super(WGAN_Cascaded_Multi, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.u_loss_fn = u_loss_fn
        self.u_optimizer = u_optimizer

    @staticmethod
    def gradient_penalty(discriminator, batch_size, real_images, fake_images, average, orog_vector,
                         unet_preds):
        """
        need to modify
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = discriminator([interpolated, average, orog_vector, unet_preds],
                                 training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @staticmethod
    def expand_conditional_inputs(X, batch_size):
        expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

        # Repeat the image to match the desired batch size
        expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

        # Create a new axis (1) on the last axis
        expanded_image = tf.expand_dims(expanded_image, axis=-1)
        return expanded_image

    def train_step(self, real_images):

        real_images, average = real_images[0]
        real_images = tf.concat([tf.expand_dims(expanded_image, axis=-1)
                                 for expanded_image in real_images], axis=-1)
        # I need to combine all the GCMs into one single batch timestep
        real_images = tf.concat([real_images[:, :, :, i, :] for i in range(real_images.shape[3])], axis=0)
        average = tf.concat([average[:, :, :, i, :] for i in range(average.shape[3])], axis=0)

        batch_size = tf.shape(real_images)[0]  # this should now be N_GCM times the average
        print(real_images.shape, batch_size, average.shape)
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)
        he_vector = self.expand_conditional_inputs(self.he, batch_size)
        vegt_vector = self.expand_conditional_inputs(self.vegt, batch_size)
        if self.train_unet:
            # running the backprop for he unet
            with tf.GradientTape() as tape:
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )

                unet_predictions = self.unet([random_latent_vectors, average,
                                              orog_vector], training=True)

                rainfall_unet, sfcwind_unet, sfcwindmax_unet, \
                tasmax_unet, tasmin_unet = unet_predictions
                loss_rain = self.u_loss_fn(real_images[:, :, :, 0:1], rainfall_unet)
                loss_sfcwind = self.u_loss_fn(real_images[:, :, :, 1:2], sfcwind_unet)
                loss_sfcwindmax = self.u_loss_fn(real_images[:, :, :, 2:3], sfcwindmax_unet)
                loss_tasmax = self.u_loss_fn(real_images[:, :, :, 3:4], tasmax_unet)
                loss_tasmin = self.u_loss_fn(real_images[:, :, :, 4:5], tasmin_unet)

                total_loss = loss_rain + loss_sfcwind + loss_sfcwindmax + loss_tasmin + loss_tasmax
            # train based on a multi-task loss function, using a shared latent-space.
            u_gradient = tape.gradient(total_loss, self.unet.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.u_optimizer.apply_gradients(zip(u_gradient, self.unet.trainable_variables))
            return {
                "unet_loss": total_loss}

        if self.train_gan:
            for n_gans in range(len(self.discriminator)):

                # Get the latent vector
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )
                unet_predictions = self.unet([random_latent_vectors, average,
                                              orog_vector], training=True)
                rainfall_unet, sfcwind_unet, sfcwindmax_unet, \
                tasmax_unet, tasmin_unet = unet_predictions
                generator_preds = self.generator([random_latent_vectors, average,
                                                  orog_vector, rainfall_unet, sfcwind_unet, sfcwindmax_unet,
                                                  tasmax_unet, tasmin_unet], training=True)
                rainfall_gan, sfcwind_gan, sfcwindmax_gan, tasmax_gan, tasmin_gan = generator_preds
                # here we introduce a gan for each individual variable
                for i in range(self.d_steps):
                    with tf.GradientTape() as tape:
                        fake_logits = self.discriminator[n_gans](
                            [generator_preds[n_gans], average, orog_vector, unet_predictions[n_gans]], training=True)
                        # Get the logits for the real images
                        # modified this line to now predict the residuals of the solution
                        real_logits = self.discriminator[n_gans](
                            [real_images[:, :, :, n_gans:n_gans + 1] - unet_predictions[n_gans], average, orog_vector,
                             unet_predictions[n_gans]],
                            training=True)

                        # Calculate the discriminator loss using the fake and real image logits
                        d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                        # Calculate the gradient penalty
                        gp = self.gradient_penalty(self.discriminator[n_gans], batch_size,
                                                   real_images[:, :, :, n_gans:n_gans + 1] - unet_predictions[n_gans],
                                                   generator_preds[n_gans],
                                                   average, orog_vector, unet_predictions[n_gans])

                        # Add the gradient penalty to the original discriminator loss
                        d_loss = d_cost + gp * self.gp_weight  # + #50 * tf.keras.losses.mean_squared_error(average, fake_image_average)

                    # Get the gradients w.r.t the discriminator loss
                    d_gradient = tape.gradient(d_loss, self.discriminator[n_gans].trainable_variables)
                    # Update the weights of the discriminator using the discriminator optimizer
                    self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator[n_gans].trainable_variables))

            # Train the generato
            # we should also explore multiples of noise levels perhaps also

            # multiplying the vector by two and examining the diversity in the response

            with tf.GradientTape() as tape:
                """
                Introducing the Maximum and Average Penalty in the Loss function for each variable 
                """
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )

                unet_predictions = self.unet([random_latent_vectors, average,
                                              orog_vector], training=True)
                generator_preds = self.generator([random_latent_vectors, average,
                                                  orog_vector, rainfall_unet, sfcwind_unet, sfcwindmax_unet,
                                                  tasmax_unet, tasmin_unet], training=True)
                rainfall_gan, sfcwind_gan, sfcwindmax_gan, tasmax_gan, tasmin_gan = generator_preds
                rainfall_unet, sfcwind_unet, sfcwindmax_unet, \
                tasmax_unet, tasmin_unet = unet_predictions
                loss_rain_gan = self.u_loss_fn(real_images[:, :, :, 0:1] - rainfall_unet, rainfall_gan)
                loss_sfcwind_gan = self.u_loss_fn(real_images[:, :, :, 1:2] - sfcwind_unet, sfcwind_gan)
                loss_sfcwindmax_gan = self.u_loss_fn(real_images[:, :, :, 2:3] - sfcwindmax_unet, sfcwindmax_gan)
                loss_tasmax_gan = self.u_loss_fn(real_images[:, :, :, 3:4] - tasmax_unet, tasmax_gan)
                loss_tasmin_gan = self.u_loss_fn(real_images[:, :, :, 4:5] - tasmin_unet, tasmin_gan)

                total_loss_mse = loss_rain_gan + loss_sfcwind_gan + loss_sfcwindmax_gan + 1.15 * (
                            loss_tasmin_gan + loss_tasmax_gan)
                adv_loss = total_loss_mse * tf.cast(0.0, 'float32')  # to avoid any issues with adding contributions
                maximum_penalty = total_loss_mse * tf.cast(0.0, 'float32')

                maximum_intensity = tf.math.reduce_max(
                    real_images[:, :, :, 0:1], axis=[-1, -2, -3])
                maximum_intensity_predicted = tf.math.reduce_max(rainfall_gan + rainfall_unet,
                                                                 axis=[-1, -2, -3])
                maximum_constraint = tf.reduce_mean(
                    tf.abs(maximum_intensity - maximum_intensity_predicted) ** 2)
                adv_losses = []
                for n_gans_adv in range(len(self.discriminator)):
                    fake_logits = self.discriminator[n_gans_adv](
                        [generator_preds[n_gans_adv], average, orog_vector, unet_predictions[n_gans_adv]],
                        training=True)

                    # add a maximum penality for each variable
                    adv_loss_individual = self.ad_loss_factor * self.g_loss_fn(fake_logits)
                    adv_losses.append(adv_loss_individual)
                # Calculate the generator loss

                g_loss = adv_losses[0] + adv_losses[1] + adv_losses[2] + adv_losses[3] + adv_losses[
                    4] + total_loss_mse + 0.3 * maximum_constraint  # + self.latent_loss * latent_loss

            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

            return {"d_loss": d_loss, "g_loss": g_loss, "gan_loss_mse": total_loss_mse, "adv_loss": adv_loss,
                    "loss_rain_gan": loss_rain_gan,
                    "adv_ind": adv_loss_individual, "loss_sfcwindmax_gan": loss_sfcwindmax_gan,
                    "max_iten_pred": tf.reduce_mean(maximum_intensity_predicted),
                    "max_iten_gt": tf.reduce_mean(maximum_intensity)}


def gamma_loss(y_true, y_pred, eps=3e-1):
    occurence = y_pred[:, :, :, -1]
    y_true = y_true[:, :, :, 0]
    shape_param = K.exp(y_pred[:, :, :, 0])
    scale_param = K.exp(y_pred[:, :, :, 1])
    bool_rain = tf.cast(y_true > 0.01, 'float32')
    eps = tf.cast(eps, 'float32')
    loss1 = ((1 - bool_rain) * tf.math.log(1 - occurence + eps) + bool_rain * (
            K.log(occurence + eps) + (shape_param - 1) * K.log(y_true + eps) -
            shape_param * tf.math.log(scale_param + eps) - tf.math.lgamma(shape_param) - y_true / (
                    scale_param + eps)))
    # bool_rain = K.flatten(bool_rain)
    # occurence = K.flatten(occurence)
    output_loss = -1 * (K.mean(loss1))
    return output_loss


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def predict(model, x_test, y_test, batch_size=32, key='Rain_bc', pred_name='simple_dense', loss='gamma', thres=0.25):
    """
    This is a function that converts a prediction to a netcdf so that it can be plotted easily
    model: tensorflow model
    x_test: input data ( e.g.. (26, 23, 5) where 26 pixels in the latitude, 23 in longitude and 5 channels)
    y_test: y_test data, please note that this should be a netcdf! not a numpy array
    loss: "gamma" or mse
    """
    data = y_test.to_dataset()
    preds = model.predict(x_test, verbose=1, batch_size=batch_size)
    if loss == "gamma":
        scale = np.exp(preds[:, :, :, 0])
        shape = np.exp(preds[:, :, :, 1])
        prob = preds[:, :, :, -1]
        rainfall = (prob > thres) * scale * shape
    else:
        rainfall = preds
    data[key].values = rainfall
    return data.rename({key: pred_name})


class GeneratorCheckpoint(Callback):
    def __init__(self, generator, filepath, period):
        super().__init__()
        self.generator = generator
        self.filepath = filepath
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.generator.save(f"{self.filepath}_epoch_{epoch + 1}.h5")


class DiscriminatorCheckpoint(Callback):
    def __init__(self, discriminator, filepath, period):
        super().__init__()
        self.discriminator = discriminator
        self.filepath = filepath
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            self.discriminator.save(f"{self.filepath}_epoch_{epoch + 1}.h5")


# changed activation function to hyperbolic tangent


def prepare_training_data(config, X, y, means, stds):
    """
    Normalizes the X training data, and stacks the features into a single dimension
    config: json file that contains a dictionary of the experimental files used in training
    X: training data, which is pre-loaded. Note this file is already in the config file, but has been loaded in another script
    mean:: normalize relative to a mean
    std: normalize relative to an std
    """

    list_of_vars = config["var_names"]
    # normalize data
    X_norm = (X[list_of_vars] - means) / stds

    stacked_X = xr.concat([X_norm[varname] for varname in list_of_vars], dim="channel")
    # stack features
    stacked_X['channel'] = (('channel'), list_of_vars)
    stacked_X = stacked_X  # .transpose("time", "lat", "lon", "channel")
    times = stacked_X.time.to_index().intersection(y.time.to_index())
    # this part should be fine
    stacked_X = stacked_X.sel(time=times)
    y = y.sel(time=times)
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
def preprocess_input_data(config):
    vegt, orog, he = prepare_static_fields(config)
    means = xr.open_dataset(config["mean"])
    stds = xr.open_dataset(config["std"])

    X = xr.open_dataset(config["train_x"])  # .sel(time = slice("2016", None))
    X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d"))

    y = xr.open_dataset(config["train_y"], chunks={"time": 5000})
    y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d"))  # .sel(time = slice("2016", None))
    try:
        y = y.drop("lat_bnds")
        y = y.drop("lon_bnds")
        y = y.drop("time_bnds")

    except:
        pass
    # preare the training data
    stacked_X, y = prepare_training_data(config, X, y, means, stds)

    return stacked_X, y, vegt, orog, he

#
# class WGAN_Cascaded_MultiGAMMA(keras.Model):
#     """
#     adapted from https://arxiv.org/pdf/2207.01561.pdf
#     https://arxiv.org/pdf/1903.05628.pdf
#
#     Also from https://arxiv.org/pdf/1903.05628.pdf
#
#     It is also likely that our GAN suffers from MODE collapse and has an inability to generate diversity
# s
#     Added static vegetation inputs as a predictor
#     """
#
#     def __init__(self, discriminator, generator, latent_dim,
#                  discriminator_extra_steps=3, gp_weight=10.0, ad_loss_factor=1e-3, latent_loss=5e-2, orog=None, he=None,
#                  vegt=None, unet=None):
#         super(WGAN_Cascaded_MultiGAMMA, self).__init__()
#
#         self.discriminator = discriminator
#         self.generator = generator
#         self.latent_dim = latent_dim
#         self.d_steps = discriminator_extra_steps
#         self.gp_weight = gp_weight
#         self.ad_loss_factor = ad_loss_factor
#         self.latent_loss = latent_loss
#         self.orog = orog
#         self.he = he
#         self.vegt = vegt
#         self.unet = unet
#
#     def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, u_loss_fn, u_optimizer):
#         super(WGAN_Cascaded_MultiGAMMA, self).compile()
#         self.d_optimizer = d_optimizer
#         self.g_optimizer = g_optimizer
#         self.d_loss_fn = d_loss_fn
#         self.g_loss_fn = g_loss_fn
#         self.u_loss_fn = u_loss_fn
#         self.u_optimizer = u_optimizer
#
#     @staticmethod
#     def gradient_penalty(discriminator, batch_size, real_images, fake_images, average, orog_vector,
#                          unet_preds):
#         """
#         need to modify
#         """
#         # Get the interpolated image
#         alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
#         diff = fake_images - real_images
#         interpolated = real_images + alpha * diff
#
#         with tf.GradientTape() as gp_tape:
#             gp_tape.watch(interpolated)
#             pred = discriminator([interpolated, average, orog_vector, unet_preds],
#                                  training=True)
#
#         grads = gp_tape.gradient(pred, [interpolated])[0]
#         norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
#         gp = tf.reduce_mean((norm - 1.0) ** 2)
#         return gp
#
#     @staticmethod
#     def expand_conditional_inputs(X, batch_size):
#         expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)
#
#         # Repeat the image to match the desired batch size
#         expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)
#
#         # Create a new axis (1) on the last axis
#         expanded_image = tf.expand_dims(expanded_image, axis=-1)
#         return expanded_image
#
#     def train_step(self, real_images):
#
#         real_images, average = real_images[0]
#         real_images = tf.concat([tf.expand_dims(expanded_image, axis=-1) for expanded_image in real_images], axis=-1)
#         print(real_images.shape)
#         batch_size = tf.shape(real_images)[0]
#         orog_vector = self.expand_conditional_inputs(self.orog, batch_size)
#         he_vector = self.expand_conditional_inputs(self.he, batch_size)
#         vegt_vector = self.expand_conditional_inputs(self.vegt, batch_size)
#
#         # running the backprop for he unet
#         with tf.GradientTape() as tape:
#             random_latent_vectors = tf.random.normal(
#                 shape=(batch_size,) + self.latent_dim[0]
#             )
#
#             unet_predictions = self.unet([random_latent_vectors, average,
#                                           orog_vector], training=True)
#
#             rainfall_unet, sfcwind_unet, sfcwindmax_unet, \
#             tasmax_unet, tasmin_unet = unet_predictions
#
#             loss_rain = gamma_loss(real_images[:, :, :, 0:1], rainfall_unet)
#             loss_sfcwind = self.u_loss_fn(real_images[:, :, :, 1:2], sfcwind_unet)
#             loss_sfcwindmax = self.u_loss_fn(real_images[:, :, :, 2:3], sfcwindmax_unet)
#             loss_tasmax = self.u_loss_fn(real_images[:, :, :, 3:4], tasmax_unet)
#             loss_tasmin = self.u_loss_fn(real_images[:, :, :, 4:5], tasmin_unet)
#             rainfall_unet_value = tf.math.exp(rainfall_unet[:, :, :, 0]) * tf.math.exp(
#                 rainfall_unet[:, :, :, 1]) * tf.cast(rainfall_unet[:, :, :, -1] > 0.01,
#                                                      'float32')
#             total_loss = loss_rain + loss_sfcwind + loss_sfcwindmax + loss_tasmin + loss_tasmax
#         # train based on a multi-task loss function, using a shared latent-space.
#         u_gradient = tape.gradient(total_loss, self.unet.trainable_variables)
#         # Update the weights of the generator using the generator optimizer
#         self.u_optimizer.apply_gradients(zip(u_gradient, self.unet.trainable_variables))
#
#         for i in range(self.d_steps):
#             # Get the latent vector
#             random_latent_vectors = tf.random.normal(
#                 shape=(batch_size,) + self.latent_dim[0]
#             )
#
#             generator_preds = self.generator([random_latent_vectors, average,
#                                               orog_vector, rainfall_unet_value, sfcwind_unet, sfcwindmax_unet,
#                                               tasmax_unet, tasmin_unet], training=True)
#             rainfall_gan, sfcwind_gan, sfcwindmax_gan, tasmax_gan, tasmin_gan = generator_preds
#             rainfall_gan_value = tf.math.exp(rainfall_gan[:, :, :, 0]) * tf.math.exp(
#                 rainfall_gan[:, :, :, 1]) * tf.cast(rainfall_gan[:, :, :, -1] > 0.01, 'float32')
#             generator_preds = [rainfall_gan_value, sfcwind_gan, sfcwindmax_gan, tasmax_gan, tasmin_gan]
#             # here we introduce a gan for each individual variable
#             for n_gans in range(len(self.discriminator)):
#                 with tf.GradientTape() as tape:
#                     fake_logits = self.discriminator[n_gans](
#                         [generator_preds[n_gans], average, orog_vector, unet_predictions[n_gans]], training=True)
#                     # Get the logits for the real images
#                     real_logits = self.discriminator[n_gans](
#                         [real_images[:, :, :, n_gans:n_gans + 1], average, orog_vector, unet_predictions[n_gans]],
#                         training=True)
#
#                     # Calculate the discriminator loss using the fake and real image logits
#                     d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
#                     # Calculate the gradient penalty
#                     gp = self.gradient_penalty(self.discriminator[n_gans], batch_size,
#                                                real_images[:, :, :, n_gans:n_gans + 1], generator_preds[n_gans],
#                                                average, orog_vector, unet_predictions[n_gans])
#
#                     # Add the gradient penalty to the original discriminator loss
#                     d_loss = d_cost + gp * self.gp_weight  # + #50 * tf.keras.losses.mean_squared_error(average, fake_image_average)
#
#                 # Get the gradients w.r.t the discriminator loss
#                 d_gradient = tape.gradient(d_loss, self.discriminator[n_gans].trainable_variables)
#                 # Update the weights of the discriminator using the discriminator optimizer
#                 self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator[n_gans].trainable_variables))
#
#         # Train the generato
#         # we should also explore multiples of noise levels perhaps also
#
#         # multiplying the vector by two and examining the diversity in the response
#
#         with tf.GradientTape() as tape:
#
#             generator_preds = self.generator([random_latent_vectors, average,
#                                               orog_vector, rainfall_unet, sfcwind_unet, sfcwindmax_unet,
#                                               tasmax_unet, tasmin_unet], training=True)
#             rainfall_gan, sfcwind_gan, sfcwindmax_gan, tasmax_gan, tasmin_gan = generator_preds
#
#             loss_rain_gan = self.u_loss_fn(real_images[:, :, :, 0:1], rainfall_gan)
#             loss_sfcwind_gan = self.u_loss_fn(real_images[:, :, :, 1:2], sfcwind_gan)
#             loss_sfcwindmax_gan = self.u_loss_fn(real_images[:, :, :, 2:3], sfcwindmax_gan)
#             loss_tasmax_gan = self.u_loss_fn(real_images[:, :, :, 3:4], tasmax_gan)
#             loss_tasmin_gan = self.u_loss_fn(real_images[:, :, :, 4:5], tasmin_gan)
#
#             total_loss_mse = loss_rain_gan + loss_sfcwind_gan + loss_sfcwindmax_gan + loss_tasmin_gan + loss_tasmax_gan
#             adv_loss = total_loss_mse * tf.cast(0.0, 'float32')  # to avoid any issues with adding contributions
#             for n_gans_adv in range(len(self.discriminator)):
#                 fake_logits = self.discriminator[n_gans_adv](
#                     [generator_preds[n_gans_adv], average, orog_vector, unet_predictions[n_gans_adv]], training=True)
#                 # Get the logits for the real images
#
#                 adv_loss_individual = self.ad_loss_factor * self.g_loss_fn(fake_logits)
#                 adv_loss += adv_loss_individual
#             # Calculate the generator loss
#
#             g_loss = adv_loss + total_loss_mse  # + self.latent_loss * latent_loss
#
#         # Get the gradients w.r.t the generator loss
#         gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
#         # Update the weights of the generator using the generator optimizer
#         self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
#
#         return {"d_loss": d_loss, "g_loss": g_loss, "gan_loss_mse": total_loss_mse, "adv_loss": adv_loss,
#                 "unet_loss": total_loss, "loss_rain_gan": loss_rain_gan, "loss_sfcwindmax_gan": loss_sfcwindmax_gan}

# def res_unet_multitask_gamma_rainfall(input_size, resize_output, num_filters, kernel_size, num_channels,
#                                           num_classes, resize=True):
#     """Residual Unet
#     Function that generates a residual unet
#     Args:
#         input_size: int, dimension of the input image
#         num_layers: int, number of layers in the encoder half, excludes bridge
#         num_filters: list, number of filters for each encoder layer
#         kernel_size: size of the kernel, applied to all convolutions
#         num_channels: int, number of channels for the input image
#         num_classes: int, number of output classes for the output
#     Returns:
#         model: tensorflow keras model for residual unet architecture
#
#         8.09.23: Modified the residual blocks to handle additive noise in the architecture
#         12.09.23: Modified to handle conditional topography inputs, giving them less importance
#     """
#
#     x = tf.keras.Input(shape=[input_size[0], input_size[1], num_channels])
#     # adding separate secondary inputs
#     topography = layers.Input(shape=[resize_output[0], resize_output[1], 1])
#
#
#     unet_rainfall = layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     unet_sfcwind = layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     unet_sfcwindmax = layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     unet_tasmax = layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     unet_tasmin = layers.Input(shape=[resize_output[0], resize_output[1], 1])
#
#     img_inputs = tf.keras.layers.Concatenate(-1)([topography, unet_rainfall, unet_sfcwind, unet_sfcwindmax,
#                                                    unet_tasmax, unet_tasmin])#, img_input4, img_input5])
#
#     x_init_ref_fields, x_init_ref_fields_high_res = conv_aux_input(img_inputs, input_size)
#     noise = layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
#     concat_noise = tf.keras.layers.Concatenate(-1)([noise, x])
#     decoder_output = interaction_of_aux_and_input_fields(concat_noise, x_init_ref_fields,
#                                                          x_init_ref_fields_high_res, num_filters)
#     rainfall_output_a = create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#
#                                                      tf.keras.layers.LeakyReLU(0.4), layer_name='alpha')
#     rainfall_output_b = create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#                                                      tf.keras.layers.LeakyReLU(0.4), layer_name='beta')
#
#     rainfall_output_c = create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#                                                      'sigmoid', layer_name='p')
#
#     rainfall_output = tf.keras.layers.Concatenate(-1)([rainfall_output_a, rainfall_output_b, rainfall_output_c])
#
#
#     sfcwind_output = create_output_prediction_layer(decoder_output,resize_output, img_inputs,
#                                                      'linear', layer_name='sfcwind')
#     sfcwindmax_output = create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#                                                      tf.keras.layers.LeakyReLU(1e-5), layer_name='sfcwindmax')
#     # hard_constraint
#     sfcwindmax_output = sfcwindmax_output + sfcwind_output
#     # weak constraint
#     tasmax_output = create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#                                                      tf.keras.layers.LeakyReLU(1e-5), layer_name='tasmax')
#     tasmin_output = create_output_prediction_layer(decoder_output,resize_output, img_inputs,
#                                                      'linear', layer_name='tasmin' )
#     tasmax_output = tasmax_output + tasmin_output
#
#     input_layers = [noise] + [x, topography, unet_rainfall, unet_sfcwind, unet_sfcwindmax,
#                                                    unet_tasmax, unet_tasmin]
#     output_layers = [rainfall_output, sfcwind_output, sfcwindmax_output, tasmax_output, tasmin_output]
#     # added multiple inputs into the tensorflow
#     # output = output + x_input
#     model = tf.keras.Model(input_layers, output_layers)
#
#     return model
#
# def unet_multitask_gamma_rainfall(input_size, resize_output, num_filters, kernel_size, num_channels, num_classes, resize=True):
#     """Residual Unet
#     Function that generates a residual unet
#     Args:
#         input_size: int, dimension of the input image
#         num_layers: int, number of layers in the encoder half, excludes bridge
#         num_filters: list, number of filters for each encoder layer
#         kernel_size: size of the kernel, applied to all convolutions
#         num_channels: int, number of channels for the input image
#         num_classes: int, number of output classes for the output
#     Returns:
#         model: tensorflow keras model for residual unet architecture
#
#         8.09.23: Modified the residual blocks to handle additive noise in the architecture
#         12.09.23: Modified to handle conditional topography inputs, giving them less importance
#     """
#
#     x = tf.keras.Input(shape=[input_size[0], input_size[1], num_channels])
#     # adding separate secondary inputs
#     topography = layers.Input(shape=[resize_output[0], resize_output[1], 1])
#     img_inputs = topography  # tf.keras.layers.Concatenate(-1)([img_input3])#, img_input4, img_input5])
#
#     x_init_ref_fields, x_init_ref_fields_high_res = conv_aux_input(img_inputs, input_size)
#     noise = layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
#     decoder_output = interaction_of_aux_and_input_fields(x, x_init_ref_fields,
#                                                          x_init_ref_fields_high_res, num_filters)
#
#     rainfall_output_a = create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#                                                      tf.keras.layers.LeakyReLU(0.4), layer_name='alpha')
#     rainfall_output_b = create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#                                                      tf.keras.layers.LeakyReLU(0.4), layer_name='beta')
#
#     rainfall_output_c = create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#                                                      'sigmoid', layer_name='p')
#
#     rainfall_output = tf.keras.layers.Concatenate(-1)([rainfall_output_a, rainfall_output_b, rainfall_output_c])
#
#
#     sfcwind_output = create_output_prediction_layer(decoder_output,resize_output, img_inputs,
#                                                      'linear', layer_name='sfcwind')
#     # these enforce the hard constraints on temperature maximum and temperature minimum
#     sfcwindmax_output = sfcwind_output + create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#                                                      tf.keras.layers.LeakyReLU(1e-5), layer_name='sfcwindmax')
#     tasmin_output = create_output_prediction_layer(decoder_output,resize_output, img_inputs,
#                                                      'linear', layer_name='tasmin')
#     tasmax_output = tasmin_output + create_output_prediction_layer(decoder_output, resize_output, img_inputs,
#                                                      tf.keras.layers.LeakyReLU(1e-5), layer_name='tasmax')
#
#     input_layers = [noise] + [x, topography]
#     output_layers = [rainfall_output,sfcwind_output, sfcwindmax_output, tasmax_output, tasmin_output]
#     # added multiple inputs into the tensorflow
#     # output = output + x_input
#     model = tf.keras.Model(input_layers, output_layers)
#
#     return model
