import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow
import xarray as xr
from dask.diagnostics import ProgressBar
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd


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


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


class WGAN_Cascaded_Residual_IP(keras.Model):
    """
    A residual GAN to downscale precipitatoin, this GAN incorparates an Intensity Constraint
    """

    def __init__(self, discriminator, generator, latent_dim,
                 discriminator_extra_steps=3, gp_weight=10.0, ad_loss_factor=1e-3,
                 latent_loss=5e-2, orog=None, he=None,
                 vegt=None, unet=None, train_unet=True, intensity_weight = 1):
        super(WGAN_Cascaded_Residual_IP, self).__init__()

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
        self.train_unet = train_unet
        self.intensity_weight = intensity_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn,
                g_loss_fn, u_loss_fn, u_optimizer):
        super(WGAN_Cascaded_Residual_IP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.u_loss_fn = u_loss_fn
        self.u_optimizer = u_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images, average, orog_vector, he_vector, vegt_vector,
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
            pred = self.discriminator([interpolated, average, orog_vector, he_vector, vegt_vector, unet_preds],
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
        if isinstance(real_images, tuple):
            real_images, average = real_images[0]
            if real_images.shape[-1] > 100:
                # only expand the dimns if required to
                real_images = tf.expand_dims(real_images, axis=-1)
            # print(real_images.shape,"rimage")

            # here the average represents the conditional input

        batch_size = tf.shape(real_images)[0]
        orog_vector = self.expand_conditional_inputs(self.orog, batch_size)
        he_vector = self.expand_conditional_inputs(self.he, batch_size)
        vegt_vector = self.expand_conditional_inputs(self.vegt, batch_size)
        # make sure the auxiliary inputs are the same shape as the training batch
        # if the U-Net is trained, apply gradients otherwise only use inference mode from the U-Net
        if self.train_unet:
            with tf.GradientTape() as tape:
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )

                init_prediction = self.unet([random_latent_vectors, average,
                                             orog_vector, he_vector, vegt_vector], training=True)
                mae_unet = self.u_loss_fn(real_images[:, :, :], init_prediction)
            u_gradient = tape.gradient(mae_unet, self.unet.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.u_optimizer.apply_gradients(zip(u_gradient, self.unet.trainable_variables))
        else:
            with tf.GradientTape() as tape:
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size,) + self.latent_dim[0]
                )

                init_prediction = self.unet([random_latent_vectors, average,
                                             orog_vector, he_vector, vegt_vector], training=True)
                mae_unet = self.u_loss_fn(real_images[:, :, :], init_prediction)
        # loop through the discriminator steps
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size,) + self.latent_dim[0]
            )

            with tf.GradientTape() as tape:

                init_prediction_unet = self.unet([random_latent_vectors, average,
                                                  orog_vector, he_vector, vegt_vector], training=True)
                # compute ground truth residuals
                residual_gt = (real_images - init_prediction_unet)
                init_prediction = init_prediction_unet
                # crete fake residuals (these are residual by default)
                fake_images = self.generator([random_latent_vectors, average,
                                              orog_vector, he_vector, vegt_vector, init_prediction], training=True)

                fake_logits = self.discriminator(
                    [fake_images, average, orog_vector, he_vector, vegt_vector, init_prediction], training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(
                    [residual_gt, average, orog_vector, he_vector, vegt_vector, init_prediction], training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, residual_gt, fake_images, average, orog_vector, he_vector,
                                           vegt_vector, init_prediction)

                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight  # + #50 * tf.keras.losses.mean_squared_error(average, fake_image_average)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator
        random_latent_vectors = tf.random.normal(
            shape=(batch_size,) + self.latent_dim[0]
        )
        # Generator steps
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            init_prediction_unet = self.unet([random_latent_vectors, average,
                                              orog_vector, he_vector, vegt_vector], training=True)

            init_prediction = init_prediction_unet  # (init_prediction_unet - min_value)/(max_value - min_value)
            # compute ground truth residuals
            residual_gt = (real_images - init_prediction_unet)
            generated_images_v1 = self.generator(
                [random_latent_vectors, average, orog_vector, he_vector, vegt_vector, init_prediction], training=True)
            generated_images = generated_images_v1  # tf.math.exp(generated_images_v1[:,:,:,0] +
            # residual predictions from the GAN

            gen_img_logits = self.discriminator(
                [generated_images, average, orog_vector, he_vector, vegt_vector, init_prediction], training=True)
            # compute the content loss or the MSE, this is the errors in the residuals
            mae = tf.keras.losses.mean_squared_error(residual_gt, generated_images_v1)
            # compute the "true" error.
            gan_mae = tf.keras.losses.mean_squared_error(real_images, generated_images_v1 + init_prediction_unet)

            # compute the intensity on the batch across each individual timestep (not the 0th dimension)
            gamma_loss_func = mae
            maximum_intensity = tf.math.reduce_max(
                real_images, axis=[-1, -2, -3])
            maximum_intensity_predicted = tf.math.reduce_max(generated_images_v1 + init_prediction_unet,
                                                             axis=[-1, -2, -3])

            average_intensity = tf.math.reduce_mean(
                real_images, axis=[-1, -4])
            average_intensity_predicted = tf.math.reduce_mean(generated_images_v1 + init_prediction_unet,
                                                              axis=[-1, -4])
            adv_loss = self.ad_loss_factor * self.g_loss_fn(gen_img_logits)
            # Calculate the generator loss
            g_loss = adv_loss + gamma_loss_func + self.intensity_weight *(tf.reduce_mean(
                tf.abs(maximum_intensity - maximum_intensity_predicted) ** 2) + tf.reduce_mean(
                tf.abs(average_intensity - average_intensity_predicted)) ** 2)  ## + self.latent_loss * latent_loss

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss, "residual_loss": gamma_loss_func, "adv_loss": adv_loss,
                "unet_loss": mae_unet, "gan_mae": gan_mae}
