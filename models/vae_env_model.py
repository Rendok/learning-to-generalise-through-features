import tensorflow as tf
import numpy as np


def make_inference_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(128, 128, 6)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2 * latent_dim)
        ], name="encoder"
    )
    return model


def make_generative_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(8192, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Reshape(target_shape=(4, 4, 512)),
            tf.keras.layers.Conv2DTranspose(
                filters=512, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=6, kernel_size=4, strides=(2, 2), padding='SAME',
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
        ], name="decoder"
    )
    return model


def make_latent_env_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim + 4)),
            tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dense(latent_dim, ),
        ], name="dynamics"
    )
    return model


class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self._latent_dim = latent_dim

        self.inference_net = make_inference_net(self._latent_dim)
        self.generative_net = make_generative_net(self._latent_dim)
        self.lat_env_net = make_latent_env_net(self._latent_dim)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    @tf.function
    def env_step(self, z, a):
        # a = a[tf.newaxis, ...]
        z = tf.concat([z, a], axis=1)
        z_pred = self.lat_env_net(z)
        return z_pred

    @tf.function
    def forward(self, x, a):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        z_pred = self.env_step(z, a)
        y_pred = self.decode(z_pred, apply_sigmoid=True)
        return y_pred

    def save_weights(self, nets, file_path, number):
        for ch in nets:
            latest = file_path + "/" + ch + "/cp-{}.ckpt".format(number)
            if ch == 'encoder' or ch == 'en':
                self.inference_net.save_weights(latest)
            elif ch == 'decoder' or ch == 'de':
                self.generative_net.save_weights(latest)
            elif ch == 'lat_env' or ch == 'le':
                self.lat_env_net.save_weights(latest)
            else:
                raise ValueError

    def load_weights(self, nets, file_path):
        for ch in nets:
            latest = tf.train.latest_checkpoint(file_path + "/" + ch)
            if ch == 'encoder' or ch == 'en':
                self.inference_net.load_weights(latest)
            elif ch == 'decoder' or ch == 'de':
                self.generative_net.load_weights(latest)
            elif ch == 'lat_env' or ch == 'le':
                self.lat_env_net.load_weights(latest)
            else:
                raise ValueError
