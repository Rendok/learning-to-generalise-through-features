import tensorflow_probability as tfp
import tensorflow as tf


def make_prior(latent_dim):
    return tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(latent_dim), scale=1),
                                         reinterpreted_batch_ndims=1)


def make_inference_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(128, 128, 6)),
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=2, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=2, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=2, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=2, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),
                                  activation=None),
            tfp.layers.MultivariateNormalTriL(latent_dim,
                                              activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                                                  make_prior(latent_dim),
                                                  weight=1.0)
                                              ),
        ], name="encoder"
    )
    return model


def make_generative_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),

            tf.keras.layers.Dense(8192, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            # tf.keras.layers.Reshape([1, 1, latent_dim]),
            tf.keras.layers.Reshape(target_shape=(4, 4, 512)),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2DTranspose(
                filters=512, kernel_size=2, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=512, kernel_size=2, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=2, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=2, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            # tf.keras.layers.Conv2DTranspose(
            #     filters=32, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu, padding='SAME',
            #     kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            # tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(
                filters=12, kernel_size=4, strides=(1, 1), activation=None, padding='SAME',
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
            tf.keras.layers.Flatten(),
            tfp.layers.IndependentLogistic((128, 128, 6), tfp.distributions.Logistic.sample),
        ], name="decoder"
    )
    return model


def make_latent_env_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim + 4)),
            tf.keras.layers.Dense(2 * latent_dim, activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dense(2 * latent_dim, activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dense(2 * latent_dim, activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dense(latent_dim, activation=None),
        ], name="dynamics"
    )
    return model


class VAE1(tf.keras.Model):
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
        z = self.inference_net(x)
        return z

    def decode(self, z):
        rv_x = self.generative_net(z)
        return rv_x

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