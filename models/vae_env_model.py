import tensorflow as tf
import numpy as np


def make_inference_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(128, 128, 6)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(512, activation='relu',
            #                       kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
            tf.keras.layers.Dense(2 * latent_dim)
        ], name="encoder"
    )
    return model


def make_generative_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            # tf.keras.layers.Dense(512, activation='relu',
            #                       kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dense(8192, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
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
            tf.keras.layers.Conv2DTranspose(
                filters=6, kernel_size=4, strides=(1, 1), activation=None, padding='SAME',
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
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
            tf.keras.layers.Dense(latent_dim),
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

    def call(self, input):
        return self.encode(input)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self._latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def infer(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def encode(self, x):
        mean, logvar = self.infer(x)
        return self.reparameterize(mean, logvar)

    def decode(self, z, apply_sigmoid=True):
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
        mean, logvar = self.infer(x)
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

    # @tf.function
    def roll_out_plan(self, x0, actions):
        all_preds = tf.TensorArray(tf.float32, actions.shape[0])

        mean, logvar = self.infer(x0[np.newaxis, ...])
        x_pred = self.reparameterize(mean, logvar)

        for i in tf.range(actions.shape[0]):
            act = actions[i, ...]
            x_pred = self.env_step(x_pred, act[tf.newaxis, ...])
            all_preds = all_preds.write(i, self.decode(x_pred))

        return x_pred, all_preds.stack()

    def plan(self, x0, xg, horizon, lr, epochs):
        actions = tf.convert_to_tensor(np.random.randn(horizon, 4).astype(np.float32))

        for i in range(epochs):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actions)
                x_pred, all_x = self.roll_out_plan(x0, actions)
                mean, logvar = self.infer(xg[np.newaxis, ...])
                zg = self.reparameterize(mean, logvar)
                loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(zg, x_pred), axis=-1))
                # loss = tf.reduce_sum(tf.losses.mean_squared_error(zg, x_pred))

            gradients = tape.gradient(loss, actions)

            # print(gradients)
            if i < 25:
                actions -= lr * gradients
            elif i < 50:
                actions -= lr * gradients
            else:
                actions -= lr * gradients

            actions = tf.clip_by_value(actions, -1., 1.)

            if i % 200 == 0:
                print(i)

        return actions, all_x