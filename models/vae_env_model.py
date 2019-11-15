import tensorflow as tf
import numpy as np


def make_inference_net(latent_dim):
    """
    Creates an inference network.

    :param int latent_dim: The number of latent dimensions
    :return: tf.model object
    :raises ValueError: if the input shape is wrong
    :raises AssertionError: if latent_dim is not integer
    """
    assert type(latent_dim) == int

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
            tf.keras.layers.Dense(2 * latent_dim)
        ], name="encoder"
    )
    return model


def make_generative_net(latent_dim):
    """
    Creates a generative network.

    :param int latent_dim: The number of latent dimensions
    :return: tf.model object
    :raises ValueError: if the input shape is wrong
    :raises AssertionError: if latent_dim is not integer
    """
    assert type(latent_dim) == int

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
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
    """
    Creates a dynamics network.

    :param int latent_dim: The number of latent dimensions
    :return: tf.model object
    :raises ValueError: if the input shape is wrong
    :raises AssertionError: if latent_dim is not integer
    """
    assert type(latent_dim) == int

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
    """
    Variational Auto Encoder
    """
    def __init__(self, latent_dim):
        """
        Class constructor

        :param int latent_dim: The number of latent dimensions
        :raises AssertionError: if latent_dim is not integer
        """
        assert type(latent_dim) == int

        super().__init__()

        self._latent_dim = latent_dim

        self._inference_net = make_inference_net(self._latent_dim)
        self._generative_net = make_generative_net(self._latent_dim)
        self._lat_env_net = make_latent_env_net(self._latent_dim)

    def call(self, input):
        """
        Standard call method. Encodes the image into the latent space.

        :param list input: input image
        :return: the encoded image
        """
        return self.encode(input)

    @property
    def inference_net(self):
        return self._inference_net

    @property
    def generative_net(self):
        return self._generative_net

    @property
    def lat_env_net(self):
        return self._lat_env_net

    @property
    def latent_dim(self):
        return self._latent_dim

    @tf.function
    def infer(self, x):
        """
        Infers means and std's from an image
        :param list x: input image, [B, H, W, C]
        :return: means and log variances
        """
        mean, logvar = tf.split(self._inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        """
        Samples a new latent state from a normal distribution.

        :param list[float] mean: means
        :param list[float] logvar: log variances
        :return:
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def encode(self, x):
        """
        Encode input into a latent space.

        :param list x: input image, [B, H, W, C]
        :return: state in latent space
        """
        mean, logvar = self.infer(x)
        return self.reparameterize(mean, logvar)

    @tf.function
    def decode(self, z, apply_sigmoid=True):
        """
        Decode from latent space.

        :param list z: state in latent space
        :param bool apply_sigmoid: apply an activation function
        :return: decoded image
        """
        logits = self._generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    @tf.function
    def env_step(self, z, a):
        """
        Predict next state.

        :param list z: current latent state
        :param list a: action
        :return: next latent state
        """
        # a = a[tf.newaxis, ...]
        z = tf.concat([z, a], axis=1)
        z_pred = self._lat_env_net(z)
        return z_pred

    @tf.function
    def forward(self, x, a):
        """
        Predict next state from an image
        :param list x: current state, image
        :param list a: action
        :return: next state, image
        """
        mean, logvar = self.infer(x)
        z = self.reparameterize(mean, logvar)
        z_pred = self.env_step(z, a)
        y_pred = self.decode(z_pred, apply_sigmoid=True)
        return y_pred

    def save_weights(self, nets, file_path, number):
        """
        Save weights.

        :param list[str] nets: the networks to save
        :param str file_path: file path
        :param int number: check point number
        :raises ValueError: if nets are wrong
        """
        for ch in nets:
            latest = file_path + "/" + ch + "/cp-{}.ckpt".format(number)
            if ch == 'encoder' or ch == 'en':
                self._inference_net.save_weights(latest)
            elif ch == 'decoder' or ch == 'de':
                self._generative_net.save_weights(latest)
            elif ch == 'lat_env' or ch == 'le':
                self._lat_env_net.save_weights(latest)
            else:
                raise ValueError

    def load_weights(self, nets, file_path):
        """
        Load weights.

        :param list[str] nets: the networks to save
        :param str file_path: file path
        :raises ValueError: if nets are wrong
        """
        for ch in nets:
            latest = tf.train.latest_checkpoint(file_path + "/" + ch)
            if ch == 'encoder' or ch == 'en':
                self._inference_net.load_weights(latest)
            elif ch == 'decoder' or ch == 'de':
                self._generative_net.load_weights(latest)
            elif ch == 'lat_env' or ch == 'le':
                self._lat_env_net.load_weights(latest)
            else:
                raise ValueError

    @tf.function
    def roll_out_plan(self, x0, actions):
        """
        Roll out the plan in latent space.

        :param x0: initial state, image
        :param list actions: list of actions
        :return: predicted sate and all intermediate sates
        """
        all_preds = tf.TensorArray(tf.float32, actions.shape[0])

        mean, logvar = self.infer(x0[np.newaxis, ...])
        x_pred = self.reparameterize(mean, logvar)

        for i in tf.range(actions.shape[0]):
            act = actions[i, ...]
            x_pred = self.env_step(x_pred, act[tf.newaxis, ...])
            all_preds = all_preds.write(i, self.decode(x_pred))

        return x_pred, all_preds.stack()


def plan(model, x0, xg, horizon, lr, epochs):
    """
    A gradient planner. Based on UPN paper

    :param model: tf.model object
    :param list x0: initial state, image
    :param list xg: goal state, image
    :param int horizon: planning horizon
    :param float lr: learning rate
    :param int epochs: the number of epochs
    :return: list of actions and all intermediate states
    """
    actions = tf.convert_to_tensor(np.random.randn(horizon, 4).astype(np.float32))

    for i in range(epochs):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(actions)

            x_pred, all_x = model.roll_out_plan(x0, actions)
            zg = model.encode(xg[np.newaxis, ...])

            loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(zg, x_pred), axis=-1))

        gradients = tape.gradient(loss, actions)

        actions -= lr * gradients

        actions = tf.clip_by_value(actions, -1., 1.)

        if i % 200 == 0:
            print("Epochs passed: ", i)

    return actions, all_x
