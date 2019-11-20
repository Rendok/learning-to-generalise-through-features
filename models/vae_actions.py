import tensorflow as tf
from models.vae_env_model import VAE
import numpy as np


def make_inference_net(latent_dims, real_dims):
    assert type(latent_dims) == int
    assert type(real_dims) == int

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(real_dims,)),
            tf.keras.layers.Dense(2 * latent_dims, activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dense(2 * latent_dims, kernel_initializer=tf.keras.initializers.he_normal(seed=None))
        ], name="encoder"
    )
    return model


def make_generative_net(latent_dims, real_dims):
    assert type(latent_dims) == int
    assert type(real_dims) == int

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dims,)),
            tf.keras.layers.Dense(2 * latent_dims, activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dense(real_dims, kernel_initializer=tf.keras.initializers.he_normal(seed=None))
        ], name="decoder"
    )
    return model


class ActionsVAE(VAE):
    """
    Variational Auto Encoder: actions to latent space
    """

    def __init__(self, latent_dim):
        """
        Class constructor

        :param int latent_dim: The number of latent dimensions
        :raises AssertionError: if latent_dim is not integer
        """

        assert type(latent_dim) == int

        super().__init__(latent_dim)

        self._latent_dim = latent_dim
        self._action_half_range = 2

        self._inference_net = make_inference_net(self._latent_dim, 4)
        self._generative_net = make_generative_net(self._latent_dim, 4)

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
            probs = self._action_half_range * tf.tanh(logits)
            return probs

        return logits


if __name__ == "__main__":
    act = ActionsVAE(16)
    data = np.random.uniform(-2, 2, [5, 4])

    z = act.encode(data)
    z, _ = tf.linalg.normalize(z, axis=1)

    # z2 = act.encode(data)
    # z2, _ = tf.linalg.normalize(z2, axis=1)

    # a = np.array([0, 0.5, 0])
    tf_data = tf.convert_to_tensor(data, dtype=tf.float32)
    tf_data, _ = tf.linalg.normalize(tf_data, axis=1)
    tf_data_sqr = tf.matmul(tf_data, tf.transpose(tf_data))
    # b = np.array([0, 1, 0])
    print(tf_data_sqr)

    z_sqr = tf.matmul(z, tf.transpose(z))
    print(z_sqr)
    distance = tf.reduce_sum(tf.math.squared_difference(z_sqr, tf_data_sqr))
    print(distance)
    # a_h = act.decode(z)
    # print(a_h)
