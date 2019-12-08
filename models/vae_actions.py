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

    def __init__(self,
                 latent_dim,
                 real_dim,
                 action_half_range):
        """
        Class constructor

        :param int latent_dim: The number of latent dimensions
        :raises AssertionError: if latent_dim is not integer
        """

        assert type(latent_dim) == int

        super().__init__(latent_dim)

        self._latent_dim = latent_dim
        self._real_dim = real_dim
        self._action_half_range = action_half_range

        self._inference_net = make_inference_net(self._latent_dim, self._real_dim)
        self._generative_net = make_generative_net(self._latent_dim, self._real_dim)

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
    act = ActionsVAE(latent_dim=256, real_dim=2, action_half_range=1)
    # path_weights = '/Users/dgrebenyuk/Research/dataset/weights/act'
    # act.load_weights(['en', 'de'], path_weights)

    # data = np.random.uniform(-2, 2, [5, 4])
    data = np.random.randn(5, 2)

    z = act.encode(data)
    z, _ = tf.linalg.normalize(z, axis=1)

    tf_data = tf.convert_to_tensor(data, dtype=tf.float32)

    tf_data, _ = tf.linalg.normalize(tf_data, axis=1)
    tf_data_sqr = tf.matmul(tf_data, tf.transpose(tf_data))
    # tf_data_sqr = tf.linalg.band_part(tf_data_sqr, 0, -1)

    print(tf_data_sqr)

    z_sqr = tf.matmul(z, tf.transpose(z))
    # z_sqr = tf.linalg.band_part(z_sqr, 0, -1)
    print(z_sqr)

    diff = tf.math.abs(z_sqr - tf_data_sqr) / tf.math.abs(tf_data_sqr)
    print(diff)
    distance = tf.reduce_sum(diff)
    print(distance)
    # a_h = act.decode(z)
    # print(a_h)
