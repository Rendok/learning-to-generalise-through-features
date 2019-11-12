import tensorflow as tf
from tf_agents.networks import value_network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils
from tf_agents.utils import common


class CriticNetwork(value_network.ValueNetwork):

    def __init__(self,
                 input_tensor_spec,
                 encoding_network,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='ValueNetwork'):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype,
            name=name)

        self._encoder = encoding_network

    def call(self, observation, step_type=None, network_state=()):

        assert observation.dtype == tf.float32

        # print(self._input_tensor_spec)
        outer_rank = nest_utils.get_outer_rank(observation, self._input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        output = tf.nest.map_structure(batch_squash.flatten, observation)

        # output = tf.cast(output, tf.float32)
        output = self._encoder.encode(output)

        value = self._postprocessing_layers(output)

        value = tf.nest.map_structure(batch_squash.unflatten, value)

        return tf.squeeze(value, -1), network_state


class AugmentedCriticNetwork(value_network.ValueNetwork):

    def __init__(self,
                 input_tensor_spec,
                 encoding_network,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='ValueNetwork'):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype,
            name=name)

        self._encoder = encoding_network

    def call(self, observation, step_type=None, network_state=()):

        # print(self._input_tensor_spec)
        outer_rank = nest_utils.get_outer_rank(observation, self._input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observation = tf.nest.map_structure(batch_squash.flatten, observation)
        # print(observation.shape)

        image = observation[..., :6]
        global_coordinates = observation[..., :7, 0, 6]
        output = tf.cast(image, tf.float32)
        output = self._encoder.encode(output)

        output = tf.concat([output, global_coordinates], axis=1)
        # print(output.shape)

        value = self._postprocessing_layers(output)

        value = tf.nest.map_structure(batch_squash.unflatten, value)
        # print(value.shape)

        return tf.squeeze(value, -1), network_state
