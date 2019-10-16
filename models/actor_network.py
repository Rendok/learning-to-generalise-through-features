import tensorflow as tf
import numpy as np

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils
from tf_agents.utils import common
from tf_agents.networks import normal_projection_network


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):

    std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        init_means_output_factor=init_means_output_factor,
        std_bias_initializer_value=std_bias_initializer_value,
        scale_distribution=False)


class ActorNetwork(network.DistributionNetwork):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 encoding_network,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 enable_last_layer_zero_initializer=False,
                 name='ActorNetwork'):

        projection_networks = tf.nest.map_structure(_normal_projection_net, action_spec)
        output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                            projection_networks)

        super().__init__(
            input_tensor_spec=observation_spec,
            state_spec=(),
            output_spec=output_spec,
            name=name
        )

        if len(tf.nest.flatten(observation_spec)) > 1:
            raise ValueError('Only a single observation is supported by this network')
        self._observation_spec = observation_spec

        # For simplicity we will only support a single action float output.
        self._action_spec = action_spec  # output tensor
        # print("Act spec", action_spec)
        flat_action_spec = tf.nest.flatten(action_spec)

        if len(flat_action_spec) > 1:
            raise ValueError('Only a single action is supported by this network')
        # print("Flat act spec", flat_action_spec)

        self._single_action_spec = flat_action_spec[0]
        # print("Sing act spec", self._single_action_spec)

        if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
            raise ValueError('Only float actions are supported by this network.')

        self._encoder = encoding_network
        self._projection_networks = projection_networks

        self._mlp_layers = utils.mlp_layers(
            conv_layer_params,
            fc_layer_params,
            dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='input_mlp')

        self._mlp_layers.append(
            tf.keras.layers.Dense(
                flat_action_spec[0].shape.num_elements(),
                activation=tf.keras.activations.tanh,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.003, maxval=0.003),
                name='action'))

    @property
    def output_tensor_spec(self):
        return self._action_spec

    def call(self, observations, step_type=(), network_state=()):
        del step_type
        # print(observations.shape)

        outer_rank = nest_utils.get_outer_rank(observations, self._observation_spec)
        # print('out rank', outer_rank)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        # print(observations.shape)

        output = tf.cast(observations, tf.float32) / 255.
        # print(observations.shape)


        output = self._encoder.encode(output)
        # print(output.shape)

        for layer in self._mlp_layers:
            output = layer(output)

        # print(output.shape)

        output = tf.nest.map_structure(batch_squash.unflatten, output)
        # print(output.shape)

        # actions = common.scale_to_spec(output, self._single_action_spec)
        # output_actions = tf.nest.pack_sequence_as(self._action_spec,
        #                                           [actions])
        # print(output_actions.shape)

        # outer_rank = nest_utils.get_outer_rank(observations, self._observation_spec)
        output_actions = tf.nest.map_structure(
            lambda proj_net: proj_net(output, outer_rank), self._projection_networks)

        # print('last out', output_actions.batch_shape)

        return output_actions, network_state
