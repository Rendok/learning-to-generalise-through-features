import tensorflow as tf

from models.critic_network import CriticNetwork
from tf_agents.environments import tf_py_environment

from tensorflow.python.framework import test_util  # TF internal
from tf_agents.networks import value_network
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv
from models.vae_env_model import VAE


train_py_env = KukaCamMultiBlocksEnv(renders=False,
                                     numObjects=4,
                                     isTest=4,  # 1 and 4
                                     operation='move_pick',
                                     )

env = tf_py_environment.TFPyEnvironment(train_py_env)


class CriticNetworkTest(tf.test.TestCase):

    @test_util.run_in_graph_and_eager_modes()
    def test_build(self):
        batch_size = 10
        obs_spec = env.observation_spec()
        action_spec = env.action_spec()
        critic_net = CriticNetwork(obs_spec,
                                   encoding_network=VAE(256),
                                   fc_layer_params=(256, 256))

        critic_net_compare = value_network.ValueNetwork(
            obs_spec, fc_layer_params=(256, 256))

        obs = tf.random.uniform([1, batch_size] + obs_spec.shape.as_list())
        acts = tf.random.uniform([1, batch_size] + action_spec.shape.as_list())
        value, _ = critic_net(obs)
        print(value)
        value_comp, _ = critic_net_compare(obs)
        print(value_comp)
        self.assertEqual([1, batch_size], value.shape)
        # self.assertEqual(58, len(actor_net.trainable_variables))


if __name__ == '__main__':
    tf.test.main()
