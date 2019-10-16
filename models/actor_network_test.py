import tensorflow as tf

from models.actor_network import ActorNetwork
from tf_agents.environments import tf_py_environment

from tensorflow.python.framework import test_util  # TF internal
from tf_agents.networks import actor_distribution_network
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv
from models.vae_env_model import VAE


train_py_env = KukaCamMultiBlocksEnv(renders=False,
                                     numObjects=4,
                                     isTest=4,  # 1 and 4
                                     operation='move_pick',
                                     )

env = tf_py_environment.TFPyEnvironment(train_py_env)


class ActorNetworkTest(tf.test.TestCase):

    @test_util.run_in_graph_and_eager_modes()
    def test_build(self):
        batch_size = 5
        obs_spec = env.observation_spec()
        action_spec = env.action_spec()

        encoding_net = VAE(256)
        encoding_net.load_weights(['en', 'de'], '/Users/dgrebenyuk/Research/dataset/weights')

        actor_net = ActorNetwork(obs_spec, action_spec, encoding_network=encoding_net)

        actor_net_compare = actor_distribution_network.ActorDistributionNetwork(
            obs_spec,
            action_spec,
            fc_layer_params=(256, 256))

        obs = tf.random.uniform([1, batch_size] + obs_spec.shape.as_list())
        actions, _ = actor_net(obs)
        act_cp = actor_net_compare(obs, (), ())
        print(actions, act_cp)

        self.assertAllEqual([1, batch_size] + action_spec.shape.as_list(),
                            actions.batch_shape.as_list())
        # self.assertEqual(58, len(actor_net.trainable_variables))


if __name__ == '__main__':
    tf.test.main()
