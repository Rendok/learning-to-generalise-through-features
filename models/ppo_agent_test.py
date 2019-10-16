from absl.testing import parameterized
import tensorflow as tf
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv

from models.vae_env_model import VAE
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork

from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import tf_py_environment
from tf_agents.utils import test_utils


class PPOAgentTest(parameterized.TestCase, test_utils.TestCase):

    def setUp(self):
        super().setUp()

        train_py_env = KukaCamMultiBlocksEnv(renders=False,
                                             numObjects=4,
                                             isTest=4,  # 1 and 4
                                             operation='move_pick',
                                             )

        env = tf_py_environment.TFPyEnvironment(train_py_env)

        tf.compat.v1.enable_resource_variables()
        self._obs_spec = env.observation_spec()
        self._action_spec = env.action_spec()
        self._time_step_spec = env.time_step_spec()
        # print(self._time_step_spec)

    def testCreateAgent(self):
        agent = ppo_agent.PPOAgent(
            self._time_step_spec,
            self._action_spec,
            tf.compat.v1.train.AdamOptimizer(),
            actor_net=ActorNetwork(self._obs_spec, self._action_spec, encoding_network=VAE(256)),
            value_net=CriticNetwork((self._obs_spec, self._action_spec), encoding_network=VAE(256)),
            check_numerics=True)
        agent.initialize()


if __name__ == '__main__':
    tf.test.main()
