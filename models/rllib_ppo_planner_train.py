import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune.registry import register_env
from acrobot.reacher_env import ReacherBulletEnv
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gym.spaces import Box

# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn

# class MyPreprocessorClass(Preprocessor):
#     def __init__(self, obs_space, options=None):
#         super().__init__(obs_space, options)
#
#     def _init_shape(self, obs_space, options):
#         """
#         returns post-processed shape
#         """
#
#         return (256, )
#
#     def transform(self, observation):
#         """
#         returns the preprocessed observation
#         """
#
#         encoding_net = ReacherBulletEnv.load_encoding_net()
#         return encoding_net.encode(observation[np.newaxis, ...])


# class VAEfcNetwork(FullyConnectedNetwork):
#     """Custom VAE vision network implemented in ModelV2 API."""
#
#     def __init__(self, obs_space, action_space, num_outputs, model_config,
#                  name):
#
#         self._encoding_net = ReacherBulletEnv.load_encoding_net()
#
#         obs_space = Box(shape=(self._encoding_net.latent_dim,), dtype=np.float32, low=0, high=1)
#
#         super().__init__(
#             obs_space, action_space, num_outputs, model_config, name)
#
#         self._encoding_net.inference_net.trainable = False
#         self._encoding_net.generative_net.trainable = False
#         self._encoding_net.lat_env_net.trainable = False
#
#         self.register_variables(self._encoding_net.variables)
#
#     def forward(self, input_dict, state, seq_lens):
#         z = self._encoding_net.encode(input_dict["obs"])
#         model_out, self._value_out = self.base_model(z)
#         return model_out, state

def env_creator(env_config):
    return KukaCamMultiBlocksEnv(renders=False,
                                 numObjects=4,
                                 isTest=4,  # 1 and 4
                                 obs_as_vector=True,
                                 delta=1.0,
                                 mode="rand_init_state",
                                 operation='move_pick')

# def env_creator(env_config):
#     return ReacherBulletEnv(max_time_step=40,
#                             render=False,
#                             obs_as_vector=True,
#                             train_env="gym",
#                             delta=(0.1, 1.0),
#                             mode="delta")


if __name__ == "__main__":
    # register_env("Reacher", env_creator)
    register_env("Kuka", env_creator)

    # ModelCatalog.register_custom_model("my_model", VAEfcNetwork)
    # ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)

    ray.init()
    tune.run(
        "PPO",
        checkpoint_freq=20,
        checkpoint_at_end=True,
        reuse_actors=True,
        local_dir="/tmp/results",
        # local_dir="/Users/dgrebenyuk/Research/results",
        # restore="/tmp/results/PPO/PPO_Reacher_c4ae3efe_2019-12-18_01-50-28vayif13f/checkpoint_200/checkpoint-200",
        # stop={"episode_reward_mean": 50},
        config={
            "env": "Kuka",
            "num_gpus": 0,
            "num_workers": 35,
            # "num_envs_per_worker": 4,
            'horizon': 40,
            # "lr": ray.tune.grid_search([0.01, 0.001, 0.0001]),
            "eager": True,
            "sample_batch_size": 40,  # 50,
            "train_batch_size": 2500,  # 2500,
            # "model": {
            #         "custom_model": "my_model",
            # },
            # 'log_level': 'INFO',
        },
    )


    # trainer = ppo.PPOTrainer(env="Reacher", config={
    #     "model": {
    #         "custom_model": "my_model",
    #         # "custom_preprocessor": "my_prep",
    #         "custom_options": {
    #         },  # extra options to pass to your preprocessor
    #     },
    #     'eager': True,
    #     # 'log_level': 'DEBUG',
    #     'horizon': 40,
    #
    # })
    #
    # # Can optionally call trainer.restore(path) to load a checkpoint.
    # # trainer.restore(path)
    # print("TEST!!!!")
    # for i in range(1000):
    #    # Perform one iteration of training the policy with PPO
    #    result = trainer.train()
    #    print(i)
    #    print(result)
    #
    #    if i % 100 == 0:
    #        checkpoint = trainer.save()
    #        print("checkpoint saved at", checkpoint)
