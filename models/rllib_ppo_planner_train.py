import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune.registry import register_env
from acrobot.reacher_env import ReacherBulletEnv
import numpy as np
import matplotlib.pyplot as plt

CLOUD = False
CHANNELS = 3
num_latent_dims = 256


class MyPreprocessorClass(Preprocessor):
    def __init__(self, obs_space, options=None):
        super().__init__(obs_space, options)


    def _init_shape(self, obs_space, options):
        """
        returns post-processed shape
        """

        return (256, )

    def transform(self, observation):
        """
        returns the preprocessed observation
        """

        encoding_net = ReacherBulletEnv.load_encoding_net()
        return encoding_net.encode(observation[np.newaxis, ...])


def env_creator(env_config):
    return ReacherBulletEnv(same_init_state=False,
                            max_time_step=40,
                            render=False,
                            train_env="gym")


register_env("Reacher", env_creator)

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)

ray.init()
trainer = ppo.PPOTrainer(env="Reacher", config={
    "model": {
        "custom_preprocessor": "my_prep",
        "custom_options": {
        },  # extra options to pass to your preprocessor
    },
    'eager': True,

})

# Can optionally call trainer.restore(path) to load a checkpoint.
# trainer.restore(path)

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(result)

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
