import pybullet
import pybullet_envs
import pybullet_data
import numpy as np

import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents import ddpg
from ray.rllib.agents import dqn
from ray.rllib.agents import ppo


def env_creator_kuka_gym(renders=True):
    import gym
    import gym_kuka_multi_blocks
    return gym.make("KukaMultiBlocks-v0")

#-----------------------------------
def env_creator_kuka_bl(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=2,
                               isDiscrete=False,
                               isTest=0,
                               maxSteps=20, # 300,
                               actionRepeat=80,
                               blockRandom=0.8,
                               operation="pick")
    return env
#------------------------------------


def init_ppo():
    from ray.rllib.models import ModelCatalog

    register_env("my_env", env_creator_kuka_bl)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 3

    #ray.init()

    env = ModelCatalog.get_preprocessor_as_wrapper(env_creator_kuka_bl(renders=True))

    agent = ppo.PPOAgent(config=config, env="my_env")
    agent.restore("/Users/dgrebenyuk/ray_results/pick/PPO_KukaMultiBlocks-v0_0_2019-03-20_12-54-40b3l4y_xo/checkpoint_340/checkpoint-340")

    #agent.restore("/Users/dgrebenyuk/ray_results/place/PPO_KukaMultiBlocks-v0_0_2019-03-13_20-40-439sc4vld7/checkpoint_120/checkpoint-120")

    return agent, env


def init_ddpg():
    from ray.rllib.models import ModelCatalog

    register_env("my_env", env_creator_kuka_bl)

    config = ddpg.apex.APEX_DDPG_DEFAULT_CONFIG.copy()
    config["num_workers"] = 3

    ray.init()
    env = ModelCatalog.get_preprocessor_as_wrapper(env_creator_kuka_bl(renders=True))
    # env = env_creator_kuka_bl(renders=True)

    agent = ddpg.apex.ApexDDPGAgent(config=config, env="my_env")
    agent.restore("/Users/dgrebenyuk/ray_results/my_experiment/APEX_DDPG_KukaMultiBlocks-v0_0_2018-11-11_09-19-105785pfg0/checkpoint_55/checkpoint-55")
    return agent, env


def test_kuka(run, iter = 1):

    if run == "PPO":
        agent, env = init_ppo()
    elif run == "DDPG":
        agent, env = init_ddpg()
    else:
        env = env_creator_kuka_bl(renders=True)

    total_reward = 0.0
    for _ in range(iter):
        reward = 0.0
        obs = env.reset()
        done = False
        while not done:
            action = agent.compute_action(obs)
            obs, rew, done, info = env.step(action)
            #obs, rew, done, info = env.step([0, 0, -1, 0])
            print("__________REWARD____________", rew, info)
            reward += rew
        total_reward += reward
    return total_reward / iter


ray.init()
#print("Kuka's mean reward", test_kuka("PPO"))
test_kuka("PPO")
# [ 0.00028128  0.02405984 -0.19820549]