import pybullet as p
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

def init_ddpg():
    from ray.rllib.models import ModelCatalog

    register_env("my_env", env_creator_kuka_bl)

    config = ddpg.apex.APEX_DDPG_DEFAULT_CONFIG.copy()
    config["num_workers"] = 3

    ray.init()
    env = ModelCatalog.get_preprocessor_as_wrapper(env_creator_kuka_bl(renders=True))
    #env = env_creator_kuka_bl(renders=True)

    agent = ddpg.apex.ApexDDPGAgent(config=config, env="my_env")
    agent.restore("/Users/dgrebenyuk/ray_results/my_experiment/APEX_DDPG_KukaMultiBlocks-v0_0_2018-11-11_09-19-105785pfg0/checkpoint_55/checkpoint-55")
    return agent, env

# ---- dump

#-----------------------------------

operation = 'move_pick'

def env_creator_kuka_bl(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_multi_blocks_gym_env as e
    env = e.KukaMultiBlocksEnv(renders=renders,
                               numObjects=2,
                               isDiscrete=False,
                               isTest=3,
                               maxSteps=40,
                               actionRepeat=80,
                               operation=operation)
    return env

def init_ppo(render):

    register_env("my_env", env_creator_kuka_bl)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 3

    env = env_creator_kuka_bl(renders=render)

    agent = ppo.PPOAgent(config=config, env="my_env")

    if operation == 'move_pick':
        pass
        # test == 0
        #agent.restore("/Users/dgrebenyuk/ray_results/move_pick/PPO_KukaMultiBlocks-v0_0_2019-03-27_11-13-30nbdyzah7/checkpoint_300/checkpoint-300")
        # test == 3 close blocks without obs and reward
        #agent.restore("/Users/dgrebenyuk/ray_results/move_pick/PPO_KukaMultiBlocks-v0_0_2019-05-07_01-53-28pb9qko3w/checkpoint_160/checkpoint-160")
        # test == 3 blocks in obs without reward
        #agent.restore("/Users/dgrebenyuk/ray_results/move_pick/PPO_KukaMultiBlocks-v0_0_2019-05-08_05-03-03rysgn407/checkpoint_300/checkpoint-300")
        # test == 3 blocks in obs with reward
        agent.restore("/Users/dgrebenyuk/ray_results/move_pick/PPO_KukaMultiBlocks-v0_0_2019-05-08_07-03-38kr507iig/checkpoint_480/checkpoint-480")
    elif operation == 'place':
        agent.restore("/Users/dgrebenyuk/ray_results/place/PPO_KukaMultiBlocks-v0_0_2019-04-03_09-59-16z2_syfpz/checkpoint_120/checkpoint-120")
    elif operation == 'move':
        agent.restore("/Users/dgrebenyuk/ray_results/move/PPO_KukaMultiBlocks-v0_0_2019-04-09_02-24-40kihke9e8/checkpoint_40/checkpoint-40")
    elif operation == 'pick':
        agent.restore("/Users/dgrebenyuk/ray_results/pick/PPO_KukaMultiBlocks-v0_0_2019-04-10_07-30-536dh0eu86/checkpoint_180/checkpoint-180")
    else:
        raise NotImplementedError

    return agent, env


def test_kuka(run="PPO", iterations=1, render=True):

    if run == "PPO":
        agent, env = init_ppo(render)
    elif run == "DDPG":
        agent, env = init_ddpg()
    else:
        env = env_creator_kuka_bl(renders=True)

    success = []
    steps = []
    rwds = []
    data = []
    for _ in range(iterations):
        reward = 0.0
        obs = env.reset()
        done = False
        i = 0
        while not done:
            action = agent.compute_action(obs)
            obs, rew, done, info = env.step(action)
            #obs, rew, done, info = env.step([0, 0, -1, 1])
            #print("__________REWARD____________", rew, info)
            reward += rew
            i += 1

        steps.append(i)
        rwds.append(reward)

        if reward >= 45:
            success.append(1)
            #data.append({'steps': i, 'reward': reward, 'success': 1})
        else:
            success.append(0)
            #data.append({'steps': i, 'reward': reward, 'success': 0})

    import pandas as pd
    data = pd.DataFrame(dict(steps=steps, reward=rwds, success=success))
    print_scatter(data)

    print("Success rate:", sum(success) / iterations, 'Average time', sum(steps) / len(steps))

    #return success, rwds, steps, data


def print_scatter(data):

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.lmplot('steps', 'reward', data=data, hue='success', fit_reg=False, palette=['r', 'g'], legend=False)
    plt.title('Reward vs Policy Length')
    plt.legend(title='Grasp Success', loc='upper right', labels=['False', 'True'])
    plt.xlabel('Policy Length (time steps)')
    plt.ylabel('Reward')
    plt.show()


ray.init()

test_kuka(iterations=100, render=False)
