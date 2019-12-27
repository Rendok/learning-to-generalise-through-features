import pybullet as p
import pybullet_envs
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import ray
import pandas as pd

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from acrobot.reacher_env import ReacherBulletEnv
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv


def kuka_env_creator(env_config):
    return KukaCamMultiBlocksEnv(renders=False,
                                 numObjects=4,
                                 isTest=4,  # 1 and 4
                                 obs_as_vector=True,
                                 delta=1.0,
                                 mode="rand_init_state",
                                 operation='move_pick')


def reacher_env_creator(env_config):
    return ReacherBulletEnv(max_time_step=40,
                            render=False,
                            obs_as_vector=True,
                            train_env="gym",
                            delta=(0.1, 1.0),
                            mode="delta")


def init_ppo(env):
    if env not in ['reacher', 'kuka']:
        raise ValueError

    if env == 'reacher':
        register_env("Reacher", reacher_env_creator)
        env_name = "Reacher"
    elif env == 'kuka':
        register_env("Kuka", kuka_env_creator)
        env_name = "Kuka"

    agent = ppo.PPOTrainer(
        env=env_name,
        config={
            'eager': True,
            "num_workers": 0,
            'log_level': 'DEBUG',
            'horizon': 40,
        })

    if env == 'reacher':
        agent.restore(
            "/Users/dgrebenyuk/Research/Backup/reacher_256_lat_dist_act_norm_data_Rand_init_st/rllib_rl/PPO_Reacher_84266e8c_2019-12-18_02-31-37c4a7htjc/checkpoint_280/checkpoint-280")
        # agent.restore("/Users/dgrebenyuk/Research/Backup/reacher_256_lat_dist_classic_vae_data_Rand_init_st/rllib_rl/PPO_Reacher_5dc3d5b6_2019-12-18_03-20-39a53gl9hs/checkpoint_40/checkpoint-40")
    elif env == 'kuka':
        pass

    return agent


def test(agent, env, iterations=1, render=True, scatter=False, kuka_stats=False, hist=False, reacher_stats=False):
    if kuka_stats:
        success = []
        steps = []
        rwds = []
        dists = []
        s_dists = []

    if reacher_stats:
        reacher_init_dist = []
        reacher_final_dist = []

    for j in range(iterations):
        reward = 0.0
        obs = env.reset()

        if render:
            plt.imshow(env.goal_img[..., :3])
            plt.title("Goal State")
            plt.show()

        action = np.array([0, 0, 0, 0])
        rew = 0
        done = False
        i = 0
        while not done:
            action = agent.compute_action(obs, prev_action=action, prev_reward=rew)
            obs, rew, done, info = env.step(action)
            img = env.get_observation(as_vector=False)

            if render and i % 3 == 0:
                # z = env.encoding_net.decode(env.encoding_net.encode(img[np.newaxis, ...]))
                # plt.imshow(z[0, ..., :3])
                plt.imshow(img[..., :3])
                plt.title("Policy State")
                plt.show()

            # obs, rew, done, info = env.step([0, 0, -1, 0])
            # print("REWARD:", rew, info)
            reward += rew
            i += 1

        if reacher_stats:
            reacher_init_dist.append(info['init_dist'])
            reacher_final_dist.append(info['final_dist'])

            if j % 50 == 0 and j > 0:
                print('iteration: ', j)

        if kuka_stats:
            steps.append(i)
            rwds.append(reward)
            dists.append(info['disturbance'])

            if reward >= 30:
                success.append(1)
                s_dists.append(info['disturbance'])
            else:
                success.append(0)

            if j % 100 == 0 and j > 0:
                print('iteration: ', j)

        # print("Total: ", reward)

    if scatter:
        data = pd.DataFrame(dict(steps=steps, reward=rwds, success=success))
        print_scatter(data)

    if reacher_stats:
        a = st.t.interval(0.95, len(reacher_init_dist) - 1, loc=np.mean(reacher_init_dist),
                          scale=st.sem(reacher_init_dist))
        b = st.t.interval(0.95, len(reacher_final_dist) - 1, loc=np.mean(reacher_final_dist),
                          scale=st.sem(reacher_final_dist))

        print(np.mean(reacher_init_dist), '+-', a)
        print(np.mean(reacher_final_dist), '+-', b)

    if kuka_stats:
        a = st.t.interval(0.95, len(success) - 1, loc=np.mean(success), scale=st.sem(success))
        b = st.t.interval(0.95, len(steps) - 1, loc=np.mean(steps), scale=st.sem(steps))
        c = st.t.interval(0.95, len(dists) - 1, loc=np.mean(dists), scale=st.sem(dists))
        d = st.t.interval(0.95, len(s_dists) - 1, loc=np.mean(s_dists), scale=st.sem(s_dists))
        print('Success rate:', sum(success) / iterations, '+-', sum(success) / iterations - a[0],
              '\nAverage time: ', sum(steps) / len(steps), '+-', sum(steps) / len(steps) - b[0],
              '\nAverage disturbance: ', sum(dists) / len(dists), '+-', sum(dists) / len(dists) - c[0],
              '\nSuccess disturbance: ', sum(s_dists) / len(s_dists), '+-', sum(s_dists) / len(s_dists) - d[0],
              '\n{$', round(sum(success) / iterations, 4), '\pm', round(sum(success) / iterations - a[0], 4),
              '$} & {$', round(sum(steps) / len(steps), 4), '\pm', round(sum(steps) / len(steps) - b[0], 4),
              '$} & {$', round(sum(dists) / len(dists), 4), '\pm', round(sum(dists) / len(dists) - c[0], 4),
              '$} & {$', round(sum(s_dists) / len(s_dists), 4), '\pm', round(sum(s_dists) / len(s_dists) - d[0], 4),
              '$}')

    if hist:
        print_hist(s_dists)

    # return sum(success) / iterations, sum(steps) / len(steps)


def print_scatter(data):
    sns.lmplot('steps', 'reward', data=data, hue='success', fit_reg=False, palette=['r', 'g'], legend=False)
    plt.title('Reward vs Policy Length')
    plt.legend(title='Grasp Success', loc='lower left', labels=['False', 'True'])
    plt.xlabel('Policy Length (time steps)')
    plt.ylabel('Reward')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -70, 70))
    plt.show()


def print_hist(data):
    sns.distplot(data, kde=False)
    plt.xlabel('Distance')
    plt.ylabel('Simulations')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2))
    plt.show()


ray.init()
agent = init_ppo('kuka')
env = kuka_env_creator({})
# agent = init_ppo('reacher')
# env = reacher_env_creator({})

test(agent, env, iterations=1, render=True, scatter=False, kuka_stats=False, hist=False, reacher_stats=False)
# test(agent, env, iterations=2000, render=False, scatter=True, kuka_stats=True, hist=True)
