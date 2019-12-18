import pybullet as p
import pybullet_envs
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import ray

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents import ppo

from models.rllib_ppo_planner_train import env_creator


def init_ppo(render):

    register_env("Reacher", env_creator)

    # config = ppo.DEFAULT_CONFIG.copy()
    # config["num_workers"] = 3

    agent = ppo.PPOTrainer(
        env="Reacher",
        config={
            'eager': True,
            "num_workers": 0,
            'log_level': 'DEBUG',
            'horizon': 40,
        })

    agent.restore("/Users/dgrebenyuk/Research/Backup/reacher_256_lat_dist_act_norm_data_Rand_init_st/rllib_rl/PPO_Reacher_84266e8c_2019-12-18_02-31-37c4a7htjc/checkpoint_280/checkpoint-280")

    return agent


def test(agent, env, iterations=1, render=True, scatter=False, stats=False, hist=False):

    success = []
    steps = []
    rwds = []
    dists = []
    s_dists = []
    for j in range(iterations):
        reward = 0.0
        obs = env.reset()

        plt.imshow(env.goal_img[..., :3])
        plt.title("Goal State")
        plt.show()

        action = [0, 0]
        rew = 0
        done = False
        i = 0
        while not done:
            action = agent.compute_action(obs, prev_action=action, prev_reward=rew)
            obs, rew, done, info = env.step(action)
            img = env.get_observation(as_vector=False)

            if i % 2 == 0:
                plt.imshow(img)
                plt.title("Policy State")
                plt.show()

            # obs, rew, done, info = env.step([0, 0, -1, 0])
            print("__________REWARD____________", rew, info)
            reward += rew
            i += 1

        if stats:
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

        print("Total: ", reward)

    if scatter:
        import pandas as pd
        data = pd.DataFrame(dict(steps=steps, reward=rwds, success=success))
        print_scatter(data)

    if stats:
        import numpy as np
        import scipy.stats as st

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

    return sum(success) / iterations, sum(steps) / len(steps)


def print_scatter(data):

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.lmplot('steps', 'reward', data=data, hue='success', fit_reg=False, palette=['r', 'g'], legend=False)
    plt.title('Reward vs Policy Length')
    plt.legend(title='Grasp Success', loc='lower left', labels=['False', 'True'])
    plt.xlabel('Policy Length (time steps)')
    plt.ylabel('Reward')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -70, 70))
    plt.show()


def print_hist(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.distplot(data, kde=False)
    plt.xlabel('Distance')
    plt.ylabel('Simulations')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2))
    plt.show()


ray.init()
agent = init_ppo(render=False)
env = env_creator({})

test(agent, env, iterations=1, render=False, scatter=False, stats=False, hist=False)
# test(agent, env, iterations=2000, render=False, scatter=True, stats=True, hist=True)
