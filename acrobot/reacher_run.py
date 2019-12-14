from acrobot.acrobot_env import AcrobotEnv
from acrobot.reacher_env import ReacherBulletEnv
from acrobot.acrobot_wrapper import AcrobotPixel
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments import tf_py_environment, suite_gym, suite_pybullet
from models.rl_planner import rl_planner
import matplotlib.pyplot as plt
from models.vae_env_model import VAE
import numpy as np

import gym
import pybulletgym


def print_img(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.show()


model = VAE(256, channels=3)
path_weights = '/Users/dgrebenyuk/Research/dataset/weights'
model.load_weights(['en', 'de'], path_weights)

environment = ReacherBulletEnv(encoding_net=model, same_init_state=False, render=False)

# environment = CartPole_Pixel(gym.make('CartPole-v0'))
# environment = suite_pybullet.load('ReacherPyBulletEnv-v0')
# environment.render(mode='human')
# print(environment.action_spec(), environment.observation_spec())

# environment = AcrobotPixel(environment)

# checkpoint_directory = '/Users/dgrebenyuk/Research/dataset/weights'

# validate_py_environment(environment, episodes=10)

eval_env = tf_py_environment.TFPyEnvironment(environment)
print(eval_env.action_spec(), eval_env.observation_spec())

# tf_agent, _, _ = rl_planner(eval_env, None, "")

time_step = environment.reset()

# print_img(environment.goal_img, "Goal State")
print_img(environment.get_observation(), "Initial State")

episode_return = 0.0

# while not time_step.is_last():
for _ in range(5):
    # action_step = tf_agent.policy.action(time_step)
    # a = action_step.action
    # print(a)
    time_step = environment.step([-1, 1])
    # img = environment.render('rgb_array')
    # environment.render(mode='human')
    obs = time_step.observation
    print(time_step.reward)
    episode_return += time_step.reward
    # print(obs.shape)
    z = model.encode(obs[np.newaxis, ...])
    z = model.decode(z)

    print_img(obs, "Intermediate State")
    print_img(z[0, ...], "Reconstructed State")

print(episode_return)
