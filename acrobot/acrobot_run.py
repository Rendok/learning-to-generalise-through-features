from acrobot.acrobot_env import AcrobotEnv
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments import tf_py_environment
from models.rl_planner import rl_planner
import matplotlib.pyplot as plt
from models.vae_env_model import VAE
import numpy as np

model = VAE(256, channels=3)
path_weights = '/Users/dgrebenyuk/Research/dataset/weights'
model.load_weights(['en', 'de'], path_weights)

environment = AcrobotEnv()
# checkpoint_directory = '/Users/dgrebenyuk/Research/dataset/weights'

# validate_py_environment(environment, episodes=10) #PASSED

eval_env = tf_py_environment.TFPyEnvironment(environment)

# tf_agent, _, _ = rl_planner(eval_env, None, "")

time_step = environment.reset()
episode_return = 0.0

# while not time_step.is_last():
for _ in range(5):
    # action_step = tf_agent.policy.action(time_step)
    # a = action_step.action
    # print(a)
    time_step = environment.step(0.9)
    obs = time_step.observation
    print(time_step.reward)
    episode_return += time_step.reward
    print(obs.shape)
    z = model.encode(obs[np.newaxis, ...])
    z = model.decode(z)
    plt.imshow(obs)
    plt.show()
    plt.imshow(z[0, ...])
    plt.show()

print(episode_return)
