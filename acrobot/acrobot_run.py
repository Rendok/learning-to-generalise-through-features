from acrobot.acrobot_env import AcrobotEnv
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments import tf_py_environment
from models.rl_planner import rl_planner
import matplotlib.pyplot as plt

environment = AcrobotEnv()
# checkpoint_directory = '/Users/dgrebenyuk/Research/dataset/weights'

# validate_py_environment(environment, episodes=10) #PASSED

eval_env = tf_py_environment.TFPyEnvironment(environment)

# tf_agent, _, _ = rl_planner(eval_env, None, "")

time_step = environment.reset()
episode_return = 0.0

# while not time_step.is_last():
for _ in range(2):
    # action_step = tf_agent.policy.action(time_step)
    # a = action_step.action
    # print(a)
    time_step = environment.step(0.9)
    z = time_step.observation
    print(time_step.reward)
    episode_return += time_step.reward
    # img = environment.render(mode='rgb_array')
    print(z.shape)
    plt.imshow(z)
    plt.show()

print(episode_return)
