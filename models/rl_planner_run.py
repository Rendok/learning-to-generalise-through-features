from tf_agents.environments import tf_py_environment
from tf_agents.environments.utils import validate_py_environment
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv
from models.rl_planner import rl_planner
from models.vae_env_model import VAE
import matplotlib.pyplot as plt

num_latent_dims = 256
checkpoint_directory = '/Users/dgrebenyuk/Research/dataset/weights'
# checkpoint_directory = '/Users/dgrebenyuk/Research/Backup/rl_vae_256_working_pick/weights'

encoding_net = VAE(num_latent_dims)
encoding_net.load_weights(['en', 'de'], checkpoint_directory)

environment = KukaCamMultiBlocksEnv(renders=False,
                                encoding_net=encoding_net,
                                numObjects=4,
                                isTest=4,  # 1 and 4
                                operation='move_pick')

# validate_py_environment(environment, episodes=10)

eval_env = tf_py_environment.TFPyEnvironment(environment)
tf_agent, _, _ = rl_planner(eval_env, encoding_net, checkpoint_directory + "/rl")

time_step = environment.reset()
episode_return = 0.0

environment.reset()
plt.imshow(environment.goal_img[..., :3])
plt.show()
# i = 0

while not time_step.is_last():
    action_step = tf_agent.policy.action(time_step)
    a = action_step.action
    print(a)
    time_step = environment.step(a)
    # z = time_step.observation / 255.
    # z = encoding_net.decode(encoding_net.encode(z[tf.newaxis, ...]))
    # plt.imshow(z[0, ..., :3])
    # plt.show()
    plt.imshow(time_step.observation[..., 3:6])
    plt.show()
    print(time_step.reward)
    episode_return += time_step.reward
    # i += 1

print(episode_return)
