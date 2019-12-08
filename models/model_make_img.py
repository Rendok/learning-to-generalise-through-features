import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv
from models.autoencoder_env_model import AutoEncoderEnvironment
from models.vae_env_model import VAE, plan
from tf_agents.environments import utils

print(tf.__version__)

# model = AutoEncoderEnvironment()
model = VAE(256)
path_weights = '/Users/dgrebenyuk/Research/dataset/weights'
model.load_weights(['en', 'de', 'le'], path_weights)

env = KukaCamMultiBlocksEnv(renders=False,
                            # encoding_net=model,
                            numObjects=4,
                            isTest=1,  # 1 and 4
                            operation='move_pick')

# utils.validate_py_environment(env, episodes=10)

ideal_actions = np.array(
    [[0, 0, -0.7, 0], [-0.7, 0, -0.7, 0], [-.7, 0, 0, 0], [0, 0, -0.7, 0], [0, 0, -0.7, 0], [0, 0, -0.7, 0]])
horizon = 6

x0 = env.reset()[3] / 255.

goal = env.goal_img
plt.imshow(goal[..., :3])
plt.show()

for i in ideal_actions[:horizon]:
    _, rew, _, obs = env.step([0, 0, -1, 0.5])
    print(rew)
xg = obs / 255.

plt.figure(figsize=(20, 20))
for i in range(2):
    plt.subplot(6, horizon, horizon * i + 1)
    if i == 0:
        plt.title('Initial State')
    plt.axis('off')
    plt.imshow(x0[..., 3 * i:3 + 3 * i])

    plt.subplot(6, horizon, horizon * i + horizon)
    if i == 0:
        plt.title('Goal State')
    plt.axis('off')
    plt.imshow(xg[..., 3 * i:3 + 3 * i])

actions, all_x = plan(model, x0, xg, horizon=horizon, lr=1e-4, epochs=100)

env.reset()

for i, a in enumerate(actions):
    _, _, _, obs = env.step(a)

    plt.subplot(6, horizon, 2 * horizon + 1 + i)
    plt.axis('off')
    plt.imshow(obs[..., :3])

    plt.subplot(6, horizon, 4 * horizon + 1 + i)
    plt.axis('off')
    plt.imshow(obs[..., 3:6])

for i, x in enumerate(all_x):
    plt.subplot(6, horizon, 3 * horizon + 1 + i)
    plt.axis('off')
    plt.imshow(x[0, ..., :3])

    plt.subplot(6, horizon, 5 * horizon + 1 + i)
    plt.axis('off')
    plt.imshow(x[0, ..., 3:6])

plt.show()
