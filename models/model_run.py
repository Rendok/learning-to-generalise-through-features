import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gym_kuka_multi_blocks.datasets.generate_data_set import env_creator_kuka_cam

print(tf.__version__)
env = env_creator_kuka_cam()

ideal_actions = np.array([[0, 0, -0.7, 0], [-0.7, 0, -0.7, 0], [-.7, 0, 0, 0], [0, 0, -0.7, 0], [0, 0, -0.7, 0], [0, 0, -0.7, 0]])
horizon = 6

x0 = env.reset() / 255.

for i in ideal_actions:
    obs, _, _, _ = env.step([0, 0, -.5, 0.5])
xg = obs / 255.

plt.figure(figsize=(20, 20))
for i in range(2):
    plt.subplot(6, horizon, horizon*i + 1)
    if i == 0:
        plt.title('Initial State')
    plt.axis('off')
    plt.imshow(x0[..., 3 * i:3 + 3 * i])
    # plt.imshow(x0[..., 6 + i])

    plt.subplot(6, horizon, horizon*i + horizon)
    if i == 0:
        plt.title('Goal State')
    plt.axis('off')
    plt.imshow(xg[..., 3*i:3 + 3*i])
#
# model = gr.AE()
# latest = tf.train.latest_checkpoint('/Users/dgrebenyuk/Research/dataset/weights')
# model.load_weights(latest)
# print('Latest checkpoint:', latest)
#
# actions, all_x = gr.plan(model, x0, xg, horizon=horizon, epochs=1000)
#
# env.reset()
#
# for i, a in enumerate(actions):
#     obs, _, _, _ = env.step(a)
#
#     plt.subplot(6, horizon, 2*horizon + 1 + i)
#     plt.axis('off')
#     plt.imshow(obs[..., :3])
#
#     plt.subplot(6, horizon, 4*horizon + 1 + i)
#     plt.axis('off')
#     plt.imshow(obs[..., 3:6])
#
# for i, x in enumerate(all_x):
#     plt.subplot(6, horizon, 3*horizon + 1 + i)
#     plt.axis('off')
#     plt.imshow(x[0, ..., :3])
#
#     plt.subplot(6, horizon, 5 * horizon + 1 + i)
#     plt.axis('off')
#     plt.imshow(x[0, ..., 3:6])

plt.show()
