from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.vae_env_model import VAE
from models.model_train import get_dataset
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv
from models.rl_planner import rl_planner
from tf_agents.environments import tf_py_environment

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

DIMENSIONS = 3
BATCH = 1024
TAKE = 9  # 9 - whole
same_init_state = True

# read data from the dataset
encoding_net = VAE(256)
path_weights = '/Users/dgrebenyuk/Research/dataset/weights'
path_val = '/Users/dgrebenyuk/Research/dataset/validation.tfrecord'
# 'en' - encoder; 'de' - decoder; 'le' - latent environment
encoding_net.load_weights(['en', 'de', 'le'], path_weights)

dataset = get_dataset(path_val)
pca = PCA(n_components=50)
tsne = TSNE(n_components=DIMENSIONS, verbose=1, perplexity=40, n_iter=300)

# to get an array from the data set
for i, (data, _, _) in enumerate(dataset.batch(BATCH).take(TAKE)):
    embedded = encoding_net.encode(data)
    # print(embedded)

    if i == 0:
        pca_result = pca.fit_transform(embedded)

    else:
        pca_result = pca.transform(embedded)
    # print(pca_result.shape)

    # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    if i == 0:
        # out = pd.DataFrame(data=pca_result, columns=['pca-one', 'pca-two'])
        out = pd.DataFrame(data=pca_result)
    if i > 0:
        # df = pd.DataFrame(data=pca_result, columns=['pca-one', 'pca-two'])
        df = pd.DataFrame(data=pca_result)
        out = pd.concat([out, df], axis=0, ignore_index=True)

# Add extra trajectory to the goal
environment = KukaCamMultiBlocksEnv(renders=False,
                                encoding_net=encoding_net,
                                numObjects=4,
                                isTest=4,  # 1 and 4
                                same_init_state=same_init_state,
                                operation='move_pick')

eval_env = tf_py_environment.TFPyEnvironment(environment)
checkpoint_directory = '/Users/dgrebenyuk/Research/dataset/weights'
tf_agent, _, _ = rl_planner(eval_env, encoding_net, checkpoint_directory + "/rl")

last_inds = []

for repeat in range(6):

    last_inds.append(out.tail(1).index.item())
    # print("last ind", last_inds, '\n', out.tail())

    # add th goal
    time_step = environment.reset()
    plt.imshow(environment.goal_img[..., :3])
    plt.show()

    goal = encoding_net.encode(environment.goal_img[tf.newaxis, ...])
    goal = pca.transform(goal)

    # add the init state
    df = pd.DataFrame(data=goal)
    out = pd.concat([out, df], axis=0, ignore_index=True)

    # plt.imshow(time_step.observation[..., :3])
    # plt.show()

    init = encoding_net.encode(time_step.observation[tf.newaxis, ...])
    init = pca.transform(init)

    df = pd.DataFrame(data=init)
    out = pd.concat([out, df], axis=0, ignore_index=True)

    a = [[0, 0, -0.5, 0],
         [0, 0, 0.5, 0],
         [0, -0.5, 0, 0],
         [0, 0.5, 0, 0],
         [-0.5, 0, 0, 0],
         [0.5, 0, 0, 0]]

    for _ in range(5):
    # while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        # a = action_step.action
        time_step = environment.step(a[repeat])
        # time_step = environment.step(a)
        z = time_step.observation
        z = encoding_net.encode(z[tf.newaxis, ...])
        z = pca.transform(z)
        df = pd.DataFrame(data=z)
        out = pd.concat([out, df], axis=0, ignore_index=True)
        # plt.imshow(z[0, ..., :3])
        # plt.show()
        # plt.imshow(time_step.observation[..., 3:6])
        # plt.show()

    # plt.imshow(time_step.observation[..., :3])
    # plt.show()

# t-SNE
tsne_results = tsne.fit_transform(out)

colours = ["#e74c3c", "#9aff24", "#ffea24", "#ff9c24", "#7140e3", "#24ffe9"]

if DIMENSIONS == 3:
    out = pd.DataFrame(data=tsne_results, columns=['tsne-one', 'tsne-two', 'tsne-three'])
    out['show'] = "#95a5a6"  # others
    for i, ind in enumerate(last_inds):
        # out['show'].iloc[i + 1] = "#2ecc71"   # goal state colour
        out['show'].iloc[ind + 2] = "#3498db"     # init state colour
        out['show'].iloc[ind + 3:] = colours[i]  # "#e74c3c"    # intermediate state colour

else:
    out = pd.DataFrame(data=tsne_results, columns=['tsne-one', 'tsne-two'])
    out['show'] = "other"
    for i in last_inds:
        out['show'].iloc[i + 1] = "goal"  # goal state colour
        out['show'].iloc[i + 2] = "initial"  # init state colour
        out['show'].iloc[i + 3:] = "intermediate"  # intermediate state colour

# print(out.tail(45))

# 2D plot
if DIMENSIONS == 2:
    flatui = ["#95a5a6", "#2ecc71", "#3498db", "#e74c3c"]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="show",
        style="show",
        palette=sns.color_palette(flatui),
        data=out,
        legend="full",
        alpha=0.5)

    plt.show()


def plot_animated_3d():

    fig = plt.figure()
    ax = Axes3D(fig)

    def init():
        ax.scatter(out["tsne-one"], out["tsne-two"], out["tsne-three"], c=out["show"], alpha=0.5)
        return fig,

    def animate(i):
        ax.view_init(elev=30., azim=i)
        return fig,
    # plt.show()

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    # Save
    anim.save('test.gif', writer="imagemagick")


# 3D plot
if DIMENSIONS == 3:
    plot_animated_3d()

    # plot ordinary img
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(out["tsne-one"], out["tsne-two"], out["tsne-three"], c=out["show"], alpha=0.5)
    ax.view_init(30, 185)
    plt.show()



