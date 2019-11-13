from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.vae_env_model import VAE
from models.model_train import get_dataset
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv
from models.rl_planner import rl_planner
from tf_agents.environments import tf_py_environment

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

# read data from the dataset
encoding_net = VAE(256)
path_weights = '/Users/dgrebenyuk/Research/dataset/weights'
path_val = '/Users/dgrebenyuk/Research/dataset/validation.tfrecord'
# 'en' - encoder; 'de' - decoder; 'le' - latent environment
encoding_net.load_weights(['en', 'de', 'le'], path_weights)

dataset = get_dataset(path_val)
pca = PCA(n_components=50)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
BATCH = 1024
TAKE = 9  # 9 - whole
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
                                operation='move_pick')

eval_env = tf_py_environment.TFPyEnvironment(environment)
checkpoint_directory = '/Users/dgrebenyuk/Research/dataset/weights'
tf_agent, _, _ = rl_planner(eval_env, encoding_net, checkpoint_directory + "/rl")

time_step = environment.reset()
# episode_return = 0.0

# add th goal
obs = environment.reset()
plt.imshow(environment.goal_img[..., :3])
plt.show()

goal = encoding_net.encode(environment.goal_img[tf.newaxis, ...])
goal = pca.transform(goal)

# add the init state
df = pd.DataFrame(data=goal)
out = pd.concat([out, df], axis=0, ignore_index=True)

plt.imshow(obs.observation[..., :3])
plt.show()

init = encoding_net.encode(obs.observation[tf.newaxis, ...])
init = pca.transform(init)

df = pd.DataFrame(data=init)
out = pd.concat([out, df], axis=0, ignore_index=True)

while not time_step.is_last():
    action_step = tf_agent.policy.action(time_step)
    a = action_step.action
    # print(a)
    time_step = environment.step(a)
    z = time_step.observation
    z = encoding_net.encode(z[tf.newaxis, ...])
    z = pca.transform(z)
    df = pd.DataFrame(data=z)
    out = pd.concat([out, df], axis=0, ignore_index=True)
    # print(out.tail())
    # plt.imshow(z[0, ..., :3])
    # plt.show()
    # plt.imshow(time_step.observation[..., 3:6])
    # plt.show()
    # print(time_step.reward)
    # episode_return += time_step.reward

# t-SNE
tsne_results = tsne.fit_transform(out)
out = pd.DataFrame(data=tsne_results, columns=['tsne-one', 'tsne-two'])
out['show'] = "other"
out['show'].iloc[BATCH*TAKE] = "goal"  # goal state colour
out['show'].iloc[BATCH*TAKE + 1] = "initial"  # init state colour
out['show'].iloc[BATCH*TAKE + 2:] = "intermediate"  # intermediate state colour
print(out.tail(45))

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
