from tf_agents.drivers import dynamic_episode_driver, dynamic_step_driver
from tf_agents.agents.ddpg import critic_network, actor_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, network
from tf_agents.networks import value_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.networks import utils
from tf_agents.utils import nest_utils
from models.critic_network import CriticNetwork
from models.actor_network import ActorNetwork
from models.model_train import train_one_step

import tensorflow as tf
from models.vae_env_model import VAE

from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv
import matplotlib.pyplot as plt
import os

# HYPER PARAMETERS
collect_episodes_per_iteration = 30  # The number of episodes to take in the environment before
replay_buffer_capacity = 101  # Replay buffer capacity per env

actor_fc_layer_params = (256, 256)
critic_fc_layer_params = (256, 256)

num_latent_dims = 256
learning_rate = 3e-4
gradient_clipping = None
num_eval_episodes = 30  # The number of episodes to run eval on.

num_parallel_environments = 1  # Number of environments to run in parallel
num_epochs = 25  # Number of epochs for computing policy updates

num_iterations = 50
log_interval = 1
eval_interval = 3

debug_summaries = False,
summarize_grads_and_vars = False
# --------

encoding_net = VAE(num_latent_dims)
encoding_net.load_weights(['en', 'de'], '/tmp/weights')
# encoding_net.load_weights(['en', 'de'], '/Users/dgrebenyuk/Research/dataset/weights')

checkpoint_directory = '/tmp/weights/rl'
# checkpoint_directory = '/Users/dgrebenyuk/Research/dataset/weights/rl'

def env():
    env = KukaCamMultiBlocksEnv(renders=False,
                                # encoding_net=encoding_net,
                                numObjects=4,
                                isTest=4,  # 1 and 4
                                operation='move_pick',
                                )
    return env


eval_env = tf_py_environment.TFPyEnvironment(env())

train_env = tf_py_environment.TFPyEnvironment(  #env())  # TODO: bug don't work in parallel with an encoding net inside
    parallel_py_environment.ParallelPyEnvironment(
        [lambda: env()] * num_parallel_environments))

# validate_py_environment(env(), episodes=10)

observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()
print("Spec", action_spec)

critic_net = CriticNetwork(
    observation_spec,
    encoding_network=encoding_net,
    fc_layer_params=critic_fc_layer_params)

actor_net = ActorNetwork(
    observation_spec,
    action_spec,
    encoding_network=encoding_net,
    fc_layer_params=actor_fc_layer_params)

# actor_net = actor_distribution_network.ActorDistributionNetwork(
#     observation_spec,
#     action_spec,
#     fc_layer_params=actor_fc_layer_params)
# continuous_projection_net=normal_projection_net)

global_step = tf.compat.v1.train.get_or_create_global_step()

tf_agent = ppo_agent.PPOAgent(
    train_env.time_step_spec(),
    action_spec,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
    # optimizer=tf.optimizers.Adam(learning_rate),
    actor_net=actor_net,
    value_net=critic_net,
    num_epochs=num_epochs,
    gradient_clipping=gradient_clipping,
    debug_summaries=debug_summaries,
    summarize_grads_and_vars=summarize_grads_and_vars,
    normalize_observations=False,
    normalize_rewards=False,
    train_step_counter=global_step)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy


def compute_avg_return(environment, policy, num_episodes=5):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity
)

# step_metrics = [tf_metrics.NumberOfEpisodes(), tf_metrics.EnvironmentSteps()]

# TODO: fix metrics; don't work in parallel
# train_metrics = [tf_metrics.AverageReturnMetric(), tf_metrics.AverageEpisodeLengthMetric()]

collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_episodes=collect_episodes_per_iteration
)

# tf_agent.train = common.function(tf_agent.train)
collect_driver.run = common.function(collect_driver.run)


def split_trajectory(trajectory, rest):
    outer_rank = nest_utils.get_outer_rank(trajectory.observation, observation_spec)
    batch_squash = utils.BatchSquash(outer_rank)
    observation, action = tf.nest.map_structure(batch_squash.flatten, (trajectory.observation, trajectory.action))
    observation = tf.cast(observation, tf.float32) / 255.

    return observation, action, observation


optimizer = tf.keras.optimizers.Adam(1e-4)
epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')
TRAIN_BUF = 2048
BATCH_SIZE = 128

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=tf_agent)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=3)
# status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
status = checkpoint.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

for _ in range(num_iterations):

    print('collecting')
    collect_driver.run()
    trajectories = replay_buffer.gather_all()

    # print(trajectories.observation.shape)

    # sample_batch_size=batch_size, num_steps=1 TODO: del
    # dataset = replay_buffer.as_dataset(num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    #     .shuffle(replay_buffer_capacity) \
    #     .batch(BATCH_SIZE) \
    #     .map(split_trajectory, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    #     .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
    #     .take(3 * int(TRAIN_BUF / BATCH_SIZE))
    #
    # # for i, (d, a, b) in enumerate(dataset):
    # #     print(i, d.shape)
    # # plt.imshow(d[0, ..., :3])
    # # plt.show()
    #
    # train_one_step(encoding_net, dataset, optimizer, epoch_loss, 'vae')

    encoding_net.inference_net.trainable = False
    encoding_net.generative_net.trainable = False
    encoding_net.lat_env_net.trainable = False

    train_loss, _ = tf_agent.train(experience=trajectories)

    # status.assert_consumed() TODO: del
    # checkpoint.save(file_prefix=checkpoint_prefix) TODO: del
    save_path = manager.save()
    print("Saved checkpoint: {}".format(save_path))

    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))


environment = KukaCamMultiBlocksEnv(renders=False,
                                # encoding_net=encoding_net,
                                numObjects=4,
                                isTest=4,  # 1 and 4
                                operation='move_pick')



time_step = environment.reset()
episode_return = 0.0

# plt.imshow(environment.goal_img[..., :3])
# plt.show()
i = 0

while not time_step.is_last():
    action_step = eval_policy.action(time_step)
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
    i += 1

print(episode_return)
