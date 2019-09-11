import h5py
import numpy as np
import matplotlib.pyplot as plt


def generate_random_data(length):
    assert type(length) is int
    states = np.random.randn(length, 150, 150, 12)  # (W, H, C)
    actions = np.random.randn(length, 4, 1)
    labels = np.random.randn(length, 150, 150, 12)  # (W, H, C)
    return states, actions, labels


def env_creator_kuka_cam(renders=False):
    import gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym as e
    env = e.KukaCamMultiBlocksEnv(renders=renders,
                                  numObjects=3,
                                  isTest=4,
                                  operation='move_pick',
                                  )
    return env


def apply_actions(actions, batch_size, planning_horizon, env):
    inputs = []
    labels = []

    for m in range(batch_size):
        obs = env.reset()
        inputs.append(obs)
        for a in range(planning_horizon):

            obs, _, _, _ = env.step(actions[m, a, :])

            if a < planning_horizon - 1:
                inputs.append(obs)
            labels.append(obs)

    return np.array(inputs), np.array(labels), np.array(actions)


def generate_data(batch_size, planning_horizon):
    env = env_creator_kuka_cam(renders=False)

    actions = 2 * np.random.random_sample((batch_size, planning_horizon, 4)) - 1  # ~N[-1, 1)

    inputs, labels, actions = apply_actions(actions, batch_size, planning_horizon, env)

    actions = np.reshape(actions, (-1, 4))

    if DEBUG:
        print(batch_size*planning_horizon, 'inputs:', np.shape(inputs), 'labels:', np.shape(labels), 'actions:', np.shape(actions))
        print('inputs:', inputs[-2:, 0, 0, 0], 'actions:', actions[-2:, 0], 'labels:', labels[-2:, 0, 0, 0])

    return inputs, actions, labels


def generate(n_batches, batch_size, planning_horizon, path):
    """
    Data set generator
    :param n_batches:
    :param batch_size:
    :param planning_horizon:
    :param path:
    :return:
    """

    data_size = n_batches * batch_size * planning_horizon
    print('Data size:', data_size)

    with h5py.File(path, 'w') as f:
        d_states = f.create_dataset('states', (data_size, 84, 84, 3), dtype='u1')
        d_actions = f.create_dataset('actions', (data_size, 4))
        d_labels = f.create_dataset('labels', (data_size, 84, 84, 3), dtype='u1')

        for i in range(n_batches):
            bg = batch_size * planning_horizon * i
            end = batch_size * planning_horizon * i + batch_size * planning_horizon
            d_states[bg:end], d_actions[bg:end], d_labels[bg:end] = generate_data(batch_size, planning_horizon)


if __name__ == "__main__":

    path_tr = '/Users/dgrebenyuk/Research/dataset/training.h5'
    path_val = '/Users/dgrebenyuk/Research/dataset/validation.h5'
    DEBUG = True

    generate(n_batches=8, batch_size=64, planning_horizon=20, path=path_tr)
    generate(n_batches=3, batch_size=64, planning_horizon=5, path=path_val)

    # read data
    with h5py.File(path_tr, 'r') as f:
        states = f['states']
        actions = f['actions']
        labels = f['labels']

        if DEBUG:
            print('Read from file')
            print(states.shape, labels.shape, actions.shape)
            print('states', states[-2:, 0, 0, 0], 'actions:', actions[-2:, 0], 'labels:', labels[-2:, 0, 0, 0])
            plt.imshow(states[-1, :, :, :])
            plt.show()
            plt.imshow(labels[-2, :, :, :])
            plt.show()
