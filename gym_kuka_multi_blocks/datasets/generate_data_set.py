import h5py
import numpy as np
import matplotlib.pyplot as plt
from gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym import KukaCamMultiBlocksEnv
from acrobot.acrobot_env import AcrobotEnv
from acrobot.reacher_env import ReacherBulletEnv
import tensorflow as tf
from models.model_train import get_dataset


def env_creator_kuka_cam(renders=False):
    env = KukaCamMultiBlocksEnv(renders=renders,
                                  numObjects=5,
                                  isTest=1,  # 1 and 4
                                  operation='move_pick',
                                  )
    return env


def env_creator_acrobot():
    return AcrobotEnv(obs_type="uint")


def env_creator_reacher():
    return ReacherBulletEnv(obs_type="uint", same_init_state=True, max_time_step=40)


def apply_actions(actions, batch_size, planning_horizon, env):
    inputs = []
    labels = []

    for m in range(batch_size):
        obs = env.reset()
        inputs.append(obs.observation)
        for a in range(planning_horizon):

            obs = env.step(actions[m, a, :])

            # print(obs.observation.shape)

            if a < planning_horizon - 1:
                inputs.append(obs.observation)
            labels.append(obs.observation)

    return np.array(inputs), np.array(labels), np.array(actions)


def generate_data(env, batch_size, planning_horizon):

    act_spec = env.action_spec()

    # actions = 2 * np.random.random_sample((batch_size, planning_horizon, 4)) - 1  # ~N[-1, 1)
    actions = np.random.uniform(act_spec.minimum, act_spec.maximum, (batch_size, planning_horizon, act_spec.shape[0]))

    inputs, labels, actions = apply_actions(actions, batch_size, planning_horizon, env)

    actions = np.reshape(actions, (-1, act_spec.shape[0])).astype('float32')

    if DEBUG:
        print(batch_size*planning_horizon, 'inputs:', np.shape(inputs), 'labels:', np.shape(labels), 'actions:', np.shape(actions))
        print('inputs:', inputs[2, 0, 0, 0], 'actions:', actions[2, ...], 'labels:', labels[2, 0, 0, 0])

    return inputs, actions, labels


def generate_h5(n_batches, batch_size, planning_horizon, path):
    """
    Data set generator
    """

    data_size = n_batches * batch_size * planning_horizon
    print('Data size:', data_size)

    with h5py.File(path, 'w') as f:
        d_states = f.create_dataset('states', (data_size, 128, 128, CHANNELS), compression="gzip")  # dtype='i4'
        d_actions = f.create_dataset('actions', (data_size, 4), compression="gzip")
        d_labels = f.create_dataset('labels', (data_size, 128, 128, CHANNELS), compression="gzip")

        for i in range(n_batches):
            bg = batch_size * planning_horizon * i
            end = batch_size * planning_horizon * i + batch_size * planning_horizon
            d_states[bg:end], d_actions[bg:end], d_labels[bg:end] = generate_data(batch_size, planning_horizon)

            print('Batch {} of {}'.format(i+1, n_batches))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, label, action):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    image_x = tf.image.encode_png(image[..., :3])
    image_y = tf.image.encode_png(image[..., 3:6])

    label_x = tf.image.encode_png(label[..., :3])
    label_y = tf.image.encode_png(label[..., 3:6])

    feature = {
        'image_x': _bytes_feature(image_x),
        'image_y': _bytes_feature(image_y),
        'label_x': _bytes_feature(label_x),
        'label_y': _bytes_feature(label_y),
        'action': _bytes_feature(tf.io.serialize_tensor(action)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def serialize_example_one_img(image, label, action):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    image_x = tf.image.encode_png(image[..., :3])
    label_x = tf.image.encode_png(label[..., :3])

    feature = {
        'image_x': _bytes_feature(image_x),
        'label_x': _bytes_feature(label_x),
        'action': _bytes_feature(tf.io.serialize_tensor(action)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def generate_tfr(env, n_batches, batch_size, planning_horizon, filename):

    data_size = n_batches * batch_size * planning_horizon
    print('Data size:', data_size)

    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(n_batches):

            x, a, y = generate_data(env, batch_size, planning_horizon)

            for xi, ai, yi in zip(x, a, y):
                if type(env).__name__ == "AcrobotEnv" or type(env).__name__ == "ReacherBulletEnv":
                    example = serialize_example_one_img(xi, yi, ai)
                elif type(env).__name__ == "KukaCamMultiBlocksEnv":
                    example = serialize_example(xi, yi, ai)
                else:
                    raise ValueError

                writer.write(example)

            print('Batch {} of {}'.format(i + 1, n_batches))


if __name__ == "__main__":

    CHANNELS = 3
    TYPE = 'tfr'
    # h5
    path_tr_h5 = '/Users/dgrebenyuk/Research/dataset/training1.h5'
    path_val_h5 = '/Users/dgrebenyuk/Research/dataset/validation1.h5'
    # tf record
    path_tr_tfr = '/Users/dgrebenyuk/Research/dataset/reacher_training.tfrecord'
    path_val_tfr = '/Users/dgrebenyuk/Research/dataset/reacher_validation.tfrecord'

    DEBUG = True

    # env = env_creator_kuka_cam(renders=False)
    env = env_creator_reacher()

    # generate_h5(n_batches=24, batch_size=16, planning_horizon=20, path=path_tr_h5)
    # generate_h5(n_batches=8, batch_size=16, planning_horizon=20, path=path_val_h5)
    generate_tfr(env, n_batches=157, batch_size=16, planning_horizon=40, filename=path_tr_tfr)  # 310
    generate_tfr(env, n_batches=16, batch_size=16, planning_horizon=40, filename=path_val_tfr)  # 31

    if DEBUG:
        if TYPE == 'h5':
            # read data
            with h5py.File(path_tr_h5, 'r') as f:
                states = f['states']
                actions = f['actions']
                labels = f['labels']

                print('Read from file')
                print(states.shape, labels.shape, actions.shape)
                print('states', states[-2:, 0, 0, 0], 'actions:', actions[-2:, 0], 'labels:', labels[-2:, 0, 0, 0])
                plt.imshow(states[-19, :, :, :3])
                plt.show()
                plt.imshow(labels[-20, :, :, :3])
                plt.show()

        if TYPE == 'tfr':

            dataset = get_dataset(path_tr_tfr)

            print('Read from file')
            for i, (image, action, label) in dataset.take(10).enumerate():

                if i == 5:
                    print("act at", i.numpy(), action.numpy())
                    plt.imshow(label[..., :3].numpy())
                    plt.show()
                elif i == 6:
                    plt.imshow(image[..., :3].numpy())
                    plt.show()

