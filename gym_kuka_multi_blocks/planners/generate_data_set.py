import h5py
import numpy as np
import matplotlib.pyplot as plt
import gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym as e
from PIL import Image
import tensorflow as tf


def generate_random_data(length):
    assert type(length) is int
    states = np.random.randn(length, 256, 256, CHANNELS)  # (W, H, C)
    actions = np.random.randn(length, 4, 1)
    labels = np.random.randn(length, 256, 256, CHANNELS)  # (W, H, C)
    return states, actions, labels


def env_creator_kuka_cam(renders=False):
    env = e.KukaCamMultiBlocksEnv(renders=renders,
                                  numObjects=5,
                                  isTest=1,
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

    actions = np.reshape(actions, (-1, 4)).astype('float32')

    if DEBUG:
        print(batch_size*planning_horizon, 'inputs:', np.shape(inputs), 'labels:', np.shape(labels), 'actions:', np.shape(actions))
        print('inputs:', inputs[2, 0, 0, 0], 'actions:', actions[2, ...], 'labels:', labels[2, 0, 0, 0])

    return inputs, actions, labels


def generate_h5(n_batches, batch_size, planning_horizon, path):
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
        d_states = f.create_dataset('states', (data_size, 128, 128, CHANNELS), compression="gzip")  # dtype='i4'
        d_actions = f.create_dataset('actions', (data_size, 4), compression="gzip")
        d_labels = f.create_dataset('labels', (data_size, 128, 128, CHANNELS), compression="gzip")

        for i in range(n_batches):
            bg = batch_size * planning_horizon * i
            end = batch_size * planning_horizon * i + batch_size * planning_horizon
            d_states[bg:end], d_actions[bg:end], d_labels[bg:end] = generate_data(batch_size, planning_horizon)

            print('Batch {} of {}'.format(i+1, n_batches))


def generate_tfr(n_batches, batch_size, planning_horizon, filename):
    def serialize_example(image, label, action):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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

    data_size = n_batches * batch_size * planning_horizon
    print('Data size:', data_size)

    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(n_batches):

            x, a, y = generate_data(batch_size, planning_horizon)

            for xi, ai, yi in zip(x, a, y):
                example = serialize_example(xi, yi, ai)
                writer.write(example)

            print('Batch {} of {}'.format(i + 1, n_batches))


if __name__ == "__main__":

    CHANNELS = 6
    TYPE = 'tfr'
    # h5
    path_tr_h5 = '/Users/dgrebenyuk/Research/dataset/training1.h5'
    path_val_h5 = '/Users/dgrebenyuk/Research/dataset/validation1.h5'
    # tf record
    path_tr_tfr = '/Users/dgrebenyuk/Research/dataset/training1.tfrecord'
    path_val_tfr = '/Users/dgrebenyuk/Research/dataset/validation1.tfrecord'

    DEBUG = True

    # generate_h5(n_batches=24, batch_size=16, planning_horizon=20, path=path_tr_h5)
    # generate_h5(n_batches=8, batch_size=16, planning_horizon=20, path=path_val_h5)
    generate_tfr(n_batches=313, batch_size=16, planning_horizon=20, filename=path_tr_tfr)
    generate_tfr(n_batches=31, batch_size=16, planning_horizon=20, filename=path_val_tfr)

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
            image_feature_description = {
                'image_x': tf.io.FixedLenFeature([], tf.string),
                'image_y': tf.io.FixedLenFeature([], tf.string),
                'label_x': tf.io.FixedLenFeature([], tf.string),
                'label_y': tf.io.FixedLenFeature([], tf.string),
                'action': tf.io.FixedLenFeature([], tf.string),
            }

            def _parse_image_function(example_proto):
                return tf.io.parse_single_example(example_proto, image_feature_description)

            def _decode_image_function(record):
                for key in ['image_x', 'image_y', 'label_x', 'label_y']:
                    record[key] = tf.image.decode_image(record[key])

                record['action'] = tf.io.parse_tensor(record['action'], out_type=tf.float32)
                return record

            filenames = [path_tr_tfr]
            raw_dataset = tf.data.TFRecordDataset(filenames)

            parsed_image_dataset = raw_dataset.map(_parse_image_function).map(_decode_image_function)
            print('Read from file')
            for i, record in enumerate(parsed_image_dataset.take(4)):

                if i == 2:
                    print(record['action'].numpy())
                    plt.imshow(record['label_y'].numpy())
                    plt.show()
                elif i == 3:
                    plt.imshow(record['image_y'].numpy())
                    plt.show()

