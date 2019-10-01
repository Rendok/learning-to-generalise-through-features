import h5py
import numpy as np
import matplotlib.pyplot as plt
import gym_kuka_multi_blocks.envs.kuka_cam_multi_blocks_gym as e
import tensorflow as tf


def env_creator_kuka_cam(renders=False):
    env = e.KukaCamMultiBlocksEnv(renders=renders,
                                  numObjects=5,
                                  isTest=1,  # 1 and 4
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


######## GET DATASET ######
def parse_image_function(example_proto):
    image_feature_description = {
        'image_x': tf.io.FixedLenFeature([], tf.string),
        'image_y': tf.io.FixedLenFeature([], tf.string),
        'label_x': tf.io.FixedLenFeature([], tf.string),
        'label_y': tf.io.FixedLenFeature([], tf.string),
        'action': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)


def decode_image_function(record):
    for key in ['image_x', 'image_y', 'label_x', 'label_y']:
        record[key] = tf.cast(tf.image.decode_image(record[key]), tf.float32) / 255.

    record['action'] = tf.io.parse_tensor(record['action'], out_type=tf.float32)

    return tf.concat((record['image_x'], record['image_y']), axis=-1), record['action'], tf.concat((record['label_x'], record['label_y']), axis=-1)


def get_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(decode_image_function)
    return dataset


if __name__ == "__main__":

    CHANNELS = 6
    TYPE = 'tfr'
    # h5
    path_tr_h5 = '/Users/dgrebenyuk/Research/dataset/training1.h5'
    path_val_h5 = '/Users/dgrebenyuk/Research/dataset/validation1.h5'
    # tf record
    path_tr_tfr = '/Users/dgrebenyuk/Research/dataset/training.tfrecord'
    path_val_tfr = '/Users/dgrebenyuk/Research/dataset/validation.tfrecord'

    DEBUG = True

    # generate_h5(n_batches=24, batch_size=16, planning_horizon=20, path=path_tr_h5)
    # generate_h5(n_batches=8, batch_size=16, planning_horizon=20, path=path_val_h5)
    # generate_tfr(n_batches=310, batch_size=16, planning_horizon=20, filename=path_tr_tfr)  # 310
    # generate_tfr(n_batches=31, batch_size=16, planning_horizon=20, filename=path_val_tfr)  # 31

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
            for i, (image, action, label) in dataset.take(4).enumerate():

                if i == 2:
                    print("act at", i, action.numpy())
                    plt.imshow(label[..., 3:6].numpy())
                    plt.show()
                elif i == 3:
                    plt.imshow(image[..., 3:6].numpy())
                    plt.show()

