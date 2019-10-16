import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
# import tensorflow_io as tfio
import boto3


class AE(tf.keras.Model):
    def __init__(self, latent_dim=256):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 128, 6)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2D(
                    filters=512, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2D(
                    filters=512, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='tanh',
                                      kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
                tf.keras.layers.Dense(latent_dim, activation='tanh',
                                      kernel_initializer=tf.keras.initializers.glorot_normal(seed=None))
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(512, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Dense(8192, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Reshape(target_shape=(4, 4, 512)),
                tf.keras.layers.Conv2DTranspose(
                    filters=512, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                    kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Conv2DTranspose(
                    filters=6, kernel_size=4, strides=(1, 1), activation='sigmoid', padding='SAME',
                    kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
            ]
        )
        self.env_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim + 4)),
                tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(latent_dim, activation='tanh',
                                      kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
            ]

        )

    @tf.function
    def encode(self, x):
        return self.inference_net(x)

    @tf.function
    def decode(self, z):
        return self.generative_net(z)

    @tf.function
    def forward(self, x, a):
        a = a[tf.newaxis, ...]
        z = tf.concat([x, a], axis=1)
        pred = self.env_net(z)
        return pred

    def predict(self, x, a, y):
        x = self.inference_net(x)
        z = tf.concat([x, a], axis=1)
        pred = self.env_net(z)
        label = self.inference_net(y)
        return pred, label


# @tf.function
def roll_out_plan(model, x0, actions):
    all_preds = tf.TensorArray(tf.float32, actions.shape[0])

    x_pred = model.infer(x0[np.newaxis, ...])
    for i in tf.range(actions.shape[0]):
        x_pred = model.forward(x_pred, actions[i, ...])
        all_preds.write(i, model.decode(x_pred))

    return x_pred, all_preds.stack()


def plan(model, x0, xg, horizon, epochs):
    actions = tf.convert_to_tensor(np.random.randn(horizon, 4).astype(np.float32))

    for i in range(epochs):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(actions)
            x_pred, all_x = roll_out_plan(model, x0, actions)
            zg = model.infer(xg[np.newaxis, ...])
            loss = tf.reduce_sum(tf.losses.mean_squared_error(zg, x_pred))

        gradients = tape.gradient(loss, actions)

        # print(gradients)
        if i < 500:
            actions -= 0.2 * gradients
        elif i < 2000:
            actions -= 0.1 * gradients
        else:
            actions -= 0.01 * gradients

        actions = tf.clip_by_value(actions, -1., 1.)

        if i % 200 == 0:
            print(i)
        # print(np.abs(new_action - actions) / actions)
        # if tf.math.reduce_max(np.abs(new_action - actions) / actions) > 0.0001:
        #     actions = new_action
        # else:
        #     break
    return actions, all_x


def compute_loss_de_en(model, x):
    z = model.infer(x)
    x_logit = model.decode(z)

    x_shape = tf.shape(x_logit)[0]

    x_logit = tf.reshape(x_logit, [x_shape, -1])
    x = tf.reshape(x, [x_shape, -1])

    loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(x, x_logit), axis=-1))

    # loss = tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)

    epoch_loss.update_state(loss)
    return loss


@tf.function
def compute_apply_gradients_enc_dec(model, x, optimizer):
    model.env_net.trainable = False

    with tf.GradientTape() as tape:
        loss = compute_loss_de_en(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def compute_loss_env(model, x, a, y):
    x_pred, label = model.forward(x, a, y)

    loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(label, x_pred), axis=-1))

    # loss = tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)

    epoch_loss.update_state(loss)
    return loss


@tf.function
def compute_apply_gradients_env(model, x, a, y, optimizer):
    model.inference_net.trainable = False
    model.generative_net.trainable = False
    with tf.GradientTape() as tape:
        loss = compute_loss_env(model, x, a, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def get_dataset(filename, demonstration=False):
    image_feature_description = {
        'image_x': tf.io.FixedLenFeature([], tf.string),
        'image_y': tf.io.FixedLenFeature([], tf.string),
        'label_x': tf.io.FixedLenFeature([], tf.string),
        'label_y': tf.io.FixedLenFeature([], tf.string),
        'action': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def train_decoder(model, epochs, path_tr, path_val):

    def _decode_image_function(record):
        for key in ['image_x', 'image_y']:
            record[key] = tf.cast(tf.image.decode_image(record[key]), tf.float32) / 255.

        return tf.concat((record['image_x'], record['image_y']), axis=-1)

    def train_decoder_one_step(model, train_dataset, test_dataset):
        for i, train_X in enumerate(train_dataset):
            compute_apply_gradients_enc_dec(model, train_X, optimizer)
            # strategy.experimental_run_v2(compute_apply_gradients_enc_dec, args=(model, train_X, optimizer))
            # loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            if i % 50 == 0:
                print(i*BATCH_SIZE, epoch_loss.result().numpy())

        train_loss = epoch_loss.result()
        epoch_loss.reset_states()

        for test_X in test_dataset:
            compute_loss_de_en(model, test_X)
            # strategy.experimental_run_v2(compute_loss_de_en, args=(model, test_X))

        return train_loss

    train_dataset = get_dataset(path_tr)
    train_dataset = train_dataset.map(_decode_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = get_dataset(path_val)
    test_dataset = test_dataset.map(_decode_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # new distribute part
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    for epoch in range(1, epochs + 1):
        train_loss = train_decoder_one_step(model, train_dataset, test_dataset)
        print('Epoch', epoch, 'train loss:', train_loss.numpy(), 'validation loss:', epoch_loss.result().numpy(), '\r')

        if CLOUD:
            model.save_weights('/tmp/weights/cp-de-{}.ckpt'.format(epoch % 3))
        else:
            model.save_weights('/Users/dgrebenyuk/Research/dataset/weights/cp-de-{}.ckpt'.format(epoch % 3))

        # s3.meta.client.upload_file('/tmp/weights/cp-de-{}.ckpt'.format(epoch), BUCKET, '/weights/cp-de-{}.ckpt'.format(epoch))

        epoch_loss.reset_states()


def train_env(model, epochs, path_tr, path_val):

    def _decode_image_function(record):
        for key in ['image_x', 'image_y', 'label_x', 'label_y']:
            record[key] = tf.cast(tf.image.decode_image(record[key]), tf.float32) / 255.

        record['action'] = tf.io.parse_tensor(record['action'], out_type=tf.float32)

        return tf.concat((record['image_x'], record['image_y']), axis=-1), record['action'], tf.concat((record['label_x'], record['label_y']), axis=-1)

    def train_env_one_step(model, train_dataset, test_dataset):
        for i, (train_X, train_A, train_Y) in enumerate(train_dataset):
            compute_apply_gradients_env(model, train_X, train_A, train_Y, optimizer)
            if i % 50 == 0:
                print(i*BATCH_SIZE, epoch_loss.result().numpy())

        train_loss = epoch_loss.result()
        epoch_loss.reset_states()

        for train_X, train_A, train_Y in test_dataset:
            compute_loss_env(model, train_X, train_A, train_Y)

        return train_loss

    train_dataset = get_dataset(path_tr)
    train_dataset = train_dataset.map(_decode_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = get_dataset(path_val)
    test_dataset = test_dataset.map(_decode_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # new distribute part
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    for epoch in range(1, epochs + 1):
        train_loss = train_env_one_step(model, train_dataset, test_dataset)

        print('Epoch', epoch, 'train loss:', train_loss.numpy(), 'validation loss:', epoch_loss.result().numpy())

        if CLOUD:
            model.save_weights('/tmp/weights/cp-de-{}.ckpt'.format(epoch % 3))
        else:
            model.save_weights('/Users/dgrebenyuk/Research/dataset/weights/cp-de-2-{}.ckpt'.format(epoch % 3))

        epoch_loss.reset_states()


if __name__ == "__main__":

    print(tf.__version__)
    # train in the cloud
    CLOUD = False

    epochs = 100
    TRAIN_BUF = 2048 * 3
    BATCH_SIZE = 128 * 2
    TEST_BUF = 2048 * 3

    if CLOUD:
        BUCKET = 'kuka-training-dataset'
        # path_tr = '/tmp/training1.h5'
        # path_val = '/tmp/validation1.h5'
        path_tr = '/tmp/training.tfrecord'
        path_val = '/tmp/validation.tfrecord'

        # upload files from the bucket
        s3 = boto3.resource('s3',
                            aws_access_key_id='AKIAZQDMP4R6P745OMOT',
                            aws_secret_access_key='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')
        s3.meta.client.download_file(BUCKET, 'training.tfrecord', path_tr)
        s3.meta.client.download_file(BUCKET, 'validation.tfrecord', path_val)

    else:
        # path_tr = '/Users/dgrebenyuk/Research/dataset/training1.h5'
        # path_val = '/Users/dgrebenyuk/Research/dataset/validation1.h5'
        path_tr = '/Users/dgrebenyuk/Research/dataset/training.tfrecord'
        path_val = '/Users/dgrebenyuk/Research/dataset/validation.tfrecord'

    # testing distributed training
    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # BATCH_SIZE_PER_REPLICA = 128
    # BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    print(tf.config.experimental.list_physical_devices())

    # with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = AE()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')

    # print(model.inference_net.summary())
    # print('\n', model.generative_net.summary())
    # print('\n', model.env_net.summary())

    # with strategy.scope():
    if CLOUD:
        latest = tf.train.latest_checkpoint('/tmp/weights')
    else:
        latest = tf.train.latest_checkpoint('/Users/dgrebenyuk/Research/dataset/weights')

    model.load_weights(latest)
    print('Latest checkpoint:', latest)

    # train_decoder(model, epochs, path_tr, path_val)
    # train_env(model, epochs, path_tr, path_val)

    number = 501
    mode = 'encode'
    # mode = 'forward'
    # mode = 'rollout'
    # mode = 'plan'

    def _decode_image_function(record):
        for key in ['image_x', 'image_y', 'label_x', 'label_y']:
            record[key] = tf.cast(tf.image.decode_image(record[key]), tf.float32) / 255.

        record['action'] = tf.io.parse_tensor(record['action'], out_type=tf.float32)

        return tf.concat((record['image_x'], record['image_y']), axis=-1), record['action'], tf.concat((record['label_x'], record['label_y']), axis=-1)

    if mode == 'plan':
        # actions = np.array([[0, 0, -1, 0], [0, 0, -1, 0], [0, 0, -1, 0], [0, 0, -1, 0]]).astype(np.float32)
        plt.imshow(states)
        plt.show()
        plt.imshow(labels)
        plt.show()
        actions = plan(model, x0=states, xg=labels, horizon=4, epochs=10)
        _, x_pred = roll_out_plan(model, states, actions)
        for x_pr in x_pred:
            plt.imshow(x_pr[0, :, :, :])
            plt.title('Decoded Predicted Transition State')
            plt.axis('off')
            plt.show()

    if mode == 'rollout':
        actions = np.array([[0, 0, -1, 0], [0, 0, -1, 0], [0, 0, -1, 0], [0, 0, -1, 0]]).astype(np.float32)
        _, x_pred = roll_out_plan(model, states, actions)
        # print(x_pred)
        for x_pr in x_pred:
            plt.imshow(x_pr[0, :, :, :])
            plt.title('Decoded Predicted Transition State')
            plt.axis('off')
            plt.show()

    if mode == 'encode':
        data = get_dataset(path_val)
        data = data.map(_decode_image_function)

        for i, (states, actions, labels) in enumerate(data.take(number)):
            if i == (number - 1):
                z = model.encode(states[np.newaxis, ...])
                x_pred = model.decode(z)

                plt.imshow(states[:, :, :3])
                plt.title('State')
                plt.axis('off')
                plt.show()
                plt.imshow(states[:, :, 3:6])
                plt.title('State')
                plt.axis('off')
                plt.show()
                plt.imshow(x_pred[0, :, :, 0:3])
                plt.title('Encoded-Decoded State')
                plt.axis('off')
                plt.show()
                plt.imshow(x_pred[0, :, :, 3:6])
                plt.title('Encoded-Decoded State')
                plt.axis('off')
                plt.show()

    elif mode == 'forward':
        z, _ = model.predict(states[np.newaxis, ...], actions[np.newaxis, ...], labels[np.newaxis, ...])
        x_pred = model.decode(z)

        plt.imshow(labels[:, :, :])
        plt.title('Transition State Label')
        plt.axis('off')
        plt.show()
        plt.imshow(x_pred[0, :, :, :])
        plt.title('Decoded Predicted Transition State')
        plt.axis('off')
        plt.show()
