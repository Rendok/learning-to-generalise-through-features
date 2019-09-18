import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import tensorflow_io as tfio
import boto3
print(tf.__version__)


class AE(tf.keras.Model):
    def __init__(self, latent_dim=128):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(84, 84, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim, activation='tanh'),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=20 * 20 * 64, activation='relu'),
                tf.keras.layers.Reshape(target_shape=(20, 20, 64)),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=4, strides=(2, 2), activation='sigmoid'),
            ]
        )
        self.env_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim + 4)),
                tf.keras.layers.Dense(2 * latent_dim, activation='relu'),
                tf.keras.layers.Dense(2 * latent_dim, activation='relu'),
                tf.keras.layers.Dense(2 * latent_dim, activation='relu'),   # new layer
                tf.keras.layers.Dense(2 * latent_dim, activation='relu'),   # new layer
                tf.keras.layers.Dense(latent_dim, activation='tanh'),
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

    x_pred = model.encode(x0[np.newaxis, ...])
    for i in tf.range(actions.shape[0]):
        x_pred = model.forward(x_pred, actions[i, ...])
        all_preds.write(i, model.decode(x_pred))

    return x_pred, all_preds.stack()


def plan(model, x0, xg, horizon, epochs):
    actions = tf.convert_to_tensor(np.random.randn(horizon, 4).astype(np.float32))

    for _ in range(epochs):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(actions)
            x_pred, _ = roll_out_plan(model, x0, actions)
            zg = model.encode(xg[np.newaxis, ...])
            loss = tf.reduce_sum(tf.losses.mean_squared_error(zg, x_pred))

        gradients = tape.gradient(loss, actions)
        # print(gradients)
        actions = actions - 0.01 * gradients
    return actions


def compute_loss_de_en(model, x):
    z = model.encode(x)
    x_logit = model.decode(z)
    x_shape = tf.shape(x_logit)[0]
    x_logit = tf.reshape(x_logit, [x_shape, -1])

    loss = tf.reduce_sum(tf.losses.mean_squared_error(tf.reshape(x, [x_shape, -1]), x_logit))

    epoch_loss(loss)
    epoch_error(tf.reshape(x, [x_shape, -1]), x_logit)

    return loss


def compute_apply_gradients_enc_dec(model, x, optimizer):
    model.env_net.trainable = False

    with tf.GradientTape() as tape:
        loss = compute_loss_de_en(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def compute_loss_env(model, x, a, y):
    x_pred, label = model.predict(x, a, y)

    loss = tf.reduce_sum(tf.losses.mean_squared_error(label, x_pred))

    epoch_loss(loss)
    epoch_error(label, x_pred)
    return loss


def compute_apply_gradients_env(model, x, a, y, optimizer):
    model.inference_net.trainable = False
    model.generative_net.trainable = False
    with tf.GradientTape() as tape:
        loss = compute_loss_env(model, x, a, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# def get_datasets(path_tr, path_val):
#     with h5py.File(path_tr, 'r') as f:
#         states = f['states'][:1000]
#
#     # dataset = io.HDF5Dataset(path_tr, '/states')  # .shuffle(TRAIN_BUF).batch(BATCH_SIZE)
#     train_dataset = tf.data.Dataset.from_tensor_slices(states).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
#     del states
#
#     with h5py.File(path_val, 'r') as f:
#         states = f['states'][:1000]
#
#     # test_dataset = io.HDF5Dataset(path_val, '/states')
#     test_dataset = tf.data.Dataset.from_tensor_slices(states).batch(BATCH_SIZE)
#     del states
#     print(train_dataset.element_spec)
#     return train_dataset, test_dataset


def train_decoder(model, epochs, path_tr, path_val):
    @tf.function
    def train_decoder_one_step(model, train_dataset, test_dataset):
        for train_X in train_dataset:
            compute_apply_gradients_enc_dec(model, train_X, optimizer)

        train_loss = epoch_loss.result()
        train_error = epoch_error.result()
        epoch_loss.reset_states()
        epoch_error.reset_states()

        for test_X in test_dataset.take(3):
            compute_loss_de_en(model, test_X)

        return train_loss, train_error

    # read datasets from hdf5
    train_dataset = tfio.IOTensor.from_hdf5(path_tr)
    train_dataset = train_dataset('/states').to_dataset().shuffle(TRAIN_BUF).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = tfio.IOTensor.from_hdf5(path_val)
    test_dataset = test_dataset('/states').to_dataset().batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # new distribute part
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    for epoch in range(1, epochs + 1):
        train_loss, train_error = train_decoder_one_step(model, train_dataset, test_dataset)
        print('Epoch', epoch, 'train loss:', train_loss.numpy(), 'error:', train_error.numpy() * 100,
          'test loss:', epoch_loss.result().numpy(), 'error:', epoch_error.result().numpy() * 100, '\r')

        if CLOUD:
            model.save_weights('/tmp/weights/cp-de-{}.ckpt'.format(epoch))
        else:
            model.save_weights('/Users/dgrebenyuk/Research/dataset/weights/cp-de-{}.ckpt'.format(epoch))

        # s3.meta.client.upload_file('/tmp/weights/cp-de-{}.ckpt'.format(epoch), BUCKET, '/weights/cp-de-{}.ckpt'.format(epoch))

        epoch_loss.reset_states()
        epoch_error.reset_states()


def train_env(model, epochs, path_tr, path_val):
    @tf.function
    def train_env_one_step(model, train_dataset, test_dataset):
        for train_X, train_A, train_Y in train_dataset:
            compute_apply_gradients_env(model, train_X, train_A, train_Y, optimizer)

        train_loss = epoch_loss.result()
        train_error = epoch_error.result()
        epoch_loss.reset_states()
        epoch_error.reset_states()

        for train_X, train_A, train_Y in test_dataset:
            compute_loss_env(model, train_X, train_A, train_Y)

        return train_loss, train_error

    # read train datasets from hdf5
    dataset = tfio.IOTensor.from_hdf5(path_tr)
    ds_states = dataset('/states').to_dataset()
    ds_actions = dataset('/actions').to_dataset()
    ds_labels = dataset('/labels').to_dataset()

    train_dataset = tf.data.Dataset.zip((ds_states, ds_actions, ds_labels))
    train_dataset = train_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    del dataset, ds_states, ds_actions, ds_labels

    # read validation datasets from hdf5
    dataset = tfio.IOTensor.from_hdf5(path_val)
    ds_states = dataset('/states').to_dataset()
    ds_actions = dataset('/actions').to_dataset()
    ds_labels = dataset('/labels').to_dataset()

    test_dataset = tf.data.Dataset.zip((ds_states, ds_actions, ds_labels))
    test_dataset = test_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    del dataset, ds_states, ds_actions, ds_labels

    # new distribute part
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    for epoch in range(1, epochs + 1):
        train_loss, train_error = train_env_one_step(model, train_dataset, test_dataset)

        print('Epoch', epoch, 'train loss:', train_loss.numpy(), 'error:', train_error.numpy() * 100,
              'test loss:', epoch_loss.result().numpy(), 'error:', epoch_error.result().numpy() * 100)

        if CLOUD:
            model.save_weights('/tmp/weights/cp-de-{}.ckpt'.format(epoch))
        else:
            model.save_weights('/Users/dgrebenyuk/Research/dataset/weights/cp-de-{}.ckpt'.format(epoch))

        epoch_loss.reset_states()
        epoch_error.reset_states()


if __name__ == "__main__":

    # train in the cloud
    CLOUD = False

    epochs = 2
    TRAIN_BUF = 1000
    BATCH_SIZE = 100
    TEST_BUF = 1000

    if CLOUD:
        BUCKET = 'kuka-training-dataset'
        path_tr = '/tmp/training.h5'
        path_val = '/tmp/validation.h5'

        # upload files from the bucket
        s3 = boto3.resource('s3',
                            aws_access_key_id='AKIAZQDMP4R6P745OMOT',
                            aws_secret_access_key='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')
        s3.meta.client.download_file(BUCKET, 'training.h5', path_tr)
        s3.meta.client.download_file(BUCKET, 'validation.h5', path_val)

    else:
        path_tr = '/Users/dgrebenyuk/Research/dataset/training1.h5'
        path_val = '/Users/dgrebenyuk/Research/dataset/validation1.h5'

    # testing distributed training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    print(tf.config.experimental.list_physical_devices())

    # BATCH_SIZE_PER_REPLICA = 64
    # GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = AE()
    epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')
    epoch_error = tf.keras.metrics.MeanAbsoluteError(name='epoch_abs_error')

    # print(model.inference_net.summary())
    # print('\n', model.generative_net.summary())
    # print('\n', model.env_net.summary())

    if CLOUD:
        latest = tf.train.latest_checkpoint('/tmp/weights')
    else:
        latest = tf.train.latest_checkpoint('/Users/dgrebenyuk/Research/dataset/weights')

    model.load_weights(latest)
    print('Latest checkpoint:', latest)

    train_decoder(model, epochs, path_tr, path_val)
    # train_env(model, epochs, path_tr, path_val)

    # model.save_weights('/Users/dgrebenyuk/Research/dataset/weights/cp4_1.ckpt')

    number = 596
    mode = 'encode'
    # mode = 'forward'
    # mode = 'rollout'
    # mode = 'plan'

    with h5py.File(path_val, 'r') as f:
        states = f['states'][number, :, :, :]
        actions = f['actions'][number, :]
        labels = f['labels'][number+4, :, :, :]
        print('label act', actions)

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
            z = model.encode(states[np.newaxis, ...])
            x_pred = model.decode(z)

            plt.imshow(states[:, :, :])
            plt.title('State')
            plt.axis('off')
            plt.show()
            plt.imshow(x_pred[0, :, :, :])
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
