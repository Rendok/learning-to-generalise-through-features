import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import tensorflow_io.hdf5 as io
# import ipykernel  # needed to fix a bug in console
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
                # tf.keras.layers.Conv2DTranspose(
                #     filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                # No activation
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
    #for a in actions:
    for i in tf.range(actions.shape[0]):
        x_pred = model.forward(x_pred, actions[i, ...])
        all_preds.write(i, model.decode(x_pred))
        # all_preds.append(model.decode(x_pred))
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


@tf.function
def compute_loss_de_en(model, x):
    z = model.encode(x)
    x_logit = model.decode(z)
    x_shape = tf.shape(x_logit)[0]
    x_logit = tf.reshape(x_logit, [x_shape, -1])

    loss = tf.reduce_sum(tf.losses.mean_squared_error(tf.reshape(x, [x_shape, -1]), x_logit))

    epoch_loss(loss)
    epoch_error(tf.reshape(x, [x_shape, -1]), x_logit)
    return loss


@tf.function
def compute_loss_env(model, x, a, y):

    x_pred, label = model.predict(x, a, y)

    loss = tf.reduce_sum(tf.losses.mean_squared_error(label, x_pred))

    epoch_loss(loss)
    epoch_error(label, x_pred)
    return loss


@tf.function
def compute_apply_gradients_enc_dec(model, x, optimizer):
    model.env_net.trainable = False
    # model.inference_net.trainable = False
    with tf.GradientTape() as tape:
        loss = compute_loss_de_en(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
def compute_apply_gradients_env(model, x, a, y, optimizer):
    model.inference_net.trainable = False
    model.generative_net.trainable = False
    with tf.GradientTape() as tape:
        loss = compute_loss_env(model, x, a, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def get_datasets(path_tr, path_val):

    with h5py.File(path_tr, 'r') as f:
        states = f['states'][:]

    train_dataset = tf.data.Dataset.from_tensor_slices(states).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

    with h5py.File(path_val, 'r') as f:
        states = f['states'][:]

    test_dataset = tf.data.Dataset.from_tensor_slices(states).batch(BATCH_SIZE)
    return train_dataset, test_dataset


@tf.function
def train_decoder(epochs, train_dataset, test_dataset):

    # with h5py.File(path_tr, 'r') as f:
    #     states = f['states'][:]  # / 255.0

    # dataset = io.HDF5Dataset(path_tr, '/states') #.shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    # train_dataset = tf.data.Dataset.from_tensor_slices(states).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

    # with h5py.File(path_val, 'r') as f:
    #     states = f['states'][:]  # / 255.0

    # test_dataset = io.HDF5Dataset(path_val, '/states')
    # test_dataset = tf.data.Dataset.from_tensor_slices(states).batch(BATCH_SIZE)

    for epoch in range(1, epochs + 1):
        for train_X in train_dataset:
            # print(train_X.shape)
            compute_apply_gradients_enc_dec(model, train_X, optimizer)

        train_loss = epoch_loss.result().numpy()
        train_error = epoch_error.result().numpy()
        epoch_loss.reset_states()
        epoch_error.reset_states()

        for i, test_X in enumerate(test_dataset):
            compute_loss_de_en(model, test_X)

        print('Epoch', epoch, 'train loss:', train_loss, 'error:', train_error * 100,
              'test loss:', epoch_loss.result().numpy(), 'error:', epoch_error.result().numpy() * 100, '\r')

        epoch_loss.reset_states()
        epoch_error.reset_states()


def train_env(epochs, path_tr, path_val):

    with h5py.File(path_tr, 'r') as f:
        states = f['states'][:] / 255.0
        actions = f['actions'][:]
        labels = f['labels'][:] / 255.0

        print(states.shape, actions.shape, labels.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices((states, actions, labels)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

    with h5py.File(path_val, 'r') as f:
        states = f['states'][:] / 255.0
        actions = f['actions'][:]
        labels = f['labels'][:] / 255.0
        test_dataset = tf.data.Dataset.from_tensor_slices((states, actions, labels)).batch(BATCH_SIZE)

    for epoch in range(1, epochs + 1):
        for train_X, train_A, train_Y in train_dataset:
            # print(np.shape(train_X), np.shape(train_A), np.shape(train_Y))
            compute_apply_gradients_env(model, train_X, train_A, train_Y, optimizer)

        train_loss = epoch_loss.result().numpy()
        train_error = epoch_error.result().numpy()
        epoch_loss.reset_states()
        epoch_error.reset_states()

        for train_X, train_A, train_Y in test_dataset:
            compute_loss_env(model, train_X, train_A, train_Y)

        print('Epoch', epoch, 'train loss:', train_loss, 'error:', train_error * 100,
              'test loss:', epoch_loss.result().numpy(), 'error:', epoch_error.result().numpy() * 100)

        epoch_loss.reset_states()
        epoch_error.reset_states()


if __name__ == "__main__":

    path_tr = '/Users/dgrebenyuk/Research/dataset/training.h5'
    path_val = '/Users/dgrebenyuk/Research/dataset/validation.h5'
    epochs = 2
    TRAIN_BUF = 1000
    BATCH_SIZE = 100
    TEST_BUF = 1000

    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = AE()
    epoch_loss = tf.keras.metrics.Mean(name='train_loss')
    epoch_error = tf.keras.metrics.MeanAbsoluteError(name='mean_abs_error')

    # print(model.inference_net.summary())
    # print('\n', model.generative_net.summary())
    # print('\n', model.env_net.summary())

    latest = tf.train.latest_checkpoint('/Users/dgrebenyuk/Research/dataset/weights')
    model.load_weights(latest)
    print('Latest checkpoint:', latest)

    tran_dataset, test_dataset = get_datasets(path_tr, path_val)
    train_decoder(epochs, tran_dataset, test_dataset)
    # train_env(epochs, path_tr, path_val)

    # model.save_weights('/Users/dgrebenyuk/Research/dataset/weights/cp4_1.ckpt')

    number = 596
    # mode = 'encode'
    # mode = 'forward'
    mode = 'rollout'
    # mode = 'plan'

    with h5py.File(path_val, 'r') as f:
        states = f['states'][number, :, :, :]  # / 255.0
        actions = f['actions'][number, :]
        labels = f['labels'][number+4, :, :, :]  # / 255.0
        print('label act', actions)

        # states = np.round(states, 0)
        # print(states)

        # for st in states:
        #     plt.imshow(st)
        #     plt.title('Env Transition States')
        #     plt.axis('off')
        #     plt.show()

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
            print(x_pred)
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
