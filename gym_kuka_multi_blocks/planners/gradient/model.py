import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
# import PIL
# import imageio
import h5py
import tensorflow as tf
import ipykernel  # needed to fix a bug in console

from IPython import display

print(tf.__version__)

path_tr = '/Users/dgrebenyuk/Research/dataset/training.h5'
path_val = '/Users/dgrebenyuk/Research/dataset/validation.h5'


class AE(tf.keras.Model):
    def __init__(self, latent_dim):
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
                tf.keras.layers.Dense(2*latent_dim, activation='relu'),
                tf.keras.layers.Dense(2*latent_dim, activation='relu'),
                tf.keras.layers.Dense(latent_dim, activation='tanh'),
            ]

        )

    def encode(self, x):
        return self.inference_net(x)

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def predict(self, x, a, y):
        x = self.inference_net(x)
        z = tf.concat([x, a], axis=1)
        pred = self.env_net(z)
        label = self.inference_net(y)
        return pred, label


@tf.function
def compute_loss(model, x):
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
def compute_apply_gradients(model, x, optimizer):
    model.env_net.trainable = False
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
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


def train_decoder(epochs, path_tr, path_val):

    with h5py.File(path_tr, 'r') as f:
        states = f['states'][:] / 255.0
        train_dataset = tf.data.Dataset.from_tensor_slices(states).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

    with h5py.File(path_val, 'r') as f:
        states = f['states'][:] / 255.0
        test_dataset = tf.data.Dataset.from_tensor_slices(states).batch(BATCH_SIZE)

    for epoch in range(1, epochs + 1):
        for train_X in train_dataset:
            compute_apply_gradients(model, train_X, optimizer)

        train_loss = epoch_loss.result().numpy()
        train_error = epoch_error.result().numpy()
        epoch_loss.reset_states()
        epoch_error.reset_states()

        for test_X in test_dataset:
            compute_loss(model, test_X)

        print('Epoch', epoch, 'train loss:', train_loss, 'error:', train_error,
              'test loss:', epoch_loss.result().numpy(), 'error:', epoch_error.result().numpy())

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


path_tr = '/Users/dgrebenyuk/Research/dataset/training.h5'
path_val = '/Users/dgrebenyuk/Research/dataset/validation.h5'
epochs = 20
TRAIN_BUF = 1000
BATCH_SIZE = 100
TEST_BUF = 1000

optimizer = tf.keras.optimizers.Adam(1e-4)
model = AE(128)
epoch_loss = tf.keras.metrics.Mean(name='train_loss')
epoch_error = tf.keras.metrics.MeanAbsoluteError(name='mean_abs_error')

# print(model.inference_net.summary())
# print('\n', model.generative_net.summary())
# print('\n', model.env_net.summary())

latest = tf.train.latest_checkpoint('/Users/dgrebenyuk/Research/dataset/weights')
model.load_weights(latest)
print('Latest checkpoint:', latest)

# train_decoder(epochs, path_tr, path_val)
# train_env(epochs, path_tr, path_val)

# model.save_weights('/Users/dgrebenyuk/Research/dataset/weights/cp3.ckpt')

with h5py.File(path_val, 'r') as f:
    states = f['states'][582, :, :, :] / 255.0
    actions = f['actions'][0, :]
    labels = f['labels'][582, :, :, :] / 255.0
    # st2 = f['states'][62, :, :, :] / 255.0
    # z = model.encode(states[np.newaxis, ...])
    z, _ = model.predict(states[np.newaxis, ...], actions[np.newaxis, ...], labels[np.newaxis, ...])
    x_pred = model.decode(z)
    # plt.imshow(states[:, :, :])
    plt.imshow(labels[:, :, :])
    plt.show()
    plt.imshow(x_pred[0, :, :, :])
    plt.show()
