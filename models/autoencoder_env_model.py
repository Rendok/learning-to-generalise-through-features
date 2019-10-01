import tensorflow as tf
import numpy as np


def make_inference_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(128, 128, 6)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu',
                                  kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(latent_dim, activation='tanh',
                                  kernel_initializer=tf.keras.initializers.glorot_normal(seed=None))
        ]
    )
    return model

def make_generative_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(512, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8192, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Reshape(target_shape=(4, 4, 512)),
            tf.keras.layers.Conv2DTranspose(
                filters=512, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=2, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(1, 1), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
                kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2DTranspose(
                filters=6, kernel_size=4, strides=(1, 1), activation='sigmoid', padding='SAME',
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
        ]
    )
    return model
#
#
# def make_inference_net(latent_dim):
#     model = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(128, 128, 6)),
#             tf.keras.layers.Conv2D(
#                 filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Conv2D(
#                 filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Conv2D(
#                 filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Conv2D(
#                 filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Conv2D(
#                 filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(512, activation='relu',
#                                   kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(latent_dim, activation='tanh',
#                                   kernel_initializer=tf.keras.initializers.glorot_normal(seed=None))
#         ], name="encoder"
#     )
#     return model
#
#
# def make_generative_net(latent_dim):
#     model = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#             tf.keras.layers.Dense(512, activation='relu',
#                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dense(4096, activation='relu',
#                                   kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Reshape(target_shape=(4, 4, 256)),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=256, kernel_size=2, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=6, kernel_size=3, strides=(2, 2), activation='relu', padding='SAME',
#                 kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
#         ], name="decoder"
#     )
#     return model


def make_latent_env_net(latent_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim + 4)),
            tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2 * latent_dim, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(2 * latent_dim, activation='relu',
            #                       kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(2 * latent_dim, activation='relu',
            #                       kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(2 * latent_dim, activation='relu',
            #                       kernel_initializer=tf.keras.initializers.he_normal(seed=None)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(latent_dim, activation='tanh',
                                  kernel_initializer=tf.keras.initializers.glorot_normal(seed=None)),
        ], name="dynamics"
    )
    return model


class AutoEncoderEnvironment(tf.keras.Model):
    def __init__(self, latent_dim=256):
        super().__init__()

        self._latent_dim = latent_dim

        self.inference_net = make_inference_net(self._latent_dim)
        self.generative_net = make_generative_net(self._latent_dim)
        self.lat_env_net = make_latent_env_net(self._latent_dim)

    def encode(self, x):
        return self.inference_net(x)

    def decode(self, z):
        return self.generative_net(z)

    @tf.function
    def env_forward(self, x, a):
        # a = a[tf.newaxis, ...]
        z = tf.concat([x, a], axis=1)
        pred = self.lat_env_net(z)
        return pred

    @tf.function
    def predict(self, x, a, y):
        x = self.inference_net(x)
        pred = self.env_forward(x, a)
        label = self.inference_net(y)
        return pred, label

    def save_weights(self, nets, file_path, number):
        for ch in nets:
            latest = file_path + "/" + ch + "/cp-{}.ckpt".format(number)
            if ch == 'encoder' or ch == 'en':
                self.inference_net.save_weights(latest)
            elif ch == 'decoder' or ch == 'de':
                self.generative_net.save_weights(latest)
            elif ch == 'lat_env' or ch == 'le':
                self.lat_env_net.save_weights(latest)
            else:
                raise ValueError

    def load_weights(self, nets, file_path):
        for ch in nets:
            latest = tf.train.latest_checkpoint(file_path + "/" + ch)
            if ch == 'encoder' or ch == 'en':
                self.inference_net.load_weights(latest)
            elif ch == 'decoder' or ch == 'de':
                self.generative_net.load_weights(latest)
            elif ch == 'lat_env' or ch == 'le':
                self.lat_env_net.load_weights(latest)
            else:
                raise ValueError

    #@tf.function
    def roll_out_plan(self, x0, actions):
        all_preds = tf.TensorArray(tf.float32, actions.shape[0])

        x_pred = self.encode(x0[np.newaxis, ...])
        for i in tf.range(actions.shape[0]):
            x_pred = self.env_forward(x_pred, actions[i, ...])
            all_preds.write(i, self.decode(x_pred))

        return x_pred, all_preds.stack()

    def plan(self, x0, xg, horizon, epochs):
        actions = tf.convert_to_tensor(np.random.randn(horizon, 4).astype(np.float32))

        for i in range(epochs):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actions)
                x_pred, all_x = self.roll_out_plan(x0, actions)
                zg = self.encode(xg[np.newaxis, ...])
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
