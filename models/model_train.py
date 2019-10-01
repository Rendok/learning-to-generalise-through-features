import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
# import tensorflow_io as tfio
import boto3
from gym_kuka_multi_blocks.datasets.generate_data_set import get_dataset
from models.autoencoder_env_model import AutoEncoderEnvironment

@tf.function
def compute_loss_de_en(model, x):
    z = model.encode(x)
    x_logit = model.decode(z)

    x_shape = tf.shape(x_logit)[0]

    x_logit = tf.reshape(x_logit, [x_shape, -1])
    x = tf.reshape(x, [x_shape, -1])

    loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(x, x_logit), axis=-1))

    # loss = tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)

    return loss


@tf.function
def compute_loss_env(model, x, a, y):
    x_pred, label = model.predict(x, a, y)

    loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(label, x_pred), axis=-1))

    # loss = tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)

    return loss


@tf.function
def compute_apply_gradients_enc_dec(model, x, optimizer):
    model.lat_env_net.trainable = False

    with tf.GradientTape() as tape:
        loss = compute_loss_de_en(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def compute_apply_gradients_env(model, x, a, y, optimizer):
    model.inference_net.trainable = False
    model.generative_net.trainable = False

    with tf.GradientTape() as tape:
        loss = compute_loss_env(model, x, a, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train(model, epochs, path_tr, path_val, mode):
    def train_one_step(model, train_dataset, test_dataset, mode):
        for i, (train_X, train_A, train_Y) in train_dataset.take(1).enumerate():
            if mode == 'ed':
                loss = compute_apply_gradients_enc_dec(model, train_X, optimizer)
                # strategy.experimental_run_v2(compute_apply_gradients_enc_dec, args=(model, train_X, optimizer))
            elif mode == 'le':
                loss = compute_apply_gradients_env(model, train_X, train_A, train_Y, optimizer)
            else:
                raise ValueError

            # loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            epoch_loss.update_state(loss)
            if i % 50 == 0:
                print(i.numpy() * BATCH_SIZE, epoch_loss.result().numpy())
                epoch_loss.reset_states()

        train_loss = epoch_loss.result()
        epoch_loss.reset_states()

        for test_X, test_A, test_Y in test_dataset.take(1):
            if mode == 'ed':
                loss = compute_loss_de_en(model, test_X)
            # strategy.experimental_run_v2(compute_loss_de_en, args=(model, test_X))
            elif mode == 'le':
                loss = compute_loss_env(model, test_X, test_A, test_Y)
            else:
                raise ValueError

            epoch_loss.update_state(loss)

        test_loss = epoch_loss.result()
        epoch_loss.reset_states()

        return train_loss, test_loss

    train_dataset = get_dataset(path_tr)
    train_dataset = train_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = get_dataset(path_val)
    test_dataset = test_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # new distribute part
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    for epoch in range(1, epochs + 1):
        train_loss, test_loss = train_one_step(model, train_dataset, test_dataset, mode)

        print('Epoch', epoch, 'train loss:', train_loss.numpy(), 'validation loss:', test_loss.numpy())

        if CLOUD:
            if mode == 'ed':
                model.save_weights(['en', 'de'], '/tmp/weights', epoch % 3)
            elif mode == 'le':
                model.save_weights(['le'], '/tmp/weights', epoch % 3)
        else:
            if mode == 'ed':
                model.save_weights(['en', 'de'], '/Users/dgrebenyuk/Research/dataset/weights', epoch % 3)
            elif mode == 'le':
                model.save_weights(['le'], '/Users/dgrebenyuk/Research/dataset/weights', epoch % 3)

        # s3.meta.client.upload_file('/tmp/weights/cp-de-{}.ckpt'.format(epoch), BUCKET, '/weights/cp-de-{}.ckpt'.format(epoch))


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
    model = AutoEncoderEnvironment()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')

    # print(model.inference_net.summary())
    # print('\n', model.generative_net.summary())
    # print('\n', model.env_net.summary())

    # with strategy.scope():
    if CLOUD:
        path_weights = '/tmp/weights'
    else:
        path_weights = '/Users/dgrebenyuk/Research/dataset/weights'

    # 'en' - encoder; 'de' - decoder; 'le' - latent environment
    model.load_weights(['en', 'de'], path_weights)
    # print('Latest checkpoints:', path_weights)

    train(model, epochs, path_tr, path_val, 'ed')
