import numpy as np
import h5py
import tensorflow as tf
import boto3
from models.autoencoder_env_model import AutoEncoderEnvironment
from models.vae_env_model import VAE
# from models.vae1_env_model import VAE1
from models.vae_actions import ActionsVAE


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

    return tf.concat((record['image_x'], record['image_y']), axis=-1), record['action'], tf.concat(
        (record['label_x'], record['label_y']), axis=-1)


def get_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(decode_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


### TRAIN ###
@tf.function
def compute_loss_de_en(model, x):
    z = model.infer(x)
    x_logit = model.decode(z)

    x_shape = tf.shape(x_logit)[0]

    x_logit = tf.reshape(x_logit, [x_shape, -1])
    x = tf.reshape(x, [x_shape, -1])

    loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(x, x_logit), axis=-1))

    # loss = tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)

    return loss


# @tf.function
# def compute_loss_all(model, x, a, y):
#     z = model.inference_net(x)
#     z = model.env_step(z, a)
#     x_pred = model.generative_net(z)
#
#     x_shape = tf.shape(x_pred)[0]
#
#     x_pred = tf.reshape(x_pred, [x_shape, -1])
#     y = tf.reshape(y, [x_shape, -1])
#
#     loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(y, x_pred), axis=-1))
#
#     # loss = tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)
#
#     return loss


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


@tf.function
def compute_loss_vae(model, x):
    mean, logvar = model.infer(x)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z, apply_sigmoid=True)

    x_pred = tf.keras.layers.Flatten()(x_pred)
    x = tf.keras.layers.Flatten()(x)

    log_px_z = -tf.reduce_sum(tf.math.squared_difference(x, x_pred), axis=-1)

    log_pz = log_normal_pdf(z, 0., 0.)
    log_qz_x = log_normal_pdf(z, mean, logvar)
    # print(log_px_z.shape, log_pz.shape, log_qz_x.shape)

    return -tf.reduce_mean(log_px_z + log_pz - log_qz_x)


# @tf.function
def compute_loss_actions(model, a):
    mean, logvar = model.infer(a)
    z = model.reparameterize(mean, logvar)
    a_pred = model.decode(z, apply_sigmoid=True)

    a_norm, _ = tf.linalg.normalize(a, axis=1)

    # a_pred = tf.keras.layers.Flatten()(a_pred)
    # a = tf.keras.layers.Flatten()(a)

    log_px_z = -tf.reduce_sum(tf.math.squared_difference(a, a_pred), axis=-1)
    # print(log_px_z)

    z_norm, _ = tf.linalg.normalize(z, axis=1)
    z_sqr = tf.matmul(z_norm, tf.transpose(z))
    a_sqr = tf.matmul(a_norm, tf.transpose(a_norm))

    # print(a_sqr)
    # print(z_sqr)
    dist = tf.reduce_sum(tf.math.abs(z_sqr - a_sqr) / tf.math.abs(a_sqr))

    # log_pz = log_normal_pdf(z, 0., 0.)
    # log_qz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(log_px_z) + dist
    # return -tf.reduce_mean(log_px_z + log_pz - log_qz_x) + dist


@tf.function
def compute_loss_vae_two_states(model, x, x_prev):
    mean, logvar = model.infer(x)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z, apply_sigmoid=True)

    x_pred = tf.keras.layers.Flatten()(x_pred)
    x = tf.keras.layers.Flatten()(x)

    log_px_z = -tf.reduce_sum(tf.math.squared_difference(x, x_pred), axis=-1)

    log_pz = log_normal_pdf(z, 0., 0.)
    log_qz_x = log_normal_pdf(z, mean, logvar)

    mean_prev, logvar_prev = model.infer(x_prev)
    log_qz_x_prev = log_normal_pdf(z, mean_prev, logvar_prev)

    # print(log_px_z.shape, log_pz.shape, log_qz_x.shape, log_qz_x_prev.shape)

    # return -tf.reduce_mean(log_px_z - log_qz_x + log_qz_x_prev)
    return -tf.reduce_mean(log_px_z + 0.00001 * log_pz - log_qz_x + 0.99999 * log_qz_x_prev)


@tf.function
def compute_loss_vae_env(model, x, a, y):
    mean, logvar = model.infer(x)
    z = model.reparameterize(mean, logvar)
    z = model.env_step(z, a)
    x_pred = model.decode(z, apply_sigmoid=True)

    x_pred = tf.keras.layers.Flatten()(x_pred)
    y = tf.keras.layers.Flatten()(y)

    loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(y, x_pred), axis=-1))

    # log_pz = log_normal_pdf(z, 0., 0.)
    # log_qz_x = log_normal_pdf(z, mean, logvar)
    # print(log_px_z.shape, log_pz.shape, log_qz_x.shape)

    return loss


# @tf.function
# def compute_loss_vae(model, x, a, y):
#     z = model.encode(x)
#     # z_pred = model.env_step(z, a)
#     rv_x = model.decode(z)
#
#     print(tf.reduce_sum(rv_x.log_prob(x)))
#     return -tf.reduce_sum(rv_x.log_prob(x))


@tf.function
def compute_apply_gradients_enc_dec(model, x, optimizer):
    model.inference_net.trainable = True
    model.generative_net.trainable = True
    model.lat_env_net.trainable = False

    with tf.GradientTape() as tape:
        loss = compute_loss_de_en(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def compute_apply_gradients_vae_env(model, x, a, y, optimizer):
    model.inference_net.trainable = False
    model.generative_net.trainable = False
    model.lat_env_net.trainable = True

    with tf.GradientTape() as tape:
        loss = compute_loss_vae_env(model, x, a, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# @tf.function
# def compute_apply_gradients_all(model, x, a, y, optimizer):
#     with tf.GradientTape() as tape:
#         loss = compute_loss_all(model, x, a, y)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#     return loss


@tf.function
def compute_apply_gradients_vae(model, x, optimizer):
    model.lat_env_net.trainable = False

    with tf.GradientTape() as tape:
        loss = compute_loss_vae(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# @tf.function
def compute_apply_gradients_actions(model, x, optimizer):
    model.lat_env_net.trainable = False

    with tf.GradientTape() as tape:
        loss = compute_loss_actions(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def compute_apply_gradients_vae_two_states(model, x, x_prev, optimizer):
    model.lat_env_net.trainable = False

    with tf.GradientTape() as tape:
        loss = compute_loss_vae_two_states(model, x,  x_prev)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train_one_step(model, train_dataset, optimizer, epoch_loss, mode, batch_size):

    for i, (train_X, train_A, train_Y) in train_dataset.enumerate():
        if mode == 'ed':
            loss = compute_apply_gradients_enc_dec(model, train_X, optimizer)
            # strategy.experimental_run_v2(compute_apply_gradients_enc_dec, args=(model, train_X, optimizer))
        elif mode == 'le':
            loss = compute_apply_gradients_vae_env(model, train_X, train_A, train_Y, optimizer)
        elif mode == 'vae':
            loss = compute_apply_gradients_vae(model, train_X, optimizer)
        elif mode == 'vae+':
            loss = compute_apply_gradients_vae_two_states(model, train_Y, train_X, optimizer)
        elif mode == 'act':
            loss = compute_apply_gradients_actions(model, train_A, optimizer)
        else:
            raise ValueError

        # loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        epoch_loss.update_state(loss)
        if i % 50 == 0:
            print(i.numpy() * batch_size, epoch_loss.result().numpy())
            epoch_loss.reset_states()

    train_loss = epoch_loss.result()
    epoch_loss.reset_states()

    return train_loss


def test_one_step(model, test_dataset, epoch_loss, mode):
    for test_X, test_A, test_Y in test_dataset:
        if mode == 'ed':
            loss = compute_loss_de_en(model, test_X)
        # strategy.experimental_run_v2(compute_loss_de_en, args=(model, test_X))
        elif mode == 'le':
            loss = compute_loss_vae_env(model, test_X, test_A, test_Y)
        elif mode == 'vae':
            loss = compute_loss_vae(model, test_X)
        elif mode == 'vae+':
            loss = compute_loss_vae_two_states(model, test_Y, test_X)
        elif mode == 'act':
            loss = compute_loss_actions(model, test_A)
        else:
            raise ValueError

        epoch_loss.update_state(loss)

    test_loss = epoch_loss.result()
    epoch_loss.reset_states()

    return test_loss


def train(model, epochs, path_tr, path_val, path_weights, mode):

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)  # TODO: incorporate into a class
    epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')

    train_dataset = get_dataset(path_tr)
    train_dataset = train_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset)

    test_dataset = get_dataset(path_val)
    test_dataset = test_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_step(model, train_dataset, optimizer, epoch_loss, mode, BATCH_SIZE)
        test_loss = test_one_step(model, test_dataset, epoch_loss, mode)

        print('Epoch', epoch, 'train loss:', train_loss.numpy(), 'validation loss:', test_loss.numpy())

        if mode == 'ed':
            model.save_weights(['en', 'de'], path_weights, epoch % 3)
        elif mode == 'le':
            model.save_weights(['le'], path_weights, epoch % 3)
        elif mode == 'vae' or mode == 'vae+' or mode == 'act':
            model.save_weights(['en', 'de'], path_weights, epoch % 3)
        else:
            raise ValueError

        # s3.meta.client.upload_file('/tmp/weights/cp-de-{}.ckpt'.format(epoch), BUCKET, '/weights/cp-de-{}.ckpt'.format(epoch))


if __name__ == "__main__":

    print(tf.__version__)
    # train in the cloud
    CLOUD = True

    epochs = 10
    TRAIN_BUF = 2048
    BATCH_SIZE = 128
    TEST_BUF = 2048

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
        path_tr = '/Users/dgrebenyuk/Research/dataset/training2.tfrecord'
        path_val = '/Users/dgrebenyuk/Research/dataset/validation2.tfrecord'

    # testing distributed training
    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # BATCH_SIZE_PER_REPLICA = 128
    # BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    print(tf.config.experimental.list_physical_devices())

    # with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(1e-4)
    # model = AutoEncoderEnvironment(256)
    # model = VAE(256)
    model = ActionsVAE(256)
    # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')

    # print(model.inference_net.summary(), '\n', model.generative_net.summary())
    # print('\n', model.lat_env_net.summary())

    # with strategy.scope():
    if CLOUD:
        # path_weights = '/tmp/weights'
        path_weights = '/tmp/weights/act'
    else:
        # path_weights = '/Users/dgrebenyuk/Research/dataset/weights'
        path_weights = '/Users/dgrebenyuk/Research/dataset/weights/act'

    # 'en' - encoder; 'de' - decoder; 'le' - latent environment
    model.load_weights(['en', 'de'], path_weights)

    # 'ed' - encoder-decoder; 'le' - latent environment
    # train(model, epochs, path_tr, path_val, 'le')
    train(model, epochs, path_tr, path_val, path_weights, 'act')
