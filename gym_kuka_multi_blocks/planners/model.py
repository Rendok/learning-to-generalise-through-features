import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers


class IMP(object):

    def __init__(self,
                 img_w=84,  # 64
                 img_h=84,  # 64
                 img_c=9,  # 6
                 act_dim=4,  # 2
                 inner_horizon=5,
                 outer_horizon=2,
                 num_plan_updates=5,
                 conv_params=[(40, 3, 2, 'VALID'), (40, 3, 2, 'VALID'), (40, 3, 2, 'VALID')],
                 n_hidden=2,
                 obs_latent_dim=128,
                 act_latent_dim=64,
                 meta_gradient_clip_value=10,
                 if_huber=True,
                 delta_huber=1.,
                 bt_num_units=10,
                 bias_transform=False,
                 nonlinearity='swish',
                 spatial_softmax=False,
                 env=None):

        ######## INITIALIZE THE INPUTS #########

        # Input Parameters
        self.img_w = img_w
        self.img_h = img_h
        self.act_dim = act_dim
        self.inner_horizon = inner_horizon
        self.outer_horizon = outer_horizon
        self.num_plan_updates = num_plan_updates
        self.conv_params = conv_params
        self.spatial_softmax = spatial_softmax
        self.n_hidden = n_hidden
        self.obs_latent_dim = obs_latent_dim
        self.act_latent_dim = act_latent_dim
        self.meta_gradient_clip_value = meta_gradient_clip_value
        self.if_huber = if_huber
        self.delta_huber = delta_huber
        self.bias_transform = bias_transform
        self.nonlinearity = nonlinearity
        self.bt_num_units = bt_num_units

        self._env = env

        # Input Hyperparameter Placeholders
        self.il_lr_0 = tf.placeholder(tf.float32, name='il_lr_0')
        self.il_lr = tf.placeholder(tf.float32, name='il_lr')
        self.ol_lr = tf.placeholder(tf.float32, name='ol_lr')

        # Input Tensor Placeholders
        self.o0 = tf.placeholder(tf.float32, [None, img_w, img_h, img_c], name='o0')
        self.atT_original = tf.placeholder(tf.float32, [None, inner_horizon, act_dim], name='atT_0')
        self.og = tf.placeholder(tf.float32, [None, img_w, img_h, img_c], name='og')
        # self.atT_target = tf.placeholder(tf.float32, [None, inner_horizon, act_dim], name='atT_target')
        # self.qt = tf.placeholder(tf.float32, [None, joint_dim], name='qt')
        # self.plan_loss_mask = tf.placeholder(tf.float32, [None, inner_horizon], name='mask')
        self.eff_horizons = tf.placeholder(tf.int32, [None], name='eff_horizons')

        # Copy placeholder for repeated plan updates
        self.atT = self.atT_original

        self._env_model()
        self._planer_model()

    def _planer_model(self):

        #######  BUILD THE COMPUTATIONAL GRAPH #######

        # Part 1: Encode the observation and the goal in a latent space, in two stages -
        # first in conv layer and then in fully connected layer

        with tf.variable_scope('gradplanner'):
            x0 = self._encode_conv(self.o0,
                                   self.conv_params,
                                   scope='obs_conv_encoding',
                                   layer_norm=True,
                                   nonlinearity='swish',
                                   spatial_softmax=self.spatial_softmax,
                                   reuse=False)

            x0 = self._encode_fc(x0,
                                 n_hidden=self.n_hidden,
                                 scope='obs_fc_encoding',
                                 layer_norm=True,
                                 latent_dim=self.obs_latent_dim,
                                 nonlinearity=self.nonlinearity,
                                 reuse=False)

            xg = self._encode_conv(self.og,
                                   self.conv_params,
                                   scope='obs_conv_encoding',
                                   layer_norm=True,
                                   nonlinearity='swish',
                                   spatial_softmax=self.spatial_softmax,
                                   reuse=True)

            xg = self._encode_fc(xg,
                                 n_hidden=self.n_hidden,
                                 scope='obs_fc_encoding',
                                 layer_norm=True,
                                 latent_dim=self.obs_latent_dim,
                                 nonlinearity=self.nonlinearity,
                                 reuse=True)

            # xt = tf.concat([xt, self.qt], axis=1)

            if self.bias_transform:
                bias_transform = tf.get_variable('bias_transform',
                                                 [1, self.bt_num_units],
                                                 initializer=tf.constant_initializer(0.1))

                bias_transform = tf.tile(bias_transform, multiples=tf.stack([tf.shape(x0)[0], 1]))

                x0 = tf.concat([x0, bias_transform], 1)

            # encode obs vector in latent space
            x0 = self._fully_connected(x0,
                                       scope='joint_encoding',
                                       out_dim=self.obs_latent_dim,
                                       nonlinearity=self.nonlinearity,
                                       layer_norm=True,
                                       reuse=False)

            # Part 2: Encode the action in the latent space if action_encode is set to True, else just keep it as it is.
            # New variable is utT. Implemented as a component in Part 3

            # Part 3: Roll out the latent plan over the planning horizon

            self._rollout_plan_in_latent_space(x0,
                                               xg,
                                               self.eff_horizons,
                                               # self.atT,
                                               self.il_lr_0,
                                               self.il_lr,
                                               num_plan_updates=self.num_plan_updates,
                                               horizon=self.inner_horizon,
                                               scope='rollout',
                                               act_dim=self.act_dim,
                                               obs_latent_dim=self.obs_latent_dim,
                                               act_latent_dim=self.act_latent_dim,
                                               nonlinearity=self.nonlinearity,
                                               if_huber=self.if_huber,
                                               delta_huber=self.delta_huber,
                                               meta_gradient_clip_value=self.meta_gradient_clip_value,
                                               layer_norm=True,
                                               reuse=False)

    def _env_model(self):

        with tf.variable_scope('env'):

            x0 = self._encode_conv(self.o0,
                                   self.conv_params,
                                   scope='obs_conv_encoding',
                                   layer_norm=True,
                                   nonlinearity='swish',
                                   spatial_softmax=self.spatial_softmax,
                                   reuse=False)

            x0 = self._encode_fc(x0,
                                 n_hidden=self.n_hidden,
                                 scope='obs_fc_encoding',
                                 layer_norm=True,
                                 latent_dim=self.obs_latent_dim,
                                 nonlinearity=self.nonlinearity,
                                 reuse=False)

            xg = self._encode_conv(self.og,
                                   self.conv_params,
                                   scope='obs_conv_encoding',
                                   layer_norm=True,
                                   nonlinearity='swish',
                                   spatial_softmax=self.spatial_softmax,
                                   reuse=True)

            xg = self._encode_fc(xg,
                                 n_hidden=self.n_hidden,
                                 scope='obs_fc_encoding',
                                 layer_norm=True,
                                 latent_dim=self.obs_latent_dim,
                                 nonlinearity=self.nonlinearity,
                                 reuse=True)

            if self.bias_transform:
                bias_transform = tf.get_variable('bias_transform',
                                                 [1, self.bt_num_units],
                                                 initializer=tf.constant_initializer(0.1))

                bias_transform = tf.tile(bias_transform, multiples=tf.stack([tf.shape(x0)[0], 1]))

                x0 = tf.concat([x0, bias_transform], 1)

            # encode obs vector in latent space
            x0 = self._fully_connected(x0,
                                       scope='joint_encoding',
                                       out_dim=self.obs_latent_dim,
                                       nonlinearity=self.nonlinearity,
                                       layer_norm=True,
                                       reuse=False)

            utT = self._encode_plan(self.atT,
                                    scope='plan_encoding',
                                    latent_dim=self.act_latent_dim,
                                    horizon=self.inner_horizon,
                                    act_dim=self.act_dim,
                                    nonlinearity='swish',
                                    reuse=False)

            utT = tf.reshape(utT, [-1, self.act_latent_dim])

            # Computational graph for the gradient
            xt_preds = self._fully_connected(tf.concat([x0, utT], axis=1),
                                             out_dim=self.obs_latent_dim,
                                             scope='dynamics',
                                             nonlinearity=self.nonlinearity,
                                             reuse=False)

            if self.if_huber:
                loss = tf.reduce_sum(
                    tf.losses.huber_loss(xg, xt_preds, delta=self.delta_huber, reduction="none"),
                    reduction_indices=[1])
            else:
                loss = tf.reduce_sum(tf.square(xg - xt_preds), reduction_indices=[1])

            env_cost = tf.reduce_mean(loss)

            # Part 5: Training and Diagnostics Ops

            optimizer = tf.train.AdamOptimizer(self.ol_lr)
            self.train_op = optimizer.minimize(env_cost)

            # self.get_inner_loss_op = self.plan_loss
            self.get_outer_loss_op = env_cost
            # self.get_outer_loss_first_step_op = bc_loss_one_step  # useful for testtime diagnostics
            self.get_plan_op = self.atT
            self.get_xt = x0
            self.get_xg = xg

    def _rollout_plan_in_latent_space(self,
                                      x0,
                                      xg,
                                      eff_horizons,
                                      # atT,
                                      il_lr_0,
                                      il_lr,
                                      num_plan_updates=5,
                                      horizon=5,
                                      scope='rollout',
                                      obs_latent_dim=16,
                                      act_latent_dim=16,
                                      act_dim=2,
                                      nonlinearity='swish',
                                      if_huber=True,
                                      delta_huber=1.,
                                      meta_gradient_clip_value=10,
                                      encode_action=True,
                                      layer_norm=True,
                                      reuse=False):

        """
        Plan updater
        """

        with tf.variable_scope(scope, reuse=reuse):
            for update_idx in range(num_plan_updates):
                xg_pred = x0
                xg_preds = []

                if update_idx == 0:
                    plan_encode_scope_reuse = False
                else:
                    plan_encode_scope_reuse = True

                utT = self._encode_plan(self.atT,
                                        scope='plan_encoding',
                                        latent_dim=act_latent_dim,
                                        horizon=horizon,
                                        act_dim=act_dim,
                                        nonlinearity='swish',
                                        reuse=plan_encode_scope_reuse)

                for time_idx in range(0, horizon):
                    if time_idx >= 1 or update_idx >= 1:
                        dynamics_scope_reuse = True
                    else:
                        dynamics_scope_reuse = False

                    # Computational graph for the gradient
                    xg_pred = self._fully_connected(tf.concat([xg_pred, utT[:, time_idx, :]], axis=1),
                                                    out_dim=obs_latent_dim,
                                                    scope='dynamics',
                                                    nonlinearity=nonlinearity,
                                                    reuse=dynamics_scope_reuse)

                    xg_preds.append(xg_pred)

                xg_preds = tf.convert_to_tensor(xg_preds)  # horizon x batch_size x obs_latent_dim
                xg_preds = tf.transpose(xg_preds, [1, 0, 2])  # batch_size x horizon x obs_latent_dim
                xg_pred = tf.gather_nd(xg_preds, tf.concat(
                    [tf.expand_dims(tf.range(tf.shape(xg_preds)[0]), 1), tf.expand_dims(eff_horizons, 1)], 1))

                if if_huber:
                    self.plan_loss = tf.reduce_sum(
                        tf.losses.huber_loss(xg, xg_pred, delta=delta_huber, reduction="none"),
                        reduction_indices=[1])
                else:
                    self.plan_loss = tf.reduce_sum(tf.square(xg_pred - xg), reduction_indices=[1])

                self.plan_loss = tf.reduce_mean(self.plan_loss)

                atT_grad = tf.gradients(self.plan_loss, self.atT)[0]
                atT_grad = tf.clip_by_value(atT_grad, -meta_gradient_clip_value, meta_gradient_clip_value)

                if update_idx == 0:
                    self.atT = self.atT - il_lr_0 * atT_grad
                else:
                    self.atT = self.atT - il_lr * atT_grad

                # print('update', update_idx)

            self.xg_preds = xg_preds  # batch_size x horizon x obs_latent_dim

    def get_plan(self, o0, og, eff_horizons, atT_original, il_lr_0, il_lr, sess):
        print('plan loss', sess.run(self.plan_loss,
                                    feed_dict={self.o0: o0,
                                               self.og: og,
                                               self.eff_horizons: eff_horizons,
                                               self.atT_original: atT_original,
                                               self.il_lr_0: il_lr_0,
                                               self.il_lr: il_lr}))
        return sess.run(self.atT,
                        feed_dict={self.o0: o0,
                                   self.og: og,
                                   self.eff_horizons: eff_horizons,
                                   self.atT_original: atT_original,
                                   self.il_lr_0: il_lr_0,
                                   self.il_lr: il_lr})

    def _encode_conv(self,
                     x,
                     conv_params,
                     scope='obs_conv_encoding',
                     layer_norm=False,
                     nonlinearity='swish',
                     spatial_softmax=False,
                     reuse=False):

        with tf.variable_scope(scope, reuse=reuse):

            out = x

            for num_outputs, kernel_size, stride, padding in conv_params:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           activation_fn=None)

                if layer_norm is True:
                    # out = layers.layer_norm(out, center=True, scale=True)
                    out = layers.layer_norm(out)
                    # Apply the non-linearity after layer-norm

                if nonlinearity == 'swish':
                    out = tf.nn.sigmoid(out) * out  # swish non-linearity
                elif nonlinearity == 'relu':
                    out = tf.nn.relu(out)

            if spatial_softmax:
                shape = tf.shape(out)
                static_shape = out.shape
                height, width, num_channels = shape[1], shape[2], static_shape[3]
                pos_x, pos_y = tf.meshgrid(tf.linspace(-1., 1., num=height),
                                           tf.linspace(-1., 1., num=width),
                                           indexing='ij')
                pos_x = tf.reshape(pos_x, [height * width])
                pos_y = tf.reshape(pos_y, [height * width])
                out = tf.reshape(tf.transpose(out, [0, 3, 1, 2]), [-1, height * width])
                softmax_attention = tf.nn.softmax(out)
                expected_x = tf.reduce_sum(pos_x * softmax_attention, [1], keep_dims=True)
                expected_y = tf.reduce_sum(pos_y * softmax_attention, [1], keep_dims=True)
                expected_xy = tf.concat([expected_x, expected_y], 1)
                feature_keypoints = tf.reshape(expected_xy, [-1, num_channels.value * 2])
                feature_keypoints.set_shape([None, num_channels.value * 2])
                return feature_keypoints
            else:
                out = layers.flatten(out)  # flatten the conv output
                return out

    def _encode_fc(self,
                   x,
                   scope='obs_fc_encoding',
                   n_hidden=2,
                   layer_norm=True,
                   latent_dim=16,
                   nonlinearity='swish',
                   reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            out = x

            for _ in range(n_hidden):
                out = layers.fully_connected(out,
                                             num_outputs=latent_dim,
                                             activation_fn=None)
                if layer_norm is True:
                    out = layers.layer_norm(out, center=True, scale=True)
                if nonlinearity == 'swish':
                    out = tf.nn.sigmoid(out) * out
                elif nonlinearity == 'relu':
                    out = tf.nn.relu(out)

            return out

    def _encode_plan(self,
                     plan,
                     scope='plan_encoding',
                     n_hidden=1,
                     nonlinearity='swish',
                     latent_dim=16,
                     horizon=5,
                     act_dim=2,
                     layer_norm=True,
                     reuse=False):

        # encode it into a plan
        with tf.variable_scope(scope, reuse=reuse):
            out = plan
            out = tf.reshape(out, [-1, act_dim])
            # out = tf.reshape(out, [-1, horizon*act_dim])
            for _ in range(n_hidden):
                out = layers.fully_connected(out,
                                             num_outputs=latent_dim,
                                             # num_outputs=latent_dim*horizon,
                                             activation_fn=None)

                if layer_norm is True:
                    out = layers.layer_norm(out, center=True, scale=True)
                if nonlinearity == 'swish':
                    out = tf.nn.sigmoid(out) * out
                if nonlinearity == 'relu':
                    out = tf.nn.relu(out)
                if nonlinearity == 'tanh':
                    out = tf.nn.tanh(out)

            out = tf.reshape(out, [-1, horizon, latent_dim])
            return out

    def _fully_connected(self,
                         x,
                         scope='fully_connected',
                         nonlinearity='swish',
                         out_dim=16,
                         layer_norm=True,
                         reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            out = x
            out = layers.fully_connected(out,
                                         num_outputs=out_dim,
                                         activation_fn=None)
            if layer_norm is True:
                out = layers.layer_norm(out, center=True, scale=True)
            if nonlinearity == 'swish':
                out = tf.nn.sigmoid(out) * out
            elif nonlinearity == 'relu':
                out = tf.nn.relu(out)
            return out

    def _conv_variable(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    @property
    def trainable_vars(self):
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gradplanner")
        # weights = dict(zip([var.name for var in weights], weights))
        return weights

    def train(self, labels, inputs, actions, eff_horizons, il_lr_0, il_lr, ol_lr, sess):

        sess.run(self.train_op,
                 feed_dict={self.o0: inputs,
                            self.og: labels,
                            self.eff_horizons: eff_horizons,
                            self.atT_original: actions,
                            self.il_lr_0: il_lr_0,
                            self.il_lr: il_lr,
                            self.ol_lr: ol_lr})

    def stats(self, o0, og, eff_horizons, atT_original, il_lr_0, il_lr, sess):

        bc_loss, xg_pred, xg = sess.run(
            [self.get_outer_loss_op, self.get_xt, self.get_xg],
            feed_dict={self.o0: o0,
                       self.og: og,
                       self.eff_horizons: eff_horizons,
                       self.atT_original: atT_original,
                       self.il_lr_0: il_lr_0,
                       self.il_lr: il_lr})

        return np.sqrt(bc_loss), xg_pred, xg

    def plan(self, o0, og, eff_horizons, atT_original, il_lr_0, il_lr, sess):

        plan = sess.run(self.get_plan_op,
                        feed_dict={self.o0: o0,
                                   self.og: og,
                                   self.eff_horizons: eff_horizons,
                                   self.atT_original: atT_original,
                                   self.il_lr_0: il_lr_0,
                                   self.il_lr: il_lr})
        return plan

    def get_latent_embedding(self, o0, og, sess):
        latent_emb, goal_emb = sess.run([self.get_xt, self.get_xg], feed_dict={self.o0: o0, self.og: og})
        return latent_emb, goal_emb

    def get_trainable_params(self, sess):
        weight_values = sess.run(self.trainable_vars)
        weight_names = [var.name for var in self.trainable_vars]
        weight_dump = dict(zip(weight_names, weight_values))
        return weight_dump


# Potentially an old code-check test. So, might not work. But this isn't relevant. It was just for debugging purposes.
if __name__ == '__main__':
    with tf.Session() as sess:
        # `sess.graph` provides access to the graph used in a `tf.Session`.
        writer = tf.summary.FileWriter("/Users/dgrebenyuk/Research/rl-task-planning", sess.graph)

        writer.close()

    # horizon = 5
    # otg = np.random.uniform(0, 1, size=(2000, 6, 84, 84, 3))
    # atT_expert = np.random.uniform(0., 1, size=(2000 * horizon * 4)).reshape((2000, horizon, 4))
    # num_plan_updates = 5
    #
    # eff_horizons = [5] * 32
    #
    #
    # def batch_sample(otg, atT, batch_size=32, max_horizon=5):

    #     total_num = otg.shape[0]
    #     batch_ot = np.zeros((batch_size, otg.shape[2], otg.shape[3], otg.shape[4]))
    #     batch_og = np.zeros((batch_size, otg.shape[2], otg.shape[3], otg.shape[4]))
    #     batch_atT = np.zeros((batch_size, max_horizon, atT.shape[-1]))
    #     batch_mask = np.ones((batch_size, max_horizon))
    #
    #     for idx in range(batch_size):
    #         t1, t2 = 0, 0
    #         while t1 == t2:
    #             t1, t2 = np.random.randint(0, max_horizon + 1, 2)
    #         t1, t2 = min(t1, t2), max(t1, t2)
    #         effective_horizon = t2 - t1
    #         assert effective_horizon <= max_horizon
    #         sample_idx = np.random.randint(0, total_num)
    #         batch_ot[idx, :, :, :] = otg[sample_idx, t1, :, :, :]
    #         batch_og[idx, :, :, :] = otg[sample_idx, t2, :, :, :]
    #         batch_atT[idx, :effective_horizon, :] = atT[sample_idx, t1:t2, :]
    #         if effective_horizon < max_horizon:
    #             batch_mask[idx, effective_horizon:] = 0
    #
    #     batch_atT_original = np.random.uniform(-2., 2.,
    #                                            size=(batch_size * horizon * 4)).reshape((batch_size, horizon, 4))
    #
    #     return batch_ot, batch_og, batch_atT, batch_mask, batch_atT_original
    #
    #
    # il_lr_0 = 0.5
    # il_lr = 0.1
    # ol_lr = 0.005
    # meta_gradient_clip_value = 5
    #
    # sess = tf.Session()
    # imp = IMP(img_c=3, outer_horizon=5, meta_gradient_clip_value=meta_gradient_clip_value)
    # sess.run(tf.global_variables_initializer())
    #
    # # Just check if loss decreases
    # for num_iter in range(1000):
    #     batch_ot, batch_og, batch_atT_target, batch_mask, batch_atT_original = batch_sample(otg, atT_expert)
    #     imp.train(batch_ot,
    #               batch_og,
    #               eff_horizons,
    #               batch_atT_original,
    #               batch_atT_target,
    #               batch_mask,
    #               il_lr_0,
    #               il_lr,
    #               ol_lr,
    #               sess)
    #
    #     bc_loss, plan_loss, xg_pred, xg, bc_loss_first_step = imp.stats(batch_ot,
    #                                                                     batch_og,
    #                                                                     batch_atT_original,
    #                                                                     batch_atT_target,
    #                                                                     batch_mask,
    #                                                                     il_lr_0,
    #                                                                     il_lr,
    #                                                                     sess)
    #
    #     print("Iter", num_iter, "BC Loss", bc_loss, "Plan Loss", plan_loss)
