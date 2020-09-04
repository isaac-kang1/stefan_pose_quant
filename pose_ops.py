import numpy as np
import tensorflow as tf

def conv1d(input, out_c, k, s, name, padding='SAME', activation=None, th=1e-6, regularizer=None):
    with tf.variable_scope(name):
        w = _weight('weights', [k, input.get_shape()[2], out_c], regularizer=regularizer)
        b = _bias('bias', [out_c,])
        conv = tf.nn.conv1d(input, w, s, padding, name='conv') + b

        if activation is None:
            act_conv = conv
        elif activation == 'threshold':
            act_conv = tf.maximum(conv, th, name='conv')
        elif activation == 'relu':
            act_conv = tf.nn.relu(conv, name='conv')
        elif activation == 'lrelu':
            act_conv = tf.nn.leaky_relu(conv, name='conv')
        elif activation == 'relu6':
            act_conv = tf.nn.relu6(conv, name='conv')

        return act_conv


def pool1d(input, k, s, name, type='max', padding='VALID'):
    with tf.variable_scope(name):
        if type=='max':
            pool = tf.nn.max_pool1d(input, k, s, padding=padding, name='max_pool')

    return pool


def conv2d(input, out_c, k, s, name, padding='SAME', norm=None, activation=None, is_train=False, regularizer=None):
    with tf.variable_scope(name):
        w = _weight('weights', [k, k, input.get_shape()[3], out_c], regularizer=regularizer)
        b = _bias('bias', [out_c,])
        conv = tf.nn.conv2d(input, w, [1, s, s, 1], padding, name='conv') + b

        norm_conv = _norm(norm, conv, is_train=is_train)

        if activation is None:
            act_conv = norm_conv
        elif activation == 'relu':
            act_conv = tf.nn.relu(norm_conv, name='conv')
        elif activation == 'lrelu':
            act_conv = tf.nn.leaky_relu(norm_conv, name='conv')
        elif activation == 'relu6':
            act_conv = tf.nn.relu6(norm_conv, name='conv')
        elif activation == 'elu':
            act_conv = tf.nn.elu(norm_conv, name='conv')

    return act_conv


def pool(input, k, s, name, type='max', padding='SAME'):
    with tf.variable_scope(name):
        if type=='max':
            pool = tf.nn.max_pool(input, [1, k, k, 1], [1, s, s, 1], padding=padding, name='max_pool')
        elif type=='avg':
            pool = tf.nn.avg_pool(input, [1, k, k, 1], [1, s, s, 1], padding=padding, name='avg_pool')
        elif type=='globalavg':
            pool = tf.layers.average_pooling2d(input, k, s, padding=padding, name=name)

    return pool



def fc(input, out_c, name, norm=None, activation=None, is_train=False, regularizer=None):
    with tf.variable_scope(name):
        w = _weight('weights', [input.get_shape()[1], out_c], regularizer=regularizer)
        b = _bias('bias', [out_c])
        fc = tf.matmul(input, w) + b

        norm_fc = _norm(norm, fc, is_train=is_train)

        if activation is None:
            act_fc = fc
        elif activation == 'sigmoid':
            act_fc = tf.nn.sigmoid(fc, name='fc')
        elif activation == 'tanh':
            act_fc = tf.nn.tanh(fc, name='fc')
        elif activation == 'softmax':
            act_fc = tf.nn.softmax(fc, name='fc')
        elif activation == 'relu':
            act_fc = tf.nn.relu(fc, name='fc')
        elif activation == 'lrelu':
            act_fc = tf.nn.leaky_relu(fc, name='fc')
        elif activation == 'relu6':
            act_fc = tf.nn.relu6(fc, name='fc')

    return act_fc


def inception(input, c1, c3_r, c3, c3xx_r, c3xx, p3_r, name, pool_type='max', norm=None, activation=None):
    """ inception v2, batch normalization and factorization"""
    with tf.variable_scope(name):
        if c1 != 0:
            conv_1 = conv2d(input, c1, 1, 1, name='conv1', norm=norm, activation=activation)
        conv_3r = conv2d(input, c3_r, 1, 1, name='conv3_r', norm=norm, activation=activation)
        conv_3 = conv2d(conv_3r, c3, 3, 1, name='conv3', norm=norm, activation=activation)
        conv_3xx_r = conv2d(input, c3xx_r, 1, 1, name='conv3xx_r', norm=norm, activation=activation)
        conv_3xx_1 = conv2d(conv_3xx_r, c3xx, 3, 1, name='conv3xx_1', norm=norm, activation=activation)
        conv_3xx = conv2d(conv_3xx_1, c3xx, 3, 1, name='conv3xx', norm=norm, activation=activation)
        pool_3 = pool(input, 3, 1, name='pool3', type=pool_type)
        if p3_r != 0:
            pool_3r = conv2d(pool_3, p3_r, 1, 1, name='pool3r', norm=norm, activation=activation)
        # conv_5r = conv2d(input, c5_r, 1, 1, name='conv5_r', norm=norm)
        # conv_5 = conv2d(conv_5r, c5, 5, 1, name='conv5', norm=norm)
        # maxpool_3 = pool(input, 3, 2, name='maxpool3', norm=norm)
        # maxpool_3r = conv2d(maxpool_3, p3_r, 1, 1, name='maxpool3_r')

        if c1 != 0 and p3_r != 0:
            inception = tf.concat([conv_1, conv_3, conv_3xx, pool_3r], axis=-1)
        elif c1 == 0 and p3_r != 0:
            inception = tf.concat([conv_3, conv_3xx, pool_3r], axis=-1)
        elif c1 != 0 and p3_r == 0:
            inception = tf.concat([conv_1, conv_3, conv_3xx, pool_3], axis=-1)
        else:
            inception = tf.concat([conv_3, conv_3xx, pool_3], axis=-1)

        if activation == 'relu':
            inception = tf.nn.relu(inception, name='inception')
        elif activation == 'lrelu':
            inception = tf.nn.leaky_relu(inception, name='inception')
        elif activation == 'relu6':
            inception = tf.nn.relu6(inception, name='inception')

    return inception


def flatten(input, dim):
    return tf.reshape(input, [-1, dim])


def _weight(name, shape, regularizer=None):
    var = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=False), regularizer=regularizer)

    return var


def _bias(name, shape, constant=0.0):
    var = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(constant))

    return var


def dropout(x, keep_prob=None, noise_shape=None, seed=None, name=None, rate=None):
    return tf.nn.dropout(x, keep_prob=keep_prob, noise_shape=noise_shape, seed=seed, name=name, rate=rate)


def _norm(norm, x, is_train=True, group_num=32, momentum=0.99, epsilon=0.001):
    if norm is None:
        return x
    elif norm == 'batch':
        norm = tf.layers.batch_normalization(x, training=is_train, momentum=momentum, epsilon=epsilon, name='norm')
    elif norm == 'instance':
        norm = tf.contrib.layers.instance_norm(x, scope='instance_norm') #reuse..?how..?
    elif norm == 'group':
        norm = tf.contrib.layers.group_norm(x, group=group_num, scope='group_norm') #reuse..?
    elif norm == 'batch_instance':
        norm = batch_instance_norm(x, scope='batch_instance_norm')

    return norm


def batch_instance_norm(x, scope='batch_instance_norm'):
    with tf.variable_scope(scope):
        c = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
        x_batch = (x - batch_mean)/(tf.sqrt(tf.square(batch_sigma)+eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1,2], keep_dims=True)
        x_ins = (x - ins_mean)/(tf.sqrt(tf.square(ins_sigma)+eps))

        rho = tf.get_variable('rho', [c], initializer=tf.constant_initializer(1.0), constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.get_variable('gamma', [c], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [c], initializer=tf.constant_initializer(0.0))

        y = rho * x_batch + (1-rho) * x_ins
        y = gamma * y + beta

    return y
