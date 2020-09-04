import tensorflow as tf
from pose_ops import fc
from pose_ops import conv2d
from pose_ops import pool

#=================
#     MODEL
#=================

NORM=None

def CNN_VIEW(x, name='view_n_ptcld', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name + '_CNN_VIEW', reuse=reuse):
        x = conv2d(x, 64, 3, 1, name='conv1_1', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = conv2d(x, 64, 3, 1, name='conv1_2', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = pool(x, 2, 2, name='pool1', type='max')
        x = conv2d(x, 128, 3, 1, name='conv2_1', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = conv2d(x, 128, 3, 1, name='conv2_2', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = pool(x, 2, 2, name='pool2', type='max')
        x = conv2d(x, 256, 3, 1, name='conv3_1', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = conv2d(x, 256, 3, 1, name='conv3_2', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = pool(x, 2, 2, name='pool3', type='max')
        x = conv2d(x, 512, 3, 1, name='conv4_1', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = conv2d(x, 512, 3, 1, name='conv4_2', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = pool(x, 2, 2, name='pool4', type='max')
        x = conv2d(x, 512, 3, 1, name='conv5_1', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = conv2d(x, 512, 3, 1, name='conv5_2', norm=NORM, activation='relu', is_train=is_train, regularizer=regularizer)
        x = pool(x, 2, 2, name='pool5', type='max')
    return x
def FC_VIEW(x, name='view_n_ptcld', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name + '_FC_VIEW', reuse=reuse):
        x = tf.layers.flatten(x)
        x = fc(x, 4096, 'fc1', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, 64, 'fc2', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
    return x
def FC_PTCLD(x, name='view_n_ptcld', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name + '_FC_PTCLD', reuse=reuse):
        x = tf.layers.flatten(x)
        x = fc(x, 64, 'fc1', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, 64, 'fc2', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
    return x
def FC_VIEW_PTCLD(x, y, name='view_n_ptcld', reuse=False, is_train=False, regularizer=None):
    with tf.variable_scope(name + '_FC_VIEW_PTCLD', reuse=reuse):
        x = tf.concat((x, y), -1)
        x = fc(x, 64, 'fc1', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, 64, 'fc2', norm=None, activation='relu', is_train=is_train, regularizer=regularizer)
        x = fc(x, 7, 'fc3', norm=None, activation=None, is_train=is_train, regularizer=regularizer)
    return x
def MODEL(input_view, input_ptcld, name='view_n_ptcld', is_train=False, regularizer=None):
    """
    
    view - CNN_VIEW - FC_VIEW \
                                FC_VIEW_PTCLD (7)
             ptcld - FC_PTCLD /

    """
    image_feature = CNN_VIEW(input_view, name=name, reuse=tf.AUTO_REUSE, is_train=is_train, regularizer=regularizer)
    image_feature = FC_VIEW(image_feature, name=name, reuse=tf.AUTO_REUSE, is_train=is_train, regularizer=regularizer)
    ptcld_feature = FC_PTCLD(input_ptcld, name=name, reuse=tf.AUTO_REUSE, is_train=is_train, regularizer=regularizer)
    rt = FC_VIEW_PTCLD(image_feature, ptcld_feature, name=name, reuse=tf.AUTO_REUSE, is_train=is_train, regularizer=regularizer) # quaternion(4), translation(3)
    return rt
