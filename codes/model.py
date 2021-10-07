# ---------------------------------------------------------------
# Copyright (c) 2021, Cheng Jin, Liang-Jian Deng, Ting-Zhu Huang,
# Gemine Vivone, All rights reserved.
#
# This work is licensed under GNU Affero General Public License 
# v3.0 International To view a copy of this license, see the 
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird and GaoFen-2), please change the corresponding
# inputs.
# ---------------------------------------------------------------

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly

pyramid_num = 5  # number of pyramid levels
recursive_block_num = 5  # number of recursive blocks
feature_num = 64  # number of feature maps, 64 for WorldView-3 datasets, 128 for QuickBird and GaoFen-2 datasets
ms_channels_num = 8  # channel number of multispectral images, 8 for WorldView-3 datasets, 4 for QuickBird and
# GaoFen-2 datasets
concat_channels_num = 9
weight_decay = 1e-5

tf.compat.v1.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # set CUDA devices for multi GPU training


# Laplacian and Gaussian Pyramid

def lap_split(img, kernel):
    with tf.name_scope('split'):
        low = tf.nn.conv2d(img, kernel, [1, 2, 2, 1], 'SAME')
        low_upsample = tf.nn.conv2d_transpose(
            low, kernel * 4, tf.shape(img), [1, 2, 2, 1])
        high = img - low_upsample
    return low, high


def LaplacianPyramid(img, kernel, n):
    levels = []
    for i in range(n):
        img, high = lap_split(img, kernel)
        levels.append(high)
    levels.append(img)
    return levels[::-1]


def GaussianPyramid(img, kernel, n):
    levels = []
    low = img  
    levels.append(img)
    for i in range(n):
        low = tf.nn.conv2d(low, kernel, [1, 2, 2, 1], 'SAME')
        levels.append(low)
    return levels[::-1]


# create kernel
def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
    new_variables = tf.compat.v1.get_variable(name=name, shape=shape, initializer=initializer, regularizer=regularizer)
    return new_variables


# sub-network
def subnet(pan, ms, feature_num):
    kernel0 = create_kernel(name='weights_0', shape=[
        3, 3, concat_channels_num, feature_num])  
    biases0 = tf.Variable(tf.constant(
        0.0, shape=[feature_num], dtype=tf.float32), trainable=True, name='biases_0')

    kernel1 = create_kernel(name='weights_1', shape=[
        3, 3, feature_num, feature_num]) 
    biases1 = tf.Variable(tf.constant(
        0.0, shape=[feature_num], dtype=tf.float32), trainable=True, name='biases_1')

    kernel2 = create_kernel(name='weights_2', shape=[
        3, 3, feature_num, feature_num])  
    biases2 = tf.Variable(tf.constant(
        0.0, shape=[feature_num], dtype=tf.float32), trainable=True, name='biases_2')

    kernel3 = create_kernel(name='weights_3', shape=[
        3, 3, feature_num, feature_num])  
    biases3 = tf.Variable(tf.constant(
        0.0, shape=[feature_num], dtype=tf.float32), trainable=True, name='biases_3')

    kernel4 = create_kernel(name='weights_4', shape=[
        3, 3, feature_num, ms_channels_num])  
    biases4 = tf.Variable(tf.constant(
        0.0, shape=[ms_channels_num], dtype=tf.float32), trainable=True, name='biases_4')

    # Concat layer
    with tf.compat.v1.variable_scope('concat_layer'):
        rs = tf.concat([pan, ms], axis=3)  

    #  1st layer
    with tf.compat.v1.variable_scope('1st_layer'):
        conv0 = tf.nn.conv2d(rs, kernel0, [1, 1, 1, 1], padding='SAME')
        bias0 = tf.nn.bias_add(conv0, biases0)
        bias0 = tf.nn.relu(bias0) 

        out_block = bias0

    for i in range(recursive_block_num):
        with tf.compat.v1.variable_scope('block_%s' % (i + 1)):
            conv1 = tf.nn.conv2d(out_block, kernel1, [
                1, 1, 1, 1], padding='SAME')
            bias1 = tf.nn.bias_add(conv1, biases1)
            bias1 = tf.nn.relu(bias1)

            conv2 = tf.nn.conv2d(bias1, kernel2, [1, 1, 1, 1], padding='SAME')
            bias2 = tf.nn.bias_add(conv2, biases2)
            bias2 = tf.nn.relu(bias2)

            conv3 = tf.nn.conv2d(bias2, kernel3, [1, 1, 1, 1], padding='SAME')
            bias3 = tf.nn.bias_add(conv3, biases3)
            bias3 = tf.nn.relu(bias3)

            out_block = tf.add(bias3, bias0) 

    #  reconstruction layer
    with tf.compat.v1.variable_scope('recons'):
        conv = tf.nn.conv2d(out_block, kernel4, [1, 1, 1, 1], padding='SAME')
        recons = tf.nn.bias_add(conv, biases4)

        final_out = tf.add(recons, ms) 

    return final_out


# LPPN structure
def LPPN(pan, ms):
    with tf.compat.v1.variable_scope('LPPN', reuse=tf.compat.v1.AUTO_REUSE):
        # kernel generation
        ms_kernel_name = './kernels/ms_kernel.mat'  # read the corresponding multispectral kernel (WorldView-3
        # (7x7x8x8), QuickBird and GaoFen-2 (7x7x4x4))
        ms_kernel = sio.loadmat(ms_kernel_name)
        ms_kernel = ms_kernel['ms_kernel'][...]
        Lap_kernel_ms = np.array(ms_kernel, dtype=np.float32)

        pan_raw_kernel_name = './kernels/pan_kernel.mat'   # read the corresponding panchromatic kernel (WorldView-3),
        # QuickBird and GaoFen-2 (7x7x1x1)
        pan_raw_kernel = sio.loadmat(pan_raw_kernel_name)
        pan_raw_kernel = pan_raw_kernel['pan_kernel'][...]
        pan_raw_kernel = np.array(pan_raw_kernel, dtype=np.float32)
        Lap_kernel_pan = pan_raw_kernel[:, :, np.newaxis, np.newaxis]

        # alignment of image spatial dimension, upsample multispectral image 4 times
        ms = ly.conv2d_transpose(ms, 8, 8, 4, activation_fn=None,
                                 weights_initializer=ly.variance_scaling_initializer(),
                                 weights_regularizer=ly.l2_regularizer(weight_decay)) # for WorldView-3 datasets
        
        # ms = ly.conv2d_transpose(ms, 4, 8, 4, activation_fn=None,
        #                          weights_initializer=ly.variance_scaling_initializer(),
        #                          weights_regularizer=ly.l2_regularizer(weight_decay)) # for QuickBird, GaoFen-2 datasets

        pan_pyramid = LaplacianPyramid(pan, Lap_kernel_pan,
                                       (pyramid_num - 1))
        ms_pyramid = LaplacianPyramid(ms, Lap_kernel_ms,
                                      (pyramid_num - 1))

        # subnet 1
        with tf.compat.v1.variable_scope('subnet1'):
            out1 = subnet(pan_pyramid[0], ms_pyramid[0],
                          int((feature_num) / 32))
            out1 = tf.nn.relu(out1)
            out1_t = tf.nn.conv2d_transpose(
                out1, Lap_kernel_ms * 4, tf.shape(ms_pyramid[1]), [1, 2, 2, 1])

        # subnet 2
        with tf.compat.v1.variable_scope('subnet2'):
            out2 = subnet(pan_pyramid[1], ms_pyramid[1],
                          int((feature_num) / 16))
            out2 = tf.add(out2, out1_t)
            out2 = tf.nn.relu(out2)
            out2_t = tf.nn.conv2d_transpose(
                out2, Lap_kernel_ms * 4, tf.shape(ms_pyramid[2]), [1, 2, 2, 1])

        # subnet 3
        with tf.compat.v1.variable_scope('subnet3'):
            out3 = subnet(pan_pyramid[2], ms_pyramid[2],
                          int((feature_num) / 8))
            out3 = tf.add(out3, out2_t)
            out3 = tf.nn.relu(out3)
            out3_t = tf.nn.conv2d_transpose(
                out3, Lap_kernel_ms * 4, tf.shape(ms_pyramid[3]), [1, 2, 2, 1])

        # subnet 4
        with tf.compat.v1.variable_scope('subnet4'):
            out4 = subnet(pan_pyramid[3], ms_pyramid[3],
                          int((feature_num) / 4))
            out4 = tf.add(out4, out3_t)
            out4 = tf.nn.relu(out4)
            out4_t = tf.nn.conv2d_transpose(
                out4, Lap_kernel_ms * 4, tf.shape(ms_pyramid[4]), [1, 2, 2, 1])

        # subnet 5
        with tf.compat.v1.variable_scope('subnet5'):
            out5 = subnet(pan_pyramid[4], ms_pyramid[4],
                          int((feature_num) / 2))
            out5 = tf.add(out5, out4_t)
            out5 = tf.nn.relu(out5)

        output_pyramid = []
        output_pyramid.append(out1)
        output_pyramid.append(out2)
        output_pyramid.append(out3)
        output_pyramid.append(out4)
        output_pyramid.append(out5)

        return output_pyramid


if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()
    input_pan = tf.Variable(tf.random.normal(
        [8806, 64, 64, 1]), trainable=False)
    input_ms = tf.Variable(tf.random.normal(
        [8806, 16, 16, 8]), trainable=False)

    output_pyramid = LPPN(input_pan, input_ms)
    var_list = tf.compat.v1.trainable_variables()
    print("Total parameters' number: %d"
          % (np.sum([np.prod(v.get_shape().as_list()) for v in var_list])))
