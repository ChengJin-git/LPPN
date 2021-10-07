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
import h5py
import scipy.io as sio
import time
import numpy as np
import tensorflow as tf
import datetime
from model import GaussianPyramid, LPNet


tf.compat.v1.reset_default_graph()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_batch_size = 32  # training batch size
test_batch_size = 32  # validation batch size
image_size = 64
model_directory = './models'  # directory to save trained model to.
# training data in .mat format
train_data_name = ''
# validation data in .mat format
test_data_name = ''
restore = False  # load pretrained model
pyramid_num = 5  # number of pyramid levels
learning_rate = 3e-4  # learning rate
iterations = int(1e5)  # iterations
ms_channels_num = 8
pan_channel_num = 1

# GPU setting
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

#Run them above
if __name__ == '__main__':

    tf.compat.v1.reset_default_graph()

    # loading data
    train_data = h5py.File(train_data_name)
    test_data = h5py.File(test_data_name)

    # placeholder for training
    gt = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                  train_batch_size, image_size, image_size, ms_channels_num])  
    lms = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                   train_batch_size, image_size, image_size, ms_channels_num])
    ms = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                  train_batch_size, image_size // 4, image_size // 4, ms_channels_num])
    pan = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                   train_batch_size, image_size, image_size, pan_channel_num])

    # placeholder for testing
    test_gt = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                       test_batch_size, image_size, image_size, ms_channels_num])
    test_lms = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                        test_batch_size, image_size, image_size, ms_channels_num])
    test_ms = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                       test_batch_size, image_size // 4, image_size // 4, ms_channels_num])
    test_pan = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                        test_batch_size, image_size, image_size, pan_channel_num])

    # kernel generation
    ms_kernel_name = './kernels/WV3_ms_kernel.mat' # corresponding multispectral MTF kernel
    ms_kernel = sio.loadmat(ms_kernel_name)
    ms_kernel = ms_kernel['ms_kernel'][...]
    ms_kernel = np.array(ms_kernel, dtype=np.float32)

    pan_kernel_name = './kernels/WV3_pan_kernel.mat' # corresponding panchromatic MTF kernel
    pan_kernel = sio.loadmat(pan_kernel_name)
    pan_kernel = pan_kernel['pan_kernel'][...]
    pan_kernel = np.array(pan_kernel, dtype=np.float32)

    gt_GaussianPyramid = GaussianPyramid(gt, ms_kernel, (pyramid_num - 1))
    test_gt_GaussianPyramid = GaussianPyramid(
        test_gt, ms_kernel, (pyramid_num - 1))

    output_pyramid = LPPN(pan, ms)
    test_output_pyramid = LPPN(test_pan, test_ms) 

    loss1 = tf.reduce_mean(
        tf.square(output_pyramid[0] - gt_GaussianPyramid[0])) 
    loss2 = tf.reduce_mean(
        tf.square(output_pyramid[1] - gt_GaussianPyramid[1]))  
    loss3 = tf.reduce_mean(
        tf.square(output_pyramid[2] - gt_GaussianPyramid[2]))  
    loss4 = tf.reduce_mean(
        tf.square(output_pyramid[3] - gt_GaussianPyramid[3]))  
    loss5 = tf.reduce_mean(
        tf.square(output_pyramid[4] - gt_GaussianPyramid[4]))  

    mse = loss1 + loss2 + loss3 + loss4 + loss5

    test_loss1 = tf.reduce_mean(
        tf.square(test_output_pyramid[0] - test_gt_GaussianPyramid[0]))  
    test_loss2 = tf.reduce_mean(
        tf.square(test_output_pyramid[1] - test_gt_GaussianPyramid[1]))  
    test_loss3 = tf.reduce_mean(
        tf.square(test_output_pyramid[2] - test_gt_GaussianPyramid[2]))  
    test_loss4 = tf.reduce_mean(
        tf.square(test_output_pyramid[3] - test_gt_GaussianPyramid[3])) 
    test_loss5 = tf.reduce_mean(
        tf.square(test_output_pyramid[4] - test_gt_GaussianPyramid[4]))

    test_mse = test_loss1 + test_loss2 + test_loss3 + test_loss4 + test_loss5

    g_optim = tf.compat.v1.train.AdamOptimizer(
        learning_rate).minimize(mse)  
    test_g_optim = tf.compat.v1.train.AdamOptimizer(
        learning_rate).minimize(test_mse)  

    all_vars = tf.compat.v1.trainable_variables()
    saver = tf.compat.v1.train.Saver(var_list=all_vars, max_to_keep=100)
    init = tf.compat.v1.global_variables_initializer()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session() as sess:
        sess.run(init)

        if restore:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # read training data
        gt1 = train_data['gt'][...]  
        pan1 = train_data['pan'][...]  
        ms_lr1 = train_data['ms'][...]  
        lms1 = train_data['lms'][...]  

        gt1 = np.array(gt1, dtype=np.float32) / \
            2047.  # normalization, WorldView L = 11
        pan1 = np.array(pan1, dtype=np.float32) / 2047.
        ms_lr1 = np.array(ms_lr1, dtype=np.float32) / 2047.
        lms1 = np.array(lms1, dtype=np.float32) / 2047.

        N = gt1.shape[0]

        # read validation data
        gt2 = test_data['gt'][...]  
        pan2 = test_data['pan'][...] 
        ms_lr2 = test_data['ms'][...]  
        lms2 = test_data['lms'][...]  

        gt2 = np.array(gt2, dtype=np.float32) / \
            2047.  # normalization, WorldView L = 11
        pan2 = np.array(pan2, dtype=np.float32) / 2047.
        ms_lr2 = np.array(ms_lr2, dtype=np.float32) / 2047.
        lms2 = np.array(lms2, dtype=np.float32) / 2047.
        N2 = gt2.shape[0]

        mse_train = []
        mse_valid = []

        for i in range(iterations):
            bs = train_batch_size
            batch_index = np.random.randint(0, N, size=bs)

            train_gt = gt1[batch_index, :, :, :]
            pan_batch = pan1[batch_index, :, :, np.newaxis]
            ms_lr_batch = ms_lr1[batch_index, :, :, :]
            train_lms = lms1[batch_index, :, :, :]

            # train_gt, train_lms, train_pan_hp, train_ms_hp = get_batch(train_data, bs = train_batch_size)
            _, Training_Loss = sess.run([g_optim, mse],
                                        feed_dict={gt: train_gt, lms: train_lms,
                                                   ms: ms_lr_batch,
                                                   pan: pan_batch})  # training

            mse_train.append(Training_Loss)  # record the mse of training

            if i % 100 == 0 and i != 0:
                print("Iter: " + str(i) + " MSE: " + str(Training_Loss))


            # compute the mse of validation data
            bs_test = test_batch_size
            batch_index2 = np.random.randint(0, N2, size=bs_test)

            test_gt_batch = gt2[batch_index2, :, :, :]
            test_pan_batch = pan2[batch_index2, :, :, np.newaxis]
            test_ms_lr_batch = ms_lr2[batch_index2, :, :, :]
            test_lms_batch = lms2[batch_index2, :, :, :]

            _, Testing_Loss = sess.run([test_g_optim, test_mse], feed_dict={
                                       test_gt: test_gt_batch, test_lms: test_lms_batch, test_pan: test_pan_batch, test_ms: test_ms_lr_batch})

            mse_valid.append(Testing_Loss)  # record the mse of trainning

            if i % 1000 == 0 and i != 0:
                print("Iter: " + str(i) + " Valid MSE: " + str(Testing_Loss))

            if i % 5000 == 0 and i != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, model_directory +
                           '/model-' + str(i) + '.ckpt')
                print("Save Model")

        # write the mse info
        # write the training error into train_mse.txt
        file = open('train_mse.txt', 'w')
        file.write(str(mse_train))
        file.close()

        # write the valid error into valid_mse.txt
        file = open('valid_mse.txt', 'w')
        file.write(str(mse_valid))
        file.close()
