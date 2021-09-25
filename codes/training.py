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
image_size = 64  # patch size
model_directory = './models'  # directory to save trained model to.
# training data in .mat format
train_data_name = ''
# validation data in .mat format
test_data_name = ''
restore = False  # load model or not
num_pyramids = 5  # number of pyramid levels
learning_rate = 3e-4  # learning rate
iterations = int(1e5)  # iterations
num_ms_channels = 8
num_pan_channels = 1

# GPU setting
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

##### Run them above #####
if __name__ == '__main__':

    tf.compat.v1.reset_default_graph()

    # loading data
    train_data = h5py.File(train_data_name)
    test_data = h5py.File(test_data_name)

    # placeholder for training
    gt = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                  train_batch_size, image_size, image_size, num_ms_channels])  # 64x64x64xnum_ms_channels
    lms = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                   train_batch_size, image_size, image_size, num_ms_channels])
    ms = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                  train_batch_size, image_size // 4, image_size // 4, num_ms_channels])
    pan = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                   train_batch_size, image_size, image_size, num_pan_channels])

    # placeholder for testing
    test_gt = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                       test_batch_size, image_size, image_size, num_ms_channels])
    test_lms = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                        test_batch_size, image_size, image_size, num_ms_channels])
    test_ms = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                       test_batch_size, image_size // 4, image_size // 4, num_ms_channels])
    test_pan = tf.compat.v1.placeholder(dtype=tf.float32, shape=[
                                        test_batch_size, image_size, image_size, num_pan_channels])

    # kernel generation
    ms_kernel_name = './kernels/ms_kernel.mat'
    ms_kernel = sio.loadmat(ms_kernel_name)
    ms_kernel = ms_kernel['ms_kernel'][...]
    ms_kernel = np.array(ms_kernel, dtype=np.float32)

    pan_kernel_name = './kernels/pan_kernel.mat'
    pan_kernel = sio.loadmat(pan_kernel_name)
    pan_kernel = pan_kernel['pan_kernel'][...]
    pan_kernel = np.array(pan_kernel, dtype=np.float32)

    # Gaussian pyramid for ground truth at （n） different scale
    gt_GaussianPyramid = GaussianPyramid(gt, ms_kernel, (num_pyramids - 1))
    # Gaussian pyramid for ground truth at （n） different scale
    test_gt_GaussianPyramid = GaussianPyramid(
        test_gt, ms_kernel, (num_pyramids - 1))

    output_pyramid = LPNet(pan, ms)  # PPN

    test_output_pyramid = LPNet(test_pan, test_ms)  # be aware of scope error

    loss1 = tf.reduce_mean(
        tf.square(output_pyramid[0] - gt_GaussianPyramid[0]))  # L2 loss
    loss2 = tf.reduce_mean(
        tf.square(output_pyramid[1] - gt_GaussianPyramid[1]))  # L2 loss
    loss3 = tf.reduce_mean(
        tf.square(output_pyramid[2] - gt_GaussianPyramid[2]))  # L2 loss
    loss4 = tf.reduce_mean(
        tf.square(output_pyramid[3] - gt_GaussianPyramid[3]))  # L2 loss
    loss5 = tf.reduce_mean(
        tf.square(output_pyramid[4] - gt_GaussianPyramid[4]))  # L2 loss

    mse = loss1 + loss2 + loss3 + loss4 + loss5

    test_loss1 = tf.reduce_mean(
        tf.square(test_output_pyramid[0] - test_gt_GaussianPyramid[0]))  # L2 loss
    test_loss2 = tf.reduce_mean(
        tf.square(test_output_pyramid[1] - test_gt_GaussianPyramid[1]))  # L2 loss
    test_loss3 = tf.reduce_mean(
        tf.square(test_output_pyramid[2] - test_gt_GaussianPyramid[2]))  # L2 loss
    test_loss4 = tf.reduce_mean(
        tf.square(test_output_pyramid[3] - test_gt_GaussianPyramid[3]))  # L2 loss
    test_loss5 = tf.reduce_mean(
        tf.square(test_output_pyramid[4] - test_gt_GaussianPyramid[4]))  # L2 loss

    test_mse = test_loss1 + test_loss2 + test_loss3 + test_loss4 + test_loss5

    g_optim = tf.compat.v1.train.AdamOptimizer(
        learning_rate).minimize(mse)  # Optimization method: SGD
    # tf.train.AdamOptimizer(learning_rate).minimize(mse)  # Optimization method: Adam
    test_g_optim = tf.compat.v1.train.AdamOptimizer(
        learning_rate).minimize(test_mse)  # Optimization method: SGD

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

        #### read training data #####
        gt1 = train_data['gt'][...]  # ground truth N*H*W*C
        pan1 = train_data['pan'][...]  # pan image N*H*W
        ms_lr1 = train_data['ms'][...]  # low resolution MS image
        lms1 = train_data['lms'][...]  # MS image interpolation to pan scale

        gt1 = np.array(gt1, dtype=np.float32) / \
            2047.  # normalization, WorldView L = 11
        pan1 = np.array(pan1, dtype=np.float32) / 2047.
        ms_lr1 = np.array(ms_lr1, dtype=np.float32) / 2047.
        lms1 = np.array(lms1, dtype=np.float32) / 2047.

        N = gt1.shape[0]

        #### read validation data #####
        gt2 = test_data['gt'][...]  # ground truth N*H*W*C
        pan2 = test_data['pan'][...]  # pan image N*H*W
        ms_lr2 = test_data['ms'][...]  # low resolution MS image
        lms2 = test_data['lms'][...]  # MS image interpolation -to pan scale

        gt2 = np.array(gt2, dtype=np.float32) / \
            2047.  # normalization, WorldView L = 11
        pan2 = np.array(pan2, dtype=np.float32) / 2047.
        ms_lr2 = np.array(ms_lr2, dtype=np.float32) / 2047.
        lms2 = np.array(lms2, dtype=np.float32) / 2047.
        N2 = gt2.shape[0]

        mse_train = []
        mse_valid = []

        start_time = datetime.datetime.now()
        file = open('train_time.txt', 'w')
        time_s = time.time()
        train_time = []
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
                # print, e.g.,: Iter: 0 MSE: 0.18406609
                print("Iter: " + str(i) + " MSE: " + str(Training_Loss))

            ###################################################################
            #### compute the mse of validation data ###########################
            bs_test = test_batch_size
            batch_index2 = np.random.randint(0, N2, size=bs_test)

            test_gt_batch = gt2[batch_index2, :, :, :]
            test_pan_batch = pan2[batch_index2, :, :, np.newaxis]
            test_ms_lr_batch = ms_lr2[batch_index2, :, :, :]
            test_lms_batch = lms2[batch_index2, :, :, :]

            # train_gt, train_lms, train_pan, train_ms = get_batch(train_data, bs = train_batch_size)
            # test_gt_batch, test_lms_batch, test_pan_batch, test_ms_batch = get_batch(test_data, bs=test_batch_size)
            _, Testing_Loss = sess.run([test_g_optim, test_mse], feed_dict={
                                       test_gt: test_gt_batch, test_lms: test_lms_batch, test_pan: test_pan_batch, test_ms: test_ms_lr_batch})
            mse_valid.append(Testing_Loss)  # record the mse of trainning

            if i % 1000 == 0 and i != 0:
                # print, e.g.,: Iter: 0 MSE: 0.18406609
                print("Iter: " + str(i) + " Valid MSE: " + str(Testing_Loss))

            if i % 5000 == 0 and i != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, model_directory +
                           '/model-' + str(i) + '.ckpt')
                time_e = time.time()
                print('time:' + str(time_e - time_s))
                train_time.append
                print("Save Model")

        file.write(str(train_time))
        file.close()
## finally write the mse info ##
        # write the training error into train_mse.txt
        file = open('train_mse.txt', 'w')
        file.write(str(mse_train))
        file.close()

        # write the valid error into valid_mse.txt
        file = open('valid_mse.txt', 'w')
        file.write(str(mse_valid))
        file.close()
