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
import model
import time

tf.reset_default_graph()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':

    test_data = './test_data/WorldView-3.mat'
    model_directory = './pretrained/WorldView-3'

    tf.reset_default_graph()

    data = sio.loadmat(test_data)

    # data normalization

    ms = data['ms'][...]  # MS image
    ms = np.array(ms, dtype=np.float32) / 2047. # 2047 for WorldView-3, QuickBird datasets, 1023 for GaoFen-2 datasets
    ms = ms[np.newaxis, :, :, :]

    lms = data['lms'][...]  # up-sampled LRMS image
    lms = np.array(lms, dtype=np.float32) / 2047. # 2047 for WorldView-3, QuickBird datasets, 1023 for GaoFen-2 datasets
    lms = lms[np.newaxis, :, :, :]

    pan = data['pan'][...]  # pan image
    pan = np.array(pan, dtype=np.float32) / 2047. # 2047 for WorldView-3, QuickBird datasets, 1023 for GaoFen-2 datasets
    pan = pan[np.newaxis, :, :, np.newaxis]

    h = pan.shape[1]  # height
    w = pan.shape[2]  # width

    # placeholder for tensor
    pan_p = tf.placeholder(shape=[1, h, w, 1], dtype=tf.float32)
    ms_p = tf.placeholder(shape=[1, h / 4, w / 4, 8], dtype=tf.float32)
    lms_p = tf.placeholder(shape=[1, h, w, 8], dtype=tf.float32)

    output_pyramid = model.LPPN(pan_p, ms_p)


    output  = tf.clip_by_value(output_pyramid[4], 0, 1)  # final output


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # loading  model
        if tf.train.get_checkpoint_state(model_directory):
            ckpt = tf.train.latest_checkpoint(model_directory)
            saver.restore(sess, ckpt)
            print("load new model")

        else:
            ckpt = tf.train.get_checkpoint_state(model_directory + "pre-trained/")
            saver.restore(sess,
                          ckpt.model_checkpoint_path)
            print("load pre-trained model")
        start_time = time.time()
        final_output = sess.run(output, feed_dict={pan_p: pan, lms_p: lms, ms_p: ms})
        end_time = time.time()
        print('running time: ', end_time-start_time)
        sio.savemat('./result/output_LPPN.mat', {'output_LPPN': final_output[0, :, :, :]})
