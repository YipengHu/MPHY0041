# This script uses TensorFlow-2
import random
import os

import tensorflow as tf
import numpy as np

import tf_utils as utils
from utils import get_image_arrays


os.environ["CUDA_VISIBLE_DEVICES"]="0"
RESULT_PATH = 'result'


## read all the data
images, test_images, test_indices = get_image_arrays()
num_data = images.shape[0]

## settings
weight_regulariser = 0.01
minibatch_size = 16
learning_rate = 1e-3
total_iterations = int(5e4+1)
freq_info_print = 500
freq_test_save = 5000

## network
reg_net = utils.UNet(num_channels_initial=32, out_channels=2)  # output ddfs in x,y two channels
reg_net.build((None,images.shape[1],images.shape[2],2))
optimizer = tf.optimizers.Adam(learning_rate)

## train step
@tf.function
def train_step(mov_images, fix_images):
    with tf.GradientTape() as tape:
        inputs = tf.stack([mov_images,fix_images],axis=3)
        ddfs = reg_net(inputs, training=True)
        pre_images = utils.warp_images(mov_images,ddfs)
        loss_similarity = tf.reduce_mean(utils.square_difference(fix_images, pre_images))
        loss_regularise = tf.reduce_mean(utils.gradient_norm(ddfs))
        loss = loss_similarity + loss_regularise*weight_regulariser
    gradients = tape.gradient(loss, reg_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, reg_net.trainable_variables))
    return loss, loss_similarity, loss_regularise

## test step
@tf.function
def test_step(mov_images, fix_images):
    inputs = tf.stack([mov_images,fix_images],axis=3)
    ddfs = reg_net(inputs, training=False)
    pre_images = utils.warp_images(mov_images,ddfs)
    loss_similarity = tf.reduce_mean(utils.square_difference(fix_images, pre_images))
    loss_regularise = tf.reduce_mean(utils.gradient_norm(ddfs))
    loss = loss_similarity + loss_regularise*weight_regulariser
    return loss, loss_similarity, loss_regularise, pre_images


## training
num_minibatch = int(num_data/minibatch_size/2)
train_indices = [i for i in range(num_data)]
for step in range(total_iterations):

    if step in range(0, total_iterations, num_minibatch):
        random.shuffle(train_indices)

    minibatch_idx = step % num_minibatch
    # random pairs
    indices_moving = train_indices[minibatch_idx*minibatch_size:(minibatch_idx+1)*minibatch_size]
    indices_fixed = train_indices[::-1][minibatch_idx*minibatch_size:(minibatch_idx+1)*minibatch_size]

    loss_train, loss_sim_train, loss_reg_train = train_step(
        mov_images=tf.convert_to_tensor(images[indices_moving,...]),
        fix_images=tf.convert_to_tensor(images[indices_fixed,...]))

    if step in range(0, total_iterations, freq_info_print):
        print('Step %d: Loss=%f (similarity=%f, regulariser=%f)' % (step, loss_train, loss_sim_train, loss_reg_train))
        print('  Moving-fixed image pair indices: %s - %s' % (indices_moving, indices_fixed))
    
    if step in range(0, total_iterations, freq_test_save):
        loss_test, loss_sim_test, loss_reg_test, pre_images_test = test_step(
            mov_images=tf.convert_to_tensor(test_images[test_indices[0],...]),
            fix_images=tf.convert_to_tensor(test_images[test_indices[1],...]))
        print('*** Test *** Step %d: Loss=%f (similarity=%f, regulariser=%f)' % (step, loss_test, loss_sim_test, loss_reg_test))
        filepath_to_save = os.path.join(RESULT_PATH, "test_step%06d-tf.npy" % step)
        np.save(filepath_to_save, pre_images_test)
        tf.print('Test data saved: {}'.format(filepath_to_save))

print('Training done.')


## save trained model
reg_net.save(os.path.join(RESULT_PATH,'saved_model_tf'))  # https://www.tensorflow.org/guide/keras/save_and_serialize
print('Model saved.')
