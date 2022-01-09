# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
# This is a tutorial using TensorFlow 2.x, in particular, low-level TF APIs without high-level Keras
import os
import random

import tensorflow as tf
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"]="0"
path_to_data = './data/datasets-promise12'
RESULT_PATH = './result' 

## Define functions for network layers
def conv3d(input, filters, downsample=False, activation=True, batch_norm=False):
    if downsample: strides = [1,2,2,2,1]
    else: strides = [1,1,1,1,1]
    y = tf.nn.conv3d(input, filters, strides=strides, padding='SAME')
    if batch_norm: y = batch_norm(y)
    if activation: y = tf.nn.relu(y)
    return y  # where bn can be added

def resnet_block(input, filters, batch_norm=False):
    y = conv3d(input, filters[..., 0])
    y = conv3d(y, filters[..., 1], activation=False) + input
    if batch_norm: y = batch_norm(y)
    return tf.nn.relu(y)  # where bn can be added

def downsample_maxpool(input, filters):
    y = conv3d(input, filters)
    return tf.nn.max_pool3d(y, ksize=[1,3,3,3,1], padding='SAME', strides=[1,2,2,2,1])

def deconv3d(input, filters, out_shape, batch_norm=False):
    y = tf.nn.conv3d_transpose(input, filters, output_shape=out_shape, strides=[1,2,2,2,1], padding='SAME')
    if batch_norm: y = batch_norm(y)
    return tf.nn.relu(y)  # where bn can be added

def batch_norm(inputs, is_training, decay = 0.999):
    # This is where to insert the implementation of batch normalisaiton
    return inputs

def add_variable(var_shape, var_list, var_name=None, initialiser=None):
    if initialiser is None:
        initialiser = tf.initializers.glorot_normal()
    if var_name is None:
        var_name = 'var{}'.format(len(var_list))
        var_list.append(tf.Variable(initialiser(var_shape), name=var_name, trainable=True))
    return var_list

## Define a model (a 3D U-Net variant) with residual layers with trinable weights
# ref: https://arxiv.org/abs/1512.03385  & https://arxiv.org/abs/1505.04597
num_channels = 32
nc = [num_channels*(2**i) for i in range(4)]
var_list=[]
# intial-layer
var_list = add_variable([5,5,5,1,nc[0]], var_list)
# encoder-s0
var_list = add_variable([3,3,3,nc[0],nc[0],2], var_list)
var_list = add_variable([3,3,3,nc[0],nc[0],2], var_list)
var_list = add_variable([3,3,3,nc[0],nc[0]], var_list)
var_list = add_variable([3,3,3,nc[0],nc[1]], var_list)
# encoder-s1
var_list = add_variable([3,3,3,nc[1],nc[1],2], var_list)
var_list = add_variable([3,3,3,nc[1],nc[1],2], var_list)
var_list = add_variable([3,3,3,nc[1],nc[1]], var_list)
var_list = add_variable([3,3,3,nc[1],nc[2]], var_list)
# encoder-s2
var_list = add_variable([3,3,3,nc[2],nc[2],2], var_list)
var_list = add_variable([3,3,3,nc[2],nc[2],2], var_list)
var_list = add_variable([3,3,3,nc[2],nc[2]], var_list)
var_list = add_variable([3,3,3,nc[2],nc[3]], var_list)
# deep-layers-s3
var_list = add_variable([3,3,3,nc[3],nc[3],2], var_list)
var_list = add_variable([3,3,3,nc[3],nc[3],2], var_list)
var_list = add_variable([3,3,3,nc[3],nc[3],2], var_list)
# decoder-s2
var_list = add_variable([3,3,3,nc[2],nc[3]], var_list)
var_list = add_variable([3,3,3,nc[2],nc[2],2], var_list)
var_list = add_variable([3,3,3,nc[2],nc[2],2], var_list)
# decoder-s1
var_list = add_variable([3,3,3,nc[1],nc[2]], var_list)
var_list = add_variable([3,3,3,nc[1],nc[1],2], var_list)
var_list = add_variable([3,3,3,nc[1],nc[1],2], var_list)
# decoder-s0
var_list = add_variable([3,3,3,nc[0],nc[1]], var_list)
var_list = add_variable([3,3,3,nc[0],nc[0],2], var_list)
var_list = add_variable([3,3,3,nc[0],nc[0],2], var_list)
# output-layer
var_list = add_variable([3,3,3,nc[0],1], var_list)

## model with corresponding layers
@tf.function
def residual_unet(input):
    # initial-layer
    skip_layers = []
    layer = conv3d(input, var_list[0])
    # encoder-s0
    layer = resnet_block(layer, var_list[1])
    layer = resnet_block(layer, var_list[2])
    skip_layers.append(layer)
    layer = downsample_maxpool(layer, var_list[3])
    layer = conv3d(layer, var_list[4])
    # encoder-s1
    layer = resnet_block(layer, var_list[5])
    layer = resnet_block(layer, var_list[6])
    skip_layers.append(layer)
    layer = downsample_maxpool(layer, var_list[7])
    layer = conv3d(layer, var_list[8])
    # encoder-s2
    layer = resnet_block(layer, var_list[9])
    layer = resnet_block(layer, var_list[10])
    skip_layers.append(layer)
    layer = downsample_maxpool(layer, var_list[11])
    layer = conv3d(layer, var_list[12])
    # deep-layers-s3
    layer = resnet_block(layer, var_list[13])
    layer = resnet_block(layer, var_list[14])
    layer = resnet_block(layer, var_list[15])
    # decoder-s2
    layer = deconv3d(layer, var_list[16], skip_layers[2].shape) + skip_layers[2]
    layer = resnet_block(layer, var_list[17])
    layer = resnet_block(layer, var_list[18])
    # decoder-s1
    layer = deconv3d(layer, var_list[19], skip_layers[1].shape) + skip_layers[1]
    layer = resnet_block(layer, var_list[20])
    layer = resnet_block(layer, var_list[21])
    # decoder-s0
    layer = deconv3d(layer, var_list[22], skip_layers[0].shape) + skip_layers[0]
    layer = resnet_block(layer, var_list[23])
    layer = resnet_block(layer, var_list[24])
    # output-layer
    layer = tf.sigmoid(conv3d(layer, var_list[25], activation=False))
    return layer

## loss function
def loss_dice(pred, target, eps=1e-6):
    dice_numerator = 2 * tf.reduce_sum(pred*target, axis=[1,2,3,4])
    dice_denominator = eps + tf.reduce_sum(pred, axis=[1,2,3,4]) + tf.reduce_sum(target, axis=[1,2,3,4])
    return  1 - tf.reduce_mean(dice_numerator/dice_denominator)


## npy data loader class
class DataReader:
    def __init__(self, folder_name):
        self.folder_name = folder_name
    def load_images_train(self, indices_mb):
        return self.load_npy_files(["image_train%02d.npy" % idx for idx in indices_mb])
    def load_images_test(self, indices_mb):
        return self.load_npy_files(["image_test%02d.npy" % idx for idx in indices_mb])
    def load_labels_train(self, indices_mb):
        return self.load_npy_files(["label_train%02d.npy" % idx for idx in indices_mb])
    def load_npy_files(self, file_names):
        images = [np.float32(np.load(os.path.join(self.folder_name, fn))) for fn in file_names]
        return np.expand_dims(np.stack(images, axis=0), axis=4)[:, ::2, ::2, ::2, :]


## training
@tf.function
def train_step(model, weights, optimizer, x, y):
    with tf.GradientTape() as tape:
        # g_tape.watched(var_list): trainable variables are automatically "watched".
        loss = loss_dice(model(x), y)
    gradients = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(gradients, weights))
    return loss

# optimisation configuration
learning_rate = 1e-4
total_iter = int(2e5)
freq_print = 100  # in iteration
freq_test = 2000  # in iteration
n = 50  # 50 training image-label pairs
size_minibatch = 4

num_minibatch = int(n/size_minibatch)  # how many minibatches in each epoch
indices_train = [i for i in range(n)]
DataFeeder = DataReader(path_to_data)
optimizer = tf.optimizers.Adam(learning_rate)
for step in range(total_iter):

    # shuffle data every time start a new set of minibatches
    if step in range(0, total_iter, num_minibatch):
        random.shuffle(indices_train)

    # find out data indices for a minibatch
    minibatch_idx = step % num_minibatch  # minibatch index
    indices_mb = indices_train[minibatch_idx*size_minibatch:(minibatch_idx+1)*size_minibatch]
    # halve image size so this can be reasonably tested, e.g. on a CPU
    input_mb = DataFeeder.load_images_train(indices_mb)
    label_mb = DataFeeder.load_labels_train(indices_mb)
    # update the variables
    loss_train = train_step(residual_unet, var_list, optimizer, input_mb, label_mb)

    # print training information
    step1 = step+1  # for display and count
    if (step1 % freq_print) == 0:
        tf.print('Step', step1, 'loss:', loss_train)

    # --- testing during training (no validation labels available)
    if (step1 % freq_test) == 0:
        indices_test = [random.randrange(30) for i in range(size_minibatch)]  # select size_minibatch test data
        input_test = DataFeeder.load_images_test(indices_test)
        pred_test = residual_unet(input_test)
        for idx in range(size_minibatch):
            filepath_to_save = os.path.join(RESULT_PATH, "label_test%02d_step%06d-tf.npy" % (indices_test[idx], step1))
            np.save(filepath_to_save, tf.squeeze(pred_test[idx, ...]))
            tf.print('Test data saved: {}'.format(filepath_to_save))

print('Training done.')


## this tutorial does not save residule_net, see e.g. https://www.tensorflow.org/api_docs/python/tf/saved_model/
