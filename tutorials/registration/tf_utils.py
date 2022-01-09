# *** This code is with an Apache 2.0 license, University College London ***
# The 2D registration network is a modified U-Net: https://www.tensorflow.org/tutorials/images/segmentation
# Part of the network is adapted from TensorFlow Examples (https://github.com/tensorflow/examples), which needs to be installed first.

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

### network and layers
class UNet(tf.keras.Model):
    def __init__(self, out_channels, num_channels_initial):
        super(UNet, self).__init__()
        # the encoder/downsampler is a series of downsample blocks implemented in TensorFlow examples.
        self.down_stack = [
            pix2pix.downsample(num_channels_initial, 3, norm_type='instancenorm'),  
            pix2pix.downsample(num_channels_initial*2, 3, norm_type='instancenorm'),
            pix2pix.downsample(num_channels_initial*4, 3, norm_type='instancenorm')
        ]
        # The decoder/upsampler is a series of upsample blocks implemented in TensorFlow examples.
        self.up_stack = [
            pix2pix.upsample(num_channels_initial*4, 3, norm_type='instancenorm'),
            pix2pix.upsample(num_channels_initial*2, 3, norm_type='instancenorm'), 
            pix2pix.upsample(num_channels_initial, 3, norm_type='instancenorm'),
        ]
        self.out_layer = tf.keras.layers.Conv2DTranspose(out_channels, 3, strides=2, padding='same', activation=None, use_bias=True)

    def call(self, inputs):
        x = inputs
        # Downsampling through the model
        skips = []
        for down in self.down_stack[:-1]:
            x = down(x)
            skips += [x]
        x = self.down_stack[-1](x)
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, reversed(skips)):
            x = up(x)
            x = tf.concat([x, skip],axis=3)  # concat = tf.keras.layers.Concatenate()
        return self.out_layer(x)

    def build(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))


### transformation utility functions
def get_reference_grid(grid_size):
    # grid_size: [batch_size, height, width]
    grid = tf.cast(tf.stack(tf.meshgrid(
                        tf.range(grid_size[1]),
                        tf.range(grid_size[2]),
                        indexing='ij'), axis=2), dtype=tf.float32)
    return tf.tile(tf.expand_dims(grid, axis=0), [grid_size[0],1,1,1])


def warp_images(images, ddfs):
    # images: [batch_size, height, width]
    # ddfs: [batch_size, height, width, 2]
    reference_grid = get_reference_grid(ddfs.shape[0:3])
    warped_grids = reference_grid + ddfs
    return bilinear_resampler(images, warped_grids)


def bilinear_resampler(grid_data, sample_grids):
    '''
    grid_data: [batch, height, width]
    sample_grids: [batch, height, width, 2]    
    '''
    batch_size, height, width = (grid_data.shape[:])
    sample_coords = tf.reshape(sample_grids, [batch_size,-1,2])
    # pad to replicate the boundaries 1-ceiling, 2-floor
    sample_coords = tf.stack([tf.clip_by_value(sample_coords[...,0],0,height-1),
                            tf.clip_by_value(sample_coords[...,1],0,width-1)], axis=2)
    i1 = tf.cast(tf.math.ceil(sample_coords[...,0]), dtype=tf.int32)
    j1 = tf.cast(tf.math.ceil(sample_coords[...,1]), dtype=tf.int32)
    i0 = tf.maximum(i1-1, 0)
    j0 = tf.maximum(j1-1, 0)
    # four data points q_ij
    q00 = tf.gather_nd(grid_data, tf.stack([i0,j0],axis=2), batch_dims=1)
    q01 = tf.gather_nd(grid_data, tf.stack([i0,j1],axis=2), batch_dims=1)
    q11 = tf.gather_nd(grid_data, tf.stack([i1,j1],axis=2), batch_dims=1)
    q10 = tf.gather_nd(grid_data, tf.stack([i1,j0],axis=2), batch_dims=1)    
    # weights with normalised local coordinates
    wi1 = sample_coords[...,0] - tf.cast(i0,dtype=tf.float32)
    wi0 = 1 - wi1
    wj1 = sample_coords[...,1] - tf.cast(j0,dtype=tf.float32)
    wj0 = 1 - wj1
    return tf.reshape(q00*wi0*wj0 + q01*wi0*wj1 + q11*wi1*wj1 + q10*wi1*wj0, [batch_size]+sample_grids.shape[1:3])



'''
def warp_grids(grid, transform):
    # grid: [batch, height, width, 2]
    # transform: [batch, 3, 3]
    batch_size, height, width = grid.shape[0:3]
    grid = tf.concat([tf.reshape(grid,[batch_size,height*width,2]), 
                    tf.ones([batch_size,height*width,1])], axis=2)
    grid_warped = tf.matmul(grid, transform)
    return tf.reshape(grid_warped[...,:2], [batch_size,height,width,2])


def random_transform_generator(batch_size, corner_scale=.1):
    # right-multiplication affine
    ori_corners = tf.tile([[[1.,1.], [1.,-1.], [-1.,1.], [-1.,-1.]]], [batch_size,1,1])
    new_corners = ori_corners + tf.random.uniform([batch_size,4,2], -corner_scale, corner_scale)    
    ori_corners = tf.concat([ori_corners,tf.ones([batch_size,4,1])], axis=2)
    new_corners = tf.concat([new_corners,tf.ones([batch_size,4,1])], axis=2)
    return tf.stack([tf.linalg.lstsq(ori_corners[n],new_corners[n]) for n in range(batch_size)], axis=0)


def random_image_transform(images):
    # images: [batch_size, height, width]
    reference_grid = get_reference_grid(images.shape[0:3])
    random_transform = random_transform_generator(images.shape[0], corner_scale=0.1)
    sample_grids = warp_grids(reference_grid, random_transform)
    return bilinear_resampler(images, sample_grids)
'''

### loss functions
def square_difference(i1, i2):
    return tf.reduce_mean(tf.square(i1 - i2), axis=[1, 2])  # use mean for normalised regulariser weighting


def gradient_dx(fv):
    return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2


def gradient_dy(fv):
    return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2


def gradient_txy(txy, fn):
    return tf.stack([fn(txy[..., i]) for i in [0, 1]], axis=3)


def gradient_norm(displacement, flag_l1=False):
    dtdx = gradient_txy(displacement, gradient_dx)
    dtdy = gradient_txy(displacement, gradient_dy)
    if flag_l1:
        norms = tf.abs(dtdx) + tf.abs(dtdy)
    else:
        norms = dtdx**2 + dtdy**2
    return tf.reduce_mean(norms, [1, 2, 3])


'''
def bending_energy(displacement):
    dtdx = gradient_txy(displacement, gradient_dx)
    dtdy = gradient_txy(displacement, gradient_dy)
    dtdxx = gradient_txy(dtdx, gradient_dx)
    dtdyy = gradient_txy(dtdy, gradient_dy)
    dtdxy = gradient_txy(dtdx, gradient_dy)
    return tf.reduce_mean(dtdxx**2 + dtdyy**2 + 2*dtdxy**2, [1, 2, 3])


def normalised_cross_correlation(ts, ps, eps=0.0):
    dp = ps - tf.reduce_mean(ps, axis=[1, 2, 3])
    dt = ts - tf.reduce_mean(ts, axis=[1, 2, 3])
    vp = tf.reduce_sum(tf.square(dp), axis=[1, 2, 3])
    vt = tf.reduce_sum(tf.square(dt), axis=[1, 2, 3])
    return tf.constant(1.0) - tf.reduce_sum(dp*dt / (tf.sqrt(vp*vt) + eps), axis=[1, 2, 3])


def normalised_cross_correlation2(ts, ps, eps=1e-6):
    mean_t = tf.reduce_mean(ts, axis=[1, 2, 3])
    mean_p = tf.reduce_mean(ps, axis=[1, 2, 3])
    std_t = tf.reduce_sum(tf.sqrt(tf.square(mean_t)-tf.reduce_mean(tf.square(ts), axis=[1, 2, 3])), axis=[1, 2, 3])
    std_p = tf.reduce_sum(tf.sqrt(tf.square(mean_p)-tf.reduce_mean(tf.square(ps), axis=[1, 2, 3])), axis=[1, 2, 3])
    return -tf.reduce_mean((ts-mean_t)*(ps-mean_p) / (std_t*std_p+eps), axis=[1, 2, 3])
'''
