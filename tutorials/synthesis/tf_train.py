# This scripts use an example of DCGAN described in a TensorFlow tutorial to simulate ultrasound images: https://www.tensorflow.org/tutorials/generative/dcgan.
import os

import tensorflow as tf
from tensorflow.keras import layers

import utils


os.environ["CUDA_VISIBLE_DEVICES"]="0"
filename = 'data/fetusphan.h5'
RESULT_PATH = './result'


## networks
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(20*15*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((20, 15, 256)))
    assert model.output_shape == (None, 20, 15, 256) # Note: None is the batch size
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 20, 15, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 40, 30, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 80, 60, 1)
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[80, 60, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()


## losses and optimisers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


## train
num_epochs = 50
batch_size = 16
noise_dim = 100
num_examples_to_generate = 64
seed = tf.random.normal([num_examples_to_generate, noise_dim])
frame_iterator = utils.H5FrameIterator(filename, batch_size)

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


## the train loop
for epoch in range(num_epochs):
    for frames in frame_iterator:

        gen_loss_train, disc_loss_train = train_step(tf.expand_dims(frames,axis=3))

    # print every epoch
    print ('Epoch {}: g-loss={:0.5f}, d-loss={:0.5f}'.format(epoch+1,gen_loss_train,disc_loss_train))

    if (epoch + 1) % 10 == 0:  # test every 10 epochs
        predictions = generator(seed, training=False)
        utils.save_images(predictions, os.path.join(RESULT_PATH,'images{:04d}-tf'.format(epoch+1)))
        print('Test images saved.')

print('Training done.')


## save trained model
generator.save(os.path.join(RESULT_PATH,'saved_generator_tf'))  # https://www.tensorflow.org/guide/keras/save_and_serialize
print('Generator saved.')
