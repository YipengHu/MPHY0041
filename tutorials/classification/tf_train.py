# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
import os
import random

import tensorflow as tf
import h5py


RESULT_PATH = 'result'

os.environ["CUDA_VISIBLE_DEVICES"]="0"
filename = './data/fetal.h5'
h5_file =  h5py.File(filename,'r')
frame_size = h5_file['/frame_size']
frame_size = [frame_size[0][0],frame_size[1][0]]
num_classes = h5_file['/num_classes'][0][0]

## build the network layers
features_input = tf.keras.Input(shape=frame_size+[1])
features = tf.keras.layers.Conv2D(32, 7, activation='relu')(features_input)

features = tf.keras.layers.MaxPool2D(3)(features)
features_block_1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(features)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_1)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features_block_2 = features + features_block_1
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features_block_2)
features = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(features)
features = features + features_block_2

features = tf.keras.layers.MaxPool2D(3)(features)
features_block_3 = tf.keras.layers.Conv2D(128, 3, activation='relu')(features)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_3)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_4 = features + features_block_3
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_4)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_5 = features + features_block_4
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_5)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features_block_6 = features + features_block_5
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features_block_6)
features = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(features)
features = features + features_block_6

features = tf.keras.layers.Conv2D(128, 3, activation='relu')(features)
features = tf.keras.layers.GlobalAveragePooling2D()(features)
features = tf.keras.layers.Dense(units=256, activation='relu')(features)
features = tf.keras.layers.Dropout(0.5)(features)
logits_output = tf.keras.layers.Dense(units=num_classes, activation='softmax')(features)

## compile the model
model = tf.keras.Model(inputs=features_input, outputs=logits_output)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['SparseCategoricalAccuracy'])


## data loader using a generator
num_subjects = h5_file['/num_subjects'][0][0]
subject_indices = range(num_subjects)
num_frames_per_subject = 1
def data_generator():
    for iSbj in subject_indices:
        dataset = '/subject%06d_num_frames' % iSbj
        num_frames = h5_file[dataset][0][0]
        idx_frame = random.sample(range(num_frames),num_frames_per_subject)[0]
        dataset = '/subject%06d_frame%08d' % (iSbj, idx_frame)
        frame = tf.transpose(h5_file[dataset][()].astype('float32')) / 255
        dataset = '/subject%06d_label%08d' % (iSbj, idx_frame)
        label = h5_file[dataset][0][0].astype('int32')
        yield (tf.expand_dims(frame, axis=2), label)

dataset = tf.data.Dataset.from_generator(generator = data_generator, 
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size+[1], ()))


## training
dataset_batch = dataset.shuffle(buffer_size=1024).batch(32)
model.fit(dataset_batch, epochs=int(1e3))
print('Training done.')

## save trained model
model.save(os.path.join(RESULT_PATH,'saved_model_tf'))  # https://www.tensorflow.org/guide/keras/save_and_serialize
print('Model saved.')
