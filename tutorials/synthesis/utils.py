import random
import os

import h5py
import numpy as np

class H5FrameIterator():
    def __init__(self, filename, batch_size):
        self.h5_file = h5py.File(filename,'r')
        self.num_frames = len(self.h5_file)
        self.batch_size = batch_size
        self.num_batches = int(self.num_frames/self.batch_size) # skip the remainders        
        self.frame_ids = [i for i in range(self.num_frames)]        
    
    def __iter__(self):
        self.batch_idx = 0
        random.shuffle(self.frame_ids)
        return self
    
    def __next__(self):
        self.batch_idx += 1
        batch_frame_ids = self.frame_ids[self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size]
        datasets = ['/frame%06d' % idx for idx in batch_frame_ids]
        if self.batch_idx>=self.num_batches:
            raise StopIteration
        frames = np.stack([self.h5_file[ds][()] for ds in datasets], axis=0).astype(np.float32)
        return frames / frames.max(axis=(1,2),keepdims=True)  # normalisation for unsigned data type


def save_images(images, filename):
    np.save(filename, np.squeeze(images))
