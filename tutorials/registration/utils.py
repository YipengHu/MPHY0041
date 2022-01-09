import os 

import numpy as np
import matplotlib.image as mpimg


def get_image_arrays():

    PATH_TO_TRAIN = 'data/datasets-hn2dct/train'
    PATH_TO_TEST = 'data/datasets-hn2dct/test'

    images = np.stack([mpimg.imread(os.path.join(PATH_TO_TRAIN, f)) for f in sorted(os.listdir(PATH_TO_TRAIN)) if f.endswith('.png')],axis=0)  # stack at dim=0 consistent with tf
    images = np.pad(images, [(0,0),(0,0),(0,1)])  # padding for an easier image size

    test_images = np.stack([mpimg.imread(os.path.join(PATH_TO_TEST, f)) for f in sorted(os.listdir(PATH_TO_TEST)) if (f.find('_')==-1 and f.endswith('.png'))],axis=0) 
    test_images = np.pad(test_images, [(0,0),(0,0),(0,1)])  # padding for an easier image size
    test_indices = [[0,0,1,1,2,2],[1,2,0,2,0,1]] # [moving,fixed]

    # image-wise normalisation 
    n = lambda im: (im-np.min(im,axis=(1,2),keepdims=True))/(np.max(im,axis=(1,2),keepdims=True)-np.min(im,axis=(1,2),keepdims=True))  
    return n(images), n(test_images), test_indices