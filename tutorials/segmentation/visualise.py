# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
# run train.py before visualise the results
import os

import numpy as np
import matplotlib.pyplot as plt


path_to_data = './data/datasets-promise12'
path_to_save = './result' 

# to plot example slices of segmentation results
for ext in ["-tf.npy","-pt.npy"]:  # find all npy files
    files = [f for f in os.listdir(path_to_save) if f.endswith(ext)]  
    fmax = []  # find the maximum step
    for test_id in set([f.split('_')[1] for f in files]):
        fmax += [max([f for f in files if f.split('_')[1]==test_id])]
    for f in fmax:
        label = np.load(os.path.join(path_to_save, f))
        image = np.load(os.path.join(path_to_data, "image_"+f.split('_')[1]+".npy"))[::2, ::2, ::2]  # change this per loader
        slices = range(0,label.shape[0],3)  # we only display a subset of data
        montage = np.concatenate([np.concatenate([image[i,...] for i in slices],axis=0),
                                  np.concatenate([label[i,...]*np.max(image) for i in slices],axis=0)], axis=1)
        plt.figure()
        plt.imshow(montage, cmap='gray')
        plt.axis('off')
        plt.title(f.split('.')[0])
        # plt.show()
        plt.savefig(os.path.join(path_to_save, f.split('.')[0]+'.jpg'),bbox_inches='tight')
        plt.close()
print('Plots saved: {}'.format(path_to_save))
