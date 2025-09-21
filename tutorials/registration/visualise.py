# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
# run either or both train_*.py before visualise the results
import os

import numpy as np
import matplotlib.pyplot as plt

from utils import get_image_arrays


PATH_TO_RESULT = 'result'
_, test_images, test_indices = get_image_arrays()

# plot example slices of registration results
for ext in ["-tf.npy","-pt.npy"]:  # find all npy files
    files = [f for f in os.listdir(PATH_TO_RESULT) if f.endswith(ext)]
    if len(files)==0: continue
    pre_images = np.load(os.path.join(PATH_TO_RESULT,max(files)))  # find the maximum step

    for ii in range(pre_images.shape[0]):
        plt.figure()
        axs = plt.subplot(1, 3, 1)
        axs.set_title('moving')
        axs.imshow(test_images[test_indices[0][ii],...], cmap='gray')
        axs.axis('off')

        axs = plt.subplot(1, 3, 2)
        axs.set_title('registered')
        axs.imshow(pre_images[ii,...], cmap='gray')
        axs.axis('off')

        axs = plt.subplot(1, 3, 3)
        axs.set_title('fixed')
        axs.imshow(test_images[test_indices[1][ii],...], cmap='gray')
        axs.axis('off')

        # plt.show()
        plt.savefig(os.path.join(PATH_TO_RESULT, '{}-{}.jpg'.format(max(files).split('.')[0],ii)),bbox_inches='tight')
        plt.close()

print('Plots saved: {}'.format(os.path.abspath(PATH_TO_RESULT)))
