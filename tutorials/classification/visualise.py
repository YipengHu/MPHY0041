# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
# run data.py before visualise the image data
import random

import h5py
from matplotlib import pyplot as plt


filename = './data/ultrasound_50frames.h5'
h5_file = h5py.File(filename, 'r')

# inspect nFrm frames from nSbj subjects
nSbj = 6
nFrm = 8

# sample random subjects
num_subjects = h5_file['/num_subjects'][0][0]
idx_subject = random.sample(range(num_subjects),nSbj)

plt.figure(figsize=[19.2,10.8])
for iSbj in range(nSbj):
    dataset = '/subject%06d_num_frames' % (idx_subject[iSbj])
    num_frames = h5_file[dataset][0][0]
    idx_frame = random.sample(range(num_frames),nFrm)
    for iFrm in range(nFrm):
        dataset = '/subject%06d_frame%08d' % (idx_subject[iSbj], idx_frame[iFrm])
        frame = h5_file[dataset][()]
        dataset = '/subject%06d_label%08d' % (idx_subject[iSbj], idx_frame[iFrm])
        label = h5_file[dataset][0][0]
        # plot in one of the subplots
        axs = plt.subplot(nSbj, nFrm, iSbj*nFrm+iFrm+1)
        axs.set_title('S{}, F{}, C{}'.format(idx_subject[iSbj], idx_frame[iFrm], label))
        axs.imshow(frame, cmap='gray')
        axs.axis('off')
# plt.show()
plt.savefig('visualise.jpg',bbox_inches='tight')
