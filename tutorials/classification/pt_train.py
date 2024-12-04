# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
import os
import random

import torch
import h5py


os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
filename = './data/fetal.h5'
RESULT_PATH = './result'

## build a vgg-16 network class
class VGGNet(torch.nn.Module):
    def __init__(self, n_in, n_out, n_config=[64, -64, 128, -128, 256, 256, -256, 512, 512, 512]):  # pooling when negative
        super(VGGNet, self).__init__()
        n_pre = n_in
        layers = []
        for _n in n_config:
            layers += self.conv2d_block(n_pre, abs(_n), _n<0)
            n_pre = abs(_n)
        # layers += [torch.nn.AvgPool2d(1)]
        self.feature_extract = torch.nn.Sequential(*layers)
        self.linear_output = torch.nn.Linear(n_config[-1], n_out)

    def forward(self, x):
        x = torch.mean(self.feature_extract(x),dim=[2,3],keepdim=False)
        return self.linear_output(x)
        
    def conv2d_block(self, ch_in, ch_out, post_pooling=False):
        block = [torch.nn.Conv2d(ch_in,ch_out,3,padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(ch_out)]
        if post_pooling:
            block += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        return block


## data loader
class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.h5_file = h5py.File(file_path, 'r')
        self.num_subjects = self.h5_file['/num_subjects'][0][0]
        self.num_frames = [self.h5_file['/subject%06d_num_frames' % idx][0][0] for idx in range(self.num_subjects)]
    
    def __len__(self):
        return self.num_subjects

    def __getitem__(self, index):
        idx_frame = random.randint(0, self.num_frames[index]-1)
        frame = torch.unsqueeze(torch.tensor(self.h5_file['/subject%06d_frame%08d' % (index, idx_frame)][()].astype('float32')), dim=0)
        label = torch.squeeze(torch.tensor(self.h5_file['/subject%06d_label%08d' % (index, idx_frame)][()].astype('int64')))
        return (frame, label)


## training
model = VGGNet(1,4)
if use_cuda:
    model.cuda()

train_set = H5Dataset(filename)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=8, 
    shuffle=True,
    num_workers=8)
'''
dataiter = iter(train_loader)
frames, labels = dataiter.next()
'''

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

freq_print = 10
for epoch in range(200):
    for step, (frames, labels) in enumerate(train_loader):
        if use_cuda:
            frames, labels = frames.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute and print loss
        if step % freq_print == (freq_print-1):    # print every 20 mini-batches
            print('[Epoch %d, iter %05d] loss: %.3f' % (epoch, step, loss.item()))
            moving_loss = 0.0

print('Training done.')


## save trained model
torch.save(model, os.path.join(RESULT_PATH,'saved_model_pt'))  # https://pytorch.org/tutorials/beginner/saving_loading_models.html
print('Model saved.')
