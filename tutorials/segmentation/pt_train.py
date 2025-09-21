# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
import os

import torch
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
folder_name = './data/promise12-data'
RESULT_PATH = './result'

## network class
class UNet(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):
        super(UNet, self).__init__()

        n_feat = init_n_feat
        self.encoder1 = UNet._block(ch_in, n_feat)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(n_feat, n_feat*2)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(n_feat*2, n_feat*4)
        self.pool3 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(n_feat*4, n_feat*8)
        self.pool4 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(n_feat*8, n_feat*16)

        self.upconv4 = torch.nn.ConvTranspose3d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((n_feat*8)*2, n_feat*8)
        self.upconv3 = torch.nn.ConvTranspose3d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((n_feat*4)*2, n_feat*4)
        self.upconv2 = torch.nn.ConvTranspose3d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((n_feat*2)*2, n_feat*2)
        self.upconv1 = torch.nn.ConvTranspose3d(n_feat*2, n_feat, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(n_feat*2, n_feat)

        self.conv = torch.nn.Conv3d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))


## loss function
def loss_dice(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    numerator = torch.sum(y_true*y_pred, dim=(2,3,4)) * 2
    denominator = torch.sum(y_true, dim=(2,3,4)) + torch.sum(y_pred, dim=(2,3,4)) + eps
    return torch.mean(1. - (numerator / denominator))


## data loader
class NPyDataset(torch.utils.data.Dataset):
    def __init__(self, folder_name, is_train=True):
        self.folder_name = folder_name
        self.is_train = is_train

    def __len__(self):
        return (50 if self.is_train else 30)

    def __getitem__(self, idx):
        if self.is_train:
            image = self._load_npy("image_train%02d.npy" % idx)
            label = self._load_npy("label_train%02d.npy" % idx)
            return image, label
        else:
            return self._load_npy("image_test%02d.npy" % idx), idx

    def _load_npy(self, filename):
        filename = os.path.join(self.folder_name, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename)[::2,::2,::2])),dim=0)


## training
model = UNet(1,1)  # input 1-channel 3d volume and output 1-channel segmentation (a probability map)
if use_cuda:
    model.cuda()

# training data loader
train_set = NPyDataset(folder_name)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    num_workers=4)
'''test
dataiter = iter(train_loader)
images, labels = dataiter.next()
preds = model(images)
'''

# test/validation data loader
test_set = NPyDataset(folder_name, is_train=False)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=4,
    shuffle=True,  # change to False for predefined test data
    num_workers=4)


# optimisation loop
freq_print = 100  # in steps
freq_test = 2000  # in steps
total_steps = int(2e5)
step = 0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
while step < total_steps:
    for ii, (images, labels) in enumerate(train_loader):
        step += 1
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_dice(preds, labels)
        loss.backward()
        optimizer.step()

        # Compute and print loss
        if (step % freq_print) == 0:    # print every freq_print mini-batches
            print('Step %d loss: %.5f' % (step,loss.item()))

        # --- testing during training (no validation labels available)
        if (step % freq_test) == 0:  
            images_test, id_test = next(iter(test_loader))  # test one mini-batch
            if use_cuda:
                images_test = images_test.cuda()
            preds_test = model(images_test)
            for idx, index in enumerate(id_test):
                filepath_to_save = os.path.join(RESULT_PATH,"label_test%02d_step%06d-pt.npy" % (index,step))
                np.save(filepath_to_save, preds_test.detach()[idx,...].cpu().numpy().squeeze())
                print('Test data saved: {}'.format(filepath_to_save))

print('Training done.')


## save trained model
torch.save(model, os.path.join(RESULT_PATH,'saved_model_pt'))  # https://pytorch.org/tutorials/beginner/saving_loading_models.html
print('Model saved.')
