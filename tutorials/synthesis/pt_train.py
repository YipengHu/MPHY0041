# This scripts use an example of DCGAN described in a TensorFlow tutorial to simulate ultrasound images: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html.
import os

import torch
import torch.nn as nn

import utils


os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
filename = 'data/fetusphan.h5'
RESULT_PATH = './result'

## networks
nc = 1    #number of input image channels
nz = 100  #size of noise
ngf = 64  #number of feature maps in generator
ndf = 64  #number of feature maps in discriminator

# weight initialisation
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_layer = nn.Linear(nz, 20*15*ngf*4, bias=True) #due to frame size
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh() 
            )
    def forward(self, input):
        input_layer = torch.reshape(self.input_layer(input), (-1,ngf*4,20,15)) #due to frame size
        return self.main(input_layer)

netG = Generator()
if use_cuda:
    netG.cuda()
netG.apply(weights_init)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, input):
        return torch.mean(self.main(input),dim=(1,2,3))

netD = Discriminator()
if use_cuda:
    netD.cuda()
netD.apply(weights_init)


## losses and optimisers
lr = 0.0002
beta1 = 0.5

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


## Training
batch_size = 16
num_epochs = 50
num_examples_to_generate = 64
fixed_noise = torch.randn(num_examples_to_generate, nz)
frame_iterator = utils.H5FrameIterator(filename, batch_size)
if use_cuda:
    fixed_noise = fixed_noise.cuda()

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for frames in frame_iterator:
        frames = torch.unsqueeze(torch.tensor(frames),dim=1)
        label = torch.full((batch_size,), 1., dtype=torch.float)
        noise = torch.randn(batch_size, nz)
        
        if use_cuda:
            frames, noise, label = frames.cuda(), noise.cuda(), label.cuda()

        # update discriminator
        netD.zero_grad()
        output = netD(frames)
        errD_real = criterion(output, label) 
        errD_real.backward()
        fake = netG(noise)
        output = netD(fake.detach())
        errD_fake = criterion(output, label.fill_(0.)) 
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        # update generator
        netG.zero_grad()
        output = netD(fake)
        errG = criterion(output, label.fill_(1.)) #real
        errG.backward()
        optimizerG.step()

    # print every epoch
    print ('Epoch {}: g-loss={:0.5f}, d-loss={:0.5f}'.format(epoch+1,errG.item(),errD.item()))

    if (epoch+1) % 10 == 0:  # test every 10 epochs
        with torch.no_grad():
            predictions = netG(fixed_noise).detach().cpu().numpy()
            utils.save_images(predictions, os.path.join(RESULT_PATH,'images{:04d}-pt'.format(epoch+1)))
            print('Test images saved.')

print('Training done.')


## save trained model
torch.save(netG, os.path.join(RESULT_PATH,'saved_generator_pt'))  # https://pytorch.org/tutorials/beginner/saving_loading_models.html
print('Generator saved.')
