import torch
import torch.nn as nn
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Dropout, BatchNormalization

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Reshape, LeakyReLU

def conv(chin, chout, kernel_size=5, stride=1):
    return nn.Sequential(nn.Conv2d(chin, chout, kernel_size=kernel_size, stride=stride),
                         nn.BatchNorm(2d(chout)),
                         nn.LeakyReLU(0.1, inplace=True))

def deconv(chin, chout, kernel_size=5, stride=1):
    return nn.Sequential(nn.ConvTranspose2d(chin, chout, kernel_size=kernel_size, stride=stride),
                         nn.LeakyReLU(0.1, inplace=True))

class DeepModel2(nn.Module):

        def __init__(self, imsize, batch_size):
            super(DeepModel2, self).__init__()
            self.conv1 = conv(2, 64, kernel_size=7)
            self.conv2 = conv(64, 128)
            self.conv3 = conv(128, 256)
            self.conv4 = conv(256, 512)
            self.conv5 = conv(512, 1024)
            self.deconv5 = deconv(1024, 512)
            self.deconv4 = deconv(512, 256)
            self.deconv3 = deconv(256, 128)
            self.deconv2 = deconv(128, 64)
            self.deconv1 = deconv(64, 2)

            for m in self.modules():
                if (isinstance(m, nn.Conv2d) | isinstance(m, nn.ConvTranspose2d)):
                    nn.init.uniform(m.bias)
                    nn.init.xavier_uniform(m.weight)

        def forward(self, batch):
            o = self.conv1(batch)
            o = self.conv2(o)
            o = self.conv3(o)
            o = self.conv4(o)
            o = self.conv5(o)
            o = self.deconv5(o)
            o = self.deconv4(o)
            o = self.deconv3(o)
            o = self.deconv2(o)
            o = self.deconv1(o)
            return o
