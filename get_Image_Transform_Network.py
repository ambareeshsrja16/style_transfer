"""
Returns ImageTransformNet

Using three types - different types of padding, and instance normalization to compare results

"""


import logging
import torch
from torch import nn
from torch.nn import functional as F


class RTST_ImgTfNetPadding(nn.Module):
    def __init__(self, log_level=logging.INFO):
        super(RTST_ImgTfNetPadding, self).__init__()

        self.log_level = log_level

        self.conv = nn.ModuleList()
        self.resblock = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.conv.append(nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4))
        self.bn.append(nn.BatchNorm2d(32))

        self.conv.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        self.bn.append(nn.BatchNorm2d(64))

        self.conv.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.bn.append(nn.BatchNorm2d(128))

        self.conv.append(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1))
        self.bn.append(nn.BatchNorm2d(64))

        self.conv.append(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1))
        self.bn.append(nn.BatchNorm2d(32))

        self.conv.append(nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1, padding=4))

        self.resblock.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.resblock.append(nn.BatchNorm2d(128))
        self.resblock.append(nn.ReLU())
        self.resblock.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.resblock.append(nn.BatchNorm2d(128))

    def forward(self, x):
        logger = logging.getLogger()
        logger.setLevel(self.log_level)

        out_sizes = [x.shape]
        logging.debug("Convolution")

        for i in range(3):
            x = F.relu(self.bn[i](self.conv[i](x)))
            out_sizes.append(x.shape)
        logging.debug(x.shape)
        out_sizes = out_sizes[::-1]
        h = x
        logging.debug("Res Blocks")
        for i in range(5):
            for j in range(len(self.resblock)):
                x = self.resblock[j](x)
            x = x + h
            h = x
        logging.debug(x.shape)

        logging.debug("Transpose Conv")
        for i in range(3, 5):
            x = F.relu(self.bn[i](self.conv[i](x, output_size=out_sizes[i - 2])))
        logging.debug(x.shape)

        logging.debug("Last Step")
        x = self.conv[-1](x, output_size=out_sizes[-1])
        logging.debug(x.shape)
        return x


class RTST_ImgTfNet_NoPadding(nn.Module):
    def __init__(self, log_level=logging.ERROR):
        super(RTST_ImgTfNet_NoPadding, self).__init__()

        self.log_level  = log_level

        self.conv = nn.ModuleList()
        self.resblock = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.conv.append(nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=0))
        self.bn.append(nn.BatchNorm2d(32))

        self.conv.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0))
        self.bn.append(nn.BatchNorm2d(64))

        self.conv.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0))
        self.bn.append(nn.BatchNorm2d(128))

        self.conv.append(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0))
        self.bn.append(nn.BatchNorm2d(64))

        self.conv.append(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0))
        self.bn.append(nn.BatchNorm2d(32))

        self.conv.append(nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1, padding=0))

        self.resblock.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.resblock.append(nn.BatchNorm2d(128))
        self.resblock.append(nn.ReLU())
        self.resblock.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.resblock.append(nn.BatchNorm2d(128))

    def forward(self, x):
        logger = logging.getLogger()
        logger.setLevel(self.log_level)

        out_sizes = [x.shape]
        logging.debug("Convolution")

        for i in range(3):
            x = F.relu(self.bn[i](self.conv[i](x)))
            out_sizes.append(x.shape)
        logging.debug(x.shape)
        out_sizes = out_sizes[::-1]
        h = x
        logging.debug("Res Blocks")
        for i in range(5):
            for j in range(len(self.resblock)):
                x = self.resblock[j](x)
            x = x + h
            h = x
        logging.debug(x.shape)

        logging.debug("Transpose Conv")
        for i in range(3, 5):
            x = F.relu(self.bn[i](self.conv[i](x, output_size=out_sizes[i - 2])))
        logging.debug(x.shape)

        logging.debug("Last Step")
        x = self.conv[-1](x, output_size=out_sizes[-1])
        logging.debug(x.shape)
        return x

class RTST_ImgTfNet_InstanceNorm(nn.Module):
    def __init__(self):
        super(RTST_ImgTfNet_InstanceNorm, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(
            128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(
            64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(
                mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

if __name__ == "__main__":
    padding_tfnet = RTST_ImgTfNetPadding()
    no_padding_tfnet = RTST_ImgTfNet_NoPadding()
    test_x = torch.rand(1, 3, 256, 256)
    res1 = padding_tfnet(test_x)
    res2 = no_padding_tfnet(test_x)