"""
Returns ImageTransformNet

Using two types - different types of padding to compare results

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


if __name__ == "__main__":
    padding_tfnet = RTST_ImgTfNetPadding()
    no_padding_tfnet = RTST_ImgTfNet_NoPadding()
    test_x = torch.rand(1, 3, 256, 256)
    res1 = padding_tfnet(test_x)
    res2 = no_padding_tfnet(test_x)