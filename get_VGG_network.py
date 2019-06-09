"""
To do: Add content layers

Forward pass through network returns style losses as namedtuple output

Freezing the network can be done outside by doing with torch.no_grad() and then using .eval()

The class as of now simply returns trainable, pre_trained model of specific style_layers



"""
import collections

import torch
from torch import nn
import torchvision as tv


class VGGNetwork(nn.Module):
    def __init__(self, list_of_style_layers = ("relu1_2", "relu2_2", "relu3_3", "relu4_3")):
        super(VGGNetwork,self).__init__()

        self.vgg_model = tv.models.vgg16(pretrained=True)

        if torch.cuda.is_available():
            self.vgg_model.cuda()

        self.name_to_layer_dict = dict(zip(["3","8","5","12"],list_of_style_layers))
        self.loss_output_dict = dict.fromkeys(list_of_style_layers)

    def forward(self, x):
        """
        https: // discuss.pytorch.org / t / how - to - extract - features - of - an - image -from-a - trained - model / 119 / 13
        """
        for name, module in self.vgg_model.features._modules.items():
            x = module(x)
            if name in self.name_to_layer_dict:
                self.loss_output_dict[self.name_to_layer_dict[name]] = x

        LossOutput = collections.namedtuple("LossOutput", " ".join(self.name_to_layer_dict.values()))
        vgg_style_loss = LossOutput(**self.loss_output_dict)

        return vgg_style_loss


if __name__ == '__main__':
    test_x = torch.rand(1,3,256,256)
    vgg_network = VGGNetwork()
    losses = vgg_network(test_x)
