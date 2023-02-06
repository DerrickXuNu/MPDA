import torch
import torch.nn.functional as F
from torch import nn
from opencood.models.da_modules.gradient_layer import GradientScalarLayer


class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.rgl = GradientScalarLayer(-9.1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # X : (b, c, h, w), b=cars_1 + cars_2 + ...
        x = self.rgl(x)
        x = F.relu(self.conv1_da(x))
        x = self.conv2_da(x)

        return x

