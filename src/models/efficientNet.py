import torch
from torch import nn

from efficientnet_pytorch import EfficientNet as efn

class EfficientNet(nn.Module):
    def __init__(self, num_classes, model_name):
        super(EfficientNet, self).__init__()

        self.model = efn.from_pretrained(model_name, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)