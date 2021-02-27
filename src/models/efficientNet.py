import torch
from torch import nn

from efficientnet_pytorch import EfficientNet as efn


class EfficientNet(nn.Module):
    def __init__(self, num_classes, model_name):
        super(EfficientNet, self).__init__()

        self.model = efn.from_pretrained(model_name)

        self.model._fc = nn.Linear(in_features=1536, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)