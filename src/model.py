from models.efficientNet import EfficientNet
from torch import nn

class MainModel(nn.Module):
    def __init__(self, num_classes, model_name):
        super(MainModel, self).__init__()

        self.model = EfficientNet(num_classes, model_name)

    def forward(self, x):
        return self.model(x)