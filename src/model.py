import torch
from torch import nn


from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained("efficientnet-b3")

model._fc = nn.Linear(in_features=1536, out_features=13, bias=True)
