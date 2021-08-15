import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

import config

def init_model():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 1)

    model.to(device)

    return model, train_loader, test_loader

