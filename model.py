import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet

from config import *
from dataset import *

def init_data_model():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 1)

    train_dataset = CatDog('/content/data/train', transform=basic_transform)
    test_dataset = CatDog('/content/data/test', transform=basic_transform)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers
    )

    model.to(device)

    return model, train_loader, test_loader


