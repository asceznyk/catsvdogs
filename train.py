import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet

from config import *
from utils import *
from dataset import *

def train():
    model = EfficientNet.from_pretrained('effieientnet-b0')
    model._fc = nn.Linear(2560, 1)

    train_dataset = CatDog('data/train', transform=basic_transform)
    test_dataset = CatDog('data/test', transform=basic_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory)


