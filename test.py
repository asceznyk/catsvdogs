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
from train import *

model = EfficientNet.from_pretrained('efficientnet-b0')
print(model)

test_dataset = CatDog('/content/data/test', transform=basic_transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers,
                         pin_memory=pin_memory)

model.to(device)

save_model_features(train_loader, model, output_size=(1,1))

print('hallelujah!')
