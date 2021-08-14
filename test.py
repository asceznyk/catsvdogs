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

train_dataset = CatDog('/content/data/train', transform=basic_transform)

loader = DataLoader(train_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers,
                         pin_memory=pin_memory)

batch = next(iter(loader))

model.to(device)

save_model_features([batch], model, output_size=(1,1))

print('hallelujah!')


