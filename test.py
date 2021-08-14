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
from features import *

model = EfficientNet.from_pretrained('efficientnet-b0')

train_dataset = CatDog('/content/data/train', transform=basic_transform)

loader = DataLoader(train_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers,
                         pin_memory=pin_memory)

batch = next(iter(loader))

model.to(device)

if load_model and os.path.exists(checkpoint_file):
    load_checkpoint(torch.loaded(checkpoint_file), model)
    print('model and successfully loaded from checkpoint!')

save_model_features([batch], model, output_size=(1,1))


