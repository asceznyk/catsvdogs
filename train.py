import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

from config import *
from utils import *
from dataset import *

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train()

    loop = tqdm(loader)
    for b, (imgs, labels) in enumerate(loop):
        imgs = imgs.to(device)
        labels = labels.unsqueeze(dim=1).to(device)

        with torch.cuda.amp.autocast():
            scores = model(imgs)
            loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def main():
    model, train_loader, test_loader = init_data_model()

    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if load_model and os.path.exists(checkpoint_file):
        load_checkpoint(torch.load(checkpoint_file), model)
        print('model and optimizer successfully loaded from checkpoint!')

    for e in range(num_epochs):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        check_accuracy(loader, model, loss_fn)

    if save_model:
        checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=checkpoint_file)

if __name__ == '__main__':
    main()
