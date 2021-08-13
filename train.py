import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

from config import *
from utils import *
from dataset import *

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    model.train()

    for b, (imgs, labels) in enumerate(loop):
        imgs = imgs.to(device)
        labels = labels.to(device)

        def forward_pass():
            scores = model(imgs)
            loss = loss_fn(scores, labels)
            return loss

        if device == 'cuda':
            with torch.cuda.amp.autocast():
                loss = forward_pass()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = forward_pass()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def save_model_features(loader, model, output_size):
    model.eval()

    images = []
    labels = []

    for b, (x, y) in enumerate(tqdm(loader)):
        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size)

        images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        labels.append(y.numpy())

    np.save('featutes.npy', np.concatenate(images, axis=0))
    np.save('labels.npy', np.concatenate(labels, axis=0))

    model.train()

def main():
    model = EfficientNet.from_pretrained('effieientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 1)

    train_dataset = CatDog('data/train', transform=basic_transform)
    test_dataset = CatDog('data/test', transform=basic_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.to(device)

    for e in range(num_epochs):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        check_accuracy(loader, model, loss_fn)

    if save_model:
        checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=checkpoint_file)

    #sav
if __name__ == '__main__':
    main()
