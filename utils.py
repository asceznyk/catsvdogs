import os
import pandas as pd
import numpy as np

import albumentations as A

from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import log_loss

from efficientnet_pytorch import EfficientNet

from config import *
from dataset import *

def init_data_model():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 1)

    train_dataset = CatDog('/content/data/train', transform=basic_transform)
    test_dataset = CatDog('/content/data/test', transform=basic_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory)

    model.to(device)

    return model, train_loader, test_loader

def check_accuracy(
    loader, model, loss_fn, input_shape=None, toggle_eval=True, print_accuracy=True
):
    """
    Check accuracy of model on data from loader
    """
    if toggle_eval:
        model.eval()
    device = next(model.parameters()).device
    num_correct = 0
    num_samples = 0

    y_preds = []
    y_true = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            if input_shape:
                x = x.reshape(x.shape[0], *input_shape)
            scores = model(x)
            predictions = torch.sigmoid(scores) > 0.5
            y_preds.append(torch.clip(torch.sigmoid(scores), 0.005, 0.995).cpu().numpy())
            y_true.append(y.cpu().numpy())
            num_correct += (predictions.squeeze(1) == y).sum()
            num_samples += predictions.size(0)

    accuracy = num_correct / num_samples

    if toggle_eval:
        model.train()

    if print_accuracy:
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(log_loss(np.concatenate(y_true, axis=0), np.concatenate(y_preds, axis=0)))

    return accuracy


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])



