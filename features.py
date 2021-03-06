import os
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

from config import *
from model import *
from utils import *
from dataset import *

def save_feature_vectors(loader, model, output_size=(1, 1), split='train'):
    model.eval()
    images, labels = [], []

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=output_size)
        images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

        if idx % 100 == 0 or idx == len(loader)-1:
            print('sleeping for 4 seconds.. reducing RAM load...')
            time.sleep(4)

            xpath = f'/content/{split}features.npy'
            ypath = f'/content/{split}labels.npy'

            if os.path.exists(xpath):
                xs = np.load(xpath)
                ys = np.load(ypath)
                xs = np.append(xs, np.concatenate(images, axis=0), axis=0)
                ys = np.append(ys, np.concatenate(labels, axis=0), axis=0)
            else:
                xs = np.concatenate(images, axis=0)
                ys = np.concatenate(labels, axis=0)

            np.save(xpath, xs)
            np.save(ypath, ys)
            images, labels = [], []

    model.train()

def main():
    model = init_model()
    train_loader = init_loader('/content/data/train', True, pin_memory)
    test_loader = init_loader('/content/data/test', False, not pin_memory)

    load_checkpoint(torch.load(checkpoint_file), model)

    save_feature_vectors(train_loader, model, output_size=(1,1))
    save_feature_vectors(test_loader, model, output_size=(1,1), split='test')

if __name__ == '__main__':
    main()
