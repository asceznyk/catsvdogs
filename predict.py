import pickle
import numpy as np

from PIL import Image, ImageFont, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet

from config import *
from utils import *
from dataset import *

def plt_images_labels(imgs, preds):
    for i, (img, pred) in enumerate(zip(imgs, preds)):
        img = np.transpose(img, (1,2,0))
        text = 'cat' if pred < 0.5 else 'dog'

        for c in range(3):
            img[:,:,c] = img[:,:,c] * basic_stds[c] + basic_means[c]

        pil_img = Image.fromarray((img * max_pixel_val).astype('uint8'))
        img_edit = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(None, 200)
        img_edit.rectangle((0,0, 40, 10), fill='black')
        img_edit.text((0,0), text, (0,255,0), font)
        pil_img.save(f'pred{i}.png')

    print('all images have been saved as .png files, you can check the predction at the top-left hand corner..')

def predict(num_batches=4):
    clf = pickle.load(open('clf.log.regressor', 'rb'))

    model, _, test_loader = init_data_model()
    load_checkpoint(torch.load(checkpoint_file), model)
    model.eval()

    imgs, preds = [], []

    for b, (x, _) in enumerate(test_loader):
        x = x.to(device)

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=(1,1))

        features = features.reshape(x.shape[0], -1).detach().cpu().numpy()
        probs = clf.predict_proba(features)[:, 1]
        imgs.append(x.detach().cpu().numpy())
        preds.append(probs)

        if b >= num_batches:
            imgs = np.concatenate(imgs, axis=0)
            preds = np.concatenate(preds, axis=0)
            plt_images_labels(imgs, preds)
            break

if __name__ == '__main__':
    predict()
