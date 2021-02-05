import os
import cv2
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .transform import train_transform
from .transform import val_transform

class RSCDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            img_trans = img_nd.transpose(2, 0, 1)
        except:
            print(img_nd.shape)
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')

        image = cv2.imread(img_file[0], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)

        #if self.transform:
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        return {
            'image': image,
            'label': mask.long()
        }

if __name__ == '__main__':
    data_dir = "/media/limzero/qyl/mmsegmentation/data/satellite_jpg/"
    train_imgs_dir = os.path.join(data_dir, "img_dir/train/")
    train_labels_dir = os.path.join(data_dir, "ann_dir/train/")

    train_data = RSCDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
    train_data[500]
