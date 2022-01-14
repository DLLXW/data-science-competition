import os
from albumentations.core.composition import OneOf
import cv2
import time
import copy
from numpy.lib.type_check import imag
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset,DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import albumentations as A
from albumentations.pytorch import ToTensorV2
train_transform = A.Compose([
    # reszie
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
        A.Transpose(p=0.5)
    ]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1023.0, p=1.0),
    ToTensorV2(p=1.0),
],p=1.0)
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

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')
        image = cv2.imread(img_file[0], cv2.IMREAD_UNCHANGED)#读取16位
        try:
            _=image.shape
        except:
            idx = self.ids[i-1]
            img_file = glob(self.imgs_dir + idx + '.*')
            image = cv2.imread(img_file[0], cv2.IMREAD_UNCHANGED)#读取16位
        #
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)-1


        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        return {
            'image':image,
            'label':mask.to(torch.int64)
        }

if __name__ == '__main__':
    data_dir = "../../data/"
    train_imgs_dir = os.path.join(data_dir, "train_images/")
    train_labels_dir = os.path.join(data_dir, "train_labels/")
    #
    train_data = RSCDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
    for batch_idx, batch_samples in enumerate(train_loader):
        image, target = batch_samples['image'], batch_samples['label']
        print(image.shape)
